"""Per-phone conversation history.

The N-ATLaS server is stateless per request: it does not remember the previous
turn, because the client sends the history every time. For a laptop that is
convenient. For a WhatsApp bot it is not — there is no terminal holding a
conversation object — so history has to live somewhere durable here.

What is stored:

    phone       normalised E.164, the primary key
    history     a JSON list of {role, content}, user and assistant only
    language    the pinned reply language, '' for auto
    updated_at  unix seconds, for TTL

What is never stored: the N-ATLaS API key, the Meta access token, the verify
token, anything from .env. `to_dict()` on a row returns messages only, so there
is no path by which a secret reaches this table.

Two backends, one interface:

    sqlite     stdlib only, one local file. Fine for one process on one host.
    supabase   the Postgres you already run for nafdac_source. Makes the bot
               fully stateless, so it can be redeployed, scaled out, or moved
               between hosts without losing anyone's conversation.

Pick `supabase` when there is any chance of running more than one instance or
restarts on ephemeral disk; pick `sqlite` for local development.

The same SQLite file holds a second table, `wa_seen`, which records inbound
message ids. Meta retries a webhook that does not answer 200 within twenty
seconds, so the same user message can arrive two or three times; without this
table a retry produces duplicate replies and duplicate billed messages.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, normalize_phone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wa_sessions (
    phone       TEXT PRIMARY KEY,
    history     TEXT NOT NULL DEFAULT '[]',
    language    TEXT NOT NULL DEFAULT '',
    updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wa_seen (
    message_id  TEXT PRIMARY KEY,
    seen_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS wa_sessions_updated ON wa_sessions(updated_at);
CREATE INDEX IF NOT EXISTS wa_seen_at ON wa_seen(seen_at);
"""


@dataclass
class Session:
    """One phone number's state."""

    phone: str
    history: list[dict] = field(default_factory=list)
    language: str = ""  # '' means auto-detect; 'en'/'ha'/'yo'/'ig' pins it

    def messages(self, cfg: Config) -> list[dict]:
        """History trimmed to what the model should see for the next turn.

        The server has a finite context budget and its own hard message-count
        limit, so an untrimmed history eventually fails every request. Trimming
        drops the oldest turns first and always keeps the newest, because the
        newest turn is the one being answered.

        Only user and assistant turns are ever returned. No system message is
        ever added here: the server drops client system messages anyway, and
        the persona is its own.
        """
        kept = [m for m in self.history if m.get("role") in ("user", "assistant")]
        kept = kept[-cfg.max_history_messages :]

        budget = cfg.max_history_tokens
        while kept and estimate_tokens(kept) > budget:
            # Drop from the front, but never the last turn: the message about
            # to be asked has not been appended yet, so the tail here is the
            # context for it and losing it loses the subject of the question.
            kept = kept[1:]
        return [dict(m) for m in kept]


def estimate_tokens(messages: list[dict]) -> int:
    """The same cheap heuristic n_atlas_client/conversation.py uses.

    Deliberately not a tokenizer. This runs on the WhatsApp host, which has no
    reason to load one, and the job is only to notice when a history has grown
    past the budget — an estimate that is within 20% is far more than enough
    for that.
    """
    count = 0
    for message in messages or []:
        text = str(message.get("content") or "")
        for word in text.split():
            count += 1 + int(len(word) / 4.0)
    return count


class SessionStore:
    """Interface both backends implement."""

    async def get(self, phone: str) -> Session:
        raise NotImplementedError

    async def save(self, session: Session) -> None:
        raise NotImplementedError

    async def clear(self, phone: str) -> None:
        raise NotImplementedError

    async def mark_seen(self, message_id: str, ttl_seconds: int) -> bool:
        """True when this id is new. False when it is a Meta retry."""
        raise NotImplementedError

    async def sweep(self, ttl_seconds: int) -> int:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class SqliteSessionStore(SessionStore):
    """Local SQLite, one file, stdlib only.

    Every statement runs in a worker thread so a slow disk write cannot stall
    the webhook's event loop. At WhatsApp traffic volumes each call is a
    fraction of a millisecond, but the cost of being wrong here is a stalled
    event loop, and the fix is three lines.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.sqlite_path)
        self._lock = asyncio.Lock()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # WAL lets the webhook read while a worker writes, which is the normal
        # access pattern here: the request path may need history and the
        # worker writes it back.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @contextmanager
    def _db(self):
        """A connection that is committed AND closed.

        `with sqlite3.connect(...) as conn` is a *transaction* context manager,
        not a resource one: it commits on success and leaves the file handle
        open. Every `_db()` call below creates a fresh connection, so without an
        explicit close the process leaks one handle per query — and on Windows
        the WAL files stay locked, which is what stops the database file from
        being deleted, moved or rotated.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    async def setup(self) -> None:
        await asyncio.to_thread(self._setup_sync)

    def _setup_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    def _run(self, fn, *args):
        async def runner():
            async with self._lock:
                return await asyncio.to_thread(fn, *args)

        return runner()

    # -- session rows ---------------------------------------------------
    def _get_sync(self, phone: str):
        with self._db() as conn:
            return conn.execute(
                "SELECT history, language FROM wa_sessions WHERE phone = ?",
                (phone,),
            ).fetchone()

    async def get(self, phone: str) -> Session:
        key = normalize_phone(phone)
        row = await self._run(self._get_sync, key)
        if row is None:
            return Session(phone=key)
        try:
            history = json.loads(row["history"])
        except (ValueError, TypeError):
            history = []
        if not isinstance(history, list):
            history = []
        return Session(phone=key, history=history, language=row["language"] or "")

    def _save_sync(self, phone: str, history: str, language: str) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO wa_sessions (phone, history, language, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(phone) DO UPDATE SET "
                "history = excluded.history, "
                "language = excluded.language, "
                "updated_at = excluded.updated_at",
                (phone, history, language, int(time.time())),
            )

    async def save(self, session: Session) -> None:
        payload = json.dumps(
            [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in session.history
            ],
            ensure_ascii=False,
        )
        await self._run(
            self._save_sync,
            session.phone,
            payload,
            session.language or "",
        )

    def _clear_sync(self, phone: str) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM wa_sessions WHERE phone = ?", (phone,))

    async def clear(self, phone: str) -> None:
        await self._run(self._clear_sync, normalize_phone(phone))

    # -- deduplication --------------------------------------------------
    def _mark_seen_sync(self, message_id: str, ttl_seconds: int) -> bool:
        cutoff = int(time.time()) - max(60, ttl_seconds)
        with self._db() as conn:
            # Purge expired ids in the same transaction, so the table cannot
            # grow without bound between sweeps.
            conn.execute("DELETE FROM wa_seen WHERE seen_at < ?", (cutoff,))
            cursor = conn.execute(
                "INSERT OR IGNORE INTO wa_seen (message_id, seen_at) VALUES (?, ?)",
                (message_id, int(time.time())),
            )
            return cursor.rowcount > 0

    async def mark_seen(self, message_id: str, ttl_seconds: int) -> bool:
        if not message_id:
            return True
        return await self._run(self._mark_seen_sync, message_id, ttl_seconds)

    def _sweep_sync(self, ttl_seconds: int) -> int:
        cutoff = int(time.time()) - max(60, ttl_seconds)
        with self._db() as conn:
            cursor = conn.execute(
                "DELETE FROM wa_sessions WHERE updated_at < ?", (cutoff,)
            )
            conn.execute("DELETE FROM wa_seen WHERE seen_at < ?", (cutoff,))
            return cursor.rowcount

    async def sweep(self, ttl_seconds: int) -> int:
        return await self._run(self._sweep_sync, ttl_seconds)


class SupabaseSessionStore(SessionStore):
    """Postgres, for more than one instance or ephemeral disk.

    Expects the same Supabase project that holds the NAFDAC greenbook, but a
    different schema: this table holds user conversation content and the
    greenbook tables hold regulatory data, and nothing here should be able to
reach the other. Set WHATSAPP_SESSION_SCHEMA (default `confamai_bot`, matching
     the package directory) and run setup once; it creates the schema and both
     tables if they are missing.

    psycopg is imported lazily so a sqlite deployment never needs it.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.schema = (
            getattr(cfg, "session_schema", "confamai_bot") or "confamai_bot"
        )
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        await asyncio.to_thread(self._setup_sync)

    def _psycopg(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise RuntimeError(
                "WHATSAPP_SESSION_BACKEND=supabase needs psycopg installed: "
                'pip install "psycopg[binary]"'
            ) from exc
        return psycopg

    def _setup_sync(self) -> None:
        """Create the schema and the tables.

        The DDL is the same text SQLite uses, with `?` swapped for `%s` and the
        schema qualified onto every name, so one statement list stays the single
        source of truth for the shape of both backends.
        """
        psycopg = self._psycopg()
        ddl = _SCHEMA.replace("?", "%s")
        ddl = re.sub(
            r"\b(wa_sessions|wa_seen)\b", r'"%s".\1' % self.schema, ddl
        )
        with psycopg.connect(self.cfg.session_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE SCHEMA IF NOT EXISTS "%s"' % self.schema)
                cur.execute(ddl)

    def _connect(self):
        psycopg = self._psycopg()
        conn = psycopg.connect(self.cfg.session_dsn, autocommit=True)
        # Pin the search path on every connection so the unqualified table names
        # below can never resolve to a same-named table in `public` — which is
        # exactly where the NAFDAC greenbook tables live.
        with conn.cursor() as cur:
            cur.execute('SET search_path TO "%s", public' % self.schema)
        return conn

    def _get_sync(self, phone: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT history, language FROM wa_sessions WHERE phone = %s",
                    (phone,),
                )
                return cur.fetchone()

    async def get(self, phone: str) -> Session:
        key = normalize_phone(phone)
        row = await asyncio.to_thread(self._get_sync, key)
        if not row:
            return Session(phone=key)
        try:
            history = json.loads(row[0])
        except (ValueError, TypeError):
            history = []
        if not isinstance(history, list):
            history = []
        return Session(phone=key, history=history, language=(row[1] or ""))

    def _save_sync(self, phone: str, history: str, language: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO wa_sessions (phone, history, language, updated_at) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (phone) DO UPDATE SET "
                    "history = EXCLUDED.history, "
                    "language = EXCLUDED.language, "
                    "updated_at = EXCLUDED.updated_at",
                    (phone, history, language, int(time.time())),
                )

    async def save(self, session: Session) -> None:
        payload = json.dumps(
            [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in session.history
            ],
            ensure_ascii=False,
        )
        async with self._lock:
            await asyncio.to_thread(
                self._save_sync, session.phone, payload, session.language or ""
            )

    def _clear_sync(self, phone: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM wa_sessions WHERE phone = %s", (phone,))

    async def clear(self, phone: str) -> None:
        await asyncio.to_thread(self._clear_sync, normalize_phone(phone))

    def _mark_seen_sync(self, message_id: str, ttl_seconds: int) -> bool:
        cutoff = int(time.time()) - max(60, ttl_seconds)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM wa_seen WHERE seen_at < %s", (cutoff,))
                cur.execute(
                    "INSERT INTO wa_seen (message_id, seen_at) VALUES (%s, %s) "
                    "ON CONFLICT (message_id) DO NOTHING",
                    (message_id, int(time.time())),
                )
                return cur.rowcount > 0

    async def mark_seen(self, message_id: str, ttl_seconds: int) -> bool:
        if not message_id:
            return True
        async with self._lock:
            return await asyncio.to_thread(
                self._mark_seen_sync, message_id, ttl_seconds
            )

    def _sweep_sync(self, ttl_seconds: int) -> int:
        cutoff = int(time.time()) - max(60, ttl_seconds)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM wa_sessions WHERE updated_at < %s", (cutoff,))
                cur.execute("DELETE FROM wa_seen WHERE seen_at < %s", (cutoff,))
                return cur.rowcount

    async def sweep(self, ttl_seconds: int) -> int:
        return await asyncio.to_thread(self._sweep_sync, ttl_seconds)


def build_store(cfg: Config) -> SessionStore:
    """The backend the configuration asked for."""
    return SupabaseSessionStore(cfg) if cfg.uses_supabase() else SqliteSessionStore(cfg)


class Sweeper:
    """Deletes expired sessions and stale message ids on an interval.

    Exists because the 24-hour TTL is a promise about data the user can be
    asked about, and a promise made in a config file is not a deletion. Run
    hourly; the table is small and the query is indexed on updated_at.
    """

    def __init__(self, store: SessionStore, cfg: Config):
        self.store = store
        self.ttl = max(1, cfg.session_ttl_hours) * 3600
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600)
                removed = await self.store.sweep(self.ttl)
                if removed:
                    print(
                        "[sessions] swept %d expired session(s)" % removed, flush=True
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a failed sweep must not kill the loop
                print("[sessions] sweep failed: %r" % (exc,), flush=True)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None