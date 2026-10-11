"""Configuration for the Confam AI WhatsApp bot.

Everything comes from environment variables or a local .env file. No secret is
ever hard-coded, printed or logged, and `describe()` exists so an operator can
see what is configured without seeing any of it.

Search order for the .env file:

    1. the explicit `env_file` argument
    2. the current working directory
    3. the project root (this package's parent)

Real environment variables always win over .env entries, so a container with
real env vars injected is never overridden by a stray file on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _env_str(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(key, default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(str(os.environ.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_list(key: str) -> tuple[str, ...]:
    """A comma-separated env value as a tuple, empties removed.

    Used for the optional allowlist. An empty result means "no allowlist", which
    is the correct default: a bot on a public number should be open until the
    operator deliberately restricts it.
    """
    raw = _env_str(key)
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass
class Config:
    # --- WhatsApp Cloud API ---------------------------------------------
    phone_number_id: str = ""      # WHATSAPP_PHONE_NUMBER_ID
    access_token: str = ""         # WHATSAPP_ACCESS_TOKEN  (never printed)
    verify_token: str = ""         # WHATSAPP_VERIFY_TOKEN  (the hub.challenge secret)
    app_secret: str = ""           # WHATSAPP_APP_SECRET    (X-Hub-Signature-256)
    graph_version: str = "v21.0"   # WHATSAPP_GRAPH_VERSION
    webhook_path: str = "/webhook" # WHATSAPP_WEBHOOK_PATH

    # --- N-ATLaS server --------------------------------------------------
    base_url: str = ""             # N_ATLAS_BASE_URL  (ends in /v1)
    api_key: str = ""              # N_ATLAS_API_KEY   (never printed)
    model: str = "n-atlas"         # N_ATLAS_MODEL
    timeout: float = 180.0         # N_ATLAS_TIMEOUT   (a T4 turn takes ~13s)
    max_tokens: int = 512          # N_ATLAS_MAX_TOKENS
    temperature: float = 0.7       # N_ATLAS_TEMPERATURE
    max_retries: int = 2           # N_ATLAS_MAX_RETRIES

    # --- throughput -------------------------------------------------------
    # The N-ATLaS server handles sync requests in a threadpool and generates on
    # one shared 4-bit GPU, so concurrency there is a queue, not a pool. This
    # semaphore is what stops one busy user from turning into 32 simultaneous
    # generations and a wall of 507s.
    max_concurrency: int = 3       # WHATSAPP_MAX_CONCURRENCY
    per_user_timeout: float = 120.0  # WHATSAPP_PER_USER_TIMEOUT

    # --- sessions ---------------------------------------------------------
    # The server is stateless per request, so history lives here.
    session_backend: str = "sqlite"    # WHATSAPP_SESSION_BACKEND sqlite|supabase
    session_ttl_hours: int = 24        # WHATSAPP_SESSION_TTL_HOURS
    max_history_messages: int = 12     # WHATSAPP_MAX_HISTORY_MESSAGES
    max_history_tokens: int = 2000     # WHATSAPP_MAX_HISTORY_TOKENS
    max_message_chars: int = 4000      # WHATSAPP_MAX_MESSAGE_CHARS
    sqlite_path: str = "sessions.db"   # WHATSAPP_SQLITE_PATH
    session_dsn: str = ""              # WHATSAPP_SESSION_DSN (supabase backend)
    session_schema: str = "confamai_bot"  # WHATSAPP_SESSION_SCHEMA

    # --- behaviour --------------------------------------------------------
    # Sent the moment a message is accepted, because a T4 turn takes ~13s and
    # an unexplained pause reads as a broken bot. One extra free message inside
    # the customer's 24h window, so it costs nothing.
    send_ack: bool = True              # WHATSAPP_SEND_ACK
    ack_text: str = ""                 # WHATSAPP_ACK_TEXT (empty: pick by language)
    debug: bool = False                # WHATSAPP_DEBUG

    # Workaround for a deterministic server defect: a message carrying a NAFDAC
    # number but no interrogative ("A4-100160", "check A4-100160") gets the
    # verified block attached and then has it ignored, and the user is asked
    # which product to verify. Measured 4/4 failures at both temperature 0.7
    # and 0.1, so it is not sampling noise. Framing the turn fixes all of them
    # without making the intent gate any more willing to search. Set this false
    # once `NAFDAC_INSTRUCTION` on the server covers bare numbers. See
    # confamai_bot/framing.py for the measurements.
    verification_frame: bool = True    # WHATSAPP_VERIFICATION_FRAME
    verification_frame_text: str = ""  # WHATSAPP_VERIFICATION_FRAME_TEXT

    # Optional allowlist of E.164 numbers. Empty means allow everyone.
    allowed_numbers: tuple[str, ...] = field(default_factory=tuple)

    def server_root(self) -> str:
        """Derive the server root from a base URL that normally ends in /v1."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base[: -len("/v1")].rstrip("/")
        return base

    def health_url(self) -> str:
        return self.server_root() + "/health"

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def transcribe_url(self) -> str:
        return self.base_url.rstrip("/") + "/audio/transcriptions"

    def graph_base(self) -> str:
        return "https://graph.facebook.com/%s/%s/messages" % (
            self.graph_version,
            self.phone_number_id,
        )

    def uses_supabase(self) -> bool:
        return self.session_backend.strip().lower() == "supabase"

    def is_allowed(self, phone: str) -> bool:
        """True when this number may talk to the bot.

        An empty allowlist allows everyone, which is the correct default for a
        bot on a number nobody has published yet.
        """
        if not self.allowed_numbers:
            return True
        return normalize_phone(phone) in {
            normalize_phone(number) for number in self.allowed_numbers
        }

    def describe(self) -> dict:
        """Safe status summary: what is set, and how big. No secret values.

        Printed by `python -m confamai_bot --status` and served on /healthz, so
        an operator can confirm a deployment without ever putting a token in a
        terminal, a log or a screenshot.
        """
        return {
            "whatsapp_configured": bool(self.phone_number_id and self.access_token),
            "phone_number_id": self.phone_number_id or None,
            "verify_token_set": bool(self.verify_token),
            "app_secret_set": bool(self.app_secret),
            "signature_check_enabled": bool(self.app_secret),
            "graph_version": self.graph_version,
            "webhook_path": self.webhook_path,
            "natlas_configured": bool(self.base_url and self.api_key),
            "natlas_base_url": self.base_url or None,
            "natlas_model": self.model,
            "session_backend": self.session_backend,
            "max_concurrency": self.max_concurrency,
            "session_ttl_hours": self.session_ttl_hours,
            "verification_frame": self.verification_frame,
            "allowed_numbers": len(self.allowed_numbers) or "all",
        }


def normalize_phone(phone: str) -> str:
    """Normalise a WhatsApp sender id to E.164-ish for use as a key.

    WhatsApp sends `to` and `from` as bare digits with no `+`
    (`2348012345678`). Adding it back here means an allowlist written with `+`
    still matches, and a session row written by one code path is found by
    another.
    """
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    return "+%s" % digits if digits else ""


def load_config(env_file: str | None = None, debug: bool = False) -> Config:
    """Load configuration from environment variables and a .env file."""
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(Path.cwd() / ".env", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    return Config(
        phone_number_id=_env_str("WHATSAPP_PHONE_NUMBER_ID"),
        access_token=_env_str("WHATSAPP_ACCESS_TOKEN"),
        verify_token=_env_str("WHATSAPP_VERIFY_TOKEN"),
        app_secret=_env_str("WHATSAPP_APP_SECRET"),
        graph_version=_env_str("WHATSAPP_GRAPH_VERSION", "v21.0"),
        webhook_path=_env_str("WHATSAPP_WEBHOOK_PATH", "/webhook") or "/webhook",
        base_url=_env_str("N_ATLAS_BASE_URL"),
        api_key=_env_str("N_ATLAS_API_KEY"),
        model=_env_str("N_ATLAS_MODEL", "n-atlas"),
        timeout=max(1.0, _env_float("N_ATLAS_TIMEOUT", 180.0)),
        max_tokens=max(1, _env_int("N_ATLAS_MAX_TOKENS", 512)),
        temperature=_env_float("N_ATLAS_TEMPERATURE", 0.7),
        max_retries=max(0, _env_int("N_ATLAS_MAX_RETRIES", 2)),
        max_concurrency=max(1, _env_int("WHATSAPP_MAX_CONCURRENCY", 3)),
        per_user_timeout=max(5.0, _env_float("WHATSAPP_PER_USER_TIMEOUT", 120.0)),
        session_backend=_env_str("WHATSAPP_SESSION_BACKEND", "sqlite") or "sqlite",
        session_ttl_hours=max(1, _env_int("WHATSAPP_SESSION_TTL_HOURS", 24)),
        max_history_messages=max(2, _env_int("WHATSAPP_MAX_HISTORY_MESSAGES", 12)),
        max_history_tokens=max(256, _env_int("WHATSAPP_MAX_HISTORY_TOKENS", 2000)),
        max_message_chars=max(256, _env_int("WHATSAPP_MAX_MESSAGE_CHARS", 4000)),
        sqlite_path=_env_str("WHATSAPP_SQLITE_PATH", "sessions.db") or "sessions.db",
        session_dsn=_env_str("WHATSAPP_SESSION_DSN"),
        session_schema=(
            _env_str("WHATSAPP_SESSION_SCHEMA", "confamai_bot") or "confamai_bot"
        ),
        send_ack=_env_bool("WHATSAPP_SEND_ACK", True),
        ack_text=_env_str("WHATSAPP_ACK_TEXT"),
        verification_frame=_env_bool("WHATSAPP_VERIFICATION_FRAME", True),
        verification_frame_text=_env_str("WHATSAPP_VERIFICATION_FRAME_TEXT"),
        debug=debug or _env_bool("WHATSAPP_DEBUG", False),
        allowed_numbers=_env_list("WHATSAPP_ALLOWED_NUMBERS"),
    )


def _looks_like_placeholder(dsn: str) -> bool:
    """True when a DSN still carries an unfilled template value.

    `validate()` only rejects an empty DSN, but a copy-pasted dashboard template
    is non-empty and contains one of these. Left unchecked it passes startup and
    fails at the first inbound message with `password authentication failed`,
    which reads like wrong credentials rather than an unfinished setup.
    """
    lowered = str(dsn or "").lower()
    markers = (
        "your-password", "your_password", "paste_password", "placeholder",
        "changeme", "change_me", "todo", "xxx", "<", ">",
    )
    if any(marker in lowered for marker in markers):
        return True
    # Also catch a password segment that is empty, i.e. `user:@host`.
    if "://" in lowered:
        credentials = lowered.split("://", 1)[1].split("@", 1)[0]
        if ":" in credentials and not credentials.split(":", 1)[1].strip():
            return True
    return False


def validate(cfg: Config, require_whatsapp: bool = True) -> None:
    """Fail fast, naming every missing variable at once.

    `require_whatsapp=False` is what the live-assay tool and the tests use: they
    need to talk to N-ATLaS without Meta credentials, so requiring a phone
    number there would be a false failure.
    """
    missing = []

    if require_whatsapp:
        if not cfg.phone_number_id:
            missing.append("WHATSAPP_PHONE_NUMBER_ID")
        if not cfg.access_token:
            missing.append("WHATSAPP_ACCESS_TOKEN")
        if not cfg.verify_token:
            missing.append("WHATSAPP_VERIFY_TOKEN")

    if not cfg.base_url:
        missing.append("N_ATLAS_BASE_URL")
    if not cfg.api_key:
        missing.append("N_ATLAS_API_KEY")

    if missing:
        raise ConfigError(
            "Missing required configuration: %s.\n"
            "Copy confamai_bot/.env.example to confamai_bot/.env (or the "
            "project-root .env) and fill in the values. See "
            "confamai_bot/README.md section 3."
            % ", ".join(missing)
        )

    if require_whatsapp and not cfg.webhook_path.startswith("/"):
        raise ConfigError(
            "WHATSAPP_WEBHOOK_PATH must start with '/', got %r." % cfg.webhook_path
        )

    if "http://" not in cfg.base_url and "https://" not in cfg.base_url:
        raise ConfigError(
            "N_ATLAS_BASE_URL must be a full URL such as "
            "https://ai.saynet.ng/v1 (got %r)." % cfg.base_url
        )

    if cfg.uses_supabase():
        if not cfg.session_dsn:
            raise ConfigError(
                "WHATSAPP_SESSION_BACKEND=supabase needs WHATSAPP_SESSION_DSN "
                "set. Copy the URI from Supabase -> Connect -> URI (session "
                "pooler, :5432) into confamai_bot/.env."
            )
        if _looks_like_placeholder(cfg.session_dsn):
            raise ConfigError(
                "WHATSAPP_SESSION_DSN still contains a placeholder password.\n"
                "Replace it in confamai_bot/.env with your real Supabase "
                "password, percent-encoded: @ -> %40, : -> %3A, / -> %2F, "
                "# -> %23, ? -> %3F.\n"
                "An unencoded special character is reported as 'missing / after "
                "host', which looks like a host typo and is not."
            )

    backend = cfg.session_backend.strip().lower()
    if backend not in {"sqlite", "supabase"}:
        raise ConfigError(
            "WHATSAPP_SESSION_BACKEND must be sqlite or supabase, got %r."
            % cfg.session_backend
        )