"""The orchestration: one inbound WhatsApp message in, one reply out.

This is the only file in the package with real branching, and the branching is
deliberately shallow. In order:

    1. is this number allowed at all?          -> refuse locally
    2. has it opted out?                       -> refuse locally
    3. is it START after an opt-out?           -> clear locally
    4. is it STOP?                             -> delete the session locally
    5. is it /hausa, /new, /lang?              -> answer locally, no model
    6. is it a photo or a document?            -> answer locally, no model
    7. otherwise                               -> hand it to the server

Steps 1 to 6 never call the model. That is not only cheaper, it is better: a
`/hausa` answered by the model would take about thirteen seconds on a T4 to say
something this package can say instantly, and would consume a slot that a real
verification question is waiting for.

Step 7 is the whole product, and it is one HTTP request. The message goes to the
server exactly as the user typed it. No NRN extraction, no product-name search,
no pre-attached greenbook block, no verdict text on this side. The server's
`_nafdac_enrich` decides whether the question is about a product at all, and
`_intent_permits_retrieval` decides whether to query the greenbook.

That restraint is the design. `n_atlas_client` reached the same conclusion about
itself: when a server is reachable it attaches nothing locally, because a
second retrieval authority runs *before* the server has seen the message, and
the concrete failure was a greeting that matched a product locally and then
talked the server into searching as well (n_atlas_client/api.py:499-510).

What this module does own is honesty about failure. When the server is
unreachable the user is told so in their own language and gets nothing else.
There is no local greenbook to fall back on — that was the deliberate cost of
one retrieval authority — so the alternative would be to answer from the
model's memory, which is precisely how a cough syrup ends up described with
ingredients nobody put in the database.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from . import framing, languages, render
from .audio import transcriber
from .config import Config, normalize_phone
from .natlas import (
    FATAL,
    EmptyResponse,
    NAtlasClient,
    NAtlasError,
)
from .queue import Concurrency, QueueTimeout, UserBusy
from .sessions import SessionStore
from .whatsapp import WhatsAppClient, WhatsAppError

# How long an inbound message id is remembered for deduplication. Meta retries a
# webhook that has not answered 200 within twenty seconds, and retries can be
# spaced minutes apart, so this is comfortably longer than any retry window.
DEDUPE_TTL = 3600


class Router:
    """Turns inbound messages into replies. One instance per process."""

    def __init__(
        self,
        cfg: Config,
        natlas: NAtlasClient,
        whatsapp: WhatsAppClient,
        store: SessionStore,
        concurrency: Concurrency,
    ):
        self.cfg = cfg
        self.natlas = natlas
        self.whatsapp = whatsapp
        self.store = store
        self.concurrency = concurrency
        # Numbers that asked to stop. Kept in memory on purpose: it must
        # outlive a session sweep, and it is deliberately not persisted, so a
        # restart gives an opted-out user a clean slate rather than a permanent
        # ban they cannot see or undo.
        self.opted_out: set[str] = set()
        # Voice notes are transcribed server-first, through the same N-ATLaS
        # client the chat path uses, so the bot needs no local model. The
        # transcriber's local NCAIR1 fallback stays available for a deployment
        # that points no server at voice or loses one (see audio.py).
        transcriber.bind(natlas)
        self.stats = {
            "text": 0,
            "local": 0,
            "refused": 0,
            "duplicate": 0,
            "errors": 0,
            "failed_sends": 0,
        }

    # ------------------------------------------------------------------
    async def handle_payload(self, payload: dict) -> list[str]:
        """Process a whole webhook body. Returns the messages that were sent.

        Called from the webhook's background task, never from the request path:
        Meta needs a 200 within twenty seconds and a T4 turn takes thirteen.
        """
        from .whatsapp import parse_inbound, parse_unsupported

        sent: list[str] = []

        for item in parse_inbound(payload):
            if item["type"] != "text":
                continue
            try:
                reply = await self.handle_text(item)
            except Exception as exc:  # a background task must not die
                self.stats["errors"] += 1
                print("[router] unhandled error: %r" % (exc,), flush=True)
                continue
            if reply:
                sent.append(reply)

        for item in parse_unsupported(payload):
            try:
                await self.handle_non_text(item)
            except Exception as exc:
                print("[router] non-text error: %r" % (exc,), flush=True)

        return sent

    # ------------------------------------------------------------------
    async def handle_text(self, item: dict) -> str:
        """One inbound text message. Returns what was sent, or ''."""
        phone = normalize_phone(item.get("from", ""))
        text = str(item.get("text") or "").strip()
        message_id = item.get("id", "")

        if not phone:
            return ""

        if not self.cfg.is_allowed(phone):
            self.stats["refused"] += 1
            return await self._send(phone, languages.message("not_allowed"))

        # Dedupe before anything else. A Meta retry of an already-answered
        # message must produce exactly zero sends: a duplicate reply is a
        # duplicate billed message and the most visible bug this bot can have.
        if not await self.store.mark_seen(message_id, DEDUPE_TTL):
            self.stats["duplicate"] += 1
            print("[router] duplicate message %s ignored" % message_id, flush=True)
            return ""

        session = await self.store.get(phone)
        lang = session.language or "en"

        # --- opt-out, before anything that would answer a question --------
        if languages.is_opt_out(text):
            return await self._handle_opt_out(phone, session, lang)

        if phone in self.opted_out:
            # Only START revokes an opt-out. Anything else is dropped in
            # silence, with no reply at all: honouring an opt-out means
            # sending nothing, and a "are you sure?" message is itself an
            # unrequested message to someone who asked to stop receiving them.
            # The session row was already deleted at opt-out time, so there is
            # nothing left to answer from even if this were allowed.
            if languages.is_start(text):
                self.opted_out.discard(phone)
                return await self._send(
                    phone, "Welcome back to Confam AI. Send a NAFDAC number to check."
                )
            self.stats["refused"] += 1
            print(
                "[router] %s is opted out; message dropped without reply" % phone,
                flush=True,
            )
            return ""
        # START when not opted out is an ordinary message and falls through to
        # the model: somebody typing START mid-flow is asking to start the
        # conversation, not asking about a product.

        # --- local commands ----------------------------------------------
        is_command, mode, argument = languages.is_language_command(text)
        if is_command:
            self.stats["local"] += 1
            if mode is None:  # /lang — read only, changes nothing
                return await self._send(phone, languages.shown(session.language))
            # Covers both a real code and AUTO (""), which is how the pin is
            # actually cleared.
            session.language = mode
            await self.store.save(session)
            return await self._send(phone, languages.confirm(mode))

        if self._is_clear_command(text):
            self.stats["local"] += 1
            session.history = []
            await self.store.save(session)
            return await self._send(phone, languages.message("cleared", lang))

        # --- the actual question ------------------------------------------
        return await self._ask(phone, session, text, lang)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_clear_command(text: str) -> bool:
        """`/new` and `/clear`, plus the language word forms of both.

        A WhatsApp user has no `/help` to read, so the two most useful commands
        also accept the words a person would guess in any of the four
        languages.
        """
        word = str(text or "").strip().lower().lstrip("/").split(" ")[0]
        return word in {
            "new", "clear", "reset", "start-over",
            "sabisha", "tso", "fefe",          # Hausa
            "titun", "ko", "reset-ni",         # Yoruba
            "ohuru", "tọọ", "malite",          # Igbo
        }

    async def _ask(self, phone: str, session, text: str, lang: str) -> str:
        """Hand the message to N-ATLaS and reply with what comes back.

        This is the whole Tier-2 path. One request, one response, no local
        interpretation of the question beyond the framing note below.
        """
        incoming = text[: self.cfg.max_message_chars]
        # A message carrying a number but no interrogative ("A4-100160",
        # "check A4-100160") currently gets the verified block attached and then
        # ignored, and the user is asked which product to verify. Framing the
        # turn fixes it. The frame carries no product, no verdict and no number,
        # and is added only when the user's own text holds one, so a greeting
        # gets nothing appended at all. See framing.py for the measurements.
        outgoing = framing.apply(
            incoming,
            enabled=self.cfg.verification_frame,
            custom=self.cfg.verification_frame_text,
        )
        history = session.messages(self.cfg)
        payload = list(history) + [{"role": "user", "content": outgoing}]

        # The acknowledgement goes out before the request, because the user is
        # watching a bubble. It is one extra free message inside the customer's
        # 24-hour window, so it costs nothing in Meta's October 2026 pricing.
        ack = ""
        if self.cfg.send_ack:
            ack = render.acknowledgement(incoming, self.cfg.ack_text)
            if ack:
                try:
                    await self.whatsapp.send_text(phone, ack)
                except WhatsAppError as exc:
                    # An unsent acknowledgement is not worth failing the reply.
                    print("[router] ack not sent: %s" % exc, flush=True)

        started = time.monotonic()

        async def call():
            return await self.natlas.chat_with_retry(
                payload,
                # '' means auto, which is the server detecting each message.
                # Anything else pins the reply for this turn and every turn
                # until the user changes it with another command.
                language=(session.language or None),
            )

        try:
            reply_text = await self.concurrency.run(phone, call)
        except UserBusy:
            self.stats["errors"] += 1
            return await self._send(phone, languages.message("busy", lang))
        except QueueTimeout:
            self.stats["errors"] += 1
            return await self._send(phone, languages.message("busy", lang))
        except NAtlasError as exc:
            self.stats["errors"] += 1
            fatal = exc.kind == FATAL
            print(
                "[router] N-ATLaS %s after %.1fs: %s"
                % ("fatal" if fatal else "transient", time.monotonic() - started, exc),
                flush=True,
            )
            # Say the truth and stop. No partial answer, no guess, no memory.
            # A verification tool that answers when it cannot verify is worse
            # than one that admits it is down.
            if isinstance(exc, EmptyResponse):
                return await self._send(phone, languages.message("blank", lang))
            return await self._send(phone, languages.message("unavailable", lang))

        self.stats["text"] += 1
        elapsed = time.monotonic() - started

        # Store what was asked and what was answered, in that order, so the next
        # turn sees a real conversation. Only once the reply exists: storing the
        # question first would leave a dangling user turn in the history if the
        # server had failed.
        #
        # The UNFRAMED text is stored, never `outgoing`. The frame is a
        # transport detail of one request; keeping it would append another copy
        # to the user's own history every turn, grow the context budget with
        # duplicated instructions, and hand the model a conversation the user
        # never had.
        session.history = list(history) + [
            {"role": "user", "content": incoming},
            {"role": "assistant", "content": reply_text},
        ]
        await self.store.save(session)

        cleaned = render.clean(reply_text)
        parts = render.chunk(cleaned)
        if len(parts) > 1:
            print(
                "[router] reply of %d chars split into %d messages"
                % (len(cleaned), len(parts)),
                flush=True,
            )

        sent = ""
        for part in parts:
            sent = await self._send(phone, part)
        print(
            "[router] %s answered in %.1fs (%d msg, %d chars)"
            % (phone, elapsed, len(parts), len(cleaned)),
            flush=True,
        )
        return sent

    # ------------------------------------------------------------------
    async def _handle_opt_out(self, phone: str, session, lang: str) -> str:
        """NDPA 2023 requires an easy stop. This is it.

        Deletes the stored conversation rather than marking it, and refuses
        everything afterwards until the user sends START. Nothing is sent to an
        opted-out number except this one confirmation, and then nothing at all.
        """
        self.opted_out.add(phone)
        await self.store.clear(phone)
        self.stats["local"] += 1
        print("[router] %s opted out; session deleted" % phone, flush=True)
        return await self._send(phone, languages.message("optout", lang))

    async def handle_non_text(self, item: dict) -> str:
        """A photo, document, sticker or voice note.

        A voice note is the one non-text message this bot understands:
        it is downloaded, transcribed with the model for the user's
        language, and the resulting text is fed through the ordinary
        text path — so a spoken "stop" opts out and a spoken NAFDAC
        number is verified exactly as if it had been typed.

        Everything else gets the local "I can only read text" answer.
        A photograph of a medicine pack is the most likely thing a real
        person sends, and silence would read as a broken bot, but OCR
        is a separate feature.
        """
        phone = normalize_phone(item.get("from", ""))
        if not phone or phone in self.opted_out or not self.cfg.is_allowed(phone):
            return ""
        if not await self.store.mark_seen(item.get("id", ""), DEDUPE_TTL):
            return ""

        if item.get("type") != "audio":
            session = await self.store.get(phone)
            return await self._send(
                phone, languages.message("image", session.language or "en")
            )

        session = await self.store.get(phone)
        lang = session.language or "en"
        audio_id = (item.get("audio") or {}).get("id") or ""
        if not audio_id:
            return await self._send(phone, languages.message("blank", lang))

        await self._send(phone, languages.message("voice_listening", lang))

        try:
            audio_bytes = await self.whatsapp.download_media(audio_id)
            # Transcription is CPU/GPU-bound and can take tens of
            # seconds; off the event loop, so every other user's
            # webhook keeps flowing while this one note is heard.
            transcribed = await asyncio.to_thread(
                transcriber.transcribe, audio_bytes, lang
            )
        except Exception as exc:
            self.stats["errors"] += 1
            print("[router] voice note failed: %r" % (exc,), flush=True)
            return await self._send(phone, languages.message("voice_failed", lang))

        if not transcribed:
            return await self._send(phone, languages.message("blank", lang))

        print("[router] transcribed voice note: %s" % transcribed, flush=True)

        # The transcription becomes an ordinary text message, so it takes
        # the whole text path: opt-out words, language commands, the
        # verification frame, the model. The id is prefixed because the
        # dedupe table already holds this note's own id from the
        # mark_seen above — reusing it would make handle_text treat the
        # turn as a Meta retry and drop it.
        mock_text_item = dict(item)
        mock_text_item["id"] = "audio-%s" % item.get("id", "")
        mock_text_item["type"] = "text"
        mock_text_item["text"] = transcribed
        return await self.handle_text(mock_text_item)

    # ------------------------------------------------------------------
    async def _send(self, phone: str, body: str) -> str:
        """Send, and never raise into the caller."""
        if not body:
            return ""
        try:
            await self.whatsapp.send_with_retry(phone, body)
        except WhatsAppError as exc:
            self.stats["failed_sends"] += 1
            print(
                "[router] send to %s failed (code=%s retryable=%s): %s"
                % (phone, exc.code, exc.retryable, exc),
                flush=True,
            )
            return ""
        return body

    def snapshot(self) -> dict[str, Any]:
        return {"stats": dict(self.stats), "queue": self.concurrency.snapshot()}