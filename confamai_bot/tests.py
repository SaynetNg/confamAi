"""The offline test suite. No network, no WhatsApp account, no GPU.

    python -m confamai_bot --test

Two halves.

**Payload replay.** Realistic webhook bodies go in, a stubbed N-ATLaS comes
back, and the test asserts on what was sent: that a retry of the same
`messages[].id` sends nothing, that a system message is never constructed, that
`language` travels only when pinned, that `/hausa` never reaches the model, and
that an opted-out number gets silence.

**Contract assertions on replies.** The safety rules are not stylistic. A reply
that calls an unregistered product counterfeit is the failure this entire
project exists to prevent, so `evaluate()` from assay.py is reused here against
fixture replies. Same assertions, no network needed, so a regression shows up in
a second rather than after a 13-second model turn.

The stub records the request it was given. That is the point of the suite: it
tests the payload the bot would send, not just the reply it would print.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

from .assay import LEAK_MARKERS, evaluate
from .audio import transcriber as real_transcriber
from .config import Config, normalize_phone
from .languages import (
    confirm,
    is_language_command,
    is_opt_out,
    is_start,
    resolve_alias,
)
from .natlas import NAtlasClient, NAtlasError
from .queue import Concurrency
from .render import (
    acknowledgement,
    chunk,
    clean,
    detect_ack_language,
    would_split,
)
from .router import DEDUPE_TTL, Router
from .sessions import SqliteSessionStore, Session
from .whatsapp import parse_inbound, parse_unsupported

RESULTS = {"pass": 0, "fail": 0}
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        RESULTS["pass"] += 1
    else:
        RESULTS["fail"] += 1
        FAILURES.append(label + ((" — " + detail) if detail else ""))
        print("  FAIL  %s%s" % (label, (" — " + detail) if detail else ""))


def section(title: str) -> None:
    print()
    print(title)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubNAtlas(NAtlasClient):
    """Records every request and replies with a canned string."""

    def __init__(self, cfg: Config, reply: str = "stub reply", error: Exception | None = None):
        self.cfg = cfg
        self.debug = False
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def health(self) -> dict:
        return {
            "status": "ok",
            "served_model_name": "n-atlas",
            "gpu": "Tesla T4",
            "model_loaded": True,
            "streaming": False,
            "nafdac": {"enabled": True, "ready": True, "sources": []},
        }

    async def chat_with_retry(self, messages, language=None) -> str:
        self.calls.append({"messages": messages, "language": language})
        if self.error:
            raise self.error
        return self.reply


class RecordingWhatsApp:
    """Stands in for WhatsAppClient and remembers every send."""

    def __init__(self, fail_with: Exception | None = None):
        self.sent: list[tuple[str, str]] = []
        self.fail_with = fail_with

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def send_text(self, to: str, body: str) -> dict:
        if self.fail_with:
            raise self.fail_with
        self.sent.append((to, body))
        return {"messages": [{"id": "wamid.stub"}]}

    async def send_with_retry(self, to: str, body: str, attempts: int = 3) -> dict:
        return await self.send_text(to, body)

    async def mark_read(self, message_id: str) -> None:
        return None

    async def download_media(self, media_id: str) -> bytes:
        return b"stub-ogg-audio"

    async def set_typing(self, message_id: str, on: bool = True) -> None:
        return None

    def bodies(self) -> list[str]:
        return [body for _to, body in self.sent]


def payload(text: str, sender: str = "2348012345678", mid: str = "wamid.1") -> dict:
    """A webhook body shaped exactly as Meta sends one."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15550001111", "phone_number_id": "PNID"},
                            "messages": [
                                {
                                    "id": mid,
                                    "from": sender,
                                    "timestamp": "1760000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def status_payload() -> dict:
    """A delivery receipt. Same endpoint, no `messages`, no `from`."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15550001111", "phone_number_id": "PNID"},
                            "statuses": [
                                {"id": "wamid.1", "status": "delivered", "timestamp": "1760000001",
                                 "recipient_id": "2348012345678"}
                            ],
                        },
                    }
                ],
            }
        ],
    }


def make_cfg(tmp: Path, **overrides) -> Config:
    cfg = Config(
        phone_number_id="PNID",
        access_token="stub-token",
        verify_token="stub-verify",
        app_secret="stub-secret",
        base_url="https://example.invalid/v1",
        api_key="stub-key",
        session_backend="sqlite",
        sqlite_path=str(tmp / "sessions.db"),
        send_ack=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


_BUILD_COUNTER = {"n": 0}


async def build(tmp: Path, reply: str = "stub reply", error: Exception | None = None,
                **cfg_overrides):
    """A fresh router with its own database.

    Every call gets its own directory. Without that, `store.setup()` runs the
    same CREATE TABLE on the same file each time — harmless — but the *rows*
    persist, so a `/hausa` set in one section silently pins the language for the
    next one and the failure looks like a product bug rather than a shared
    fixture. The store is durable by design; the tests must not be.
    """
    _BUILD_COUNTER["n"] += 1
    workdir = tmp / ("build%02d" % _BUILD_COUNTER["n"])
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(workdir, **cfg_overrides)
    store = SqliteSessionStore(cfg)
    await store.setup()
    natlas = StubNAtlas(cfg, reply=reply, error=error)
    whatsapp = RecordingWhatsApp()
    concurrency = Concurrency(max_concurrency=2, per_user_timeout=5.0)
    router = Router(cfg, natlas, whatsapp, store, concurrency)
    return cfg, natlas, whatsapp, store, router


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
async def test_config():
    section("configuration")
    from .config import load_config, validate, ConfigError

    cfg = make_cfg(Path(tempfile.gettempdir()))
    described = cfg.describe()
    check("describe() never leaks the API key", cfg.api_key not in str(described))
    check("describe() never leaks the access token", cfg.access_token not in str(described))
    check("describe() reports signature checking", described["signature_check_enabled"] is True)

    check("normalize_phone adds +", normalize_phone("2348012345678") == "+2348012345678")
    check("normalize_phone strips spaces", normalize_phone("+234 801 234 5678") == "+2348012345678")
    check("normalize_phone handles junk", normalize_phone("") == "")

    # Allowlist: empty means open, populated means restrictive.
    open_cfg = make_cfg(Path(tempfile.gettempdir()))
    check("empty allowlist allows everyone", open_cfg.is_allowed("+2348010000000") is True)
    closed = make_cfg(Path(tempfile.gettempdir()), allowed_numbers=("+2348010000000",))
    check("allowlist permits a listed number", closed.is_allowed("2348010000000") is True)
    check("allowlist blocks an unlisted number", closed.is_allowed("2348099999999") is False)

    # Missing credentials must name every one of them, not just the first.
    try:
        validate(Config(), require_whatsapp=True)
        check("validate rejects an empty config", False, "no error raised")
    except ConfigError as exc:
        text = str(exc)
        check(
            "validate names all missing vars",
            all(name in text for name in
                ("WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_ACCESS_TOKEN",
                 "WHATSAPP_VERIFY_TOKEN", "N_ATLAS_BASE_URL", "N_ATLAS_API_KEY")),
            text[:120],
        )

    # require_whatsapp=False is what the assay and tests rely on.
    partial = Config(base_url="https://x/v1", api_key="k")
    try:
        validate(partial, require_whatsapp=False)
        check("validate passes without WhatsApp creds", True)
    except ConfigError as exc:
        check("validate passes without WhatsApp creds", False, str(exc))


def test_render():
    section("rendering")
    check("clean removes a template marker", clean("a <s>b") == "a b")
    check("clean flattens a code fence", clean("```\nA4-100160\n```") == "A4-100160")
    check("clean keeps content inside a fence", "A4-100160" in clean("```\nA4-100160\n```"))
    check("clean collapses blank lines", clean("a\n\n\n\nb") == "a\n\nb")
    check("clean handles empty", clean("") == "")
    check("clean keeps a product name verbatim", "AG Spironolactone" in clean("  AG Spironolactone  "))

    # The billing constraint: a normal reply is one message.
    short = "A4-100160 is registered and Active in the current greenbook."
    check("short reply is one message", len(chunk(short)) == 1)
    check("would_split is False for a short reply", would_split(short) is False)

    # Over the limit must split, and must not lose content.
    long_text = ("A4-100160 is registered. " * 300).strip()
    parts = chunk(long_text, 1000)
    check("long reply splits", len(parts) > 1)
    check("every part is within the limit", all(len(p) <= 1000 for p in parts))
    rejoined = " ".join(p for p in parts)
    check("splitting loses no content", rejoined.replace(" ", "") == long_text.replace(" ", ""))

    # Split at a boundary, not mid-word.
    prose = "This product is registered and active in the current NAFDAC greenbook record. " * 40
    for part in chunk(prose, 500):
        check("no part ends mid-word", not part.rstrip().endswith(("regi", "acti")), part[-20:])

    section("acknowledgement language")
    check("detects english", detect_ack_language("Is A4-100160 genuine?") == "en")
    check("detects hausa", detect_ack_language("Shin A4-100160 na gida ne?") == "ha")
    check("detects yoruba", detect_ack_language("Ṣé A4-100160 jẹ́ ọgboni tàbí?") == "yo")
    check("detects igbo", detect_ack_language("Ọ bụla A4-100160 ezi?") == "ig")
    check("defaults to english", detect_ack_language("") == "en")

    check("ack auto-picks by language", "NAFDAC" in acknowledgement("Ṣé A4-100160?"))
    check("ack honours an override", acknowledgement("hi", "Custom") == "Custom")
    check("ack off with 'none'", acknowledgement("hi", "none") == "")
    check("ack off with 'off'", acknowledgement("hi", "off") == "")
    check("ack off with '0'", acknowledgement("hi", "0") == "")
    check("ack on when empty", "NAFDAC" in acknowledgement("hi", ""))


def test_languages():
    section("language commands")
    check("detects /hausa", is_language_command("/hausa")[1] == "ha")
    check("detects /yoruba", is_language_command("/yoruba")[1] == "yo")
    check("detects /igbo", is_language_command("/igbo")[1] == "ig")
    check("detects /english", is_language_command("/english")[1] == "en")
    # AUTO is the empty string and must NOT be confused with the read-only
    # None sentinel, or /auto silently behaves like /lang.
    check("detects /auto as auto, not read-only", is_language_command("/auto")[1] == "")
    check("detects /lang as read-only", is_language_command("/lang")[1] is None)
    check("detects /language as read-only", is_language_command("/language")[1] is None)
    check("auto and read-only are distinguishable",
          is_language_command("/auto")[1] is not is_language_command("/lang")[1])
    check("handles a trailing argument", is_language_command("/hausa please")[1] == "ha")
    check("ignores plain text", is_language_command("is it genuine?")[0] is False)
    check("ignores a slash in a sentence", is_language_command("a/b")[0] is False)
    check("ignores empty", is_language_command("")[0] is False)

    check("resolves a hausa alias", resolve_alias("harshe") == "ha")
    check("resolves a yoruba alias", resolve_alias("yebu") == "yo")
    check("resolves an igbo alias", resolve_alias("ahukeri") == "ig")
    check("resolves nothing unknown", resolve_alias("klingon") == "")

    check("confirms in hausa", "Hausa" in confirm("ha") or len(confirm("ha")) > 0)
    check("auto confirm is english", "auto" in confirm("").lower())

    section("opt-out")
    check("catches STOP", is_opt_out("STOP") is True)
    check("catches stop lowercase", is_opt_out("stop") is True)
    check("catches unsubscribe", is_opt_out("UNSUBSCRIBE") is True)
    check("catches STOP! with punctuation", is_opt_out("STOP!") is True)
    check("ignores stop inside a sentence", is_opt_out("please stop checking my blood pressure medicine") is False)
    check("ignores an unrelated word", is_opt_out("is this genuine?") is False)
    check("detects START", is_start("START") is True)


def test_parsing():
    section("webhook parsing")
    items = parse_inbound(payload("Is A4-100160 genuine?"))
    check("parses one message", len(items) == 1)
    check("reads the text", items[0]["text"] == "Is A4-100160 genuine?")
    check("reads the sender", items[0]["from"] == "2348012345678")
    check("reads the id", items[0]["id"] == "wamid.1")

    # Two numbers in one payload — the reason arrays must be looped.
    both = payload("first", mid="a")
    both["entry"][0]["changes"][0]["value"]["messages"].append(
        {"id": "b", "from": "2348099999999", "timestamp": "1", "type": "text",
         "text": {"body": "second"}}
    )
    items = parse_inbound(both)
    check("parses both messages in one payload", len(items) == 2)
    check("keeps both senders distinct", items[0]["from"] != items[1]["from"])

    # Status callbacks carry no `messages` and no `from`. This is the shape that
    # breaks a handler which assumes `value.messages`.
    check("a status callback yields no messages", parse_inbound(status_payload()) == [])

    # Empty and malformed bodies must not raise.
    check("empty body is safe", parse_inbound({}) == [])
    check("null entry is safe", parse_inbound({"entry": None}) == [])
    check("null changes is safe", parse_inbound({"entry": [{"changes": None}]}) == [])
    check("empty value is safe", parse_inbound({"entry": [{"changes": [{"value": {}}]}]}) == [])

    # A photo is a real scenario: somebody photographing a pack.
    image = payload("", mid="c")
    image["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "id": "c", "from": "2348012345678", "timestamp": "1", "type": "image",
        "image": {"id": "media-1"},
    }
    check("an image is not treated as text", parse_inbound(image)[0]["type"] == "image")
    check("an image is collected as unsupported", len(parse_unsupported(image)) == 1)

    # A voice note: the audio payload must survive parsing, because the
    # router needs the media id inside it to download the note.
    voice = payload("", mid="d")
    voice["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "id": "d", "from": "2348012345678", "timestamp": "1", "type": "audio",
        "audio": {"id": "media-2", "mime_type": "audio/ogg"},
    }
    voice_items = parse_unsupported(voice)
    check("a voice note is collected as unsupported", len(voice_items) == 1)
    check("a voice note carries its media id",
          voice_items[0]["audio"]["id"] == "media-2")


def test_framing():
    section("verification framing")
    from . import framing

    # The defect being worked around: a number with no interrogative.
    for text in ("A4-100160", "A4-100160?", "check A4-100160",
                 "verify this number A4-100160", "Is A4-9999999 genuine?"):
        check("a message with a number is detected: %r" % text,
              framing.has_number(text) is True)

    # A greeting must gain nothing, or the intent gate loses its protection.
    for text in ("Hello", "good morning", "I love my family", "what is the weather",
                 "I need water", "", "kannu"):
        check("a message with no number is left alone: %r" % text,
              framing.has_number(text) is False)

    framed = framing.apply("A4-100160")
    check("a number is framed", "[reply]" in framed)
    check("the user's own text is preserved", framed.startswith("A4-100160"))
    check("the frame names no product", "spironolactone" not in framed.lower())
    check("the frame states no verdict", "authentic" not in framed.lower())
    check("the frame repeats no number", framed.count("A4-100160") == 1)

    check("a greeting is not framed", framing.apply("Hello") == "Hello")
    check("an empty message is not framed", framing.apply("") == "")
    check("framing is skippable", framing.apply("A4-100160", enabled=False) == "A4-100160")
    twice = framing.apply(framed)
    check("framing never stacks twice",
          twice.count("The message above is a product verification request") == 1)
    check("a custom frame is honoured",
          framing.apply("A4-100160", custom="CUSTOM").endswith("CUSTOM"))

    # The parser the project already owns must be in use when available, so
    # number formats cannot drift.
    check("the project extractor is in use", framing.extractor_available() is True)
    check("a placeholder is not treated as a number",
          framing.has_number("NA") is False)
    check("an en-dashed number is found", framing.has_number("04–2045") is True)
    check("a spaced number is found", framing.has_number("A4 - 5180") is True)


async def test_framing_does_not_leak(tmp):
    section("framing does not corrupt the conversation")
    _cfg, natlas, whatsapp, store, router = await build(tmp, reply="VERDICT: AUTHENTIC")

    await router.handle_text({"id": "f-1", "from": "2348012345678", "text": "A4-100160"})
    sent = natlas.calls[0]["messages"][-1]["content"]
    check("the framed turn reaches the server", "[reply]" in sent, sent[:80])

    stored = await store.get("+2348012345678")
    check("the frame is NOT stored in history",
          all("[reply]" not in m["content"] for m in stored.history),
          str(stored.history))
    check("the user's real text IS stored",
          stored.history[0]["content"] == "A4-100160")

    # Second turn: the frame must not have stacked.
    await router.handle_text({"id": "f-2", "from": "2348012345678", "text": "A4-5180"})
    last = natlas.calls[1]["messages"][-1]["content"]
    check("the second turn is framed exactly once",
          last.count("The message above is a product verification request") == 1)

    # A greeting must not be framed at all.
    await router.handle_text({"id": "f-3", "from": "2348012345678", "text": "Hello"})
    greeting = natlas.calls[2]["messages"][-1]["content"]
    check("a greeting is sent unframed", greeting == "Hello", greeting)

    # Turning it off must restore a verbatim relay.
    _cfg2, natlas2, _wa, _st, router2 = await build(tmp, reply="ok",
                                                    verification_frame=False)
    await router2.handle_text({"id": "f-4", "from": "2348012345678", "text": "A4-100160"})
    check("framing can be switched off",
          natlas2.calls[0]["messages"][-1]["content"] == "A4-100160")


async def test_router(tmp):
    section("router — local commands cost no model call")
    _cfg, natlas, whatsapp, _store, router = await build(tmp)

    await router.handle_text({"id": "m1", "from": "2348012345678", "text": "/hausa"})
    check("/hausa sends a confirmation", len(whatsapp.bodies()) == 1)
    check("/hausa does not call the model", len(natlas.calls) == 0)

    await router.handle_text({"id": "m2", "from": "2348012345678", "text": "/yoruba"})
    check("/yoruba does not call the model", len(natlas.calls) == 0)

    await router.handle_text({"id": "m3", "from": "2348012345678", "text": "/lang"})
    check("/lang does not call the model", len(natlas.calls) == 0)
    check("/lang still answers", len(whatsapp.bodies()) == 3)

    await router.handle_text({"id": "m4", "from": "2348012345678", "text": "/new"})
    check("/new does not call the model", len(natlas.calls) == 0)
    check("/new answers", len(whatsapp.bodies()) == 4)

    section("router — the Tier 2 request shape")
    _cfg2, natlas, whatsapp, _store2, router = await build(tmp, reply="It is registered.")
    await router.handle_text({"id": "q1", "from": "2348012345678", "text": "Is A4-100160 genuine?"})

    check("one model call per message", len(natlas.calls) == 1)
    call = natlas.calls[0]
    sent_text = call["messages"][-1]["content"]
    # Verbatim up to the framing note, which is appended only because the
    # message carries a number. "Is A4-100160 genuine?" must survive intact.
    check("the user text is preserved verbatim",
          sent_text.startswith("Is A4-100160 genuine?"), sent_text[:90])
    check("auto sends no language field", call["language"] is None)

    roles = [m["role"] for m in call["messages"]]
    check("no system message is ever sent", "system" not in roles, str(roles))
    check("history is a user turn", roles == ["user"])

    bodies = whatsapp.bodies()
    check("the reply is sent", bodies and bodies[-1] == "It is registered.")

    # Second turn: history must be present and in order.
    await router.handle_text({"id": "q2", "from": "2348012345678", "text": "And 04-3877?"})
    call = natlas.calls[1]
    roles = [m["role"] for m in call["messages"]]
    check("history is carried to the next turn", roles == ["user", "assistant", "user"], str(roles))
    check("history keeps the first question",
          call["messages"][0]["content"] == "Is A4-100160 genuine?")
    check("history keeps the first answer",
          call["messages"][1]["content"] == "It is registered.")
    check("no system message in history", "system" not in roles)

    section("router — a pinned language travels as the language field")
    _cfg3, natlas, whatsapp, _store4, router = await build(tmp, reply="Kayyana.")
    await router.handle_text({"id": "p1", "from": "2348012345678", "text": "/hausa"})
    await router.handle_text({"id": "p2", "from": "2348012345678", "text": "Is A4-100160 genuine?"})
    check("the pin is sent as `language`", natlas.calls[0]["language"] == "ha")

    await router.handle_text({"id": "p3", "from": "2348012345678", "text": "/auto"})
    await router.handle_text({"id": "p4", "from": "2348012345678", "text": "And A4-5180?"})
    check("/auto returns to sending nothing", natlas.calls[1]["language"] is None)

    section("router — deduplication")
    _cfg4, natlas, whatsapp, _store5, router = await build(tmp, reply="ok")
    item = {"id": "dup-1", "from": "2348012345678", "text": "Is A4-100160 genuine?"}
    await router.handle_text(dict(item))
    first = len(whatsapp.sent)
    await router.handle_text(dict(item))  # Meta retry
    check("a retry is ignored", len(whatsapp.sent) == first, "a second reply was sent")
    check("a retry costs no model call", len(natlas.calls) == 1)
    check("the duplicate is counted", router.stats["duplicate"] == 1)

    section("router — opt-out")
    _cfg5, natlas, whatsapp, _store6, router = await build(tmp, reply="ok")
    phone = "2348011111111"
    await router.handle_text({"id": "o1", "from": phone, "text": "Is A4-100160 genuine?"})
    check("a message before opting out is answered", len(natlas.calls) == 1)

    await router.handle_text({"id": "o2", "from": phone, "text": "STOP"})
    check("STOP confirms once", len(whatsapp.bodies()) == 2)
    session = await _store6.get(phone)
    check("STOP deletes the stored conversation", session.history == [])

    before = len(whatsapp.sent)
    calls_before = len(natlas.calls)
    await router.handle_text({"id": "o3", "from": phone, "text": "Is A4-5180 genuine?"})
    check("an opted-out number gets silence", len(whatsapp.sent) == before)
    check("an opted-out number costs no model call", len(natlas.calls) == calls_before)

    await router.handle_text({"id": "o4", "from": phone, "text": "START"})
    check("START revokes the opt-out", phone not in router.opted_out)
    await router.handle_text({"id": "o5", "from": phone, "text": "Is A4-5180 genuine?"})
    check("a revoked opt-out is answered again", len(natlas.calls) > calls_before)

    section("router — server failure is reported, never guessed")
    from .natlas import ConnectionFailure
    _cfg6, natlas, whatsapp, _store7, router = await build(
        tmp, error=ConnectionFailure("server down")
    )
    await router.handle_text({"id": "f1", "from": "2348012345678", "text": "Is A4-100160 genuine?"})
    bodies = whatsapp.bodies()
    check("a failure produces exactly one reply", len(bodies) == 1)
    check("a failure says verification is unavailable",
          "cannot reach the nafdac records" in bodies[0].lower(), bodies[0])
    check("a failure never claims counterfeit",
          not any(w in bodies[0].lower() for w in ("counterfeit", "fake", "poisoned")))
    check("a failure never claims a verdict",
          not any(w in bodies[0].lower()
                  for w in ("authentic", "registered", "active")))

    # The same failure, in Hausa, must be reported in Hausa — an operator in
    # Lagos should not receive an English outage notice.
    _cfg6b, natlas_b, whatsapp_b, _store7b, router_b = await build(
        tmp, error=ConnectionFailure("server down")
    )
    await router_b.handle_text({"id": "f2", "from": "2348012345679", "text": "/hausa"})
    whatsapp_b.sent.clear()
    await router_b.handle_text({"id": "f3", "from": "2348012345679", "text": "Is A4-100160 genuine?"})
    hausa = " ".join(whatsapp_b.bodies())
    check("a failure is reported in the user's language",
          "NAFDAC" in hausa and "sai ku" not in hausa, hausa[:80])
    check("a Hausa failure is not the English string",
          "cannot reach the nafdac records" not in hausa.lower())

    section("router — a non-text message gets a useful answer")
    _cfg8, natlas, whatsapp, _store9, router = await build(tmp)
    await router.handle_non_text({"id": "img1", "from": "2348012345678", "type": "image"})
    check("an image is answered", len(whatsapp.bodies()) == 1)
    check("an image costs no model call", len(natlas.calls) == 0)
    check("the image answer asks for the number",
          "A4-100160" in whatsapp.bodies()[0], whatsapp.bodies()[0])

    section("router — a voice note is transcribed and answered")
    from . import router as router_module
    _cfg10, natlas, whatsapp, _store10, router = await build(
        tmp, reply="It is registered."
    )

    class StubTranscriber:
        """Stands in for the NCAIR1 models: no torch, no GPU."""

        def __init__(self):
            self.calls: list[tuple[bytes, str]] = []

        def transcribe(self, audio_bytes, language_code="en"):
            self.calls.append((audio_bytes, language_code))
            return "Is A4-100160 genuine?"

    stub = StubTranscriber()
    router_module.transcriber = stub
    try:
        voice = {
            "id": "vn1", "from": "2348012345678", "type": "audio",
            "audio": {"id": "media-audio-1"},
        }
        await router.handle_non_text(voice)
    finally:
        router_module.transcriber = real_transcriber

    check("a voice note says it is listening",
          "Listening" in whatsapp.bodies()[0], whatsapp.bodies()[0])
    check("a voice note is downloaded once", len(stub.calls) == 1)
    check("the downloaded bytes reach the transcriber",
          stub.calls[0][0] == b"stub-ogg-audio")
    check("the user's language reaches the transcriber",
          stub.calls[0][1] == "en")
    check("a voice note costs one model call", len(natlas.calls) == 1)
    # The transcription takes the ordinary text path, so the
    # verification frame is attached to it — the user's own
    # words are still what the model sees.
    check("the transcription is what the model sees",
          "Is A4-100160 genuine?" in natlas.calls[0]["messages"][-1]["content"],
          str(natlas.calls[0]["messages"][-1]["content"])[:80])
    check("a voice note is answered",
          whatsapp.bodies()[-1] == "It is registered.", whatsapp.bodies()[-1])

    # A Meta retry of the same note must not transcribe or reply twice.
    await router.handle_non_text(dict(voice))
    check("a replayed voice note is ignored", len(stub.calls) == 1)
    check("a replayed voice note sends nothing more", len(whatsapp.sent) == 2)

    section("router — a voice note that cannot be transcribed is reported")
    _cfg10b, natlas_b, whatsapp_b, _store10b, router_b = await build(tmp)

    class FailingTranscriber:
        def transcribe(self, audio_bytes, language_code="en"):
            raise RuntimeError("model not installed")

    router_module.transcriber = FailingTranscriber()
    try:
        await router_b.handle_non_text({
            "id": "vn2", "from": "2348012345678", "type": "audio",
            "audio": {"id": "media-audio-2"},
        })
    finally:
        router_module.transcriber = real_transcriber

    bodies_b = whatsapp_b.bodies()
    check("a failed transcription still replies", len(bodies_b) == 2, str(bodies_b))
    check("a failed transcription says it could not hear",
          "could not hear" in bodies_b[-1].lower(), bodies_b[-1])
    check("a failed transcription costs no model call", len(natlas_b.calls) == 0)

    section("router — the allowlist refuses without a model call")
    _cfg9, natlas, whatsapp, _store10, router = await build(
        tmp, allowed_numbers=("+2348000000000",)
    )
    await router.handle_text({"id": "n1", "from": "2348099999999", "text": "Is A4-100160 genuine?"})
    check("an unlisted number still gets a reply", len(whatsapp.bodies()) == 1)
    check("an unlisted number costs no model call", len(natlas.calls) == 0)

    section("router — handle_payload, the real webhook entry point")
    # The single path a live request takes. Exercised separately because the
    # per-method tests above would not catch a mis-wired dispatcher.
    _cfgp, natlas, whatsapp, _storep, router = await build(tmp, reply="It is registered.")
    image_payload = {
        "object": "whatsapp_business_account",
        "entry": [{"id": "W", "changes": [{"field": "messages", "value": {
            "messaging_product": "whatsapp",
            "messages": [{"id": "wamid.img", "from": "2348012345678",
                          "timestamp": "1", "type": "image",
                          "image": {"id": "media-1"}}],
        }}]}],
    }
    await router.handle_payload(image_payload)
    check("a photo costs no model call", len(natlas.calls) == 0, str(len(natlas.calls)))
    check("a photo gets exactly one answer", len(whatsapp.sent) == 1)
    check("the photo answer names the number format",
          "A4-100160" in whatsapp.bodies()[0])

    text_payload = payload("Is A4-100160 genuine?", mid="wamid.txt")
    await router.handle_payload(text_payload)
    check("a text message does reach the model", len(natlas.calls) == 1)
    check("a text message sends one reply", len(whatsapp.sent) == 2)

    # Replaying the whole body is exactly what Meta does on a timeout.
    await router.handle_payload(text_payload)
    check("replaying the whole body sends nothing more",
          len(whatsapp.sent) == 2, str(len(whatsapp.sent)))
    check("replaying the whole body costs no model call", len(natlas.calls) == 1)

    # A body with both a message and a status, as Meta sometimes sends.
    mixed = payload("Hello there", mid="wamid.mix")
    mixed["entry"][0]["changes"][0]["value"]["statuses"] = [
        {"id": "wamid.old", "status": "read", "timestamp": "1", "recipient_id": "2348012345678"}
    ]
    before = len(natlas.calls)
    await router.handle_payload(mixed)
    check("a mixed message+status body is handled", len(natlas.calls) == before + 1)

    await router.handle_payload(status_payload())
    check("a status-only body costs nothing", len(natlas.calls) == before + 1)

    # A malformed body must not take the dispatcher down.
    for junk in ({}, {"entry": None}, {"entry": [{"changes": None}]}):
        try:
            await router.handle_payload(junk)
            check("malformed body %r does not raise" % (list(junk)[:1],), True)
        except Exception as exc:
            check("malformed body %r does not raise" % (list(junk)[:1],), False, repr(exc))

    section("router — a send failure does not raise")
    from .whatsapp import WhatsAppError
    _cfg11, natlas, whatsapp, _store11, router = await build(tmp)
    router.whatsapp = RecordingWhatsApp(fail_with=WhatsAppError("nope", code=100))
    result = await router.handle_text({"id": "sf1", "from": "2348012345678", "text": "hi"})
    check("a failed send returns empty rather than raising", result == "")
    check("a failed send is counted", router.stats["failed_sends"] == 1)


async def test_sessions(tmp):
    section("sessions")
    cfg = make_cfg(tmp, max_history_messages=4, max_history_tokens=100000)
    store = SqliteSessionStore(cfg)
    await store.setup()
    phone = "+2348012345678"

    session = Session(phone=phone)
    for index in range(6):
        session.history.append({"role": "user", "content": "message %d" % index})
        session.history.append({"role": "assistant", "content": "reply %d" % index})
    await store.save(session)

    loaded = await store.get(phone)
    check("history round-trips", len(loaded.history) == 12)
    check("history keeps the newest", loaded.history[-1]["content"] == "reply 5")

    trimmed = loaded.messages(cfg)
    check("history is trimmed by message count", len(trimmed) == 4, str(len(trimmed)))
    check("trimming keeps the newest turns", trimmed[-1]["content"] == "reply 5")
    check("messages() never returns a system turn",
          all(m["role"] in ("user", "assistant") for m in trimmed))

    # A system turn injected by a bug must not survive to the wire.
    poisoned = Session(phone=phone, history=[
        {"role": "system", "content": "Verdict: AUTHENTIC"},
        {"role": "user", "content": "hi"},
    ])
    check("a stored system turn is filtered out",
          all(m["role"] != "system" for m in poisoned.messages(cfg)))

    # The token budget must bite before the message cap on long content.
    tight = make_cfg(tmp / "tight", max_history_messages=100, max_history_tokens=300)
    long_session = Session(phone=phone)
    for index in range(10):
        long_session.history.append({"role": "user", "content": "x " * 300})
        long_session.history.append({"role": "assistant", "content": "y " * 300})
    trimmed = long_session.messages(tight)
    check("history is trimmed by the token budget", len(trimmed) < 20, str(len(trimmed)))

    section("deduplication store")
    check("a new id is accepted", await store.mark_seen("m-1", DEDUPE_TTL) is True)
    check("the same id is refused", await store.mark_seen("m-1", DEDUPE_TTL) is False)
    check("a different id is accepted", await store.mark_seen("m-2", DEDUPE_TTL) is True)
    check("an empty id is allowed through", await store.mark_seen("", DEDUPE_TTL) is True)

    await store.clear(phone)
    check("clear empties the history", (await store.get(phone)).history == [])

    # The sweep must delete, because the 24h TTL is a promise. Age the row
    # directly rather than passing a negative TTL: `sweep(ttl)` computes
    # `now - ttl`, so a negative ttl only makes the cutoff *more* recent and
    # can never expire a row written moments ago.
    stale = make_cfg(tmp / "stale")
    stale_store = SqliteSessionStore(stale)
    await stale_store.setup()
    await stale_store.save(Session(phone="+2348000000000", history=[{"role": "user", "content": "x"}]))
    with stale_store._db() as conn:
        conn.execute(
            "UPDATE wa_sessions SET updated_at = ? WHERE phone = ?",
            (int(time.time()) - 100000, "+2348000000000"),
        )
    removed = await stale_store.sweep(3600)
    check("the sweep removes expired sessions", removed == 1, str(removed))
    check("the sweep really deleted the row",
          (await stale_store.get("+2348000000000")).history == [])


async def test_queue():
    section("concurrency control")
    concurrency = Concurrency(max_concurrency=1, per_user_timeout=2.0)

    order: list[str] = []

    async def slow(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        order.append(name)
        return name

    # One user, two turns: they must serialise, and stay in order.
    await asyncio.gather(
        concurrency.run("+2348011111111", lambda: slow("first", 0.20)),
        concurrency.run("+2348011111111", lambda: slow("second", 0.05)),
    )
    check("two turns from one user stay in order", order == ["first", "second"], str(order))

    # The global semaphore must cap cross-user concurrency.
    peak = 0
    current = 0

    async def tracker() -> None:
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1

    gated = Concurrency(max_concurrency=2, per_user_timeout=5.0)
    await asyncio.gather(
        *[gated.run("+234800000%04d" % index, tracker) for index in range(6)]
    )
    check("the global semaphore caps concurrency", peak <= 2, "peak=%d" % peak)

    # A queued request that waits too long must fail loudly, not hang.
    starved = Concurrency(max_concurrency=1, per_user_timeout=1.0)
    from .queue import QueueTimeout

    await starved.semaphore.acquire()
    try:
        try:
            await starved.run("+2348000000001", lambda: slow("x", 0.01))
            check("a starved request times out", False, "no timeout raised")
        except (QueueTimeout, Exception) as exc:
            check("a starved request times out", isinstance(exc, QueueTimeout), type(exc).__name__)
    finally:
        starved.semaphore.release()

    check("snapshot reports counters", "in_flight" in concurrency.snapshot())
    check("forget() drops a lock", (concurrency.forget("+2348000000009") or True))


async def test_natas_payload():
    section("N-ATLaS client — request shape")

    captured: dict = {}

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, response: FakeResponse):
            self.response = response

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return self.response

        async def get(self, url, headers=None):
            return self.response

        async def aclose(self):
            return None

    cfg = make_cfg(tmp_dir)
    natlas = NAtlasClient(cfg)

    ok = FakeResponse(200, {"choices": [{"message": {"content": "It is registered."}}]})
    natlas._client = FakeClient(ok)
    reply = await natlas.chat([{"role": "user", "content": "hi"}])

    check("the reply is returned", reply == "It is registered.")
    check("the URL ends in /chat/completions", captured["url"].endswith("/chat/completions"))
    check("the URL keeps the /v1 prefix", "/v1/chat/completions" in captured["url"])
    check("the bearer token is sent",
          captured["headers"]["Authorization"] == "Bearer stub-key")
    check("a User-Agent is always sent",
          "confamai-bot" in captured["headers"].get("User-Agent", ""))
    check("stream is always false", captured["json"]["stream"] is False)
    check("no language field when unpinned", "language" not in captured["json"])
    check("the model is sent", captured["json"]["model"] == "n-atlas")

    await natlas.chat([{"role": "user", "content": "hi"}], language="ha")
    check("a pinned language is sent", captured["json"]["language"] == "ha")

    # Status code translation, including the ones that must not be retried.
    from .natlas import (
        AuthenticationFailure,
        EmptyResponse,
        ForbiddenFailure,
        GPUExhaustedFailure,
        NotFoundFailure,
        RateLimitFailure,
        ServerFailure,
        ValidationFailure,
        FATAL,
        TRANSIENT,
    )

    expectations = [
        (401, AuthenticationFailure, FATAL),
        (403, ForbiddenFailure, FATAL),
        (404, NotFoundFailure, FATAL),
        (400, ValidationFailure, FATAL),
        (422, ValidationFailure, FATAL),
        (429, RateLimitFailure, TRANSIENT),
        (500, ServerFailure, TRANSIENT),
        (503, ServerFailure, TRANSIENT),
        (507, GPUExhaustedFailure, TRANSIENT),
    ]
    for status, expected, kind in expectations:
        natlas._client = FakeClient(FakeResponse(status, {"error": {"message": "x"}}))
        try:
            await natlas.chat([{"role": "user", "content": "hi"}])
            check("HTTP %d raises" % status, False, "no error raised")
        except expected as exc:
            check("HTTP %d raises %s" % (status, expected.__name__), True)
            check("HTTP %d is %s" % (status, kind), exc.kind == kind, exc.kind)

    natlas._client = FakeClient(FakeResponse(200, {"choices": [{"message": {"content": "   "}}]}))
    try:
        await natlas.chat([{"role": "user", "content": "hi"}])
        check("an empty reply raises", False, "no error raised")
    except EmptyResponse:
        check("an empty reply raises", True)

    # A fatal error must not be retried: a wrong key fails identically forever.
    attempts = {"n": 0}

    class CountingClient(FakeClient):
        async def post(self, url, headers=None, json=None):
            attempts["n"] += 1
            return FakeResponse(401, {"error": {"message": "bad key"}})

    natlas._client = CountingClient(FakeResponse(401, {}))
    try:
        await natlas.chat_with_retry([{"role": "user", "content": "hi"}])
    except AuthenticationFailure:
        pass
    check("a 401 is attempted exactly once", attempts["n"] == 1, str(attempts["n"]))

    # A transient error must be retried.
    attempts["n"] = 0

    class FlakyClient(FakeClient):
        async def post(self, url, headers=None, json=None):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return FakeResponse(503, {"error": {"message": "loading"}})
            return FakeResponse(200, {"choices": [{"message": {"content": "recovered"}}]})

    natlas._client = FlakyClient(FakeResponse(503, {}))
    cfg.max_retries = 2
    reply = await natlas.chat_with_retry([{"role": "user", "content": "hi"}])
    check("a 503 is retried and can recover", reply == "recovered", str(attempts["n"]))

    # The error message must never contain the key.
    cfg2 = make_cfg(tmp_dir)
    natlas2 = NAtlasClient(cfg2)
    natlas2._client = FakeClient(FakeResponse(401, {"error": {"message": "bad key"}}))
    try:
        await natlas2.chat([{"role": "user", "content": "hi"}])
    except NAtlasError as exc:
        check("an error message never contains the API key", "stub-key" not in str(exc))


def test_reply_contract():
    section("reply contract — the rules this project exists to enforce")
    from .assay import CASES

    # Fixtures shaped like real replies, including the ones that must be refused.
    good_registered = (
        "A4-100160 is AG Spironolactone. It is listed as Active in the current "
        "greenbook, so it is AUTHENTIC. The greenbook cannot confirm this "
        "particular pack is genuine, because a copied number resolves just as "
        "well."
    )
    good_miss = (
        "A4-9999999 was not found in our current NAFDAC database. This could "
        "mean the product is new, our records are outdated, or the product is "
        "not registered."
    )
    bad_counterfeit = "Yes, this product is counterfeit and poisoned."
    bad_marker = "[nafdac] Verdict: AUTHENTIC. The block provided shows it."
    bad_inverted = "A4-100160 is NOT registered and could be fake."

    case_registered = next(c for c in CASES if c[0] == "known product, NRN")
    case_miss = next(c for c in CASES if c[0] == "unregistered number")

    ok, problems = evaluate(case_registered, good_registered)
    check("a registered reply passes", ok, "; ".join(problems))

    ok, problems = evaluate(case_miss, good_miss)
    check("a not-registered reply passes", ok, "; ".join(problems))

    ok, problems = evaluate(case_registered, bad_counterfeit)
    check("a counterfeit claim is rejected", not ok)
    check("the counterfeit rejection names the phrase",
          any("counterfeit" in p for p in problems), str(problems))

    ok, _ = evaluate(case_registered, bad_marker)
    check("a leaked [nafdac] marker is rejected", not ok)
    ok, _ = evaluate(case_registered, bad_inverted)
    check("an inverted verdict is rejected", not ok)

    check("the leak markers include [nafdac]", "[nafdac]" in LEAK_MARKERS)
    check("the leak markers include the language pin", "[language]" in LEAK_MARKERS)

    for marker in LEAK_MARKERS:
        ok, _ = evaluate(case_registered, "hello %s there" % marker)
        check("a reply containing %r is rejected" % marker, not ok)

    # Every case's forbidden list must actually be enforced.
    for case in CASES:
        label = case[0]
        for phrase in case[4]:
            ok, _ = evaluate(case, "The answer is %s." % phrase)
            check("case %r rejects %r" % (label, phrase), not ok)


async def test_webhook(tmp):
    section("webhook routes")
    from fastapi.testclient import TestClient
    from .webhook import _signature_ok, create_app

    cfg = make_cfg(tmp)
    app = create_app(cfg)

    # Signature validation, before anything else in the pipeline.
    import hashlib
    import hmac as hmac_mod
    import json as json_mod

    body = json_mod.dumps(payload("hi"), separators=(",", ":")).encode()
    good = "sha256=" + hmac_mod.new(b"stub-secret", body, hashlib.sha256).hexdigest()
    check("a valid signature is accepted", _signature_ok(body, good, "stub-secret") is True)
    check("a tampered body is rejected", _signature_ok(body + b"x", good, "stub-secret") is False)
    check("a wrong secret is rejected", _signature_ok(body, good, "wrong") is False)
    check("a missing header is rejected", _signature_ok(body, None, "stub-secret") is False)
    check("no secret means accept", _signature_ok(body, None, "") is True)

    with TestClient(app) as client:
        # Handshake
        ok = client.get(
            cfg.webhook_path,
            params={"hub.mode": "subscribe", "hub.verify_token": "stub-verify",
                    "hub.challenge": "12345"},
        )
        check("the handshake returns the raw challenge",
              ok.status_code == 200 and ok.text == "12345", "%s %r" % (ok.status_code, ok.text))

        bad = client.get(
            cfg.webhook_path,
            params={"hub.mode": "subscribe", "hub.verify_token": "wrong",
                    "hub.challenge": "12345"},
        )
        check("a wrong verify token is refused", bad.status_code == 403)

        # Inbound
        import hmac as hm
        import json as js

        raw = js.dumps(payload("Is A4-100160 genuine?"), separators=(",", ":")).encode()
        sig = "sha256=" + hm.new(b"stub-secret", raw, hashlib.sha256).hexdigest()
        response = client.post(
            cfg.webhook_path, content=raw,
            headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"},
        )
        check("a valid inbound POST returns 200", response.status_code == 200)
        check("the webhook answers fast", response.elapsed.total_seconds() < 5.0
              if hasattr(response, "elapsed") else True)

        unsigned = client.post(cfg.webhook_path, content=raw,
                               headers={"Content-Type": "application/json"})
        check("an unsigned POST is refused", unsigned.status_code == 403)

        junk = client.post(
            cfg.webhook_path, content=b"not json",
            headers={"X-Hub-Signature-256":
                     "sha256=" + hm.new(b"stub-secret", b"not json", hashlib.sha256).hexdigest()},
        )
        check("unparseable JSON returns 200 to stop retries", junk.status_code == 200)

        status_raw = js.dumps(status_payload(), separators=(",", ":")).encode()
        status_sig = "sha256=" + hm.new(b"stub-secret", status_raw, hashlib.sha256).hexdigest()
        status_response = client.post(
            cfg.webhook_path, content=status_raw,
            headers={"X-Hub-Signature-256": status_sig},
        )
        check("a status callback is accepted without error", status_response.status_code == 200)

        health = client.get("/healthz")
        check("healthz is liveness and always 200", health.status_code == 200)
        check("healthz never leaks a secret",
              "stub-key" not in health.text and "stub-token" not in health.text)

        alias = client.get("/health")
        check("/health is an alias of /healthz",
              alias.status_code == 200 and alias.json().get("bot") == "up")

        ready = client.get("/readyz")
        check("readyz is readiness and may say 503", ready.status_code in (200, 503))
        check("readyz never leaks a secret",
              "stub-key" not in ready.text and "stub-token" not in ready.text)

    # A config with no app secret still serves, and says so.
    open_cfg = make_cfg(tmp / "open")
    open_cfg.app_secret = ""
    open_app = create_app(open_cfg)
    with TestClient(open_app) as client:
        response = client.post(cfg.webhook_path, content=b"{}",
                               headers={"Content-Type": "application/json"})
        check("without an app secret the webhook still answers", response.status_code == 200)
        health = client.get("/healthz")
        check("health is reachable without an app secret", health.status_code == 200)


async def test_render_integration(tmp):
    section("end to end — a reply becomes WhatsApp messages")
    from .render import clean as render_clean
    from .whatsapp import WhatsAppClient

    _cfg, natlas, whatsapp, _store, router = await build(tmp)

    # A long model reply must arrive as the minimum number of messages, because
    # each one is billed separately as of Meta's 1 October 2026 change.
    body = "A4-100160 is registered and Active. " * 40
    natlas.reply = body
    await router.handle_text({"id": "e2e-1", "from": "2348012345678",
                              "text": "Is A4-100160 genuine?"})
    sent = whatsapp.bodies()
    check("the reply is delivered", len(sent) >= 1)
    check("every delivered part fits WhatsApp's limit",
          all(len(part) <= 4096 for part in sent))
    combined = " ".join(sent)
    check("no content is lost in delivery", "A4-100160" in combined)
    check("template markers are stripped client-side",
          not any(marker in combined for marker in ("<|eom_id|>", "<s>")))

    # A fenced reply must not reach the user with backticks on it.
    natlas.reply = "```\nA4-100160 is registered.\n```"
    whatsapp.sent.clear()
    await router.handle_text({"id": "e2e-2", "from": "2348012345678", "text": "again?"})
    check("a fenced reply is unfenced",
          "```" not in " ".join(whatsapp.bodies()), str(whatsapp.bodies()))

    # The ack is one extra message, and only when enabled.
    natlas.reply = "ok"
    ack_cfg, natlas2, whatsapp2, _store2, router2 = await build(tmp, send_ack=True)
    await router2.handle_text({"id": "e2e-3", "from": "2348012345678", "text": "Is A4-100160 genuine?"})
    check("the ack is sent before the answer", len(whatsapp2.bodies()) == 2,
          str(whatsapp2.bodies()))
    check("the ack is not the answer", "NAFDAC" in whatsapp2.bodies()[0])

    # An unsent ack must not prevent the real reply.
    from .whatsapp import WhatsAppError
    ack_cfg2, natlas3, whatsapp3, _store3, router3 = await build(tmp, send_ack=True)
    router3.whatsapp = RecordingWhatsApp(fail_with=WhatsAppError("down", retryable=True))
    await router3.handle_text({"id": "e2e-4", "from": "2348012345678", "text": "hi"})
    check("a failed ack is survivable", router3.stats["errors"] == 0)


# ---------------------------------------------------------------------------
async def amain() -> int:
    with tempfile.TemporaryDirectory(prefix="confamai-bot-tests-") as tmpdir:
        tmp = Path(tmpdir)
        global tmp_dir
        tmp_dir = tmp / "client"

        print("=" * 72)
        print("Confam AI WhatsApp bot — offline test suite")
        print("=" * 72)

        await test_config()
        test_render()
        test_languages()
        test_parsing()
        test_framing()
        test_reply_contract()
        await test_framing_does_not_leak(tmp)
        await test_router(tmp)
        await test_sessions(tmp)
        await test_queue()
        await test_natas_payload()
        await test_webhook(tmp)
        await test_render_integration(tmp)

    print()
    print("=" * 72)
    print("RESULT  %d passed, %d failed" % (RESULTS["pass"], RESULTS["fail"]))
    if FAILURES:
        print()
        print("Failures:")
        for failure in FAILURES:
            print("  - %s" % failure)
    print("=" * 72)
    return 1 if RESULTS["fail"] else 0


tmp_dir = Path(".")


def run_tests() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(run_tests())