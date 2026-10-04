"""Turning a model reply into text WhatsApp renders cleanly.

Three jobs, in order of how much damage they do when wrong:

1. **Length.** WhatsApp rejects a text body over 4096 characters. The chunker
   exists so that never happens.

2. **Cost.** Since Meta's 1 October 2026 rate change, a service message inside
   the customer's 24-hour window is charged per message delivered — about
   $0.0101 (₦14) to a Nigerian number. So every chunk this file produces is a
   separate bill. `chunk()` therefore works hard to return exactly one message
   for a normal reply and treats splitting as the failure case, not the
   default. A one-line length warning beats a two-message bill.

3. **Legibility.** WhatsApp gives no formatting language of its own; it
   interprets WhatsApp markup (`*bold*`, `_italic_`, ``` ``` ```). The model is
   instructed to answer in plain text, so anything that looks like a code fence
   is an accident and is flattened rather than left to render as a grey box.

Nothing here rewrites, translates or summarises the reply. The model's wording
is the product: the safety rules about never calling a product counterfeit live
in the server's NAFDAC instruction block, and every transformation below is
proved in tests.py not to touch a product name, a number, a date or a verdict.
"""

from __future__ import annotations

import re

# The hard limit Meta enforces on a text message body.
WHATSAPP_MAX_CHARS = 4096

# Kept as a safety margin rather than filling 4096 exactly, so that a reply one
# character over the edge still produces a single message on the first pass.
_SOFT_LIMIT = 4000

# Artifacts that occasionally leak out of remote chat templates. Copied from
# n_atlas_client/api.py so the bot and the laptop strip exactly the same set;
# two lists that drift is how a stray "<s>" ends up in a customer's inbox.
TEMPLATE_ARTIFACTS = (
    "",
    "",
    "",
    "<|eom_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "<s>",
    "</s>",
)

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?|```")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_LONG_TAIL = re.compile(r"(.{160,}?)(?:\n+|$)", re.DOTALL)


def strip_template_artifacts(text: str) -> str:
    """Remove chat-template control tokens from a reply.

    Deliberately narrow. It strips known control markers and nothing else,
    because a wider "cleaner" on a safety-critical answer risks deleting the
    evidence it was meant to tidy.
    """
    if not text:
        return ""
    out = str(text)
    for token in TEMPLATE_ARTIFACTS:
        if token and token in out:
            out = out.replace(token, "")
    return out.strip()


def flatten_fences(text: str) -> str:
    """Remove ``` code fences, keeping the text they wrapped.

    The system prompt asks for plain text, so a fence is always an artifact.
    The inner lines are kept: if the model did format something as code, that
    content is still the answer and dropping it would lose a product number.
    """
    if not text or "```" not in text:
        return text or ""
    return _FENCE.sub("", text)


def tidy(text: str) -> str:
    """Whitespace normalisation only. Never touches wording."""
    if not text:
        return ""
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = _TRAILING_WS.sub("\n", out)
    out = _EXCESS_BLANK_LINES.sub("\n\n", out)
    return out.strip()


def clean(text: str) -> str:
    """The full cleanup chain, in the one order that is safe.

    Artifacts first: a template token can otherwise glue two lines together and
    defeat the whitespace pass. Fences before whitespace: removing a fence
    leaves the blank lines that surrounded it. Tidy last, because everything
    before it can leave trailing space behind.
    """
    out = strip_template_artifacts(text)
    out = flatten_fences(out)
    out = tidy(out)
    return out


def _split_soft(text: str, limit: int) -> list[str]:
    """Break one long message at the best boundary available.

    Preference order, because readability degrades but correctness does not:
    a blank line, then any line break, then a sentence end, then a space. Only
    a wall of unbroken characters forces a hard cut.
    """
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]

        cut = -1
        for pattern in (r"\n\n", r"\n", r"[.!?]\s", r"\s"):
            found = -1
            for match in re.finditer(pattern, window):
                found = match.end()
            if found > 0:
                cut = found
                break

        if cut <= 0:
            # No boundary anywhere in the window: hard cut, and the extra
            # message is the price of exceeding Meta's limit at all.
            cut = limit

        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining.rstrip())
    return [part for part in parts if part]


def chunk(text: str, limit: int = _SOFT_LIMIT) -> list[str]:
    """Split a reply into WhatsApp-sized messages.

    `limit` defaults below the 4096 hard cap so a reply just over the edge
    still arrives as one message. Returns a single-element list for anything
    that already fits, which is the overwhelmingly common case and the one
    that costs nothing.
    """
    if not text:
        return []
    limit = max(200, min(int(limit), WHATSAPP_MAX_CHARS))
    if len(text) <= limit:
        return [text]
    return _split_soft(text, limit)


def would_split(text: str, limit: int = _SOFT_LIMIT) -> bool:
    """True when this reply will cost more than one delivered message.

    Exposed so a test can assert the normal case never splits, and so an
    operator can log the rate without shipping the reply twice.
    """
    return len(chunk(text, limit)) > 1


# --- acknowledgement language --------------------------------------------
# Used ONLY for the short "checking..." line sent the moment a message is
# accepted, because that is sent before the model has replied and therefore
# before the server's language detector has run. The reply itself is never
# routed on this — the server owns that decision, and this module has no say
# in it.
#
# It is a character-and-keyword heuristic, not a language model, and it is
# allowed to be wrong in exactly one direction: an English ack for a Hausa
# question is harmless, because the real answer arrives a few seconds later in
# the correct language.

# Yoruba and Igbo share ọ (with a dot below), so ọ alone identifies neither.
# These two sets hold only characters that are exclusive to one of the two, and
# they are checked in this order for that reason: an Igbo message checked
# against the Yoruba set first gets claimed as Yoruba on the strength of a
# letter both languages use.
_YOROBA_ONLY = set("ẹṣńìẹọ") - set("ọ")
_IGBO_ONLY = set("ịụṅ")  # ụ and ị do not occur in standard Yoruba orthography
_SHARED = set("ọ")

# Hausa is written in plain ASCII, so it has no exclusive characters at all and
# is recognised from its function words instead. "na" is excluded on purpose: it
# is also an English preposition, and admitting it would classify half of
# English as Hausa.
_HAUSA_WORDS = frozenset(
    {
        "sannu", "yaya", "ina", "sai", "ka", "ki", "ku", "ke",
        "sauraya", "wahala", "kuma", "yadda", "wannan", "shine", "ta",
        "don", "idan", "samu", "gaisuwa", "barka", "daga", "gida",
        "shin", "wai", "kwatanta", "tabbatar", "mai", "kuma", "yayin",
        "gobai", "jima", "gari", "rani", "dar", "yawa", "safi", "kafin",
        "tuma", "sosai", "amfani", "maganin", "zauna", "za", "zaa",
        "yana", "tana", "akwai", "babu", "ina", "kuma", "wai", "sai",
        "ina", "ina", "ina", "ina", "ina", "ina", "ina",
    }
)
_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "it", "am", "i", "in", "on", "of", "to", "and", "na"}
)


def detect_ack_language(text: str) -> str:
    """Best-effort language code for the acknowledgement line only.

    Returns one of `en`, `ha`, `yo`, `ig`. Diacritics decide first because they
    are unambiguous; Hausa, having none, is recognised from its function words.
    The default is English, which is the correct fallback for an unknown
    language — and it is a safe direction to be wrong in, because the real
    answer arrives seconds later in the correct language.
    """
    if not text:
        return "en"
    lowered = str(text).lower()
    chars = set(lowered)

    # Exclusive diacritics beat everything, in either order, because a message
    # can legitimately mix scripts.
    if chars & _IGBO_ONLY:
        return "ig"
    if chars & _YOROBA_ONLY:
        return "yo"

    words = set(re.findall(r"[a-z']+", lowered)) - _STOPWORDS
    hits = len(words & _HAUSA_WORDS)
    if hits >= 2:
        return "ha"
    if hits == 1 and len(words) <= 3:
        # One Hausa word in a short message is meaningful; the same word buried
        # in a long English sentence is not.
        return "ha"

    # ọ with nothing to disambiguate it: Hausa has no ọ, so it can only be
    # Yoruba or Igbo, and Hausa has already been ruled out above.
    if chars & _SHARED:
        return "yo"

    return "en"


_ACKS = {
    "en": "Checking the NAFDAC records — one moment.",
    "ha": "Na dubi bayanai na NAFDAC — ka waita da dama.",
    "yo": "Mo ń wo àkọsọ NAFDAC — kọ́kọ́rò kíákíọ.",
    "ig": "Anyị na-akọta nke NAFDAC — bịa nnọọ.",
}


# Values that switch the acknowledgement off entirely. An operator who would
# rather have exactly one message per exchange sets WHATSAPP_ACK_TEXT to one of
# these; an unset or empty value means "auto-pick a language", which is not the
# same intent as "off".
ACK_OFF_VALUES = frozenset({"0", "off", "no", "none", "false", "disable", "disabled"})


def acknowledgement(text: str = "", configured: str = "") -> str:
    """The one-line acknowledgement for this message.

    Three states, deliberately distinct because they are three different
    operator intentions:

        unset / empty    pick a wording in the message's own language
        off / none       send no acknowledgement at all
        any other text   send exactly that, in one language, unchanged
    """
    value = str(configured or "").strip()
    if value.lower() in ACK_OFF_VALUES:
        return ""
    if value:
        return value
    return _ACKS[detect_ack_language(text)]