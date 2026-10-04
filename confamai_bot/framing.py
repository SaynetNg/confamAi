"""Rephrasing aid for messages that carry a NAFDAC number but no question.

Why this exists
---------------

Measured against the live server on 2026-10-04, the retrieval gate attaches a
verified block correctly for a bare number — `prompt_tokens` 1658 against a 599
baseline — and the model then throws it away and asks which product to verify:

    "A4-100160"                    -> "Would you like to verify a product?"   FAIL
    "A4-100160?"                   -> "Would you like to verify a product?"   FAIL
    "check A4-100160"              -> "Which product would you like...?"      FAIL
    "verify this number A4-100160" -> "Would you like to verify a product?"   FAIL
    "What is A4-100160?"           -> full record, VERDICT: AUTHENTIC         ok
    "Tell me about A4-100160"      -> full record                            ok

The trigger is grammatical framing, not content: a message with an interrogative
is answered, and a bare noun phrase or bare imperative is redirected. The
Confam AI persona's `BOUNDARIES` rule wins over the server's `NAFDAC_INSTRUCTION`
when there is nothing that looks like a question.

It is deterministic, not sampling noise. Four trials at temperature 0.7 and four
at 0.1 all failed. It is also the highest-frequency way anyone uses a
verification service, so it is the most damaging defect found.

What this module does
--------------------

When the user's own text already contains a NAFDAC number, append one short
instruction line that makes the request explicit. That fixes all four cases:

    "A4-100160"  + frame -> "A4-100160 - AG Spironolactone
                                VERDICT: AUTHENTIC - listed as Active"      ok

This is not a second retrieval authority, and the distinction matters
(`n_atlas_client/api.py:499-510` records what happens when there is one). Two
properties keep it safe, and both were measured rather than assumed:

1. **No greenbook data is injected.** The frame contains no product name, no
   verdict and no number. `nafdac_source` is imported for `extract_nrn` alone,
   which is pure text — it reads no database and no file, and importing it does
   not load psycopg.

2. **The intent gate still refuses non-product messages.** Applying the frame to
   every message moved "Hello" from 599 to 644 prompt tokens, a `+45` delta
   that is the frame text itself. A real greenbook block adds about 1000. So no
   greeting starts searching: the gate still says no.

The frame is only added when a number is present, which makes the blast radius
smaller still — "Hello" and "good morning" get nothing appended at all.

The real fix belongs in the server: a clause in `NAFDAC_INSTRUCTION` saying a
message that is only a NAFDAC number is a verification request. Set
`WHATSAPP_VERIFICATION_FRAME=false` once that is deployed, and this module goes
dormant.
"""

from __future__ import annotations

# Appended to the user's own turn. Deliberately names no product, no verdict and
# no number: the safety-critical values come from the server's block, and a frame
# that carried any would be both redundant and a second thing to keep in step.
DEFAULT_FRAME = (
    "\n\n[reply] The message above is a product verification request. Answer it "
    "directly, using the verified NAFDAC record for the number in it: name the "
    "product and state its VERDICT. Do not ask which product to verify — a "
    "number was given."
)

# Used when nafdac_source cannot be imported. Deliberately wider than the real
# NRN grammar, because being wrong here only means an unnecessary frame on a
# message that is plainly about a product; missing a number would mean shipping
# the bug. Requires a letter before the digits so a bare "100160" is not framed.
_FALLBACK_HINT = ("nafdac", "nafdac number", "nrn", "a4-", "a5-", "a1-", "a2-",
                  "a3-", "04-", "05-", "06-", "07-", "08-", "09-")

try:  # pragma: no cover - exercised by whichever branch the deployment takes
    from nafdac_source.extract import extract_nrn as _extract_nrn

    _EXTRACTOR_OK = True
except Exception:  # pragma: no cover - nafdac_source absent from the image
    _extract_nrn = None
    _EXTRACTOR_OK = False


def extractor_available() -> bool:
    """True when the real NRN parser is in use, False when guessing."""
    return _EXTRACTOR_OK


def has_number(text: str) -> bool:
    """Does this message carry a NAFDAC number?

    Uses the project's own `extract_nrn` when it can be imported, because that
    parser already handles `A4 - 5180`, `a4-5180`, an en-dashed `04–2045`,
    `4/1/9086` and the placeholders (`NA`, `Not available yet`) that must NOT
    be treated as numbers. A second, looser regex here would drift from it, and
    drifting parsers are exactly how a date or a dose gets looked up as a
    registration number.
    """
    if not text:
        return False

    if _extract_nrn is not None:
        try:
            return bool(_extract_nrn(text))
        except Exception:
            # A broken extractor must never be the reason a message is not
            # framed, so fall through to the hint rather than returning False.
            pass

    lowered = str(text).lower()
    return any(hint in lowered for hint in _FALLBACK_HINT)


def apply(text: str, enabled: bool = True, custom: str = "") -> str:
    """Return the turn to send, framed or not.

    Never appends to an empty message, and never appends twice: a caller that
    already framed this turn must not stack a second frame.
    """
    if not enabled or not text:
        return text
    if not has_number(text):
        return text
    if "[reply] The message above is a product verification request" in text:
        return text

    frame = str(custom or "").strip() or DEFAULT_FRAME
    return "%s%s" % (text, frame)