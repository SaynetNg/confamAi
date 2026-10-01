"""Pull a NAFDAC registration number (NRN) out of free text.

The greenbook is messy: NRNs look like `A4-100160`, `B4-6238`, `04-3877`,
`C4-1142`, sometimes with stray spaces (`A4 - 5180`), sometimes with an en dash
(`04-2045`), and a handful are written as a fraction (`4/1/9086`) or carry a
suffix letter (`A1-4924L`). Users paste any of these, plus typos, plus
sentences around them.

Two things are therefore kept separate:

    normalize_nrn()      the display form, kept close to what NAFDAC published
    normalize_nrn_key()  the lookup key, so `a4 - 5180` finds `A4-5180`

`normalize_nrn_key` is what the sources index on, so a lookup only succeeds if
the digits agree. Everything else is cosmetic.
"""

import re
import unicodedata

# The greenbook writes an NRN as an optional letter prefix, one or two digits, a
# separator, then the serial: `A4-100160`, `B4-6238`, `04-3877`, `C4-1142`,
# `A11-100325`, `BA-0906`. The separator class is loose because the source data
# uses a hyphen, every dash variant, a slash and plain spaces, sometimes with
# stray padding (`A4 - 5180`, `04- 9502`). The trailing letter group absorbs the
# `L` in `A1-4924L` and the `A` in `A9-9999A`.
_SERIAL = r"\d{2,7}"
_SEP = r"[-‐-―/]"
_PREFIX = r"[A-Za-z]{0,2}"
_SUFFIX = r"[A-Za-z]{0,2}"

# Two numeric groups is by far the common form. The trailing lookahead stops it
# from matching only the first two groups of a three-group value, so
# `B4-9999-0000` is not silently truncated to `B4-9999`.
EXTRACTION_PATTERN = re.compile(
    rf"\b(?P<prefix>{_PREFIX})(?P<first>\d{{1,2}})"
    rf"\s*(?P<sep>{_SEP})\s*(?P<serial>{_SERIAL})"
    rf"(?![-‐-―/]\s*\d)"
    rf"(?P<suffix>[A-Za-z]{{0,2}})\b"
)

# Three numeric groups covers the fraction-shaped rows such as `4/1/9086`, which
# are year/month/serial typeset with slashes. The group count is preserved when
# rebuilding the value, so the extracted key still matches the stored key.
FRACTION_PATTERN = re.compile(
    rf"\b(?P<prefix>{_PREFIX})(?P<first>\d{{1,2}})"
    rf"\s*(?P<sep>{_SEP})\s*(?P<second>\d{{1,2}})"
    rf"\s*(?P<sep2>{_SEP})\s*(?P<serial>{_SERIAL})"
    rf"(?![-‐-―/]\s*\d)"
    rf"(?P<suffix>[A-Za-z]{{0,2}})\b"
)

# A looser shape, used only as a last resort: pure digits such as `100160`.
_BARE_DIGITS = re.compile(r"\b(\d{4,8})\b")

# Values the greenbook itself uses to mean "no number was recorded". These are
# not NRNs and must never be treated as a lookup key.
NOT_A_NUMBER = {
    "",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "nil",
    "not available",
    "not available yet",
    "not applicable",
    "unknown",
    "pending",
}

# Phrases that mark a message as a NAFDAC lookup even when the number itself is
# missing or garbled. Checked against folded text (lowercased, diacritics
# stripped) so "NAFDAC" and "Nafdac" and "nafdak" all match.
_NAFDAC_WORDS = ("nafdac", "nafdak", "nafdat", "greenbook", "green book")
_LOOKUP_VERBS = (
    "verify",
    "verifies",
    "verified",
    "check",
    "checks",
    "confirm",
    "confirms",
    "validate",
    "validates",
    "lookup",
    "look up",
    "search",
    "genuine",
    "real",
    "original",
    "fake",
    "original",
    "register",
    "registered",
    "registration",
    "approved",
    "authorised",
    "authorized",
    "legit",
    "legitimate",
    "valid",
    "exist",
    "exists",
    "sabis",
    "tabbatar",
    "duba",
    "gaske",
    "sir",
)


def fold(text: str) -> str:
    """Lowercase and strip diacritics so 'NAFDAC' and 'nafdac' both match."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", str(text)).lower()
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def looks_like_nrn(value: str) -> bool:
    """True when the *whole* value is one NRN and not a placeholder.

    The full-string anchors matter: a sentence such as `is A4-100160 real?`
    contains an NRN but is not one, and treating it as a whole NRN would send
    the entire sentence to the database as a lookup key.
    """
    text = str(value or "").strip()
    if not text or text.lower() in NOT_A_NUMBER:
        return False
    if not re.search(r"\d", text):
        return False
    if re.fullmatch(r"\s*\d{3,8}\s*", text):
        return True
    for pattern in (FRACTION_PATTERN, EXTRACTION_PATTERN):
        if pattern.fullmatch(fold(text)):
            return True
    return False


def normalize_nrn(value: str) -> str:
    """Return the display form of an NRN.

    Uppercases, collapses every dash/slash variant to `-`, and removes the
    spaces around the separator. Placeholders such as `NA` return an empty
    string so callers never look them up.
    """
    text = str(value or "").strip()
    if not text or text.lower() in NOT_A_NUMBER:
        return ""
    text = fold(text).upper()
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-")
    text = text.replace("\u2014", "-").replace("\u2015", "-")
    text = re.sub(r"\s*[-/]\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_nrn_key(value: str) -> str:
    """Return the lookup key for an NRN.

    Letters, digits and dashes only, with every dash run collapsed to one. This
    makes `A4 - 5180`, `a4-5180` and `A4–5180` share the key `A4-5180`, and
    makes `04-2045` share `04-2045` whether the source wrote an en dash or a
    hyphen. Returns an empty string for placeholders.
    """
    text = normalize_nrn(value)
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def _rebuild(match: "re.Match") -> str:
    """Rebuild an NRN from a regex match, keeping the group count.

    `A4-100160` stays `A4-100160`; `4/1/9086` becomes `4-1-9086`, which is the
    same key the CSV loader builds for that row, so the lookup matches without
    needing a separate alias table.

    The suffix is attached with no separator, because the source attaches it that
    way: `A1-4924L`, `A7-4942L`, `A9-9999A`. It is dropped when it merely
    continues the surrounding word rather than belonging to the number, so
    `product 04-3877 is genuine` yields `04-3877` and not `04-3877IS`.
    """
    prefix = (match.group("prefix") or "").strip().upper()
    suffix = (match.group("suffix") or "").strip().upper()
    first = (match.group("first") or "").strip()
    serial = (match.group("serial") or "").strip()
    second = match.groupdict().get("second")

    head = f"{prefix}{first}"
    if second:
        body = f"{head}-{second}-{serial}"
    else:
        body = f"{head}-{serial}"

    if not suffix:
        return body
    # Keep the suffix only when the character after it does not continue a word
    # of letters, which is the difference between `A1-4924L` and `04-3877 is`.
    end = match.end("suffix")
    following = match.string[end : end + 1]
    if following and following.isalpha():
        return body
    return f"{body}{suffix}"


def extract_nrn_candidates(text: str) -> List[str]:
    """Every NRN-shaped token in `text`, in order, de-duplicated.

    Order is preserved and display forms are returned, so the first candidate is
    the one most likely to be what the user meant. The three-group pattern is
    tried first so a fraction-shaped value is not mis-read as its prefix alone.
    """
    if not text:
        return []
    folded = fold(text)

    # Collect matches from both patterns with their spans, then drop any match
    # that overlaps one already taken. Without this, `4/1/9086` yields both
    # `4-1-9086` (from the fraction pattern) and `1-9086` (from the plain
    # pattern matching its tail), and the tail would be reported first.
    candidates = []
    for priority, pattern in ((0, FRACTION_PATTERN), (1, EXTRACTION_PATTERN)):
        for match in pattern.finditer(folded):
            candidates.append((match.start(), match.end(), priority, _rebuild(match)))

    candidates.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))

    found: List[str] = []
    seen: set = set()
    consumed_to = -1
    for start, end, _priority, value in candidates:
        if start < consumed_to:
            continue
        key = normalize_nrn_key(value)
        if key and key not in seen:
            seen.add(key)
            found.append(value)
        consumed_to = end

    return found


def extract_nrn(text: str) -> str:
    """The single best NRN in `text`, or an empty string when there is none.

    The first match wins. When a message contains several numbers, callers that
    care should use `extract_nrn_candidates` and decide how to disambiguate.
    """
    candidates = extract_nrn_candidates(text)
    return candidates[0] if candidates else ""


def mentions_nafdac(text: str) -> bool:
    """True when the message is asking about a NAFDAC product.

    Matches on the word `NAFDAC` (any common misspelling) or on a lookup verb
    together with `greenbook`. A bare number with no surrounding intent is not
    enough, because ordinary quantities look the same.
    """
    folded = fold(text)
    if not folded:
        return False
    if any(word in folded for word in _NAFDAC_WORDS):
        return True
    has_verb = any(verb in folded for verb in _LOOKUP_VERBS)
    has_number = bool(EXTRACTION_PATTERN.search(folded) or _BARE_DIGITS.search(folded))
    return has_verb and has_number
