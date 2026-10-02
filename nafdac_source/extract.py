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
    """Lowercase and strip diacritics so 'NAFDAC' and 'nafdac' both match.

    Also the reason `Báwo` and `Bawo`, or `ọdụ` and `odu`, reach the same
    entry: a user typing without diacritics is not typing a different word.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", str(text)).lower()
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


# Nouns that mean a medicine, a drug or a packaged product. These widen the
# trigger beyond the word NAFDAC so a user who asks "is this malaria tablet
# genuine?" in their own language reaches the greenbook at all.
#
# Written in the languages' own orthography, then folded at import time.
# `fold()` strips diacritics, so `ọdụ` is matched as `odu` and a user who types
# either form hits the same entry.
#
# Kept deliberately short and high-precision. A loose word like Yoruba `ewu`
# (bag) or `ikun` (rope) would fire on unrelated conversation, and a trigger
# that fires on ordinary chat is worse than no trigger at all, because every
# ordinary message would then cost a greenbook lookup.
_MULTILINGUAL_PRODUCT_NOUNS = (
    # English
    "drug",
    "drugs",
    "medicine",
    "medicines",
    "medication",
    "medications",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    # Hausa
    "magari",
    "kwayoyi",
    "ruwan tabo",
    # Yoruba
    "oro",
    "oogun",
    "abọ",
    "ojú",
    # Igbo
    "ọdụ",
    "ọdọ",
    "nwanwị",
    "ụdụ",
)


_WORD = re.compile(r"[a-z0-9]+")
_FOLDED_PRODUCT_NOUNS = frozenset(
    fold(word) for word in _MULTILINGUAL_PRODUCT_NOUNS
)


def _tokens(text: str) -> List[str]:
    """Word-split a folded string, so short tokens cannot match inside words."""
    return _WORD.findall(fold(text))


def extract_search_terms(text: str, max_terms: int = 3) -> List[str]:
    """Pull greenbook search terms out of free text.

    Returns a list, most distinctive first, and an empty list when nothing
    usable is left — the caller's signal not to search at all.

    A list rather than one joined string because the greenbook is ranked by
    substring match: joining "paracetamol syrup" into a single needle would match
    nothing, since no product name contains that exact pair. The terms are
    matched independently and the best hit wins.

    The greenbook indexes English product names, ingredients and applicants, so
    this deliberately returns plain terms. A Yoruba or Igbo message is expected
    to have been translated by the model, which knows the language, before it
    reaches this function.

    Stop words are dropped and the longest terms are kept, because a greenbook
    name is specific ("Spironolactone") while the sentence around it is not.
    """
    folded = fold(text)
    if not folded:
        return []

    # An NRN in the text is a far better query than any word around it.
    nrn = extract_nrn(text)
    if nrn:
        return [nrn]

    tokens = _tokens(folded)
    terms = [
        token
        for token in tokens
        if len(token) >= _MIN_QUERY_TERM and token not in _QUERY_STOPWORDS
    ]
    if not terms:
        return []

    # Longest first, so a distinctive multi-word drug name beats the short
    # filler around it. dict.fromkeys de-duplicates while keeping first-seen
    # order, which matters because the original order is the user's emphasis.
    unique = list(dict.fromkeys(terms))
    ordered = sorted(unique, key=lambda token: -len(token))
    return ordered[: max(1, max_terms)]


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


# Words that carry no product meaning and would otherwise end up as the search
# query. English filler plus the local function words that crowd out a real
# product name.
_BASE_STOPWORDS = frozenset(
    {
        # English
        "a", "an", "the", "this", "that", "these", "those", "is", "it", "are",
        "am", "was", "were", "be", "been", "do", "does", "did", "can", "could",
        "will", "would", "should", "have", "has", "had", "please", "want", "need",
        "know", "tell", "help", "check", "verify", "genuine", "real", "original",
        "fake", "safe", "use", "used", "using", "buy", "bought", "get", "got",
        "any", "some", "about", "for", "from", "with", "me", "my", "i",
        "you", "your", "and", "or", "not", "no", "yes", "of", "in", "on", "at",
        "to", "so", "if", "then", "what", "which", "who", "how", "when", "where",
        "why", "there", "here", "pack", "packet", "box", "nafdac", "greenbook",
        # Hausa
        "wannan", "wancan", "shi", "ita", "su", "mu", "ku", "na", "ta", "ba",
        "tafi", "yana", "yake", "ina", "don", "kane", "kanwa", "ina", "san",
        "saboda", "tunda", "yanzu", "gaske", "gaskiya",
        # Yoruba
        "nibii", "mo", "ko", "ati", "ibi", "yen", "won", "wonii", "ti", "si",
        "fun", "pelu", "lati", "ati", "julo", "jo", "kini", "se", "ba",
        # Igbo
        "maka", "mgbe", "ndi", "gi", "ka", "nwa", "onye", "nke", "a", "na",
        "ya", "ha", "m", "di", "ga", "ere",
    }
)

# Minimum length for a search term. One or two letter fragments produce
# nonsense matches against 8930 rows.
_MIN_QUERY_TERM = 3

# The product nouns are deliberately NOT stop words. `magari`, `oro` and `odu`
# are the most useful search terms a local-language message can offer, because
# they are often the only product-related word the user typed. Subtracting the
# noun set here rather than hand-editing it below means a noun added to
# `_MULTILINGUAL_PRODUCT_NOUNS` can never be silently swallowed by this list.
#
# The authenticity words (`genuine`, `to`, `eziokwu`) stay in, because they
# describe the question rather than the product and match nothing in the
# greenbook.
_QUERY_STOPWORDS = _BASE_STOPWORDS - _FOLDED_PRODUCT_NOUNS
