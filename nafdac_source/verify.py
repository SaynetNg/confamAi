"""verify_nafdac_product(): resolve one NRN against every available source.

This is the function the client and the server both call. It is the single
place that decides what "verified" means, so the two never disagree.

The result is a `VerificationResult` rather than a bare record, because the
answer depends on more than yes/no:

    found           did any source have this NRN?
    nrn             the number that was actually looked up
    matches         every record returned, merged across sources
    ambiguous       more than one distinct product carries this NRN
    authorised      is the product currently authorised, or None if unknown
    sources_checked which sources ran, and which failed and why
    confidence      how much weight to put in the answer

Sources are merged, not raced. The first source that returns a record for a
field wins, later sources only fill gaps, and the record keeps a list of every
source that confirmed it. A field present in PostgreSQL but missing from the
CSV therefore comes from PostgreSQL, and vice versa.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extract import (
    extract_nrn,
    looks_like_nrn,
    normalize_nrn,
    normalize_nrn_key,
)
from .records import (
    STATUS_UNKNOWN,
    VERDICT_AUTHENTIC,
    VERDICT_FLAGGED,
    VERDICT_NOT_REGISTERED,
    VERDICT_UNKNOWN,
    ProductRecord,
    is_authorised,
    merge_records,
    render_record,
)
from .sources import NafdacSource, SourceResult, build_sources

# Where the greenbook files live, relative to the project root. Overridable
# through NAFDAC_CSV_PATH / NAFDAC_XLSX_PATH / NAFDAC_DATABASE_URL.
NAFDAC_DEFAULT_PATHS: Dict[str, str] = {
    "csv": "nafdac_scraper/nafdac_greenbook.csv",
    "xlsx": "nafdac_scraper/nafdac_greenbook.xlsx",
}

NAFDAC_URL_TEMPLATE = "https://greenbook.nafdac.gov.ng/products/details/{product_id}"

# Confidence levels.
CONFIDENCE_VERIFIED = "verified"      # PostgreSQL and at least one file agree
CONFIDENCE_SINGLE = "single-source"   # only one source had it
CONFIDENCE_LOCAL = "local-file"       # files only, no database configured
CONFIDENCE_SUPERSEDED = "superseded"  # only in the history table
CONFIDENCE_NONE = "none"              # not found anywhere


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


@dataclass
class VerificationResult:
    """The full outcome of one verification."""

    nrn: str = ""
    found: bool = False
    matches: List[ProductRecord] = field(default_factory=list)
    ambiguous: bool = False
    authorised: Optional[bool] = None
    confidence: str = CONFIDENCE_NONE
    sources_checked: List[str] = field(default_factory=list)
    sources_failed: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    searched: bool = False

    @property
    def primary(self) -> Optional[ProductRecord]:
        """The record to show when there is exactly one, else the first."""
        return self.matches[0] if self.matches else None

    @property
    def status(self) -> str:
        record = self.primary
        return record.status if record else STATUS_UNKNOWN

    @property
    def verdict(self) -> str:
        """The user-facing answer: authentic, flagged, not_registered or unknown.

        Aggregated across matches with the worst verdict winning. A number
        carrying one flagged product and one active product must never be
        reported as authentic, so `flagged` beats `authentic` here and the
        `ambiguous` flag separately tells the user to ask which product they
        meant.
        """
        if not self.searched:
            return VERDICT_UNKNOWN
        if not self.matches:
            return VERDICT_NOT_REGISTERED

        verdicts = {record.verdict for record in self.matches}
        if VERDICT_FLAGGED in verdicts:
            return VERDICT_FLAGGED
        if verdicts == {VERDICT_AUTHENTIC}:
            return VERDICT_AUTHENTIC
        return VERDICT_UNKNOWN

    @property
    def verdict_reason(self) -> str:
        """Why the verdict came out that way, in the user's terms."""
        verdict = self.verdict
        if verdict == VERDICT_NOT_REGISTERED:
            return (
                "no available source lists this number, which is not proof the "
                "product is counterfeit"
            )
        if verdict == VERDICT_UNKNOWN and not self.searched:
            return "no NAFDAC number was found to look up"
        if verdict == VERDICT_UNKNOWN:
            return "the status recorded for this number is not one this code recognises"

        reasons: List[str] = []
        for record in self.matches:
            reason = record.verdict_reason
            if reason not in reasons:
                reasons.append(reason)
        return "; ".join(reasons)

    @property
    def greenbook_url(self) -> str:
        record = self.primary
        if not record or not record.product_id:
            return ""
        if record.source_url:
            return record.source_url
        return NAFDAC_URL_TEMPLATE.format(product_id=record.product_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nrn": self.nrn,
            "found": self.found,
            "ambiguous": self.ambiguous,
            "authorised": self.authorised,
            "status": self.status,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "confidence": self.confidence,
            "greenbook_url": self.greenbook_url,
            "match_count": len(self.matches),
            "matches": [record.to_dict() for record in self.matches],
            "sources_checked": list(self.sources_checked),
            "sources_failed": dict(self.sources_failed),
            "notes": list(self.notes),
            "searched": self.searched,
        }

    def render(self) -> str:
        """A plain-text block suitable for a terminal or a model prompt.

        Written to be read by a person first. N-ATLaS quotes these values
        rather than re-deriving them, which is the point: the model must not
        invent a product name or an approval date that is not in this block.
        """
        if not self.searched:
            return (
                "No NAFDAC number was found in the message. Ask the user for the "
                "NAFDAC registration number, which looks like A4-100160 or 04-3877."
            )

        if not self.found:
            lines = [
                f"NAFDAC {self.nrn}: NOT FOUND in the NAFDAC Greenbook.",
                f"  Verdict: {self.verdict.upper()} - {self.verdict_reason}.",
                "  This number does not appear in any available source.",
            ]
            if self.sources_checked:
                lines.append(f"  Sources checked: {', '.join(self.sources_checked)}")
            if self.sources_failed:
                lines.append(
                    "  Could not check: "
                    + ", ".join(f"{name} ({why})" for name, why in self.sources_failed.items())
                )
            lines.append(
                "  Do not treat this as proof the product is fake. The greenbook "
                "may be out of date, or the number may be mistyped."
            )
            lines.append(
                "  NEVER call a product counterfeit or fake: the greenbook "
                "records registrations, not counterfeiting. Tell the user the "
                "number was not found in this export, and ask them to check the "
                "number printed on the pack."
            )
            lines.append(
                "  Answer in the language the user wrote in, including Hausa, "
                "Yoruba or Igbo."
            )
            return "\n".join(lines)

        lines: List[str] = []
        if self.ambiguous:
            lines.append(
                f"NAFDAC {self.nrn}: found {len(self.matches)} different products "
                "registered under this number. Show all of them and ask which one "
                "the user means."
            )
        else:
            state = (
                "AUTHORISED"
                if self.authorised is True
                else "NOT CURRENTLY AUTHORISED"
                if self.authorised is False
                else "STATUS NOT RECORDED"
            )
            lines.append(f"NAFDAC {self.nrn}: found - {state}.")

        lines.append(f"  Verdict: {self.verdict.upper()} - {self.verdict_reason}.")
        if self.verdict == VERDICT_AUTHENTIC:
            lines.append(
                "  The greenbook lists this product as currently authorised. Tell "
                "the user it is registered, but that the greenbook cannot confirm "
                "this particular pack is genuine: a copied number also resolves "
                "here."
            )
        elif self.verdict == VERDICT_FLAGGED:
            lines.append(
                "  Tell the user this product is flagged and should not be used, "
                "and give the reason above. Do not soften this into a suggestion."
            )

        # Before the records, not after: callers cap this block and an ambiguous
        # number can produce several, so anything placed at the end risks being
        # truncated away. These rules are the whole point of the block.
        lines.append(
            "  Report only the values listed below. If the user asks about a "
            "field that is not listed, say the greenbook does not record it. "
            "Never supply a product name, ingredient, strength, applicant, "
            "manufacturer, date or status from memory."
        )
        lines.append(
            "  Use the VERDICT line as the answer to 'is it genuine?', but do not "
            "invent a stronger one. NEVER call a product counterfeit or fake: the "
            "greenbook records registrations, not counterfeiting."
        )
        lines.append(
            "  Answer in the language the user wrote in, including Hausa, Yoruba "
            "or Igbo. Keep product names, NAFDAC numbers, dates and statuses "
            "exactly as written."
        )
        lines.append("")

        for record in self.matches:
            lines.append(render_record(record))

        if self.greenbook_url:
            lines.append(f"  Greenbook page: {self.greenbook_url}")
        lines.append(f"  Confidence: {self.confidence}.")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def _merge_matches(results: List[SourceResult]) -> List[ProductRecord]:
    """Collapse per-source results into one list, merging on Product ID.

    Two sources returning the same product produce one record carrying both
    source names. Two different products sharing an NRN stay separate, because
    that ambiguity is real and the user needs to see it.
    """
    merged: Dict[Any, ProductRecord] = {}
    order: List[Any] = []

    for result in results:
        for record in result.matches:
            # Product ID is the greenbook's real primary key. Fall back to the
            # NRN key plus the name so a row without a Product ID still merges
            # consistently across sources.
            identity: Any = record.product_id
            if identity is None:
                identity = (record.nrn_key, record.product_name)

            if identity in merged:
                merged[identity] = merge_records(merged[identity], record)
            else:
                merged[identity] = record
                order.append(identity)

    return [merged[identity] for identity in order]


def _score_confidence(
    matches: List[ProductRecord],
    source_results: List[SourceResult],
    sources_failed: Dict[str, str],
) -> str:
    """How much to trust this answer, and why.

    The distinctions that matter:

      verified     the database and a local file agree
      single-source exactly one place had this product
      local-file   only the local files had it, and the database was not
                   available, so a superseding update would not show
      superseded   the product is no longer in the current greenbook and only
                   its history row matched

    Decided from the per-source results rather than from a record's fields,
    because whether the database ran and found nothing is a different fact
    from whether the database ran and found something.
    """
    if not matches:
        return CONFIDENCE_NONE

    matched = [result for result in source_results if result.matches]
    matched_names = {result.source for result in matched}
    from_history = any(result.from_history for result in matched)

    if from_history and "postgres" in matched_names:
        return CONFIDENCE_SUPERSEDED

    file_confirmed = bool(matched_names & {"csv", "xlsx"})
    database_confirmed = "postgres" in matched_names

    if database_confirmed and file_confirmed:
        return CONFIDENCE_VERIFIED

    if database_confirmed:
        return CONFIDENCE_SINGLE

    if file_confirmed:
        # The database did not confirm. That is only a weakness if the database
        # actually ran: an unconfigured or unreachable one is the local-file
        # case, whereas a database that ran and had no row means the two are
        # simply out of step, which is a single-source answer.
        postgres_ran = any(
            result.source == "postgres" and result.ok for result in source_results
        )
        if postgres_ran and "postgres" not in sources_failed:
            return CONFIDENCE_SINGLE
        return CONFIDENCE_LOCAL

    return CONFIDENCE_NONE


def classify(status: str) -> str:
    from .records import classify_status

    return classify_status(status)


def verify_nafdac_product(
    nrn_or_text: str,
    sources: Optional[List[NafdacSource]] = None,
    include_history: bool = True,
) -> VerificationResult:
    """Verify one NAFDAC registration number.

    `nrn_or_text` may be a bare number (`A4-100160`) or a whole user message
    (`"Is A4-100160 a real drug?"`); an NRN is extracted from it when needed.

    When `sources` is omitted the default list is built from the environment,
    so the same call works on the laptop and on Kaggle without configuration.
    """
    result = VerificationResult()

    if sources is None:
        sources = build_sources(
            csv_path=_env_path("NAFDAC_CSV_PATH"),
            xlsx_path=_env_path("NAFDAC_XLSX_PATH"),
            database_url=os.environ.get("NAFDAC_DATABASE_URL", ""),
        )

    if not nrn_or_text or not str(nrn_or_text).strip():
        return result

    text = str(nrn_or_text).strip()
    # A bare number is taken as-is, but a whole sentence ("is A4-100160 real?")
    # must be mined for the number, so only trust the direct form when the input
    # really is a single NRN-shaped token.
    nrn = normalize_nrn(text) if looks_like_nrn(text) else ""
    if not normalize_nrn_key(nrn):
        nrn = extract_nrn(text)
    result.nrn = nrn
    result.searched = bool(normalize_nrn_key(nrn))
    if not result.searched:
        return result

    if not include_history:
        sources = [
            source for source in sources if getattr(source, "name", "") != "postgres"
        ]

    key = normalize_nrn_key(nrn)
    source_results: List[SourceResult] = []
    for source in sources:
        try:
            source_result = source.search(key)
        except Exception as exc:  # a source must never break the lookup
            source_result = SourceResult(
                source=getattr(source, "name", "unknown"),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        source_results.append(source_result)

        if source_result.ok:
            result.sources_checked.append(source_result.source)
        else:
            result.sources_failed[source_result.source] = (
                source_result.error or "unavailable"
            )

    matches = _merge_matches(source_results)
    result.matches = matches
    result.found = bool(matches)
    if not matches:
        return result

    # Ambiguity is about distinct products, not about how many sources agreed.
    identities = {
        (record.product_id, record.product_name) for record in matches
    }
    result.ambiguous = len(identities) > 1

    if result.ambiguous:
        result.notes.append(
            "Several distinct products share this NAFDAC number. The greenbook "
            "allows this, so confirm which product the user means."
        )

    primary = result.primary
    if primary is not None:
        result.authorised = primary.authorised

    result.confidence = _score_confidence(
        matches, source_results, result.sources_failed
    )

    if result.confidence == CONFIDENCE_LOCAL:
        result.notes.append(
            "Confirmed from the local greenbook files only. The PostgreSQL "
            "database was not available, so a superseding update would not show."
        )
    elif result.confidence == CONFIDENCE_SUPERSEDED:
        result.notes.append(
            "This product is no longer in the current greenbook table. Only its "
            "superseded history row matched."
        )

    if any(record.duplicate_marker for record in matches):
        result.notes.append(
            "The greenbook lists at least one of these as a duplicate entry."
        )

    return result


def _as_needles(query: Any) -> List[str]:
    """Accept a string or a sequence of terms and return folded needles.

    A single string is treated as one needle, so `search_products("paracetamol
    syrup")` keeps behaving exactly as before: it matches a product whose name
    contains that whole phrase. Callers that want per-term matching pass a list.
    """
    if query is None:
        return []
    if isinstance(query, str):
        text = query.strip().lower()
        return [text] if text else []
    needles: List[str] = []
    for item in query:
        text = str(item or "").strip().lower()
        if text and text not in needles:
            needles.append(text)
    return needles


def search_products(
    query: Any,
    sources: Optional[List[NafdacSource]] = None,
    limit: int = 20,
) -> List[ProductRecord]:
    """Free-text search across product name, ingredients and applicant.

    `query` is either one string or a list of search terms, matched
    independently. Returns a plain ranked list, which is what the REST endpoint
    and the `/search` command want. For anything that feeds the model, use
    `find_products()` instead: it returns a `SearchResult` that renders itself as
    an anti-hallucination block and carries the per-record verdicts.

    `query` is normally the output of `extract_search_terms()`, which drops stop
    words and keeps the distinctive ones. Matching is whole-word, so an ordinary
    verb does not hit a brand that contains it.
    """
    needles = _as_needles(query)
    if not needles:
        return []
    if sources is None:
        sources = build_sources(
            csv_path=_env_path("NAFDAC_CSV_PATH"),
            xlsx_path=_env_path("NAFDAC_XLSX_PATH"),
            database_url=os.environ.get("NAFDAC_DATABASE_URL", ""),
        )

    scored: List[tuple] = []

    for source in sources:
        loader = getattr(source, "load", None)
        if not callable(loader):
            continue
        try:
            index = loader()
        except Exception:
            continue
        scored.extend(_candidates(index, needles, with_counts=True))

    # Ordered by how much of the query each record matched, then by rank. Ranking
    # on rank alone let a one-word overlap beat a record that matched every term:
    # "Alben Paracetamol Drops" returned "ALBEN AMLODIPINE 5MG TABLET", because
    # both matched "alben" and the tie fell through to alphabetical order.
    deduped: Dict[Any, tuple] = {}
    for rank, matched, record in scored:
        identity = record.product_id or (record.nrn_key, record.product_name)
        sort_key = (-matched, rank)
        current = deduped.get(identity)
        if current is None or sort_key < current[0]:
            deduped[identity] = (sort_key, record)

    ordered = sorted(
        deduped.values(),
        key=lambda item: (item[0], item[1].product_name.lower()),
    )
    return [record for _sort_key, record in ordered[: max(1, limit)]]


_WORD_RE = re.compile(r"[a-z0-9]+")

# Built once per loaded source index. Keyed on id() of that index, with the
# index itself kept in the value so its id cannot be reused by a later object.
_SCAN_INDEX: Dict[int, tuple] = {}


def _scan_index(index: Any) -> tuple:
    """An inverted word index over one source's loaded rows.

    Every message is searched now, so this is the hot path. Scanning the raw
    strings of 8930 rows for every message measured about 0.9s; scoring from
    per-record word sets brought it to about 35s of work over the whole run, and
    this removes the scan entirely: a lookup becomes three dict gets instead of
    8930 record comparisons.

    Whole words, not substrings. Greenbook names are ordinary English, and a
    substring test makes `treat` match "Levotreat" and `cold` match "Coldkris",
    which attaches a list of candidates to "how do I treat a cold?".
    """
    cached = _SCAN_INDEX.get(id(index))
    if cached is not None:
        return cached[1]

    by_identity: Dict[Any, ProductRecord] = {}
    for records in index.values():
        for record in records:
            identity = record.product_id or (record.nrn_key, record.product_name)
            if identity not in by_identity:
                by_identity[identity] = record

    names: Dict[str, List[Any]] = {}
    ingredients: Dict[str, List[Any]] = {}
    applicants: Dict[str, List[Any]] = {}
    exact: Dict[str, List[Any]] = {}
    nrns: List[tuple] = []

    for identity, record in by_identity.items():
        name = record.product_name.lower()
        for word in _WORD_RE.findall(name):
            names.setdefault(word, []).append(identity)
        for word in _WORD_RE.findall(record.active_ingredients.lower()):
            ingredients.setdefault(word, []).append(identity)
        for word in _WORD_RE.findall(record.applicant_name.lower()):
            applicants.setdefault(word, []).append(identity)
        if name:
            exact.setdefault(name, []).append(identity)
        nrns.append((record.nrn_key.lower(), identity))

    built = (by_identity, names, ingredients, applicants, exact, nrns)
    _SCAN_INDEX[id(index)] = (index, built)
    return built


def _candidates(index: Any, needles: List[str], with_counts: bool = False) -> List[tuple]:
    """Every (rank, record) matching any needle, best rank per record.

    `with_counts` additionally reports how many distinct needles each record
    matched, as a 3-tuple. It is off by default because the 2-tuple shape is
    depended on elsewhere, and it exists because "matched at least one word" is
    not the same as "matched the query": `Alben Paracetamol Drops` and
    `ALBEN AMLODIPINE 5MG TABLET` both match "alben" at the same rank, so
    ordering on rank alone returned the wrong product first.

    Ranks, lower being better:

      1  the NRN or the whole product name, exactly
      2  a substring of the NRN
      3  a whole word at the start of the name
      4  a whole word anywhere in the name
      5  a whole word in the active ingredients
      6  a whole word in the applicant name

    Only the NRN is matched as a substring, and only when the needle contains a
    digit, so `/search 5180` still finds `A4-5180` without paying for 8930
    string tests on every message.
    """
    by_identity, names, ingredients, applicants, exact, nrns = _scan_index(index)
    best: Dict[Any, int] = {}
    counts: Dict[Any, int] = {}

    def offer(identity: Any, rank: int) -> None:
        current = best.get(identity)
        if current is None or rank < current:
            best[identity] = rank
        counts[identity] = counts.get(identity, 0) + 1

    for needle in needles:
        if not needle:
            continue
        for identity in exact.get(needle, ()):
            offer(identity, 1)
        for identity in names.get(needle, ()):
            name = by_identity[identity].product_name.lower()
            offer(identity, 3 if name.startswith(needle) else 4)
        for identity in ingredients.get(needle, ()):
            offer(identity, 5)
        for identity in applicants.get(needle, ()):
            offer(identity, 6)
        if any(char.isdigit() for char in needle):
            for nrn_key, identity in nrns:
                if needle in nrn_key:
                    offer(identity, 2)

    if with_counts:
        return [
            (rank, counts[identity], by_identity[identity])
            for identity, rank in best.items()
        ]
    return [(rank, by_identity[identity]) for identity, rank in best.items()]


@dataclass
class SearchResult:
    """Free-text search candidates, plus enough context to report honestly.

    The same caution as a single verification applies here, and more sharply: a
    search that returns nothing is not evidence the product is counterfeit, and
    a search that returns twelve things has not answered anything until the user
    says which pack they hold. `sources_checked` and `sources_failed` are kept
    for the same reason they exist on `VerificationResult` — a lookup that ran
    against nothing must not be reported as a lookup that found nothing.
    """

    query: str = ""
    records: List[ProductRecord] = field(default_factory=list)
    total: int = 0
    searched: bool = False
    best_rank: int = 0
    sources_checked: List[str] = field(default_factory=list)
    sources_failed: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.records)

    @property
    def strong_match(self) -> bool:
        """True when the best hit identifies a product rather than a word.

        Not a gate — every message is searched, and a hit is always passed on.
        This tells the caller how much to trust the candidates, which the
        rendered block states in words so the model can discount them.

        A match at product level or better (rank 1-5: the number, the name, or an
        ingredient) is something the user could plausibly be holding. A match that
        only appears in a company's name is a coincidence of wording, and since
        greenbook names are ordinary English, so are a few unexpected ones:
        "good morning" really does match products called "Morning".
        """
        return 0 < self.best_rank <= 5

    @property
    def flagged_count(self) -> int:
        return sum(1 for record in self.records if record.verdict == VERDICT_FLAGGED)

    @property
    def authoritative_count(self) -> int:
        return sum(
            1 for record in self.records if record.verdict == VERDICT_AUTHENTIC
        )

    @property
    def verdict(self) -> str:
        """A verdict only when the candidates agree; otherwise unknown.

        With one candidate the verdict is meaningful. With several it is not,
        because "flagged" here means "at least one of these is flagged", which
        is a statement about the candidate list rather than about the pack in
        the user's hand.
        """
        if not self.searched:
            return VERDICT_UNKNOWN
        if not self.records:
            return VERDICT_NOT_REGISTERED
        if len(self.records) == 1:
            return self.records[0].verdict
        if self.flagged_count and self.flagged_count == len(self.records):
            return VERDICT_FLAGGED
        return VERDICT_UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "found": self.found,
            "searched": self.searched,
            "total": self.total,
            "count": len(self.records),
            "verdict": self.verdict,
            "flagged_count": self.flagged_count,
            "matches": [record.to_dict() for record in self.records],
            "sources_checked": list(self.sources_checked),
            "sources_failed": dict(self.sources_failed),
            "notes": list(self.notes),
        }

    def render(self, limit: int = 8) -> str:
        """A plain-text block for the model, in the spirit of VerificationResult.

        Two behaviours are deliberate. A miss says the greenbook export does not
        list it rather than implying the product is fake. A multi-candidate hit
        refuses to pick one, because the user asked about a pack and the data
        only knows about registrations.
        """
        if not self.searched:
            return (
                "No product name was extracted from the message, so no greenbook "
                "search was run. Ask the user for the product name or its NAFDAC "
                "registration number."
            )

        if not self.records:
            lines = [
                f"Greenbook search for {self.query!r}: no product matched.",
                f"  Verdict: {self.verdict.upper()} - not listed in this greenbook "
                "export.",
            ]
            if self.sources_checked:
                lines.append(f"  Sources searched: {', '.join(self.sources_checked)}")
            if self.sources_failed:
                lines.append(
                    "  Could not search: "
                    + ", ".join(f"{name} ({why})" for name, why in self.sources_failed.items())
                )
            lines.append(
                "  This is not proof the product is counterfeit or unregistered. "
                "The export may be out of date, or the name may be spelled "
                "differently on the pack."
            )
            lines.append(
                "  Ask the user for the NAFDAC registration number printed on the "
                "pack, which looks like A4-100160 or 04-3877, and look that up "
                "instead."
            )
            return "\n".join(lines)

        shown = self.records[: max(1, limit)]
        lines: List[str] = []
        if len(self.records) == 1:
            record = shown[0]
            lines.append(
                f"Greenbook search for {self.query!r}: one product matched."
            )
            lines.append(f"  Verdict: {self.verdict.upper()} - {record.verdict_reason}.")
        else:
            lines.append(
                f"Greenbook search for {self.query!r}: {len(self.records)} "
                "products matched, shown ranked by how closely they match."
            )
            lines.append(
                "  Verdict: UNKNOWN - these are different products, so this is not "
                "an answer about one pack."
            )
            lines.append(
                "  Show the candidates and ask which one the user means. Do not "
                "choose the top-ranked one for them."
            )
            if self.flagged_count:
                lines.append(
                    f"  {self.flagged_count} of them are flagged by NAFDAC."
                )

        # The instructions come before the records, not after them. Callers cap
        # this block at a few thousand characters, and a long candidate list is
        # truncated at the end — which would otherwise cut off the rules that
        # matter most. Putting them first means they survive every cut.
        lines.append("")
        lines.append(
            "  These matched because the message shared words with the greenbook, "
            "not necessarily because the user asked about them. Greenbook names "
            "are ordinary English, so an unrelated sentence can match by accident."
        )
        if not self.strong_match:
            lines.append(
                "  The closest match is weak: a word appears in a name or an "
                "ingredient rather than the product being named."
            )
        lines.append(
            "  If the user is not asking about a medicine, a product or their "
            "health, ignore this block completely and answer what they actually "
            "asked. Never open with an unprompted product list."
        )
        lines.append(
            "  Report only the values listed below. Never supply a product name, "
            "ingredient, strength, applicant, manufacturer, date or status from "
            "memory, and never call a product counterfeit or fake: the greenbook "
            "records registrations, not counterfeiting."
        )
        lines.append(
            "  Answer in the language the user wrote in, including Hausa, Yoruba "
            "or Igbo. Keep product names, NAFDAC numbers, dates and statuses "
            "exactly as written."
        )
        lines.append("")

        for record in shown:
            lines.append(render_record(record))
        if len(self.records) > len(shown):
            lines.append(f"  ...and {len(self.records) - len(shown)} more not shown.")

        if len(self.records) == 1:
            record = shown[0]
            if record.verdict == VERDICT_AUTHENTIC:
                lines.append(
                    "  The greenbook lists this as currently authorised. Tell the "
                    "user it is registered, but that the greenbook cannot confirm "
                    "this particular pack is genuine: a copied number resolves here "
                    "too."
                )
            elif record.verdict == VERDICT_FLAGGED:
                lines.append(
                    "  Tell the user this product is flagged and should not be used, "
                    "and give the reason above. Do not soften this into a suggestion."
                )

        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def find_products(
    query: Any,
    sources: Optional[List[NafdacSource]] = None,
    limit: int = 8,
) -> SearchResult:
    """Free-text search that reports its own provenance.

    `query` is either one string or a list of search terms, matched
    independently. Unlike `search_products()`, a source that could not be read is
    recorded rather than skipped silently, so "nothing matched" is never
    reported when the truth is "nothing was searched".
    """
    needles = _as_needles(query)
    label = " ".join(needles) if len(needles) != 1 else needles[0]
    result = SearchResult(query=label)
    if not needles:
        return result

    result.searched = True
    if sources is None:
        sources = build_sources(
            csv_path=_env_path("NAFDAC_CSV_PATH"),
            xlsx_path=_env_path("NAFDAC_XLSX_PATH"),
            database_url=os.environ.get("NAFDAC_DATABASE_URL", ""),
        )

    scored: List[tuple] = []
    deduped: Dict[Any, tuple] = {}

    for source in sources:
        name = getattr(source, "name", "unknown")
        loader = getattr(source, "load", None)
        if not callable(loader):
            result.sources_failed[name] = "no searchable index"
            continue
        try:
            index = loader()
        except Exception as exc:
            result.sources_failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        result.sources_checked.append(name)

        for rank, record in _candidates(index, needles):
            identity = record.product_id or (record.nrn_key, record.product_name)
            current = deduped.get(identity)
            if current is None or rank < current[0]:
                deduped[identity] = (rank, record)

    if not result.sources_checked:
        result.searched = False
        result.notes.append(
            "No greenbook source was readable, so nothing was searched. Do not "
            "tell the user the product is unregistered."
        )
        return result

    ordered = sorted(
        deduped.values(), key=lambda item: (item[0], item[1].product_name.lower())
    )
    result.total = len(ordered)
    result.best_rank = ordered[0][0] if ordered else 0
    result.records = [record for _rank_value, record in ordered[: max(1, limit)]]
    return result
