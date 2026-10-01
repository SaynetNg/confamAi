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

        for record in self.matches:
            lines.append(render_record(record))

        if self.greenbook_url:
            lines.append(f"  Greenbook page: {self.greenbook_url}")
        lines.append(f"  Confidence: {self.confidence}.")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append(
            "  Report only the values listed above. If the user asks about a "
            "field that is not listed, say the greenbook does not record it. "
            "Never supply a product name, ingredient, strength, applicant, "
            "manufacturer, date or status from memory."
        )
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


def search_products(
    query: str,
    sources: Optional[List[NafdacSource]] = None,
    limit: int = 20,
) -> List[ProductRecord]:
    """Free-text search across product name, ingredients and applicant.

    Used by the server so N-ATLaS can answer "do you have anything for malaria?"
    rather than only exact-number lookups. Results are ranked: an exact NRN or
    product-ID hit first, then a name prefix, then a substring, then a match in
    the ingredients or applicant.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return []
    if sources is None:
        sources = build_sources(
            csv_path=_env_path("NAFDAC_CSV_PATH"),
            xlsx_path=_env_path("NAFDAC_XLSX_PATH"),
            database_url=os.environ.get("NAFDAC_DATABASE_URL", ""),
        )

    scored: List[tuple] = []
    seen: set = set()

    for source in sources:
        loader = getattr(source, "load", None)
        if not callable(loader):
            continue
        try:
            index = loader()
        except Exception:
            continue
        for records in index.values():
            for record in records:
                identity = record.product_id or (record.nrn_key, record.product_name)
                if identity in seen:
                    continue
                rank = _rank(record, needle)
                if rank:
                    seen.add(identity)
                    scored.append((rank, record))

    scored.sort(key=lambda item: (item[0], item[1].product_name.lower()))
    return [record for _rank_value, record in scored[: max(1, limit)]]


def _rank(record: ProductRecord, needle: str) -> int:
    """Lower is better; 0 means no match."""
    nrn_key = record.nrn_key.lower()
    name = record.product_name.lower()
    ingredients = record.active_ingredients.lower()
    applicant = record.applicant_name.lower()

    if needle == nrn_key or needle == name:
        return 1
    if nrn_key and needle in nrn_key and len(needle) >= 4:
        return 2
    if name.startswith(needle):
        return 3
    if name and needle in name:
        return 4
    if needle in ingredients:
        return 5
    if needle in applicant:
        return 6
    return 0
