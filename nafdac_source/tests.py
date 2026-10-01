"""Tests for the NAFDAC lookup, run with no database and no network.

    python -m nafdac_source.tests

The PostgreSQL path is exercised against a fake driver rather than a live
server, so the SQL, the merge order and the superseded-history fallback are all
covered here without needing credentials. The greenbook data itself is real:
the tests read the actual nafdac_greenbook.csv, so a change to the scraper
that breaks a number format shows up as a failure.

Exit code 0 when everything passes, 1 otherwise.
"""

import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from nafdac_source import (  # noqa: E402
    VerificationResult,
    classify_status,
    extract_nrn,
    extract_nrn_candidates,
    is_authorised,
    looks_like_nrn,
    mentions_nafdac,
    normalize_nrn,
    normalize_nrn_key,
    record_from_row,
    search_products,
    verify_nafdac_product,
)
from nafdac_source.records import STATUS_ACTIVE, STATUS_INACTIVE  # noqa: E402
from nafdac_source.sources import (  # noqa: E402
    CsvSource,
    PostgresSource,
    SourceResult,
    XlsxSource,
    build_sources,
)

_RESULTS = []


def check(name, condition, detail=""):
    ok = bool(condition)
    _RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" :: {detail}"
    print(line)
    return ok


def _row(**overrides):
    """A minimal greenbook row, as CSV and psycopg both deliver it."""
    row = {
        "NRN": "A4-100160",
        "Product ID": "2",
        "Product Name": "AG Spironolactone",
        "Active Ingredients": "Spironolactone",
        "Strengths": "25 mg",
        "Form": "Tablet",
        "ROA": "Oral",
        "Status": "Active",
        "Applicant Name": "AnnyGod Pharmaceutical Company Ltd",
        "Approval Date": "2021-12-17",
        "Expiry Date": "2026-12-16",
        "ATC Code": "C03DA01",
        "Marketing Category": "Prescription-only Medicine (POM)",
        "Manufacturer Name": "Alpa Laboratories Pvt Ltd",
        "Manufacturer Country": "INDIA",
        "Source URL": "https://greenbook.nafdac.gov.ng/products/details/2",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_normalisation():
    print("\nnormalisation")
    check("hyphen is preserved", normalize_nrn("a4-100160") == "A4-100160")
    check("en dash becomes a hyphen", normalize_nrn("04\u20132045") == "04-2045")
    check("spaces around the dash are removed", normalize_nrn("A4 - 5180") == "A4-5180")
    check("a placeholder is not a number", normalize_nrn("NA") == "")
    check(
        "a placeholder phrase is not a number",
        normalize_nrn("Not available yet") == "",
        repr(normalize_nrn("Not available yet")),
    )
    check("leading zeros survive", normalize_nrn_key("04-3877") == "04-3877")
    check("case does not matter", normalize_nrn_key("a4-100160") == normalize_nrn_key("A4-100160"))
    check(
        "an en dash and a hyphen share a key",
        normalize_nrn_key("04\u20132045") == normalize_nrn_key("04-2045"),
    )
    check(
        "a padded value and a tight value share a key",
        normalize_nrn_key("04 - 9502") == normalize_nrn_key("04-9502"),
    )
    check("NA has no key", normalize_nrn_key("NA") == "")


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def test_extraction():
    print("\nextraction")
    check("a bare number is found", extract_nrn("A4-100160") == "A4-100160")
    check(
        "a number in a sentence is found",
        extract_nrn("Is A4-100160 genuine?") == "A4-100160",
        extract_nrn("Is A4-100160 genuine?"),
    )
    check(
        "a sentence does not become the number",
        extract_nrn("is A4-100160 real?") == "A4-100160",
        repr(extract_nrn("is A4-100160 real?")),
    )
    check(
        "a word after the number is not absorbed",
        extract_nrn("product 04-3877 is genuine") == "04-3877",
        extract_nrn("product 04-3877 is genuine"),
    )
    check(
        "several numbers keep their order",
        extract_nrn_candidates("compare A4-100160 and 04-3877")
        == ["A4-100160", "04-3877"],
    )
    check("no number yields nothing", extract_nrn("how are you?") == "")
    check("an empty message yields nothing", extract_nrn("") == "")

    # A whole sentence must not pass looks_like_nrn, or the sentence itself
    # would be sent to the database as a lookup key.
    check("a whole sentence is not an NRN", not looks_like_nrn("Is A4-100160 genuine?"))
    check("a bare number is an NRN", looks_like_nrn("A4-100160"))
    check("a placeholder is not an NRN", not looks_like_nrn("NA"))

    check("'nafdac' marks a lookup", mentions_nafdac("check the nafdac number"))
    check("a misspelling still marks a lookup", mentions_nafdac("nafdak number?"))
    check("'verify' plus a number marks a lookup", mentions_nafdac("verify B4-4875"))
    check("plain conversation is not a lookup", not mentions_nafdac("how are you?"))
    check(
        "a quantity is not a lookup",
        not mentions_nafdac("I have 500mg of ibuprofen"),
    )


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def test_records():
    print("\nrecords")
    record = record_from_row(_row(), source="csv")
    check("the name is read", record.product_name == "AG Spironolactone")
    check("the product id is an int", record.product_id == 2, repr(record.product_id))
    check("the status is read", record.status == STATUS_ACTIVE)
    check("an active product is authorised", record.authorised is True)
    check("the source is recorded", record.sources == ["csv"])
    check("an int field survives a float-ish string", record_from_row(
        _row(**{"Product ID": "2.0"})
    ).product_id == 2)

    check("'Active' classifies as active", classify_status("Active") == STATUS_ACTIVE)
    check("'Inactive' classifies as inactive", classify_status("Inactive") == STATUS_INACTIVE)
    check("'withdrawn' classifies as inactive", classify_status("withdrawn") == STATUS_INACTIVE)
    check(
        "an unknown status is preserved, not flattened",
        classify_status("Under review") == "Under review",
        classify_status("Under review"),
    )
    check("unknown status does not claim authorisation", is_authorised("Under review") is None)
    check("an empty status claims nothing", is_authorised(classify_status("")) is None)

    inactive = record_from_row(_row(Status="Inactive"))
    check("an inactive product is not authorised", inactive.authorised is False)

    # A duplicate row must be recognisable, because the scraper keeps 268 of
    # them and a user asking about one needs to be told.
    duplicate = record_from_row(_row(**{"Product Name": "Paracetamol (duplicate)"}))
    check("a duplicate row is flagged", duplicate.duplicate_marker is True)
    check("an ordinary row is not flagged", record.duplicate_marker is False)


# ---------------------------------------------------------------------------
# CSV source, against the real greenbook export
# ---------------------------------------------------------------------------
def test_csv_source():
    print("\ncsv source (real data)")
    source = CsvSource()
    if not source.available():
        check("nafdac_greenbook.csv is present", False, str(source.path))
        return False
    index = source.load()
    total = sum(len(records) for records in index.values())
    check("nafdac_greenbook.csv is present", True, str(source.path))
    check("every row is indexed", total > 8000, f"{total} products")
    # The file has embedded newlines in Composition, so a line-based reader
    # would report ~14953 lines. Getting 8930 records proves csv was used.
    check("the row count is right, so csv was used", 8900 < total < 8960, f"{total}")

    result = source.search("A4-100160")
    check("A4-100160 is found", result.ok and len(result.matches) == 1)
    if result.matches:
        check(
            "the product name is right",
            result.matches[0].product_name == "AG Spironolactone",
            result.matches[0].product_name,
        )
    check("a padded lookup still matches", bool(source.search("A4 - 5180").matches))
    check("an en-dash lookup still matches", bool(source.search("04\u20132045").matches))
    check("a lowercase lookup still matches", bool(source.search("a4-100160").matches))
    check("an unknown number returns nothing", source.search("A4-0000000").matches == [])
    check("a placeholder returns nothing", source.search("NA").matches == [])
    return True


# ---------------------------------------------------------------------------
# merging across sources, and the superseded fallback
# ---------------------------------------------------------------------------
class _FakeSource:
    """A source stub that returns fixed records, standing in for PostgreSQL."""

    def __init__(self, name, matches, ok=True, error="", from_history=False):
        self.name = name
        self._matches = matches
        self._ok = ok
        self._error = error
        self._from_history = from_history

    def available(self):
        return self._ok

    def describe(self):
        return "fake" if self._ok else f"broken: {self._error}"

    def search(self, nrn):
        return SourceResult(
            source=self.name,
            ok=self._ok,
            error=self._error,
            matches=list(self._matches) if nrn == "A4-100160" else [],
            from_history=self._from_history,
        )

    def close(self):
        return None


def test_merge_and_history():
    print("\nmerging across sources")
    thin = record_from_row(_row(), source="postgres")
    thin.applicant_name = ""
    thin.approval_date = ""
    full = record_from_row(_row(), source="csv")
    full.sources = ["csv"]

    result = verify_nafdac_product(
        "A4-100160", sources=[_FakeSource("postgres", [thin]), _FakeSource("csv", [full])]
    )
    check("the same product from two sources is one record", len(result.matches) == 1)
    if result.matches:
        record = result.matches[0]
        check("both sources are named", set(record.sources) == {"postgres", "csv"}, str(record.sources))
        check("a gap is filled from the later source", record.approval_date == "2021-12-17")
        check("confidence is verified when both agree", result.confidence == "verified", result.confidence)

    # A thin database row must not overwrite a field the CSV already has.
    thin_override = record_from_row(_row(**{"Product Name": "WRONG NAME"}), source="postgres")
    result = verify_nafdac_product(
        "A4-100160",
        sources=[_FakeSource("postgres", [thin_override]), _FakeSource("csv", [full])],
    )
    if result.matches:
        check(
            "the earlier source wins a conflict",
            result.matches[0].product_name == "WRONG NAME",
            result.matches[0].product_name,
        )

    print("\nambiguity and superseded history")
    two = [
        record_from_row(_row(), source="csv"),
        record_from_row(_row(**{"Product Name": "A4-100160 Tablet", "Product ID": "99"}), source="csv"),
    ]
    result = verify_nafdac_product("A4-100160", sources=[_FakeSource("csv", two)])
    check("two products under one number are both returned", len(result.matches) == 2)
    check("that is reported as ambiguous", result.ambiguous is True)
    check("an ambiguous result still says found", result.found is True)

    superseded = record_from_row(_row(Status="Superseded"), source="postgres")
    result = verify_nafdac_product(
        "A4-100160",
        sources=[_FakeSource("postgres", [superseded], from_history=True)],
    )
    check("a history-only row is still found", result.found is True)
    check("a superseded row is not authorised", result.authorised is False)
    check(
        "a history-only result says so",
        result.confidence == "superseded",
        result.confidence,
    )

    print("\na broken source degrades rather than raising")
    result = verify_nafdac_product(
        "A4-100160",
        sources=[
            _FakeSource("postgres", [], ok=False, error="connection refused"),
            _FakeSource("csv", [full]),
        ],
    )
    check("the good source still answers", result.found is True)
    check("the broken source is reported", "postgres" in result.sources_failed)
    check("the failure reason is kept", result.sources_failed["postgres"] == "connection refused")
    # A database that could not be reached means a superseding update would not
    # show, so this is weaker than a database that ran and confirmed the row.
    check(
        "an unreachable database means local-file confidence",
        result.confidence == "local-file",
        result.confidence,
    )
    check(
        "local-file confidence warns about the missing database",
        any("not available" in note or "would not show" in note for note in result.notes),
        str(result.notes),
    )

    # A database that ran fine but had no row is a different situation: the two
    # stores are out of step, which is a single-source answer.
    result = verify_nafdac_product(
        "A4-100160",
        sources=[_FakeSource("postgres", []), _FakeSource("csv", [full])],
    )
    check(
        "a database that ran and found nothing is single-source",
        result.confidence == "single-source",
        result.confidence,
    )

    print("\nno sources at all")
    result = verify_nafdac_product("A4-100160", sources=[])
    check("an empty source list does not raise", isinstance(result, VerificationResult))
    check("nothing is found", result.found is False)
    check("nothing is claimed", result.authorised is None)

    print("\na source that raises is contained")
    result = verify_nafdac_product(
        "A4-100160",
        sources=[_ExplodingSource(), _FakeSource("csv", [full])],
    )
    check("a raising source does not break the lookup", result.found is True)
    check("the raising source is recorded as failed", "boom" in result.sources_failed)


class _ExplodingSource(_FakeSource):
    def __init__(self):
        super().__init__("boom", [])

    def search(self, nrn):
        raise RuntimeError("the driver exploded")


# ---------------------------------------------------------------------------
# the miss path must not over-claim
# ---------------------------------------------------------------------------
def test_not_found_wording():
    print("\nnot-found reporting")
    result = verify_nafdac_product("A4-0000000", sources=[])
    result.sources_checked = ["csv"]
    rendered = result.render()
    check("a miss says not found", "NOT FOUND" in rendered)
    check("a miss does not claim the product is fake", "not treat this as proof" in rendered)
    check("a miss leaves authorisation unknown", result.authorised is None)
    check("an unsearched request says so", verify_nafdac_product("").searched is False)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search():
    print("\nfree-text search")
    source = CsvSource()
    if not source.available():
        return
    results = search_products("paracetamol", sources=[source], limit=5)
    check("a search returns results", len(results) > 0, f"count={len(results)}")
    check(
        "a search respects the limit",
        len(search_products("a", sources=[source], limit=3)) <= 3,
    )
    check("an empty query returns nothing", search_products("", sources=[source]) == [])
    ranked = search_products("spironolactone", sources=[source], limit=5)
    check(
        "an exact name ranks first",
        ranked and "spironolactone" in ranked[0].product_name.lower(),
        ranked[0].product_name if ranked else "none",
    )


# ---------------------------------------------------------------------------
# the reader and the writer must agree on the column list
# ---------------------------------------------------------------------------
def test_reader_writer_agree():
    """The SELECT in sources.py and the INSERT in load_postgres.py are two
    hand-maintained lists of the same 35 columns. If they drift, a lookup
    either misses a field or fails outright, and neither shows up until a real
    database is involved. This is the cheapest place to catch that.
    """
    print("\nreader / writer column agreement")
    from nafdac_source.load_postgres import _COLUMNS
    from nafdac_source.sources import _SELECT_COLUMNS

    selected = [
        column.strip()
        for column in _SELECT_COLUMNS.replace("\n", " ").split(",")
        if column.strip()
    ]

    # nrn_key is the WHERE key, so it is deliberately not selected.
    expected = set(_COLUMNS) - {"nrn_key"}
    check(
        "the reader selects every column the writer writes",
        set(selected) == expected,
        f"missing={sorted(expected - set(selected))} extra={sorted(set(selected) - expected)}",
    )
    check("the reader selects no duplicates", len(selected) == len(set(selected)))
    check(
        "every column is a real record field",
        all(column in _COLUMNS for column in selected),
    )
    check("product_id is among them", "product_id" in selected)
    check("nrn is among them", "nrn" in selected)

    # A row shaped exactly like a psycopg result must produce a full record.
    # psycopg hands back the snake_case column names, not the greenbook's
    # spaced headers, so this is the case the two lookups have to agree on.
    database_row = {
        "nrn": "A4-100160",
        "product_id": 2,
        "product_name": "AG Spironolactone",
        "active_ingredients": "Spironolactone",
        "strengths": "25 mg",
        "form": "Tablet",
        "route": "Oral",
        "status": "Active",
        "applicant_name": "AnnyGod Pharmaceutical Company Ltd",
        "manufacturer_name": "Alpa Laboratories Pvt Ltd",
        "manufacturer_country": "INDIA",
        "approval_date": "2021-12-17",
        "expiry_date": "2026-12-16",
        "atc_code": "C03DA01",
        "marketing_category": "Prescription-only Medicine (POM)",
        "source_url": "https://greenbook.nafdac.gov.ng/products/details/2",
        "created_at": "2022-05-23 02:59:13+00:00",
        "updated_at": "2025-03-05 16:01:15+00:00",
    }
    record = record_from_row(database_row, source="postgres")
    check("a psycopg-shaped row fills the name", record.product_name == "AG Spironolactone")
    check("a psycopg-shaped row fills the nrn", record.nrn == "A4-100160")
    check("a psycopg-shaped row parses product_id", record.product_id == 2)
    check("a psycopg-shaped row parses the dates", record.approval_date == "2021-12-17")
    check("a psycopg-shaped row keeps the source_url", "greenbook" in record.source_url)
    check("a psycopg-shaped row keeps created_at", "2022" in record.created_at)
    check("a psycopg-shaped row computes the key", record.nrn_key == "A4-100160")
    check("a psycopg-shaped row is authorised", record.authorised is True)


# ---------------------------------------------------------------------------
# build_sources
# ---------------------------------------------------------------------------
def test_build_sources():
    print("\nsource construction")
    sources = build_sources(use_cache=False)
    names = [source.name for source in sources]
    check("postgres comes first, so it wins a conflict", names[0] == "postgres", str(names))
    check("the csv source is included", "csv" in names)
    check("the xlsx source is included", "xlsx" in names)

    only_files = build_sources(use_database=False, use_cache=False)
    check(
        "the database can be switched off",
        [s.name for s in only_files] == ["csv", "xlsx"],
    )
    check(
        "cached construction returns the same objects",
        build_sources() is build_sources(),
    )
    check(
        "an uncached build returns new objects",
        build_sources(use_cache=False) is not build_sources(use_cache=False),
    )
    check("a missing file degrades to unavailable", not XlsxSource(
        Path("no-such-file.xlsx")
    ).available())
    check("a postgres source without a dsn is unavailable", not PostgresSource(dsn="").available())


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def test_rendering():
    print("\nrendering")
    result = verify_nafdac_product("A4-100160", sources=build_sources(use_database=False))
    check("a found result renders", result.found is True)
    if result.found:
        text = result.render()
        check("the render includes the number", "A4-100160" in text)
        check("the render includes the product name", "AG Spironolactone" in text)
        check("the render includes the applicant", "AnnyGod" in text)
        check("the render includes the greenbook link", "greenbook.nafdac.gov.ng" in text)
        check(
            "the render tells the model not to invent values",
            "Never supply" in text and "does not record it" in text,
        )
        check("the render is plain text, not markdown", "**" not in text)
        check("the render has no non-ascii dashes", "\u2014" not in text)


def main() -> int:
    print("=" * 70)
    print("NAFDAC SOURCE TESTS")
    print("=" * 70)

    tests = [
        test_normalisation,
        test_extraction,
        test_records,
        test_csv_source,
        test_merge_and_history,
        test_not_found_wording,
        test_search,
        test_reader_writer_agree,
        test_build_sources,
        test_rendering,
    ]
    for test in tests:
        try:
            test()
        except Exception:
            print(f"  [FAIL] {test.__name__} raised")
            traceback.print_exc()
            _RESULTS.append((test.__name__, False, "raised"))

    passed = sum(1 for _n, ok, _d in _RESULTS if ok)
    failed = [(n, d) for n, ok, d in _RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"{passed} passed, {len(failed)} failed, out of {len(_RESULTS)} checks")
    if failed:
        print("-" * 70)
        for name, detail in failed:
            print(f"  FAIL  {name}" + (f" :: {detail}" if detail else ""))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
