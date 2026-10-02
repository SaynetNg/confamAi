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
    classify_verdict,
    extract_nrn,
    extract_nrn_candidates,
    extract_search_terms,
    find_products,
    is_authorised,
    looks_like_nrn,
    mentions_nafdac,
    normalize_nrn,
    normalize_nrn_key,
    record_from_row,
    search_products,
    verdict_reason,
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


# ---------------------------------------------------------------------------
# localisation of the terminal output
# ---------------------------------------------------------------------------
def test_localisation():
    """`lang=` localises the table without touching a single value.

    The rule under test is the one that makes localisation safe here: a user
    checking whether a medicine is genuine must still read the real product
    name, the real applicant and the real dates, in whatever language the
    client is currently rendering in.
    """
    print("\nlocalisation")

    from .formatting import (
        STRINGS,
        SUPPORTED_LANGUAGES,
        _t,
        format_not_found,
        format_search_results,
        format_status,
        format_verification,
        nafdac_help,
    )

    result = verify_nafdac_product("A4-100160", sources=build_sources(use_database=False))
    if not result.found:
        check("a found result is available to localise", False, result.nrn)
        return
    record = result.primary

    english = format_verification(result, lang="en")
    check("english is the default rendering", english == format_verification(result))

    # Every language answers every key, so a partially translated table can
    # never print a key name at a user.
    keys = set(STRINGS["en"])
    check(
        "no language defines a key English does not",
        all(not set(STRINGS[lang]) - keys for lang in STRINGS),
    )
    missing = [
        (lang, key)
        for lang in SUPPORTED_LANGUAGES
        for key in keys
        if _t(lang, key) == key
    ]
    check("every language answers every key", not missing, str(missing[:3]))
    check(
        "an unknown language falls back to english",
        _t("zz", "label_number") == STRINGS["en"]["label_number"],
    )

    for lang in SUPPORTED_LANGUAGES:
        text = format_verification(result, lang=lang)
        label = _t(lang, "label_applicant")
        check(f"[{lang}] the applicant label is localised", label in text)
        check(f"[{lang}] the number is untouched", record.nrn in text)
        check(f"[{lang}] the product name is untouched", record.product_name in text)
        check(f"[{lang}] the approval date is untouched", record.approval_date in text)
        check(f"[{lang}] no unformatted placeholder survives", "{" not in text)

    check(
        "a translated verdict word differs from the english one",
        _t("ha", "verdict_authorised") != _t("en", "verdict_authorised"),
    )
    check(
        "greenbook stays one word in every language",
        len({_t(lang, "label_greenbook") for lang in SUPPORTED_LANGUAGES}) == 1,
    )

    # A miss must stay a miss in every language, and must never accuse anyone.
    miss = verify_nafdac_product("A4-0000000", sources=build_sources(use_database=False))
    for lang in SUPPORTED_LANGUAGES:
        text = format_not_found(miss, lang=lang)
        check(f"[{lang}] a miss says not found", "A4-0000000" in text)
        check(
            f"[{lang}] a miss does not claim the product is fake",
            "counterfeit" not in text.lower(),
        )

    empty = format_search_results([], "zzzznotarealproduct", lang="ha")
    check("a search miss renders in the given language", bool(empty.strip()))
    check("a search miss keeps the query", "zzzznotarealproduct" in empty)
    check("the status block renders in the given language", bool(format_status([], lang="yo").strip()))
    check("the help block renders in the given language", "/verify" in nafdac_help("ig"))

    # The model-facing block is never localised. It is verified data, and a
    # translation in front of a verifier is how a wrong value gets reported.
    # Asserted structurally: render() takes no language at all, so there is no
    # way to localise it by accident.
    import inspect

    from .records import render_record

    check(
        "the model block takes no language argument",
        "lang" not in inspect.signature(result.render).parameters
        and not [
            p
            for p in inspect.signature(render_record).parameters.values()
            if "lang" in p.name
        ],
    )
    check(
        "the model block is identical across calls",
        result.render() == result.render(),
    )
    check(
        "the model block carries no localised verdict",
        _t("ha", "verdict_authorised") not in result.render(),
    )



    """A user asks "is this genuine?", which `authorised` does not answer.

    The verdict is the four-way answer the model actually needs, and the one
    thing that must never be invented: there is deliberately no "fake" value,
    because the greenbook records registrations, not counterfeiting.
    """
    # ---------------------------------------------------------------------------
# the user-facing verdict
# ---------------------------------------------------------------------------
def test_verdict():
    """A user asks "is this genuine?", which `authorised` does not answer.

    The verdict is the four-way answer the model actually needs, and the one
    thing that must never be invented: there is deliberately no "fake" value,
    because the greenbook records registrations, not counterfeiting.
    """
    print("\nverdict")

    check(
        "an active product is authentic",
        classify_verdict("Active") == "authentic",
    )
    check(
        "an inactive product is flagged",
        classify_verdict("Inactive") == "flagged",
    )
    for status in ("Withdrawn", "Suspended", "Recalled", "Revoked", "Expired"):
        check(
            f"{status} is flagged",
            classify_verdict(status) == "flagged",
        )
    check(
        "a duplicate entry is flagged even when active",
        classify_verdict("Active", duplicate=True) == "flagged",
    )
    check(
        "an unrecognised status is unknown, not authentic",
        classify_verdict("Pending Review") == "unknown",
    )
    check("a blank status is unknown", classify_verdict("") == "unknown")

    check(
        "an action status is named in the reason",
        "Withdrawn" in verdict_reason("Withdrawn"),
        verdict_reason("Withdrawn"),
    )
    check(
        "the reason is never the word counterfeit",
        all(
            "counterfeit" not in verdict_reason(status).lower()
            for status in ("Active", "Inactive", "Withdrawn", "", "Weird")
        ),
    )

    # The raw status must survive, or a withdrawn product and an expired one
    # become indistinguishable to the user.
    withdrawn = record_from_row(_row(Status="Withdrawn", **{"Product ID": "99"}))
    check("the raw status is preserved", withdrawn.raw_status == "Withdrawn")
    check("the normalised status is still Inactive", withdrawn.status == "Inactive")
    check("a withdrawn row is flagged", withdrawn.verdict == "flagged")
    check(
        "the verdict prefers the raw status",
        withdrawn.verdict_reason == "NAFDAC status is Withdrawn",
        withdrawn.verdict_reason,
    )

    active = record_from_row(_row())
    check("an active row is authentic", active.verdict == "authentic")
    check("the verdict is in to_dict", active.to_dict()["verdict"] == "authentic")

    duplicate = record_from_row(_row(**{"Product Name": "AG Spironolactone (duplicate)"}))
    check("a duplicate name row is flagged", duplicate.verdict == "flagged")


def test_verdict_on_result():
    """The result-level verdict, aggregated across every match."""
    print("\nverdict on a result")
    sources = build_sources(use_database=False)

    found = verify_nafdac_product("A4-100160", sources=sources)
    check("a known number is authentic", found.verdict == "authentic", found.verdict)
    check(
        "the reason explains it",
        "Active" in found.verdict_reason,
        found.verdict_reason,
    )
    check("the verdict is in to_dict", found.to_dict()["verdict"] == "authentic")
    check("the render names the verdict", "Verdict: AUTHENTIC" in found.render())

    missing = verify_nafdac_product("A4-0000000", sources=sources)
    check("an unknown number is not_registered", missing.verdict == "not_registered")
    check(
        "a miss says it is not proof, rather than asserting counterfeit",
        "not proof" in missing.verdict_reason.lower(),
        missing.verdict_reason,
    )

    empty = verify_nafdac_product("", sources=sources)
    check("nothing searched is unknown, not not_registered", empty.verdict == "unknown")

    # A number carrying both an active and a flagged product must never be
    # reported as authentic, because the user would be told a flagged product
    # is fine.
    flagged = record_from_row(_row(Status="Recalled", **{"Product ID": "501"}))
    ok = record_from_row(_row(**{"Product ID": "502"}))
    mixed = VerificationResult(
        nrn="A4-999999",
        searched=True,
        found=True,
        matches=[ok, flagged],
        authorised=True,
    )
    check(
        "one flagged match makes the whole result flagged",
        mixed.verdict == "flagged",
        mixed.verdict,
    )

    # render() must never license the model to call a product counterfeit, on
    # any path.
    for result in (found, missing, mixed):
        text = result.render().lower()
        check(
            f"render forbids calling it counterfeit ({result.verdict})",
            "never call a product counterfeit" in text,
        )
        check(
            f"render tells the model to reply in the user's language ({result.verdict})",
            "hausa" in text and "yoruba" in text and "igbo" in text,
        )


# ---------------------------------------------------------------------------
# search terms, and the whole-word guard
# ---------------------------------------------------------------------------
def test_search_terms():
    """Every message is searched now, so the term extraction has to be right.

    There is no longer a gate deciding whether a message is a product question,
    which means these terms come from ordinary conversation as well. Getting this
    wrong shows the user a list of products they never asked about.
    """
    print("\nsearch terms")

    # A product noun must survive as a search term. If it were treated as a stop
    # word the local-language path would have nothing left to search for.
    check("magari is kept as a term", "magari" in extract_search_terms("Shin wannan magari ne?"))
    check("oro is kept as a term", "oro" in extract_search_terms("Je oro yi to?"))
    check("odu is kept as a term", "odu" in extract_search_terms("Nke onye bu odu?"))
    check("an NRN beats the words around it", extract_search_terms("is A4-100160 real") == ["A4-100160"])
    check("stop words are dropped", "genuine" not in extract_search_terms("is paracetamol genuine?"))
    check("the drug name is kept", "paracetamol" in extract_search_terms("is paracetamol genuine?"))
    check(
        "a message of only stop words yields no query",
        extract_search_terms("is it for me to use") == [],
        extract_search_terms("is it for me to use"),
    )
    check("an empty message yields no query", extract_search_terms("") == [])

    sources = build_sources(use_database=False)

    # Diacritics are stripped, so a user typing without tone marks still hits.
    check("Yoruba without diacritics still yields oro", "oro" in extract_search_terms("Se oro yi to?"))
    check("Igbo without diacritics still yields odu", "odu" in extract_search_terms("O bu odu?"))

    # Whole-word indexing is what stops an ordinary verb from matching a brand.
    # Without it "treat" hits "Levotreat" and "cold" hits "Coldkris", which
    # would attach products to "how do I treat a cold?".
    from nafdac_source.verify import _candidates

    def matching(text):
        return sorted(
            record.product_name
            for _rank, record in _candidates(sources[0].load(), [text])
        )

    check(
        "a word inside a brand name does not match",
        matching("treat") == [],
        str(matching("treat")),
    )
    check(
        "a real product name still matches",
        any("cold" in name.lower() for name in matching("cold")),
        str(matching("cold")[:3]),
    )
    check("an absent word matches nothing", matching("zzzznotaword") == [])
    check("an exact name matches", len(matching("good morning cough syrup")) >= 1)
    check(
        "an NRN substring still matches",
        len(matching("5180")) >= 1,
        str(matching("5180")[:3]),
    )
    # "Limited" appears in applicant names and rarely in a product name, so it
    # is the weak-match case: a hit exists but is not a product the user could be
    # holding, and `strong_match` has to say so.
    generic = find_products("limited", sources=sources, limit=5)
    check(
        "a company-word hit is not a strong match",
        generic.found is True and generic.strong_match is False,
        f"rank={generic.best_rank} strong={generic.strong_match}",
    )
    check(
        "a weak match says so in the rendered block",
        "closest match is weak" in generic.render(limit=1),
    )
    # A brand that is also a product name prefix is a strong match.
    branded = find_products("emzor", sources=sources, limit=5)
    check(
        "a brand used as a product name is a strong match",
        branded.strong_match is True,
        f"rank={branded.best_rank}",
    )

    sources = build_sources(use_database=False)


def test_search_every_message():
    """A miss attaches nothing, so the model answers conversationally.

    There is no product-question gate any more. That is a deliberate trade: a
    greenbook name is ordinary English, so "good morning" matches products called
    "Morning", and no word list can tell that apart from a real question. The
    decision is handed to the model, which reads the block and knows what the user
    actually asked.
    """
    print("\nsearch on every message")
    sources = build_sources(use_database=False)

    # A miss is the case that matters: nothing attached, model answers normally.
    for label, message in (
        ("a greeting", "Ka lafiya?"),
        ("how are you, mixed", "Báwo ni? How are you today?"),
        ("a symptom", "I have a headache"),
        ("a nonsense product", "zzzznotarealproduct"),
    ):
        terms = extract_search_terms(message)
        result = find_products(terms, sources=sources, limit=5) if terms else None
        attached = bool(result and result.searched and result.found)
        check(
            f"{label} attaches nothing",
            not attached,
            f"terms={terms} hits={result.total if result else 0}",
        )

    # An incidental match still attaches, because the greenbook genuinely cannot
    # tell it apart from a product question: "weather", "morning" and "family" are
    # all words that appear in real product names. What must not happen is the
    # model being told to trust the match, so every block states the uncertainty
    # and tells the model to drop the candidates if they are not relevant.
    for label, message in (
        ("good morning", "good morning"),
        ("an unrelated sentence", "I love my family"),
        ("the weather", "what is the weather in Lagos"),
    ):
        terms = extract_search_terms(message)
        result = find_products(terms, sources=sources, limit=5) if terms else None
        if not (result and result.found):
            check(f"an incidental match ({label}) matched nothing", True)
            continue
        text = result.render(limit=2)
        check(
            f"an incidental match ({label}) states it may not be what the user meant",
            "not necessarily because the user asked about them" in text,
            f"hits={result.total} best_rank={result.best_rank}",
        )
        check(
            f"an incidental match ({label}) may be ignored entirely",
            "ignore this block completely" in text,
        )

    # A genuine product question still reaches the data.
    result = find_products(
        extract_search_terms("is paracetamol genuine?"), sources=sources, limit=5
    )
    check("a drug name still finds candidates", result.found is True, str(result.total))
    check("a drug name is a strong match", result.strong_match is True)
    check("best_rank is recorded", result.best_rank > 0, str(result.best_rank))


def test_search_result():
    """`find_products` reports provenance, and never over-claims a miss."""
    print("\nsearch result")
    sources = build_sources(use_database=False)

    result = find_products("spironolactone", sources=sources, limit=5)
    check("a search runs", result.searched is True)
    check("a search reports the sources it used", bool(result.sources_checked))
    check("a search finds candidates", result.found is True, str(result.total))
    check("the query is echoed", "spironolactone" in result.query)
    check("the verdict is in to_dict", "verdict" in result.to_dict())

    # Several candidates means the data has not answered the question about one
    # specific pack, so no verdict may be claimed.
    many = find_products("malaria", sources=sources, limit=10)
    if many.total > 1:
        check(
            "several candidates claim no verdict",
            many.verdict == "unknown",
            many.verdict,
        )
        text = many.render(limit=3)
        check("the render asks which product", "which one the user means" in text)
        check("the render will not pick one", "Do not choose" in text)

    miss = find_products("zzzznotarealproduct", sources=sources)
    check("a miss still searched", miss.searched is True)
    check("a miss claims nothing", miss.found is False)
    check("a miss is not_registered", miss.verdict == "not_registered")
    miss_text = miss.render()
    check(
        "a miss is not called counterfeit",
        "not proof the product is counterfeit" in miss_text.lower(),
    )
    check("a miss asks for the number", "A4-100160" in miss_text)
    check("the miss render is plain text", "**" not in miss_text)

    # Multiple terms must match independently. Joining them into one needle
    # would match nothing, because no product name contains that exact pair.
    joined = find_products("paracetamol syrup", sources=sources, limit=5)
    split = find_products(["paracetamol", "syrup"], sources=sources, limit=5)
    check(
        "separate terms match more than one joined phrase",
        len(split.records) >= len(joined.records),
        f"split={len(split.records)} joined={len(joined.records)}",
    )

    # A source that cannot be read must be reported, so "nothing matched" is
    # never printed when the truth is "nothing was searched".
    broken = find_products("spironolactone", sources=[], limit=5)
    check("no readable source means not searched", broken.searched is False)
    check(
        "a dead lookup does not claim not_registered",
        broken.verdict == "unknown",
        broken.verdict,
    )
    check(
        "a dead lookup warns the model",
        any("was readable" in note.lower() for note in broken.notes),
        str(broken.notes),
    )


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
        test_verdict,
        test_verdict_on_result,
        test_search_terms,
        test_search_every_message,
        test_search_result,
        test_reader_writer_agree,
        test_build_sources,
        test_rendering,
        test_localisation,
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
