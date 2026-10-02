"""NAFDAC Greenbook product lookup, shared by the N-ATLaS client and server.

The flow this package implements is:

    user message
        -> extract_nrn()          pull a NAFDAC number out of free text
        -> verify_nafdac_product() resolve it against every available source
        -> NafdacResult           a normalised record plus where it came from

Sources are consulted in order and merged, never silently overwritten:

    1. postgres  the remote PostgreSQL database (authoritative, optional)
    2. csv       nafdac_scraper/nafdac_greenbook.csv   (local, 8935 rows)
    3. xlsx      nafdac_scraper/nafdac_greenbook.xlsx (local, Products sheet)

Every source is optional. With no PostgreSQL connection and no files, lookup
still runs and simply reports that the number is not found; it never raises
because a source is missing.
"""

from .extract import (
    EXTRACTION_PATTERN,
    extract_nrn,
    extract_nrn_candidates,
    extract_search_terms,
    looks_like_nrn,
    mentions_nafdac,
    normalize_nrn,
    normalize_nrn_key,
)
from .formatting import (
    NAFDAC_HELP,
    STRINGS,
    SUPPORTED_LANGUAGES,
    format_not_found,
    format_no_number_found,
    format_searched_nothing,
    format_search_results,
    format_search_verdict,
    format_status,
    format_usage_search,
    format_usage_verification,
    format_usage_verification_words,
    format_usage_verification_words_short,
    format_verification,
    nafdac_help,
    source_status,
)
from .records import (
    CORE_FIELDS,
    VERDICT_AUTHENTIC,
    VERDICT_FLAGGED,
    VERDICT_NOT_REGISTERED,
    VERDICT_UNKNOWN,
    ProductRecord,
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_UNKNOWN,
    classify_status,
    classify_verdict,
    is_authorised,
    record_from_row,
    render_record,
    verdict_reason,
)
from .sources import (
    NafdacSource,
    SourceResult,
    PostgresSource,
    CsvSource,
    XlsxSource,
    build_sources,
)
from .sources import clear_source_cache
from .verify import (
    NAFDAC_DEFAULT_PATHS,
    NAFDAC_URL_TEMPLATE,
    SearchResult,
    VerificationResult,
    find_products,
    verify_nafdac_product,
    search_products,
)

__all__ = [
    "CORE_FIELDS",
    "EXTRACTION_PATTERN",
    "NAFDAC_HELP",
    "STRINGS",
    "SUPPORTED_LANGUAGES",
    "NAFDAC_DEFAULT_PATHS",
    "NAFDAC_URL_TEMPLATE",
    "SearchResult",
    "STATUS_ACTIVE",
    "STATUS_INACTIVE",
    "STATUS_UNKNOWN",
    "VERDICT_AUTHENTIC",
    "VERDICT_FLAGGED",
    "VERDICT_NOT_REGISTERED",
    "VERDICT_UNKNOWN",
    "CsvSource",
    "NafdacSource",
    "PostgresSource",
    "ProductRecord",
    "SourceResult",
    "VerificationResult",
    "XlsxSource",
    "build_sources",
    "classify_status",
    "classify_verdict",
    "clear_source_cache",
    "extract_nrn",
    "extract_nrn_candidates",
    "extract_search_terms",
    "find_products",
    "format_not_found",
    "format_no_number_found",
    "format_searched_nothing",
    "format_search_results",
    "format_search_verdict",
    "format_status",
    "format_usage_search",
    "format_usage_verification",
    "format_usage_verification_words",
    "format_usage_verification_words_short",
    "format_verification",
    "is_authorised",
    "looks_like_nrn",
    "mentions_nafdac",
    "normalize_nrn",
    "normalize_nrn_key",
    "nafdac_help",
    "record_from_row",
    "render_record",
    "search_products",
    "source_status",
    "verdict_reason",
    "verify_nafdac_product",
]

__version__ = "1.0.0"
