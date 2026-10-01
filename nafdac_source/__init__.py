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
    looks_like_nrn,
    mentions_nafdac,
    normalize_nrn,
    normalize_nrn_key,
)
from .formatting import (
    NAFDAC_HELP,
    format_not_found,
    format_search_results,
    format_status,
    format_verification,
    source_status,
)
from .records import (
    CORE_FIELDS,
    ProductRecord,
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_UNKNOWN,
    classify_status,
    is_authorised,
    record_from_row,
    render_record,
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
    VerificationResult,
    verify_nafdac_product,
    search_products,
)

__all__ = [
    "CORE_FIELDS",
    "EXTRACTION_PATTERN",
    "NAFDAC_HELP",
    "NAFDAC_DEFAULT_PATHS",
    "NAFDAC_URL_TEMPLATE",
    "STATUS_ACTIVE",
    "STATUS_INACTIVE",
    "STATUS_UNKNOWN",
    "CsvSource",
    "NafdacSource",
    "PostgresSource",
    "ProductRecord",
    "SourceResult",
    "VerificationResult",
    "XlsxSource",
    "build_sources",
    "classify_status",
    "clear_source_cache",
    "extract_nrn",
    "extract_nrn_candidates",
    "format_not_found",
    "format_search_results",
    "format_status",
    "format_verification",
    "is_authorised",
    "looks_like_nrn",
    "mentions_nafdac",
    "normalize_nrn",
    "normalize_nrn_key",
    "record_from_row",
    "render_record",
    "search_products",
    "source_status",
    "verify_nafdac_product",
]

__version__ = "1.0.0"
