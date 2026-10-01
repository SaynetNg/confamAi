"""Configuration for the NAFDAC Greenbook scraper.

Three data sources are used, all public and all GET-only:

1. Bulk product JSON (DataTables server-side on the home page) - every product
   with its core fields. ~19 requests for the whole dataset.
2. Product detail pages ``/products/details/{product_id}`` - adds Manufacturer
   Name, Manufacturer Country and Marketing Category, which the bulk JSON does
   not carry. One request per product.
3. Reference index pages - manufacturers, applicants, ingredients, categories
   and the ATC classification tree.

Every tunable value lives here.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------
BASE_URL = "https://greenbook.nafdac.gov.ng/"

PRODUCT_CATEGORIES_URL = BASE_URL + "productCategories"
ATC_URL = BASE_URL + "atc-classification"
MANUFACTURERS_URL = BASE_URL + "manufacturers"
APPLICANTS_URL = BASE_URL + "applicants"
INGREDIENTS_URL = BASE_URL + "ingredients"

DETAIL_URL_TEMPLATE = BASE_URL + "products/details/{product_id}"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Only the bulk JSON endpoint needs the AJAX header. Sending it to HTML pages
# is harmless, but keeping it separate makes the intent obvious.
AJAX_HEADERS = {
    **REQUEST_HEADERS,
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------
REQUEST_TIMEOUT = 60          # seconds per request
MAX_RETRIES = 5               # attempts per request (1 = no retry)
BACKOFF_FACTOR = 2.0
BACKOFF_CAP = 60.0
REQUEST_DELAY = 1.0           # polite pause between bulk page requests
RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

# Detail crawl is the expensive phase (~8.9k requests). Detail pages are far
# more fragile than the JSON endpoint: at 4 workers the site starts returning
# HTTP 500 and dropping connections. 2 workers with a 0.4s pause sustains a
# stable ~4 requests/second and never trips the errors.
DETAIL_WORKERS = 2
DETAIL_DELAY = 0.4            # pause per worker between detail requests
DETAIL_CACHE = "detail_cache.jsonl"   # resume support for the detail crawl

# Product ids whose detail page returns HTTP 500 unconditionally - a server-side
# bug, not throttling. Retrying them costs minutes and never succeeds, so their
# ids are remembered in a dead-letter file and skipped on later runs. Delete the
# file to force a fresh attempt if NAFDAC fixes the bug.
DETAIL_DEAD_FILE = "detail_dead_ids.json"

# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------
PAGE_SIZE = 500
DEFAULT_LIMIT = 0             # 0 = no limit (full crawl)

# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------
FETCH_DETAILS = True          # crawl detail pages for the extra 3 fields
FETCH_INDEXES = True          # crawl manufacturers/applicants/ingredients/ATC

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
CSV_FILENAME = "nafdac_greenbook.csv"
XLSX_FILENAME = "nafdac_greenbook.xlsx"
LOG_FILENAME = "nafdac_scraper.log"

# The ten columns originally requested. Kept first and in this exact order so
# the export stays a drop-in replacement for the earlier CSV.
CORE_COLUMNS = [
    "Product Name",
    "Active Ingredients",
    "Product Category",
    "NRN",
    "Form",
    "ROA",
    "Strengths",
    "Applicant Name",
    "Approval Date",
    "Status",
]

# Every other field the site exposes, grouped so related values sit together.
EXTRA_COLUMNS = [
    # identity
    "Product ID",
    # classification
    "Product Category ID",
    "ATC Code",
    # composition
    "Ingredient ID",
    "Ingredient Synonym",
    "Form ID",
    "Route ID",
    # presentation
    "Product Description",
    "Composition",
    "Pack Size",
    # parties
    "Applicant ID",
    "Applicant Address",
    "Manufacturer ID",
    "Manufacturer Name",
    "Manufacturer Country",
    "Country ID",
    # regulatory
    "Marketing Category ID",
    "Marketing Category",
    "Expiry Date",
    "SMPC URL",
    "Biosimilar",
    # record metadata
    "Created At",
    "Updated At",
    "Source URL",
]

OUTPUT_COLUMNS = CORE_COLUMNS + EXTRA_COLUMNS

SAMPLE_ROWS = 10              # rows printed before saving

# The site's own JavaScript strips '#' and '*' markers from product names before
# display. Keep True to match what NAFDAC shows; set False for raw API values.
STRIP_PRODUCT_NAME_MARKERS = True
