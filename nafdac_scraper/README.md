# NAFDAC Greenbook Scraper

Extracts every field the NAFDAC Greenbook site publishes into a structured
dataset, exported as CSV plus a multi-sheet Excel workbook.

Last verified against the live site: **8,935 products, 34 columns, plus 5
reference sheets** (1,427 manufacturers, 1,661 applicants, 2,393 ingredients,
6 product categories, 902 ATC codes).

All ten originally requested columns are 100% populated except NRN, which is
blank on 3 products because the site has no NRN for them.

---

## How the site works

The Greenbook is a Laravel application. There is **no public REST API** — the
data lives behind jQuery DataTables (`serverSide: true`) and plain server-rendered
HTML pages. Three access patterns are needed to get everything.

### 1. Bulk product JSON

| | |
|---|---|
| URL | `https://greenbook.nafdac.gov.ng/` (same URL as the home page) |
| Method | `GET` |
| Auth | none — no key, no login, no CSRF token |
| Required header | `X-Requested-With: XMLHttpRequest` |

That header is the only real trap: without it the server returns the HTML home
page instead of JSON. A `csrf-token` meta tag exists on the page but is not
needed for GETs.

**Pagination** uses the DataTables protocol: `draw` (echo counter), `start`
(0-based row offset), `length` (page size). `recordsTotal` gives the row count to
stop on. The server honours large `length` values — `length=10000` returns
everything in one request — but the scraper pages at 500 so an interrupted run
costs one page, not the whole crawl.

**Response shape:**

```jsonc
{
  "draw": 1,
  "recordsTotal": 8935,
  "recordsFiltered": 8935,
  "data": [ { /* one product */ } ]
}
```

Each product row carries the core fields plus its related models
(`ingredient`, `product_category`, `form`, `route`, `applicant`):

```jsonc
{
  "product_id": 2,
  "product_name": "AG Spironolactone##",
  "NAFDAC": "A4-100160",              // this is the NRN
  "strength": "25 mg",
  "approval_date": "2021-12-17",
  "expiry_date": "2026-12-16",
  "status": "Active",
  "product_description": "Film Coated Tablet",
  "pack_size": "1 x 500's (in a jar)",
  "composition": "Each film coated tablet contains: Spironolactone BP 25 mg",
  "atc": "C03DA01",
  "smpc": "https://admin.greenbook.nafdac.gov.ng/uploadImage/...",
  "ingredient":        { "ingredient_name": "Spironolactone", "synonym": "" },
  "product_category":  { "name": "Drugs" },
  "form":              { "name": "Tablet" },
  "route":             { "name": "Oral" },
  "applicant":         { "name": "AnnyGod Pharmaceutical Company Ltd" }
}
```

The query is filtered server-side to `status IN ('Active','Inactive')` and
`deleted_at IS NULL`.

### 2. Product detail pages

`https://greenbook.nafdac.gov.ng/products/details/{product_id}` — server-rendered
HTML, no JS needed. Each field is an `<h1>` label followed by a `<p>` value.

Three fields exist **only** here and not in the bulk JSON:

- **Manufacturer Name**
- **Manufacturer Country**
- **Marketing Category**

The detail page also repeats the core ten, so the two sources agree.

### 3. Reference index pages

Server-rendered, paginated with `?page=N`, 50 cards per page:

| Page | Pages | Entries |
|---|---|---|
| `/manufacturers` | 29 | 1,427 |
| `/applicants` | 34 | 1,661 |
| `/ingredients` | 48 | 2,393 |
| `/productCategories` | 1 | 6 |
| `/atc-classification` | 1 | 902 codes |

Each becomes its own workbook sheet.

---

## Field mapping

The ten originally requested columns come first, in the required order. Every
other field the site publishes follows.

| Column | Source |
|---|---|
| **Product Name** | `product_name` |
| **Active Ingredients** | `ingredient.ingredient_name` |
| **Product Category** | `product_category.name` |
| **NRN** | `NAFDAC` |
| **Form** | `form.name` |
| **ROA** | `route.name` |
| **Strengths** | `strength` |
| **Applicant Name** | `applicant.name` |
| **Approval Date** | `approval_date` |
| **Status** | `status` |
| Product ID | `product_id` |
| Product Category ID | `product_category_id` |
| ATC Code | `atc` |
| Ingredient ID | `ingredient_id` |
| Ingredient Synonym | `ingredient.synonym` |
| Form ID | `form_id` |
| Route ID | `route_id` |
| Product Description | `product_description` |
| Composition | `composition` (line breaks preserved) |
| Pack Size | `pack_size` |
| Applicant ID | `applicant_id` |
| Applicant Address | `applicant.address` |
| Manufacturer ID | `manufacturer_id` |
| **Manufacturer Name** | detail page |
| **Manufacturer Country** | detail page |
| Country ID | `country_id` |
| Marketing Category ID | `marketing_category_id` |
| **Marketing Category** | detail page |
| Expiry Date | `expiry_date` |
| SMPC URL | `smpc` |
| Biosimilar | `biosimilar` |
| Created At / Updated At | `created_at` / `updated_at` |
| Source URL | constructed from `product_id` |

Nothing is inferred. Missing values become empty strings, never placeholders.

### Columns that are legitimately empty

Three columns are empty for every record because the site never populates them:
`Applicant Address`, `Country ID`, and — for all but 38 products —
`Biosimilar`. They are kept because the fields exist in the API and are populated
for *some* records.

---

## The 26 broken detail pages

**26 of 8,935 products have a detail page that returns HTTP 500 on every
request.** Verified serially at 3-second intervals, four attempts each, plus
both crawl passes and two full independent runs. This is a bug on NAFDAC's
server, not throttling — a control product fetched 200 immediately afterwards in
the same session.

Their ids are listed in `detail_cache`-adjacent `detail_dead_ids.json` and in
`scrape_manifest.json`, and are skipped on later runs so the scraper doesn't burn
several minutes retrying a page that can never load. Delete that file to force a
fresh attempt if NAFDAC fixes the bug.

**Two of the three detail-only fields are still recovered for them**, using only
keys the site itself published:

- **Manufacturer Name** — 25 of 26 filled by joining `manufacturer_id` to the
  `/manufacturers` index. The 26th has a `manufacturer_id` absent from that
  index, so it stays empty.
- **Marketing Category** — 24 of 26 filled by mapping `marketing_category_id`
  through the values observed on pages that did load. The id space is small and
  one-to-one in practice: `0` = unset, `1` = Prescription-only Medicine (POM),
  `2` = Over-the-counter (OTC), `5` = NA. Ids seen with conflicting names are
  dropped rather than guessed.
- **Manufacturer Country** — 26 stay empty. No other page on the site carries
  it, and guessing a country from manufacturer name would be fabrication.

---

## Install

```bash
cd nafdac_scraper
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Only three dependencies. No BeautifulSoup or lxml — the markup is simple and
targeted regexes parse it reliably.

## Run

```bash
python scraper.py --limit 20    # 20-record smoke test
python scraper.py               # full crawl, all sources
python scraper.py --no-details  # skip ~8.9k detail requests (3 columns stay empty)
```

A cold full run makes roughly **8,900 detail requests** and takes about 40–70
minutes, depending on how the site is behaving. Progress and ETA are logged every
250 products. Interrupting is safe: re-running picks up from
`detail_cache.jsonl` and finishes in minutes.

| Flag | Default | Purpose |
|---|---|---|
| `--limit N` | `0` (all) | stop after N products |
| `--max-pages N` | `0` (all) | stop after N bulk pages |
| `--page-size N` | `500` | rows per bulk request |
| `--delay S` | `1.0` | pause between bulk page requests |
| `--detail-workers N` | `2` | concurrent detail workers |
| `--detail-delay S` | `0.4` | pause per detail worker |
| `--detail-cache PATH` | `./detail_cache.jsonl` | detail resume cache |
| `--detail-dead-file PATH` | `./detail_dead_ids.json` | ids with permanently broken detail pages |
| `--no-details` | off | skip the detail crawl |
| `--no-indexes` | off | skip the reference index crawl |
| `--timeout S` | `60` | per-request timeout |
| `--max-retries N` | `5` | attempts before giving up |
| `--output-dir DIR` | script folder | output location |
| `--no-xlsx` | off | CSV only |
| `-v` | off | debug logging |

## Output

```
nafdac_scraper/
├── scraper.py            # CLI, orchestration, export
├── client.py             # HTTP layer, retries, bulk + detail + index crawls
├── mapping.py            # JSON row → 34 columns, detail merge, dedup keys
├── reference.py          # manufacturers/applicants/ingredients/ATC
├── config.py             # all tunable settings
├── requirements.txt
├── README.md
├── nafdac_greenbook.csv        # 8,935 rows × 34 columns
├── nafdac_greenbook.xlsx       # Products + 5 reference sheets
├── scrape_manifest.json        # what was crawled, counts, timing
├── detail_cache.jsonl          # detail resume cache
├── detail_dead_ids.json        # 26 products with permanently broken detail pages
└── nafdac_scraper.log
```

**CSV** — UTF-8 **with BOM** so Excel on Windows renders accented applicant names
correctly. Flat, one row per product, 34 columns in the order above.

**XLSX** — six sheets:

| Sheet | Rows |
|---|---|
| Products | 8,935 |
| Manufacturers | 1,427 |
| Applicants | 1,661 |
| Ingredients | 2,393 |
| Product Categories | 6 |
| ATC Classification | 902 |

Each sheet has a frozen header row, an auto-filter, and sized columns. The
reference sheets are separated deliberately — they're different entities, and
mixing them into the product table would break the row-per-product shape.

---

## Disinfectants (category id 12)

You asked about this, and it needs a clear answer: **the Disinfectants category
exists but contains zero published products.**

Verified three ways:

- The API returns `recordsFiltered: 0` when filtering on `product_category_id = 12`.
- `https://greenbook.nafdac.gov.ng/productCategory/products/12` renders the
  heading "Products of Disinfectants" with an empty list.
- `/productCategories` renders a Disinfectants card, but the counts on that card
  are **inside an HTML comment** — the site has disabled them. They are
  decorative and do not reflect reality.

Category counts actually published by the API:

| Category | id | Products |
|---|---|---|
| Drugs | 1 | 7,448 |
| Medical devices | 5 | 1,107 |
| Vaccines and Biologics | 2 | 151 |
| Herbals and Nutraceuticals | 7 | 125 |
| Veterinary | 6 | 93 |
| **Disinfectants** | **12** | **0** |
| *(uncategorised)* | — | 11 |

Ids 3, 4 and 8–13 are unassigned. The 11 uncategorised rows carry
`Product Category = N/A`, which is what the API returns for medical devices
without a resolved category.

The scraper reads the unfiltered table, so this is not a filtering artefact —
Disinfectants genuinely has no rows.

---

## De-duplication

The primary key is **`product_id`**, the backend's true row identifier. An
earlier version keyed on NRN, which was wrong: NRN is not unique upstream.

**14 NRNs are each shared by two or more distinct products.** These are upstream
data problems, not duplicates. Both products are kept, and the collisions are
reported in the summary and in `scrape_manifest.json`. Examples:

- `A4-6223` → Getryl 2 mg Tablets (Getz Pharma, Active) *and* Amonex Capsules
  (Vixa Pharmaceutical, Inactive)
- `B4-4875` → Hanmal *and* Hanmal Dispersible Tablet (same applicant)

About 268 rows NAFDAC itself labels `(duplicate)` in the product name are
retained, because they are separately registered `product_id`s with real
approval dates. Suppressing them would be a judgement call about NAFDAC's
publications, not a de-duplication decision. Filter on the `Product Name`
column if you want them gone.

---

## Pagination and stopping

The bulk crawl reads `recordsTotal` on the first request, then advances `start`
by the number of rows returned. It stops on whichever comes first:
`start >= recordsTotal`, an empty `data` array, `--limit`, or `--max-pages`.
The final page came back partial (435 rows), which is the expected
end-of-dataset signal. No page count is hard-coded.

Index crawls read the pagination block to find the last page number, then walk
forward until a page yields no new cards.

## Reliability

- Session reuse, per-thread sessions for the detail crawl
- 60 s timeout per request
- 5 attempts with exponential backoff (1 s → 2 s → 4 s … capped at 60 s) plus
  jitter, on connection errors, 429 and 5xx
- **The detail crawl makes two full passes.** Detail pages are far more fragile
  than the JSON endpoint — at 4 workers the site began returning HTTP 500s and
  dropping connections. Two workers with a 0.4 s pause sustains a stable ~4
  requests/second with no errors. Anything still missing after pass two is
  logged by product ID and its fields stay empty.
- Detail cache is rewritten atomically via `os.replace`, so an interrupted run
  can't corrupt it
- Missing relationships degrade to empty strings rather than raising
- Anything already collected is still summarised and saved if the run aborts
- `scrape_manifest.json` records record counts, timing and phase flags for the run

## Compliance

`robots.txt` is:

```
User-agent: *
Disallow:
```

Nothing is disallowed. The scraper identifies itself with a normal browser
User-Agent, reads only public data, and paces itself. It's worth being explicit
about the request budget: the full run sends ~8,900 detail requests over roughly
40 minutes. That's the cost of the three fields that only exist on detail pages.
If you don't need Manufacturer Name, Manufacturer Country or Marketing Category,
`--no-details` gets you the full dataset in about 60 seconds with 19 requests.
That's a public-sector site — keep `--detail-delay` at 0.4 s or higher.

## Known limitations

- **Statuses other than Active/Inactive are unreachable.** The backend filters to
  those two values server-side, so suspended, withdrawn and recalled products
  never appear — not in the table, not in the API, not scrapable.
- **No product detail history.** `created_at` and `updated_at` are the only
  change data available; there is no audit trail.
- **`Marketing Category` is blank for 10 products.** Eight have
  `marketing_category_id = 0` (unset upstream) and two are among the 26 broken
  pages. Nothing is inferred for them.
- **ATC codes are stored per product, not validated** against the ATC sheet.
  A product's `ATC Code` may reference a code that isn't in the 902-entry
  classification list.
- **`/productCategories` counts are commented out upstream**, so the workbook's
  Product Categories sheet has a `Counts As Published` column that will be
  empty for every row. Use the live counts printed in the summary instead.
- Product names are cleaned of `#` and `*` markers because the site's own
  JavaScript strips them before display. Set `STRIP_PRODUCT_NAME_MARKERS = False`
  in `config.py` for raw API values.
- Approval dates are strings (`YYYY-MM-DD`) so the displayed value is preserved
  exactly. Cast to date in pandas if you need chronological sorting.
