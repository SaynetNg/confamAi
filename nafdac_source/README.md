# NAFDAC greenbook lookup

The data layer behind N-ATLaS product verification. One implementation, shared
by the laptop client and the Kaggle server, so the two can never disagree about
whether a product is authorised.

```
    user message
        |
        v
    extract_nrn()             pull a NAFDAC number out of free text
        |
        v
    verify_nafdac_product()   resolve it against every available source
        |
        +---> postgres         current + superseded products  (hosted, remote)
        +---> csv              nafdac_greenbook.csv           (local, 8930 rows)
        +---> xlsx             nafdac_greenbook.xlsx         (local, same rows)
        |
        v
    VerificationResult         merged record, provenance, confidence
```

## Quick start

```bash
python -m nafdac_source.tests              # 109 checks, no network, no database
python -m nafdac_source.load_postgres --dry-run
```

```python
from nafdac_source import verify_nafdac_product, search_products

result = verify_nafdac_product("Is A4-100160 genuine?")
if result.found:
    print(result.primary.product_name)   # AG Spironolactone
    print(result.authorised)             # True
    print(result.confidence)             # verified | single-source | local-file
else:
    print(result.render())

for record in search_products("malaria", limit=5):
    print(record.nrn, record.product_name)
```

## The result, not a verdict

`verify_nafdac_product()` returns a `VerificationResult` rather than a bool,
because the honest answer is rarely yes or no:

| Field                | Meaning                                                        |
|----------------------|----------------------------------------------------------------|
| `found`              | some source had this number                                    |
| `matches`            | every record, merged across sources                            |
| `ambiguous`          | more than one distinct product carries this number              |
| `authorised`         | `True`, `False`, or `None` when the status is not recognised    |
| `confidence`         | how much weight the answer carries, and why                     |
| `sources_checked`    | which sources ran                                              |
| `sources_failed`     | which did not, and the reason                                  |
| `greenbook_url`      | the NAFDAC page, when a product id is known                     |

`render()` produces the plain-text block handed to the model. It carries an
explicit instruction not to supply a value that is not in the block, because the
failure this whole package exists to prevent is a model confidently inventing a
product name or an approval date.

## Design decisions worth knowing

**NRN is not unique.** 257 of the 8930 rows share a number with another row.
`Product ID` is the real primary key. A lookup that returns several records is
reporting a genuine ambiguity, so all of them are shown and the user is asked
which product they meant. Nothing is silently picked.

**Sources merge, they do not race.** The first source with a value wins; later
sources only fill empty fields. A thin database row is enriched from the CSV,
and a field the database knows is never overwritten by the file.

**A miss is not a verdict.** Not being in the export is not proof a product is
counterfeit, so the wording says so. This matters for a real user: being told a
genuine product is fake because of a stale export destroys trust in every other
answer.

**A status is never guessed.** The greenbook publishes `Active` and `Inactive`.
`withdrawn`, `recalled` and similar map to the same "not authorised" state, but
anything unrecognised returns `authorised=None` rather than being flattened to
`True`.

**One bad source never breaks a lookup.** Every source returns a `SourceResult`
with `ok=False` and a reason instead of raising, so an unreachable database
degrades the answer rather than failing the question.

**Number formats are messy, so normalisation is two-stage.** `normalize_nrn()`
produces a display form close to what NAFDAC published; `normalize_nrn_key()`
produces the lookup key. `A4 - 5180`, `a4-5180` and an en-dashed `04–2045` all
resolve. Placeholders (`NA`, `Not available yet`, `2 X 7`) are recognised as
"no number recorded" and never looked up. A sentence is never treated as a
number, or the whole sentence would be sent to the database as a key.

## The PostgreSQL database

Three tables, in `schema.sql`:

| Table                     | Holds                                            |
|---------------------------|--------------------------------------------------|
| `nafdac_products`         | the current greenbook, keyed on `product_id`      |
| `nafdac_product_history`  | superseded, withdrawn, suspended, recalled        |
| `nafdac_load_runs`        | what was loaded, when, and how many rows          |

The history table is the reason for the split. A number that has been withdrawn
must be answerable as "this used to be registered", and that is a different
answer from "this was never registered". Collapsing them would tell a user a
real product is fake.

`nrn_key` is stored normalised and indexed, so a lookup for `A4 - 5180` hits
the row stored as `A4-5180`.

```bash
pip install "psycopg[binary]"

export NAFDAC_DATABASE_URL='postgresql://USER:PASS@HOST:5432/DBNAME?sslmode=require'

python -m nafdac_source.load_postgres --dry-run   # read the CSV, write nothing
python -m nafdac_source.load_postgres             # create schema, load, archive
python -m nafdac_source.verify_postgres           # prove the round trip
```

Re-running the loader is safe. The schema is `IF NOT EXISTS` throughout, rows
are upserted on `product_id`, and a row already in history is not re-archived —
so history accumulates real supersessions, not loader runs.

`verify_postgres.py` writes to both tables. Point it at a scratch database.

Connection strings come from `--url`, `NAFDAC_DATABASE_URL` or `DATABASE_URL`,
and are never printed: the tools log only the host and database name.

## Performance

| Operation                | Time    |
|--------------------------|---------|
| CSV load (first)         | ~1.4 s  |
| CSV load (cached in RAM) | instant |
| XLSX load (first)        | ~40 s   |
| XLSX load (cached CSV)   | ~1.5 s  |
| Any single lookup        | <10 ms  |

Reading the workbook with openpyxl is slow, so the parsed rows are cached
beside it as a small CSV keyed on the workbook's size and mtime. The cache is
plain CSV rather than pickle on purpose: it is fast, inspectable, and cannot
execute anything when loaded. It is gitignored, and a stale or missing cache
just means the workbook is read again.

`build_sources()` caches its sources process-wide, so the files are read once
per process no matter how many messages are sent.

## Configuration

| Variable              | Default                       | Effect                        |
|-----------------------|-------------------------------|-------------------------------|
| `NAFDAC_DATABASE_URL` | unset                         | enables the PostgreSQL source |
| `NAFDAC_CSV_PATH`     | auto-detected                 | greenbook CSV                 |
| `NAFDAC_XLSX_PATH`    | auto-detected                 | greenbook workbook             |

Every source is optional. With none of them configured, the package still runs
and reports the number as not found — it never raises because a source is
missing.

## Files

| File                 | Purpose                                            |
|----------------------|----------------------------------------------------|
| `extract.py`         | NRN extraction and normalisation                  |
| `records.py`         | `ProductRecord`, status handling, merging          |
| `sources.py`         | `CsvSource`, `XlsxSource`, `PostgresSource`        |
| `verify.py`          | `verify_nafdac_product()`, `search_products()`     |
| `formatting.py`      | terminal and prompt rendering                      |
| `schema.sql`         | the PostgreSQL schema                             |
| `load_postgres.py`   | CSV to PostgreSQL, with history archiving          |
| `verify_postgres.py` | checks the database path against a real server     |
| `tests.py`           | 109 checks, no network or database required        |
