# NAFDAC greenbook lookup

The data layer behind N-ATLaS product verification. One implementation, shared
by the laptop client and the Kaggle server, so the two can never disagree about
whether a product is authorised.

```
    user message
        |
        +-- number present --> extract_nrn() --> verify_nafdac_product()
        |
        +-- no number ------> extract_search_terms() --> find_products()
        |                                                      |
        |                                             no hit? -> nothing
        v                                                      |
    VerificationResult | SearchResult <-----------------------+
        |
        v
    rendered block: verified values, a verdict, and the rules the model
    must follow. Instructions sit ABOVE the records, because callers cap
    the block and a long candidate list is truncated at the end.
```

Both paths resolve against every available source, in order:

```
    postgres   current + superseded products  (hosted, remote)
    csv        nafdac_greenbook.csv           (local, 8930 rows)
    xlsx       nafdac_greenbook.xlsx         (local, same rows)
```

## Quick start

```bash
python -m nafdac_source.tests              # 232 checks, no network, no database
python -m nafdac_source.load_postgres --dry-run
```

```python
from nafdac_source import verify_nafdac_product, find_products

result = verify_nafdac_product("Is A4-100160 genuine?")
if result.found:
    print(result.primary.product_name)   # AG Spironolactone
    print(result.verdict)                # authentic | flagged | not_registered | unknown
    print(result.authorised)             # True
    print(result.confidence)             # verified | single-source | local-file
else:
    print(result.render())

# No number, just product words, in any supported language.
for record in find_products("paracetamol", limit=5).records:
    print(record.nrn, record.product_name, record.verdict)
```

## Verifying by name, not just by number

A user rarely has the number to hand. `find_products()` searches product name,
active ingredients and applicant, ranked: an exact NRN or product-ID hit first,
then a name prefix, then a substring, then a match in the ingredients or the
applicant.

```python
result = find_products("malaria", limit=5)
result.records       # ranked candidates
result.total         # how many matched at all
result.flagged_count # how many are flagged
result.verdict       # only meaningful with a single candidate
result.render()      # the block handed to the model
```

Terms are matched independently rather than as one joined string, because the
match is a substring test and no product name contains the exact phrase
"paracetamol syrup". `extract_search_terms()` produces them: an NRN in the text
wins outright, otherwise stop words are dropped and the longest remaining terms
are kept.

Several candidates means the data has not answered the question about the pack
in the user's hand, so no verdict is claimed and the block tells the model to ask
which product they mean. It never picks one.

## Every message is searched

There is no test for "is this about a product". Every message goes through the
greenbook, because a hit is cheap and a miss costs nothing:

```
    hit   -> the block is attached, and the model decides whether it is relevant
    miss  -> nothing is attached, and the model answers conversationally
```

The alternative â€” deciding up front whether a message is about a product â€” does
not survive contact with the data. A greenbook name is ordinary English, so:

| Message                  | Matches                                     |
|--------------------------|---------------------------------------------|
| `good morning`           | "Good Morning Cough Syrup"                  |
| `I love my family`       | 16 products containing *Family*             |
| `what is the weather`    | 4 products, via applicant names            |
| `I need water`           | "Water Guard Plus Powder"                   |
| `Shin wannan magari ne?` | nothing (the greenbook is in English)       |

A word list cannot separate those from a genuine product question, because the
difference is one of meaning. So the decision is handed to the model, which
reads the block, understands the question in any of the four languages, and is
told in as many words to drop the candidates if they are not what the user meant.

`SearchResult.strong_match` reports whether the closest hit was a product or only
a word in a company's name, and a weak match says so in the block.

## Four languages

Diacritics are folded, so `BÃ¡wo` and `Bawo`, and `á»dá»¥` and `odu`, reach the same
entry. `extract_search_terms()` keeps a product noun as a search term even though
it is an ordinary word elsewhere (`magari`, `oro`, `odu`), because it is often
the only product-related word the user typed.

The greenbook indexes English names, so a Yoruba or Igbo message usually finds
nothing and the model answers conversationally. Translating the query is the
model's job, not this package's â€” the model already speaks all four languages,
and a translation layer in Python would be a worse version of something it does
better.

## The result, not a verdict

`verify_nafdac_product()` returns a `VerificationResult` rather than a bool,
because the honest answer is rarely yes or no:

| Field                | Meaning                                                        |
|----------------------|----------------------------------------------------------------|
| `found`              | some source had this number                                    |
| `verdict`            | `authentic`, `flagged`, `not_registered` or `unknown`          |
| `verdict_reason`     | why, in plain words                                            |
| `matches`            | every record, merged across sources                            |
| `ambiguous`          | more than one distinct product carries this number              |
| `authorised`         | `True`, `False`, or `None` when the status is not recognised    |
| `confidence`         | how much weight the answer carries, and why                     |
| `sources_checked`    | which sources ran                                              |
| `sources_failed`     | which did not, and the reason                                  |
| `greenbook_url`      | the NAFDAC page, when a product id is known                     |

`verdict` is the answer to "is this genuine?", which `authorised` is not: a user
holding a pack wants to know whether to trust it, not whether a boolean is set.

| Verdict          | Means                                                          |
|------------------|----------------------------------------------------------------|
| `authentic`      | listed as Active in the current greenbook                      |
| `flagged`        | withdrawn, suspended, recalled, expired, revoked, or a duplicate entry |
| `not_registered` | no available source lists it                                   |
| `unknown`        | found, but the status is unrecognised, or the number is ambiguous |

There is deliberately **no `fake` verdict.** The greenbook records registrations,
not counterfeiting: a number that is absent from an export is a fact about the
export, not about the product. Telling a real product is counterfeit because of a
stale export destroys trust in every other answer, so every render carries an
explicit instruction not to.

`AUTHENTIC` does not mean the pack in the user's hand is genuine either â€” a
copied number resolves just as well. The rendered block says so whenever the
verdict is `authentic`.

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
produces the lookup key. `A4 - 5180`, `a4-5180` and an en-dashed `04â€“2045` all
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
are upserted on `product_id`, and a row already in history is not re-archived â€”
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
| Search index (first)     | ~2.8 s  |
| Search index (cached)    | instant |
| Any single lookup        | <10 ms  |
| Search after indexing    | ~1 ms   |

Every message is searched, so the search path is the hot one, and it is
built on a **whole-word inverted index** rather than a scan. Scanning the raw
strings of 8930 rows per message measured about 0.9 s; the index brings a
lookup to about 1 ms, and a message that matches nothing to 0.1 ms. Both indexes
are built once per process and cached alongside the parsed rows.

Words are indexed whole, not as substrings. A substring test makes `treat`
match "Levotreat" and `cold` match "Coldkris", which would attach a list of
candidates to "how do I treat a cold?".

`build_sources()` caches its sources process-wide, so the files are read once
per process no matter how many messages are sent.

## Configuration

| Variable              | Default                       | Effect                        |
|-----------------------|-------------------------------|-------------------------------|
| `NAFDAC_DATABASE_URL` | unset                         | enables the PostgreSQL source |
| `NAFDAC_CSV_PATH`     | auto-detected                 | greenbook CSV                 |
| `NAFDAC_XLSX_PATH`    | auto-detected                 | greenbook workbook             |

Every source is optional. With none of them configured, the package still runs
and reports the number as not found â€” it never raises because a source is
missing.

## Files

| File                 | Purpose                                            |
|----------------------|----------------------------------------------------|
| `extract.py`         | NRN extraction, multilingual triggers, normalisation           |
| `records.py`         | `ProductRecord`, status handling, verdicts, merging             |
| `sources.py`         | `CsvSource`, `XlsxSource`, `PostgresSource`                     |
| `verify.py`          | `verify_nafdac_product()`, `find_products()`, `search_products()` |
| `formatting.py`      | terminal and prompt rendering                                   |
| `schema.sql`         | the PostgreSQL schema                                          |
| `load_postgres.py`   | CSV to PostgreSQL, with history archiving                       |
| `verify_postgres.py` | checks the database path against a real server                  |
| `tests.py`           | 232 checks, no network or database required                     |
