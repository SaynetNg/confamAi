# ConfamAI — Product Verification Platform

ConfamAI is the official product verification assistant for **CONFAM**, powered by **saynet.ng**. It helps Nigerians check whether a product is registered with **NAFDAC** before they buy or use it — covering food products, drugs & pharmaceuticals, controlled substances, cosmetics & personal care, medical devices, chemicals & agro-chemicals, and veterinary products.

The assistant is multilingual (**English, Hausa, Yoruba, Igbo**), runs a fine-tuned **N-ATLaS** language model on a Kaggle Tesla T4 GPU, and is reachable over an OpenAI-compatible API, a laptop client, and a WhatsApp bot.

```
                 ┌─────────────────────────────────────────────┐
                 │  WhatsApp user  (confamai_bot)              │
                 └───────────────┬─────────────────────────────┘
                                 │ Meta Cloud API
                 ┌───────────────▼─────────────────────────────┐
                 │  Laptop client  (n_atlas_client)            │
                 │  NAFDAC lookup happens HERE (local data)    │
                 └───────────────┬─────────────────────────────┘
                                 │ HTTPS + Bearer key
                 ┌───────────────▼─────────────────────────────┐
                 │  Kaggle server  (n_atlas_server)            │
                 │  FastAPI + N-ATLaS (4-bit NF4) on Tesla T4  │
                 └─────────────────────────────────────────────┘
                                 ▲
                 ┌───────────────┴─────────────────────────────┐
                 │  Training  (n_atlas_train)  → HF adapter    │
                 └─────────────────────────────────────────────┘
```

---

## Repository layout

```
confamAi/
├── confam_prompt.txt          # ★ THE persona — single source of truth for ConfamAI
├── confam_training_dataset.jsonl
├── n_atlas_server/            # ★ Stage 1: Kaggle GPU server (FastAPI + model)
├── n_atlas_client/            # ★ Laptop client (package + standalone single-file)
├── n_atlas_train/             # QLoRA fine-tuning (Notebook A)
├── nafdac_source/             # ★ Greenbook lookup — shared by client & server
├── nafdac_scraper/            # Scrapes the NAFDAC Greenbook site → CSV/XLSX
├── confamai_bot/              # WhatsApp front end for the API
├── conversations/             # Saved chats, written by the clients
└── README.md                  # this file
```

**The one file that matters most is `confam_prompt.txt`** at the project root. It defines who ConfamAI is. Both the client and the server read it, so editing that one file changes the assistant's behaviour everywhere — no Python, no notebook cell, no constant to keep in step.

---

# 1. `n_atlas_server/` — Kaggle GPU Server (Stage 1)

`NCAIR1/N-ATLaS` (4-bit NF4) on a Tesla T4, exposed as an authenticated OpenAI-compatible FastAPI server. This folder is the **server half** of the project; it runs on Kaggle.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Full pinned dependency set. Safe to `pip install -r` on Kaggle. |
| `natlas_notebook.py` | **Source of truth.** 19 `# %%`-delimited cells, plain sequential Python. |
| `natlas_kaggle.ipynb` | Built from the above by `build_notebook.py`. **Upload this to Kaggle.** |
| `build_notebook.py` | Regenerates the `.ipynb`. Run after editing `natlas_notebook.py`. |
| `validate_notebook.py` | Checks the `.ipynb` parses, every cell compiles, no content is lost. |
| `verify_offline.py` | Runs Cells 1–18 plus all three scripts, with a mock model + real HTTP. Cell 19 skipped. |
| `verify_nafdac.py` | Checks Cell 18 (the greenbook) on its own. No GPU needed. |
| `mock_backend.py` | Test doubles for the tokenizer/model (used only by the verifier). |
| `notebook_cells.py` | Loads/parses the cells. Shared by the scripts and the builder. |
| `kaggleserver.py` | Script entry point: the **generation engine** (Cells 2–8), no HTTP. |
| `apiserver.py` | Script entry point: **engine + FastAPI server + greenbook** (Cells 1–11 plus Cell 18). |
| `kaggle.py` | Runs the notebook cells as a plain script (`--list`, `--cells 2-11`). |
| `check_prompt_drift.py` | Fails when any prompt copy differs from `confam_prompt.txt`. |
| `build_kaggle_dataset.py` | Builds the `n-atlas-source` Kaggle dataset (carries `prompt/confam_prompt.txt`). |

The scripts contain **no duplicated logic** — they execute the very same cells the notebook does, so nothing can drift out of sync. Editing `natlas_notebook.py` updates the notebook *and* every script at once.

## Dependencies

`requirements.txt` lists everything: model runtime (`transformers`, `accelerate`, `safetensors`, `bitsandbytes`), web server (`fastapi`, `uvicorn[standard]`, `pydantic`, `requests`, `nest-asyncio`), and test tooling (`openai`, `httpx`, `nbformat`). Two caveats:

- **`torch` is deliberately not listed.** It must match your CUDA build. Kaggle already ships `torch 2.10.0+cu128` for the T4; reinstalling it is the most common way to break the notebook.
- **`kaggle_secrets` is not on PyPI.** Kaggle injects it into the image.
- **`bitsandbytes` is the one hard requirement Kaggle does not ship.** Without it Cell 5 cannot build the 4-bit NF4 `BitsAndBytesConfig`.

## Running on Kaggle

1. Create a notebook, enable the **GPU T4 x2** accelerator.
2. Set `Internet` on (needed once to download the model).
3. Upload `natlas_kaggle.ipynb`.
4. Run **Cell 1** first — it *is* the credentials step (`N_ATLAS_API_KEY` + Hugging Face login).
5. Run **Cells 2 → 18 in order, once each**.
6. Read the **Cell 17** summary. Stage 1 is done only when it says `ALL CHECKS PASSED` with **no skipped checks**.
7. Only then run **Cell 19**, the public tunnel. It blocks until you interrupt the kernel.

```
Cell 1  credentials ─→ Cells 2 … 18 ─→ ALL CHECKS PASSED ─→ Cell 19  tunnel
```

## Configuration (all environment variables)

| Variable | Default | Notes |
|---|---|---|
| `N_ATLAS_API_KEY` | random ephemeral dev key | **Set this before exposing the server.** |
| `N_ATLAS_DEV_KEY` | unset | Known temporary key, dev only. |
| `N_ATLAS_ALLOW_EPHEMERAL_KEY` | `1` | Set `0` to hard-fail when no key is configured. |
| `N_ATLAS_SYSTEM_PROMPT` | unset — `confam_prompt.txt` is served | Inline system prompt override. |
| `N_ATLAS_SYSTEM_PROMPT_FILE` | unset — `confam_prompt.txt` is found on disk | Path to a prompt file; wins over inline. |
| `N_ATLAS_ADAPTER_ID` | unset — base model only | QLoRA adapter repo, e.g. `Saynetng/confam-ai-adapter`. |
| `N_ATLAS_MAX_NEW_TOKENS` | `512` | Hard ceiling on client `max_tokens`. |
| `N_ATLAS_DEFAULT_MAX_TOKENS` | `512` | Used when the client omits it. |
| `N_ATLAS_MAX_INPUT_TOKENS` | `4096` | Prompt ceiling. T4-safe — see below before raising. |
| `N_ATLAS_CONTEXT_BUDGET` | `8192` | N-ATLaS context target; the VRAM budget binds first. |
| `N_ATLAS_OOM_RETRY_FRACTION` | `0.5` | Second attempt uses this fraction of the budget. |
| `N_ATLAS_MAX_MESSAGES` | `64` | Most recent turns retained. |
| `N_ATLAS_MAX_MESSAGES_ABSOLUTE` | `512` | Above this, the request is rejected. |
| `N_ATLAS_MAX_REQUEST_BYTES` | `1048576` | Body size cap → `413`. |
| `N_ATLAS_LOG_CONTENT` | `0` | `1` logs message text (off by default). |
| `N_ATLAS_LANGUAGE_TAGGING` | `1` | `0` disables the deterministic language directive. |
| `N_ATLAS_STRICT_GPU` | `1` | `0` permits CPU (dev only). |
| `N_ATLAS_HOST` / `N_ATLAS_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `NAFDAC_DATABASE_URL` | unset | Supabase **session** pooler DSN (port 5432) for the greenbook source. |
| `NAFDAC_SOURCE_DIR` | unset | Where to find `nafdac_source/`. |
| `NAFDAC_CSV_PATH` / `NAFDAC_XLSX_PATH` | auto-detected | Local greenbook files. |
| `N_ATLAS_NAFDAC_ENABLED` | `true` | Master switch for Cell 18. |
| `N_ATLAS_NAFDAC_MAX_CONTEXT_CHARS` | `1800` | Cap on the injected block. |
| `N_ATLAS_NAFDAC_MAX_SEARCH_RESULTS` | `10` | Cap on `/v1/nafdac/search`. |

**Prompt precedence** (strongest first):

```
N_ATLAS_SYSTEM_PROMPT_FILE > N_ATLAS_SYSTEM_PROMPT > confam_prompt.txt on disk
                            > DEFAULT_SYSTEM_PROMPT in the notebook
```

Cell 2 prints which one won after `System prompt:`. `bundled:` means your file is being served; `builtin` means the dataset was not attached and the notebook fell back to its own copy.

## Measured on the Tesla T4

From a real run (T4 x2, 4-bit NF4, 14.56 GB usable per card):

| Prompt tokens | Peak VRAM | Result |
|---|---|---|
| ~1,350 | 1.6 GB steady | 200 OK |
| ~6,700 | 12.7 GB peak, wanted 4.3 GB more | **507 CUDA OOM** |

That is ~1.6 MB of VRAM **per prompt token** (large-vocab logits tensor + KV cache). Weights alone are only ~1.6 GB, so a long prompt costs several times the model's own size. Hence `N_ATLAS_MAX_INPUT_TOKENS` defaults to **4096** (~8.4 GB projected peak), and on CUDA OOM the server empties the cache and retries once at half the budget before returning `507`.

## Language handling

Two layers work together:

1. **Language tagging (server, deterministic).** Before each generation the last user message is classified into `en` / `ha` / `yo` / `ig` and an explicit directive is appended to the system prompt. Diacritics are folded first (`Báwo ni` → `bawo ni`), weighted marker scoring is used, an explicit request ("answer in Igbo") always wins, and English is the fallback. There is a **deliberate anti-English bias**: when a message mixes English with a well-attested Nigerian language, the Nigerian language wins.
2. **Model adherence.** The server relays whatever the model produces. Cell 14 checks the tagger and the model separately, so the report distinguishes "my code chose the wrong language" from "the model ignored a correct instruction".

**Pinned language mode:** a client can pin one language per request with a top-level `language` field (`"en"`, `"ha"`, `"yo"`, `"ig"`; `"auto"`/null/absent keeps detection). A pin replaces the detected directive with a HARD LANGUAGE MODE directive that holds even when the user's message is in another language.

## API

```python
from openai import OpenAI

client = OpenAI(base_url="https://YOUR_TUNNEL/v1", api_key="YOUR_SECRET")
reply = client.chat.completions.create(
    model="n-atlas",
    messages=[{"role": "user", "content": "Sannu, yaya kake?"}],
)
print(reply.choices[0].message.content)
```

| Route | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Readiness. GPU, quantization, load time, model_loaded, nafdac status. |
| `GET /v1/models` | bearer | OpenAI-compatible model list. |
| `POST /v1/chat/completions` | bearer | Chat completion. Full message history. |
| `POST /v1/nafdac/verify` | bearer | Check one NAFDAC number against the greenbook. |
| `POST /v1/nafdac/search` | bearer | Free-text product search. |
| `GET /v1/nafdac/sources` | bearer | Which greenbook sources are reachable. |

Streaming (`stream=true`) is **not** supported and returns `400`.

### Error mapping

| Code | Cause |
|---|---|
| `400` | Malformed JSON, empty/whitespace messages, no user turn, too many messages, stream requested, legacy `/v1/completions`, or a NAFDAC request with no number in it. |
| `401` | Missing or wrong bearer key. Adds `WWW-Authenticate`. |
| `404` | Unknown model name. |
| `413` | Body over `N_ATLAS_MAX_REQUEST_BYTES`. |
| `422` | Schema violations (bad role, `temperature` out of range, negative `max_tokens`). |
| `500` | Generation failed. |
| `503` | Model not loaded, or the generation queue timed out. |
| `507` | CUDA OOM. `torch.cuda.empty_cache()` runs; the server stays up. |

Clients never receive a traceback. Details go to the notebook log.

## Stage 1 gate (Cells 12–17)

- [x] model loads, GPU recognised, 4-bit NF4 active
- [x] English / Hausa / Yoruba / Igbo generation
- [x] mixed-language input and language switching
- [x] conversation memory across a language switch
- [x] response cleaning (markers, template artifacts, recovery)
- [x] `/health`, `/v1/models`, `/v1/chat/completions` → `200`
- [x] invalid API key → `401`; invalid requests → `400`/`404`/`422`/`413`
- [x] CUDA OOM → `507` after one halved-budget retry; no tracebacks leaked
- [x] long conversations trimmed so the **real** prompt fits the budget
- [x] OpenAI-compatible response body, verified through the real OpenAI SDK
- [x] one generation at a time via a lock; 3 concurrent requests all succeed
- [x] API key never echoed or logged
- [ ] replies actually in the requested language — **model-dependent**

## Script entry points

```bash
python kaggle.py --list                 # cell map
python kaggleserver.py                  # load model, smoke-test English/Hausa/Yoruba/Igbo/mixed
python apiserver.py                     # serve the API (blocks)
N_ATLAS_PORT=9000 N_ATLAS_API_KEY='…' python apiserver.py
python verify_offline.py                # full offline verification (mock model + real HTTP)
python verify_nafdac.py                 # Cell 18 only, no GPU
python check_prompt_drift.py            # prompt sync check
```

## Next stage (the tunnel)

Only after Stage 1 fully passes, Cell 19 installs cloudflared from the `.deb` and runs `cloudflared tunnel run --token "<tunnel token>"` against a **named** tunnel. The hostname is the Public Hostname configured in the Cloudflare Zero Trust dashboard, and it must point at `http://localhost:8000`. The cell blocks until interrupted.

---

# 2. `n_atlas_client/` — Laptop client

This is a **CLIENT**. The N-ATLaS model runs on Kaggle (Tesla T4). Nothing model-related is installed or runs on the laptop — no torch, no transformers, no bitsandbytes, no CUDA, no GPU, no Hugging Face download.

```
         LAPTOP
            |
            |  NAFDAC number in the message?
            +----> verify_nafdac_product()  ->  PostgreSQL + greenbook CSV/XLSX
            |                                  (on this machine, no network)
            |                                        |
            |                                  verified facts
            |                                        v
            | HTTPS (OpenAI-compatible API + Bearer key)
            ▼
       Secure Tunnel URL     <- the real URL goes in .env (N_ATLAS_BASE_URL)
            |
            ▼
      KAGGLE SERVER (FastAPI)
            |
            ▼
         N-ATLaS  ---- writes the reply in the user's language, using the
            |       verified facts rather than its own memory
            ▼
        Tesla T4 GPU
```

NAFDAC verification is the one part that runs **locally**, so a product question still gets an answer when the Kaggle session has ended.

## Files

| File | Purpose |
|---|---|
| `n_atlas.py` | **Self-contained single-file client** you can drop onto any machine on its own. |
| `main.py` / `__main__.py` | `python -m n_atlas_client` entry point. |
| `api.py` | OpenAI-compatible HTTP client. |
| `config.py` | `.env` resolution and settings. |
| `prompt.py` | System-prompt resolver + `CONFAM_AI_SYSTEM_PROMPT` fallback. |
| `language.py` | Deterministic language detection (en/ha/yo/ig). |
| `nafdac.py` / `nafdac_source.py` | Local greenbook verification. |
| `conversation.py` / `storage.py` | Saved chats. |
| `terminal.py` | Interactive terminal UI. |
| `tests.py` | Offline test suite. |
| `requirements.txt` | Just `openai`, `python-dotenv`, `requests`. |

## Configuration

Both entry points read the first `.env` they find, in this order: (1) the current working directory, (2) the project root (`../.env`), (3) this folder (`./.env`).

```
N_ATLAS_BASE_URL=https://REAL-TUNNEL-URL/v1
N_ATLAS_API_KEY=YOUR_SECRET_KEY
```

- Never use `localhost:8000` — localhost on your laptop is not the Kaggle server.
- The API key is read from the environment / `.env` and is **never** printed or written into conversation files.

## System prompt (ConfamAI)

The model answers as **ConfamAI** because both halves read one file: **`confam_prompt.txt` at the project root**.

- **Client** — `.env` sets `N_ATLAS_SYSTEM_PROMPT_FILE=confam_prompt.txt`; the resolver also finds the file beside the client or at the project root.
- **Server** — `build_kaggle_dataset.py` puts the file in the uploaded dataset as `prompt/confam_prompt.txt`, and Cell 2 finds it on disk.

Precedence on either side, strongest first:

```
N_ATLAS_SYSTEM_PROMPT_FILE > N_ATLAS_SYSTEM_PROMPT > confam_prompt.txt on disk > built-in fallback
```

---

# 3. `nafdac_source/` — Greenbook lookup (shared data layer)

The data layer behind N-ATLaS product verification. **One implementation, shared by the laptop client and the Kaggle server, so the two can never disagree about whether a product is authorised.**

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
    xlsx       nafdac_greenbook.xlsx          (local, same rows)
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

A user rarely has the number to hand. `find_products()` searches product name, active ingredients and applicant, ranked: an exact NRN or product-ID hit first, then a name prefix, then a substring, then a match in the ingredients or the applicant.

Several candidates means the data has not answered the question about the pack in the user's hand, so **no verdict is claimed** and the block tells the model to ask which product they mean. It never picks one.

## Every message is searched

There is no test for "is this about a product". Every message goes through the greenbook, because a hit is cheap and a miss costs nothing:

```
    hit   -> the block is attached, and the model decides whether it is relevant
    miss  -> nothing is attached, and the model answers conversationally
```

The alternative — deciding up front whether a message is about a product — does not survive contact with the data. A greenbook name is ordinary English, so `good morning` matches "Good Morning Cough Syrup", `I love my family` matches 16 products containing *Family*, and `I need water` matches "Water Guard Plus Powder".

## Files

| File | Purpose |
|---|---|
| `sources.py` | The three sources (postgres / csv / xlsx) behind one interface. |
| `verify.py` | `verify_nafdac_product()` — number lookup + verdict. |
| `records.py` | Record model, ranking, rendering. |
| `extract.py` | `extract_nrn()` / `extract_search_terms()`. |
| `formatting.py` | The rendered block handed to the model. |
| `load_postgres.py` | Loads the greenbook into PostgreSQL. |
| `verify_postgres.py` | Postgres-specific verification. |
| `schema.sql` | Database schema. |
| `tests.py` | 232 offline checks. |

---

# 4. `nafdac_scraper/` — Greenbook scraper

Extracts every field the NAFDAC Greenbook site publishes into a structured dataset, exported as CSV plus a multi-sheet Excel workbook.

Last verified against the live site: **8,935 products, 34 columns, plus 5 reference sheets** (1,427 manufacturers, 1,661 applicants, 2,393 ingredients, 6 product categories, 902 ATC codes). All ten originally requested columns are 100% populated except NRN, which is blank on 3 products because the site has no NRN for them.

## How the site works

The Greenbook is a Laravel application. There is **no public REST API** — the data lives behind jQuery DataTables (`serverSide: true`) and plain server-rendered HTML pages. Three access patterns are needed:

1. **Bulk product JSON** — `GET https://greenbook.nafdac.gov.ng/` with header `X-Requested-With: XMLHttpRequest` (the only real trap: without it the server returns the HTML home page instead of JSON). Pagination uses the DataTables protocol (`draw`, `start`, `length`); the scraper pages at 500 so an interrupted run costs one page, not the whole crawl.
2. **Product detail pages** — `https://greenbook.nafdac.gov.ng/products/details/{product_id}`, server-rendered HTML. Three fields exist **only** here: **Manufacturer Name**, **Manufacturer Country**, **Marketing Category**.
3. **Reference index pages** — server-rendered, paginated with `?page=N`, 50 cards per page (`/manufacturers`, `/applicants`, `/ingredients`, …).

## Files

| File | Purpose |
|---|---|
| `scraper.py` | The scraper itself. |
| `client.py` | HTTP client for the three access patterns. |
| `mapping.py` | JSON/HTML field → column mapping. |
| `reference.py` | Reference-sheet scraping. |
| `config.py` | URLs, paths, settings. |
| `nafdac_greenbook.csv` / `.xlsx` | The exported dataset (feeds `nafdac_source/`). |
| `detail_cache.jsonl` | Cached detail pages. |
| `scrape_manifest.json` | Run metadata. |

---

# 5. `n_atlas_train/` — QLoRA fine-tuning (Notebook A)

Trains the ConfamAI adapter with QLoRA and pushes it to Hugging Face. This folder is the **training** half of the project and serves nothing.

| | This folder (Notebook A) | `n_atlas_server/` (Notebook B) |
|---|---|---|
| Job | trains the adapter, pushes it, ends | serves chat requests on port 8000 |
| Kaggle dataset attached | `confam-training` | `n-atlas-source` |
| Lifetime | one run, ~1.5 h, then stop it | as long as you leave it running |
| Your laptop client points at it | never | **only this one** |

The only thing that crosses between them is the adapter on the Hugging Face Hub. Notebook B picks it up by setting `N_ATLAS_ADAPTER_ID`; nothing else changes, and `served_model_name` stays `n-atlas` so no client is reconfigured.

## Files

| File | Role |
|---|---|
| `train_notebook.py` | **Source of truth.** Plain sequential Python with `# %%` cell markers. |
| `confam_train_kaggle.ipynb` | **Build artefact** — upload this to Kaggle. Never hand-edit it. |
| `build_train_notebook.py` | `train_notebook.py` → `confam_train_kaggle.ipynb`. |
| `validate_train_notebook.py` | Compiles every cell, checks nbformat line encoding, proves the notebook reassembles exactly. |
| `confam_training_dataset.jsonl` | The 643-record training set Notebook A consumes. |
| `prepare_training_data.py` | Rebuilds that dataset from a raw source, or validates it in place. |
| `check_transcript_coverage.py` | Compares real saved conversations against what the dataset covers. |
| `qlora_finetuning_guide.md` | The original cell-by-cell walkthrough (reference; superseded by the README). |
| `time.md` | The training-time estimate: arithmetic, research anchors, Kaggle limits. |

## Running it

**Before the first run**

1. Upload `confam_training_dataset.jsonl` to Kaggle as a dataset named `confam-training`.
2. Create a Kaggle secret named `HF_TOKEN` with a Hugging Face token that has **write** permission (Cell 8 pushes the adapter).
3. Accelerator: **GPU T4 x2**. The second T4 is deliberately left idle.
4. **Accept N-ATLaS's licence on Hugging Face first** — `NCAIR1/N-ATLaS` is gated.
5. Budget **~1.5 hours** for the default 200 steps on one T4.

**Then**

```bash
python build_train_notebook.py
python validate_train_notebook.py   # must print VALID
# upload confam_train_kaggle.ipynb to Kaggle, attach confam-training + HF_TOKEN, run all cells
```

## The nine cells

| Cell | Job | Fails loudly if |
|---|---|---|
| 1 | config, finds the dataset in `/kaggle/input/**` | the JSONL is not attached |
| 2 | installs libraries, `trl` pinned to 0.12.2 | no GPU, or more than one is visible |
| 3 | Hugging Face login | `HF_TOKEN` is missing |
| 4 | **dataset integrity checks** | personas are mixed, a verdict has no block, or a record claims counterfeit / safety |
| 5 | base model in 4-bit NF4 + LoRA | the base model exposes no chat template |
| 6 | renders records with the model's own template, **measures token lengths** | any record exceeds `MAX_SEQ_LENGTH` |
| 7 | trains 200 steps | — |
| 8 | pushes the adapter to the Hub | the pushed repo is implausibly large |
| 9 | three behavioural checks | prints PASS/CHECK; `identity` not saying ConfamAI means the adapter did not take |

Cell 4 runs **before** the 5-minute model load, so a bad upload fails in seconds. Cell 6's length measurement matters because all 643 records repeat the same ~850-token system prompt, so a too-small `max_seq_length` silently truncates the assistant turn — the only part of a record that teaches anything.

## After it works

Set `N_ATLAS_ADAPTER_ID` on Notebook B (a Kaggle secret named `N_ATLAS_ADAPTER_ID` is the clean way), then re-run Cells 1, 2, 3 and 5 of `natlas_kaggle.ipynb`. Cell 2's banner must print `Adapter: Saynetng/confam-ai-adapter` and `GET /v1/health` must report `"adapter_id"`. A configured adapter that fails to load stops Notebook B rather than falling back to the base model, so the two can never silently disagree.

---

# 6. `confamai_bot/` — WhatsApp bot

A WhatsApp front end for the N-ATLaS NAFDAC verification API. It receives a message, hands it to the server, and sends back the reply.

```
   WhatsApp user
       │
       │  Meta Cloud API
       ▼
   POST /webhook ──► validate signature ──► dedupe ──► 200 to Meta
       │
       │  background task
       ▼
   Router ──► local commands (no model)  │  everything else
                                             ▼
                               POST /v1/chat/completions  ──►  ai.saynet.ng
                                             │              (N-ATLaS, Tesla T4,
                                             │               NAFDAC intent gate)
                                             ▼
                                 reply ──► render ──► Graph API send
```

## What it owns — and what it deliberately does not

**Owns:** transport (the webhook and the sender), sessions (per-phone history), formatting (WhatsApp-safe text), and honesty about failure.

**Does not own, because the server already does and owns it better:** the NAFDAC greenbook lookup, the intent gate, the verdict, the system prompt. So this package queries no greenbook and holds no product data. It imports exactly one thing from `nafdac_source` — `extract_nrn`, a pure text function that reads no database and no file.

That restraint is the design, not laziness. A second retrieval authority on this side would run *before* the server has seen the message. The cost is explicit: **when the N-ATLaS server is down, this bot cannot verify anything and says so.** There is no local greenbook to fall back on, by design — the alternative would be answering from the model's memory, which is exactly how a cough syrup ends up described with ingredients nobody put in the database.

## Files

| File | Purpose |
|---|---|
| `__main__.py` | `python -m confamai_bot` entry point. |
| `webhook.py` | Meta webhook receiver + signature validation. |
| `whatsapp.py` | Graph API sender. |
| `router.py` | Message routing (local commands vs. model). |
| `sessions.py` | Per-phone conversation history. |
| `render.py` | WhatsApp-safe text formatting. |
| `framing.py` | Message framing. |
| `languages.py` | Language handling. |
| `queue.py` | Background task queue. |
| `assay.py` | Live test harness (13 real questions). |
| `natlas.py` | N-ATLaS API client. |
| `config.py` | Settings from `.env`. |
| `tests.py` | Offline test suite. |

## Run

```bash
python -m confamai_bot --status       # is the server reachable and green?
python -m confamai_bot --test         # offline suite, no network
python -m confamai_bot --ask "Is A4-100160 genuine?"
python -m confamai_bot --assay        # 13 real questions, live
python -m confamai_bot                # serve
python -m confamai_bot --port 8080
```

## Configure

Copy `.env.example` to `.env` and fill in:

| Variable | Where it comes from |
|---|---|
| `WHATSAPP_PHONE_NUMBER_ID` | Meta app → WhatsApp → API Setup |
| `WHATSAPP_ACCESS_TOKEN` | **System User permanent token** |
| `WHATSAPP_VERIFY_TOKEN` | invent one; do not reuse anything |
| `WHATSAPP_APP_SECRET` | Meta app → Basic Settings |
| `N_ATLAS_BASE_URL` | `https://ai.saynet.ng/v1` |
| `N_ATLAS_API_KEY` | the key the N-ATLaS server issued |

A dashboard temporary token expires in about a day, and the bot is discovered to be broken roughly then. Before going public, rotate both secrets — the bot puts an internet-facing webhook in front of the N-ATLaS key.

---

# 7. `confam_prompt.txt` — the persona (single source of truth)

The ConfamAI system prompt. Both the client and the server read this file, and every copy of the prompt in the codebase (`n_atlas_client/prompt.py`, `n_atlas_client/n_atlas.py`, `n_atlas_server/natlas_notebook.py`, `n_atlas_server/natlas_kaggle.ipynb`) must stay identical to it.

**Key rules it encodes:**

- **Identity:** the assistant is **ConfamAI** — never N-ATLaS, never "Confam AI".
- **Verdicts:** exactly `AUTHENTIC` (registered), `FLAGGED` (recalled, banned or suspended), or `NOT REGISTERED` (not found in database) — copied word-for-word from the data, never restated.
- **NOT REGISTERED ≠ fake.** The required sentence is always: *"This product was not found in our current NAFDAC database. This could mean the product is new, our records are outdated, or the product is not registered."*
- **Never claim** a product is counterfeit, fake, poisoned or adulterated.
- **Safety:** if a food, drink, cosmetic, medicine, or chemical comes back NOT REGISTERED or FLAGGED, say plainly that the user should not buy or use it, and suggest they report it to NAFDAC.
- **Data-only:** never invent registration numbers, approval dates, manufacturers, product names, or verdicts; if a field is not recorded, say it is not recorded.
- **Language:** always match the user's language (English / Hausa / Yoruba / Igbo); default to English, never Hausa, when unsure.
- **Boundaries:** product verification and NAFDAC questions only; never provide medical, nutritional or health advice; the service is **not** officially endorsed by NAFDAC.

Verify sync at any time:

```bash
cd n_atlas_server && python check_prompt_drift.py
# VALID: every copy matches confam_prompt.txt.
```

---

# 8. `conversations/` — saved chats

Saved conversation transcripts, written by the clients (`n_atlas_client/` and `confamai_bot/`). Used by `n_atlas_train/check_transcript_coverage.py` to measure what the training dataset covers against real usage.

---

# End-to-end data flow

```
1. nafdac_scraper/      scrapes greenbook.nafdac.gov.ng  →  nafdac_greenbook.csv / .xlsx
2. nafdac_source/       loads CSV/XLSX + PostgreSQL        →  verify_nafdac_product() / find_products()
3. n_atlas_train/       fine-tunes N-ATLaS on 643 records →  Saynetng/confam-ai-adapter (HF Hub)
4. n_atlas_server/      serves the model + greenbook on Kaggle  →  OpenAI-compatible API
5. n_atlas_client/      laptop client: local NAFDAC lookup + API call
6. confamai_bot/        WhatsApp front end → same API
```

**The persona (`confam_prompt.txt`) and the greenbook data (`nafdac_source/`) are the two things every layer shares**, which is why a product answer is identical whether it arrives over the API, the laptop client, or WhatsApp.

---

## Quick reference

| Task | Command |
|---|---|
| Rebuild the Kaggle notebook | `cd n_atlas_server && python build_notebook.py` |
| Validate the notebook | `cd n_atlas_server && python validate_notebook.py` |
| Check prompt sync | `cd n_atlas_server && python check_prompt_drift.py` |
| Verify server offline | `cd n_atlas_server && python verify_offline.py` |
| Verify greenbook only | `cd n_atlas_server && python verify_nafdac.py` |
| Run greenbook tests | `cd nafdac_source && python -m nafdac_source.tests` |
| Rebuild training notebook | `cd n_atlas_train && python build_train_notebook.py` |
| Run laptop client | `cd n_atlas_client && python -m n_atlas_client` |
| Run WhatsApp bot | `cd confamai_bot && python -m confamai_bot` |
