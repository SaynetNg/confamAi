# ConfamAI — Product Verification Platform

ConfamAI is the official product verification assistant for **CONFAM**, powered by **saynet.ng**. It helps Nigerians check whether a product is registered with **NAFDAC** before they buy or use it — covering Food Products (packaged foods, beverages, water, supplements), Drugs & Pharmaceuticals (medications, vaccines, herbal remedies), Controlled Substances, Cosmetics & Personal Care, Medical Devices, Chemicals & Agro-chemicals, and Veterinary Products.

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
                 │  FastAPI + N-ATLaS (4-bit NF4) on Tesla T4 │
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
├── confam_training_dataset.jsonl   # 643-record QLoRA training set
├── mermaid-diagram.png        # architecture diagram
├── n-atlas-source.zip         # Kaggle dataset archive (built by build_kaggle_dataset.py)
├── .env                       # client + server configuration (gitignored)
├── .gitignore
├── README.md                  # this file
│
├── n_atlas_server/            # ★ Stage 1: Kaggle GPU server (FastAPI + model)
├── n_atlas_client/            # ★ Laptop client (package + standalone single-file)
├── nafdac_source/             # ★ Greenbook lookup — shared by client & server
├── nafdac_scraper/            # Scrapes the NAFDAC Greenbook site → CSV/XLSX
├── n_atlas_train/             # QLoRA fine-tuning (Notebook A)
├── confamai_bot/              # WhatsApp front end for the API
└── conversations/             # Saved chats, written by the clients
```

**The one file that matters most is `confam_prompt.txt`** at the project root. It defines who ConfamAI is. Both the client and the server read it, so editing that one file changes the assistant's behaviour everywhere — no Python, no notebook cell, no constant to keep in step.

---

# 1. `n_atlas_server/` — Kaggle GPU Server (Stage 1)

`NCAIR1/N-ATLaS` (4-bit NF4) on a Tesla T4, exposed as an authenticated OpenAI-compatible FastAPI server. This folder is the **server half** of the project; it runs on Kaggle.

## Files

| File | Size | Purpose |
|---|---|---|
| `natlas_notebook.py` | 186 KB | **Source of truth.** 19 `# %%`-delimited cells, plain sequential Python. |
| `natlas_kaggle.ipynb` | 240 KB | Built from the above by `build_notebook.py`. **Upload this to Kaggle.** |
| `build_notebook.py` | 3.8 KB | Regenerates the `.ipynb`. Run after editing `natlas_notebook.py`. |
| `validate_notebook.py` | 3.1 KB | Checks the `.ipynb` parses, every cell compiles, no content is lost. |
| `notebook_cells.py` | 4.0 KB | Shared loader for the `# %%` cells. Used by the scripts and the builder. |
| `kaggleserver.py` | 4.7 KB | Script entry point: the **generation engine** (Cells 2–8), no HTTP. |
| `apiserver.py` | 3.1 KB | Script entry point: **engine + FastAPI server + greenbook** (Cells 1–11 plus Cell 18). |
| `kaggle.py` | 2.4 KB | Runs the notebook cells as a plain script (`--list`, `--cells 2-11`). |
| `verify_offline.py` | 38.7 KB | Runs Cells 1–18 plus all three scripts, with a mock model + real HTTP. Cell 19 skipped. |
| `verify_nafdac.py` | 43.4 KB | Checks Cell 18 (the greenbook) on its own. No GPU needed. |
| `mock_backend.py` | 8.2 KB | Test doubles for the tokenizer/model (used only by the verifier). |
| `check_prompt_drift.py` | 4.1 KB | Fails when any prompt copy differs from `confam_prompt.txt`. |
| `build_kaggle_dataset.py` | 6.5 KB | Builds the `n-atlas-source` Kaggle dataset (carries `prompt/confam_prompt.txt`). |
| `requirements.txt` | 2.5 KB | Full pinned dependency set. Safe to `pip install -r` on Kaggle. |
| `.env` | 1.2 KB | Server env: HF token, API key, tunnel config (gitignored). |
| `natlas_api_key.txt` | 50 B | Ephemeral key written when `N_ATLAS_API_KEY` is unset (gitignored). |
| `readme.md` | 26.4 KB | The server's own detailed README. |
| `.offline-workdir/` | dir | Scratch directory for `verify_offline.py`. |
| `__pycache__/` | dir | Compiled bytecode (gitignored). |

The scripts contain **no duplicated logic** — they execute the very same cells the notebook does via `notebook_cells.py`, so nothing can drift out of sync. Editing `natlas_notebook.py` updates the notebook *and* every script at once.

### The three script entry points

| Script | Runs | Use |
|---|---|---|
| `kaggleserver.py` | Cells 2–8 | Engine only, no HTTP. Smoke-tests all four languages. |
| `apiserver.py` | Cells 1–11 plus Cell 18 | Engine + FastAPI on `0.0.0.0:8000`, blocks until Ctrl+C. |
| `kaggle.py` | any range | `python kaggle.py --list`, `--cells 2-8`, `--cells 2-11`. |

```bash
python kaggle.py --list                 # cell map
python kaggleserver.py                  # load model, smoke-test English/Hausa/Yoruba/Igbo/mixed
python apiserver.py                     # serve the API (blocks)
N_ATLAS_PORT=9000 N_ATLAS_API_KEY='…' python apiserver.py
```

`kaggleserver.py` re-binds `CONFIG`, `GPU`, `DEVICE`, `MODEL`, `generate`, `generate_response`, `build_prompt_messages`, `prepare_context`, `clean_response` and the eight `ServerError` subclasses so its public surface is greppable. Its `run_smoke_test()` generates one reply per prompt for five canned prompts (English "How are you?", Hausa "Kana lafiya?", Yoruba "Báwo ni?", Igbo "Kedu ka ị mere?", Mixed "How are you? Yaya kake?") and fails any reply containing a refusal phrase ("i can't understand", "as an ai", "i don't speak", …).

`apiserver.py` imports `kaggleserver` as `_ENGINE`, executes Cells 9–11 into it to get `app` / `start_server` / `stop_server` / `serve_forever`, then executes Cell 18 to register the NAFDAC routes and fill `NAFDAC_HOOKS` **before** the server starts, so the routes exist when the first request arrives.

## Dependencies

`requirements.txt` lists everything: model runtime (`transformers>=5.0.0`, `accelerate>=1.0.0`, `safetensors>=0.4.3`, `huggingface-hub>=0.30.0`, `tokenizers>=0.21.0`, `bitsandbytes>=0.46.0`, `numpy>=1.26.0`), web server (`fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `pydantic>=2.9.0`, `requests>=2.32.0`, `nest-asyncio>=1.6.0`), and test tooling (`openai>=1.50.0`, `httpx>=0.27.0`, `nbformat>=5.10.0`). Three caveats:

- **`torch` is deliberately not listed.** It must match your CUDA build. Kaggle already ships `torch 2.10.0+cu128` for the T4; reinstalling it is the most common way to break the notebook. Install with `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **`kaggle_secrets` is not on PyPI.** Kaggle injects it into the image.
- **`bitsandbytes` is the one hard requirement Kaggle does not ship.** Without it Cell 5 cannot build the 4-bit NF4 `BitsAndBytesConfig`.

Verified on the target stack: **Python 3.12 · torch 2.10.0+cu128 · transformers 5.0.0 · Tesla T4 (sm_75)**.

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

Cell 3 only pip-installs what is genuinely missing, in two groups: **server** (`fastapi`, `uvicorn`, `pydantic`, `requests`, `nest_asyncio`) and **model runtime** (`bitsandbytes`, `accelerate`, `safetensors`, installed only when a CUDA GPU is present). It never touches torch or transformers. Set `N_ATLAS_AUTO_INSTALL=0` to make a missing package a hard error instead.

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
| `N_ATLAS_MAX_CHARS_PER_MESSAGE` | `6000` | Per-message character cap. |
| `N_ATLAS_MAX_REQUEST_BYTES` | `1048576` | Body size cap → `413`. |
| `N_ATLAS_MAX_QUEUE_SECONDS` | `120.0` | Generation-lock wait before `503`. |
| `N_ATLAS_GENERATION_TIMEOUT` | `300.0` | Per-generation wall-clock cap. |
| `N_ATLAS_LOG_CONTENT` | `0` | `1` logs message text (off by default). |
| `N_ATLAS_LANGUAGE_TAGGING` | `1` | `0` disables the deterministic language directive. |
| `N_ATLAS_STRICT_GPU` | `1` | `0` permits CPU (dev only). |
| `N_ATLAS_AUTO_INSTALL` | `1` | `0` makes a missing package a hard error. |
| `N_ATLAS_HOST` / `N_ATLAS_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `N_ATLAS_TEMPERATURE` | `0.1` | Default sampling temperature (low by design — see below). |
| `N_ATLAS_MAX_TEMPERATURE` | `2.0` | Ceiling on client temperature. |
| `N_ATLAS_TOP_P` | `0.9` | Nucleus sampling. |
| `N_ATLAS_REPETITION_PENALTY` | `1.10` | Repetition penalty. |
| `N_ATLAS_TEMPLATE_HEADROOM` | `64` | Extra tokens reserved for the chat template. |
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

**Temperature is low by default (0.1)** because most of what this server returns has to match a database verbatim — a product name, an approval date, a verdict. At 0.7 the same question returned three different answers across three runs, and a two-number comparison came back with a different drug every time. Sampling is only useful for the open-ended part of the job, so the floor is low and an operator who wants a chatty assistant should raise it deliberately.

## Measured on the Tesla T4

From a real run (T4 x2, 4-bit NF4, 14.56 GB usable per card):

| Prompt tokens | Peak VRAM | Result |
|---|---|---|
| ~1,350 | 1.6 GB steady | 200 OK |
| ~6,700 | 12.7 GB peak, wanted 4.3 GB more | **507 CUDA OOM** |

That is ~1.6 MB of VRAM **per prompt token** (large-vocab logits tensor + KV cache). Weights alone are only ~1.6 GB, so a long prompt costs several times the model's own size. Hence `N_ATLAS_MAX_INPUT_TOKENS` defaults to **4096** (~8.4 GB projected peak — real headroom on a 16 GB card), and on CUDA OOM the server empties the cache and retries once at half the budget before returning `507`.

`PYTORCH_ALLOC_CONF` is set to `expandable_segments:True` in Cell 2 before CUDA initialises. Without it the caching allocator held 1.55 GiB "reserved but unallocated" during long prompts, which is the difference between fitting in 16 GB and OOMing. It only takes effect in a fresh kernel, so re-run Cell 2 if you restart mid-notebook.

## Language handling

Two layers work together:

1. **Language tagging (server, deterministic).** Before each generation the last user message is classified into `en` / `ha` / `yo` / `ig` and an explicit directive is appended to the system prompt:
   ```
   [language] The user's last message is Yoruba.
   Write the entire reply in Yoruba. Do not use English, Hausa or Igbo.
   ```
   - Diacritics are folded first (`Báwo ni` → `bawo ni`, `ị` → `i`, `ṣ` → `s`), so missing diacritics classify identically.
   - Weighted marker scoring; distinctive words (`bawo`, `kedu`, `kana`) outweigh function words (`i`, `o`, `na`).
   - An explicit request ("answer in Igbo") always wins over the score.
   - English is the fallback when nothing is clear.
   - **Deliberate anti-English bias:** English is this model's default fallback, so when a message mixes English with a well-attested Nigerian language, the Nigerian language wins. `How are you? Yaya kake?` targets Hausa, not English.
   - Set `N_ATLAS_LANGUAGE_TAGGING=0` to turn it off.

2. **Model adherence.** The server relays whatever the model produces. Cell 14 checks the tagger and the model separately, so the report distinguishes "my code chose the wrong language" from "the model ignored a correct instruction".

**Pinned language mode:** a client can pin one language per request with a top-level `language` field (`"en"`, `"ha"`, `"yo"`, `"ig"`; `"auto"`, `null` or absent keeps detection). A pin replaces the detected directive with a HARD LANGUAGE MODE directive that holds even when the user's message is in another language:
```
[language] HARD LANGUAGE MODE: every reply must be written in Hausa.
The user's message may be written in another language; that does not change
this mode. Write the entire reply in Hausa and do not use English, Yoruba, Igbo.
```
Two properties worth relying on: **additive** (`ChatCompletionRequest` is `extra="allow"`, so an older server ignores the field rather than rejecting the request) and **reported back** (the response echoes `"language"` with what was applied).

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

### NAFDAC verification (Cell 18)

A chat message carrying a NAFDAC number is verified before generation, and the verified record is injected as a system message. The model then writes the reply in the user's language from those values, so a product name, status, applicant or approval date can only come from the data.

Cell 18 loads the shared `nafdac_source` package and reads, in order:
1. **PostgreSQL** via `NAFDAC_DATABASE_URL`. The only source reachable from Kaggle, and the one carrying superseded, withdrawn and recalled products.
2. **CSV** via `NAFDAC_CSV_PATH`, or auto-detected.
3. **XLSX** via `NAFDAC_XLSX_PATH`, or auto-detected. Needs `openpyxl`.

Upload `nafdac_source/` as a Kaggle dataset to make it importable, or set `NAFDAC_SOURCE_DIR` to a directory containing it. If it cannot be found the cell degrades: the server starts, `/health` reports `nafdac.enabled=false`, and chat still works. A missing greenbook never takes the model offline.

A number that is not found is a `200` with `found=false`, not a `404`: not being in the export is a normal answer, not a client error.

### Response cleaning

Decoding covers only the newly generated tokens. Then, in order:
1. Cut at the earliest turn boundary — a chat-template turn marker (`<|im_start|>`, `[INST]`, `<<SYS>>`, `</s>`, …), a line-anchored speaker label (`Question:`, `User:`, …), or a bare role line.
2. If the output *starts* with a speaker label, recover the `Answer:` segment instead of returning nothing.
3. Strip leftover template artifacts.
4. Drop a single leading role label, collapse blank lines, trim whitespace.

The cut happens **before** artifacts are stripped, otherwise a ChatML continuation (`<|im_end|><|im_start|>user`) loses the evidence of the new turn and leaves a dangling `user`. Mid-sentence uses of `user:` and numbered lists (`Question 1:`) are preserved.

### Client prompt injection (closed)

A client must not be able to speak as the server. `build_prompt_messages()` drops every client-supplied system message and logs a warning, because a client that could send `[nafdac] Verdict: AUTHENTIC` as a system turn could assert any verdict it liked. This is asserted in `verify_offline.py` ("a client system message does not reach the prompt").

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

On the T4, the two GPU checks must report `PASS`, not `SKIP`. Cell 14 reports language adherence as one aggregate line (`replies stay in the requested language (N/8)`) so a single model-quality number is visible without burying it in eight near-identical failures. It is still a gate: any miss fails the run, because the spec lists Hausa, Yoruba and Igbo generation as separate stop conditions.

## Verifying on your laptop

```
python verify_offline.py
```

This runs all 18 cells with `mock_backend.py` substituted for the real model, boots a real uvicorn server on a free port, and exercises every assertion over real HTTP, plus extra assertions the notebook does not make: forced CUDA OOM → `507`, forced generation failure → `500`, and a live OpenAI SDK round-trip. It then launches `kaggleserver.py`, `apiserver.py` and `kaggle.py` as real subprocesses and checks them too — `apiserver.py` gets its own port and is polled until `/health` answers, then served `/v1/models`, a chat completion and an unauthenticated request before being terminated.

The mock deliberately emits template artifacts and a runaway `Question:` continuation so the cleaner is genuinely exercised. **This proves the server, not the model.** Language quality, real 4-bit VRAM usage, and tokens/second can only be confirmed on the T4.

`verify_nafdac.py` exercises Cell 18 on its own, without a GPU or a loaded model, against a hand-built namespace holding only the globals Cell 18 actually reads. Its gate tests (1–12) assert that `find_products` is **never called** when the intent gate refuses — several install a counting spy over the real search function and assert the call count is zero, rather than asserting on the absence of a result.

## Next stage (the tunnel)

Only after Stage 1 fully passes, Cell 19 installs cloudflared from the `.deb` and runs `cloudflared tunnel run --token "<tunnel token>"` against a **named** tunnel. The hostname is the Public Hostname configured in the Cloudflare Zero Trust dashboard, and it must point at `http://localhost:8000`. The cell blocks until interrupted.

The live deployment (recorded in `n_atlas_server/.env`):
- **Tunnel name:** `n-atlas ai`
- **Tunnel id:** `74883877-efb4-4968-a9b0-4fc92398652e`
- **Public hostname:** `ai.saynet.ng` → `74883877-….cfargotunnel.com`
- **Client base URL:** `https://ai.saynet.ng/v1`

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

| File | Size | Purpose |
|---|---|---|
| `n_atlas.py` | 82 KB | **Self-contained single-file client** you can drop onto any machine on its own. Imports nothing from the package. |
| `main.py` | 3.1 KB | `python -m n_atlas_client` entry point (`--debug`, `--test`, `--version`, `--env`). |
| `__main__.py` | 124 B | Allows `python -m n_atlas_client`. |
| `__init__.py` | 1.8 KB | Package init; re-exports the public API. `__version__ = "1.0.0"`. |
| `config.py` | 8.8 KB | `Config` dataclass, `load_config`, `validate`, `apply_nafdac_env`. |
| `api.py` | 27 KB | `NAtlasClient` — health, models, chat, streaming, retries (699 lines). |
| `language.py` | 10 KB | The `/hausa /yoruba /igbo /english /auto` language modes (249 lines). |
| `terminal.py` | 9.4 KB | Terminal banner, REPL loop and local commands (274 lines). |
| `conversation.py` | 4.2 KB | Local conversation memory and client-side context trimming. |
| `storage.py` | 1.9 KB | Optional local JSON conversation saving. |
| `nafdac.py` | 12.5 KB | `NafdacInterceptor` — the local greenbook lookup (304 lines). |
| `nafdac_source.py` | 3.1 KB | Locates the shared `nafdac_source` package at the project root. |
| `prompt.py` | 7.4 KB | System-prompt resolver + `CONFAM_AI_SYSTEM_PROMPT` fallback. |
| `tests.py` | 24.6 KB | `--test` suite: connection, health, models, four languages, memory, NAFDAC. |
| `requirements.txt` | 1.0 KB | Just `openai>=1.30.0`, `python-dotenv>=1.0.0`, `requests>=2.31.0`. |
| `.env` | 5.1 KB | Client config (gitignored). |
| `README.md` | 23.4 KB | The client's own detailed README. |

## Configuration

Both entry points read the first `.env` they find, in this order: (1) the explicit `--env` path, (2) the current working directory, (3) the project root (`../.env`), (4) this folder (`./.env`). Real environment variables always win over `.env` entries.

```
N_ATLAS_BASE_URL=https://REAL-TUNNEL-URL/v1
N_ATLAS_API_KEY=YOUR_SECRET_KEY
```

- Never use `localhost:8000` — localhost on your laptop is not the Kaggle server.
- The API key is read from the environment / `.env` and is **never** printed or written into conversation files.

Full `Config` fields (all from env, with defaults): `base_url`, `api_key`, `model` (`n-atlas`), `timeout` (180.0), `max_tokens` (512), `temperature` (0.7), `stream` (True), `language` (`auto`), `max_conversation_messages` (40), `max_conversation_tokens` (3500), `max_message_chars` (6000), `max_retries` (2), `save_conversations` (True), `conversations_dir` (`conversations`), `system_prompt`, `nafdac_enabled` (True), `nafdac_database_url`, `nafdac_csv_path`, `nafdac_xlsx_path`, `nafdac_show_data` (True), `nafdac_max_search_results` (10), `nafdac_server_authoritative` (True), `debug` (False).

`apply_nafdac_env()` pushes the NAFDAC settings into the environment so the shared `nafdac_source` package (which reads its config from the environment) works identically on the laptop and on Kaggle.

## System prompt (ConfamAI)

The model answers as **ConfamAI** because both halves read one file: **`confam_prompt.txt` at the project root**.

- **Client** — `.env` sets `N_ATLAS_SYSTEM_PROMPT_FILE=confam_prompt.txt`; the resolver also finds the file beside the client or at the project root.
- **Server** — `build_kaggle_dataset.py` puts the file in the uploaded dataset as `prompt/confam_prompt.txt`, and Cell 2 finds it on disk.

Precedence on either side, strongest first:

```
N_ATLAS_SYSTEM_PROMPT_FILE > N_ATLAS_SYSTEM_PROMPT > confam_prompt.txt on disk > built-in fallback
```

`N_ATLAS_SYSTEM_PROMPT=none` (also `off`/`0`/`false`/`no`/`disable`/`disabled`) switches the system prompt off entirely, which is how the server's own prompt takes over again.

## Language modes

`/hausa`, `/yoruba`, `/igbo`, `/english`, `/auto` pin the reply language. The mode is a **hard pin, not a suggestion**: in Yoruba mode a Hausa message is answered in Yoruba, because the mode is the user's choice about the *reply*. It changes only when the user runs one of the commands, never because a message arrived in another language.

The mode reaches the server two ways, and both are always sent: (1) `language` in the request body, which a patched server treats as authoritative and skips its own detector for that request (an unpatched server ignores it because `ChatCompletionRequest` is `extra="allow"`); (2) one directive line on the outgoing turn, which an unpatched server still reads. There is deliberately **no language detector** in the client — the server has a good one, and a second copy would be a second thing to keep in step with it.

Synonyms resolve to the same mode in four languages, including local names: `harshe`/`sauraya` → Hausa, `yebu`/`ede yoruba` → Yoruba, `ahukeri` → Igbo.

## Local commands (never sent to the server)

`/help`, `/new`, `/clear`, `/history`, `/status`, `/hausa`, `/yoruba`, `/igbo`, `/english`, `/auto`, `/lang` (show current mode), `/verify <nrn>`, `/search <terms>`, `/nafdac`, `/exit`.

## Conversation memory

The conversation history lives on the laptop. `Conversation.build_messages_for()` puts the ConfamAI system prompt first, then the local history, then the new message. History is trimmed to `MAX_CONVERSATION_MESSAGES` and to an estimated `MAX_CONVERSATION_TOKENS`, preserving the opening exchange and the most recent turns. `estimate_tokens()` is a cheap heuristic (no tokenizer on the laptop): `1 + len(word)/4` per word.

## NAFDAC verification (local)

`NafdacInterceptor` intercepts a message before it reaches N-ATLaS. When the message carries a NAFDAC number, the number is verified against the greenbook **here, on the laptop**, and the verified facts are handed to the model as context, prepended with:

```
[nafdac] Verified greenbook data for this question. Report these values …
```

Two paths: `/verify <nrn>` (a local command, answered without touching the server) and ordinary chat (an intercepted message, answered with the server).

**`nafdac_server_authoritative` (default True):** when a server is reachable, the client never searches locally on its own initiative, because a second retrieval authority would decide for itself that a message was about a product — the exact failure the server's gate exists to prevent. Local data is still used as a fallback when no server is reachable, and on explicit commands like `/verify`, which are the user asking directly.

## Run

```bash
python -m n_atlas_client                    # chat
python -m n_atlas_client --debug            # chat with developer output
python -m n_atlas_client --test             # test suite
python -m n_atlas_client --version
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
    csv        nafdac_greenbook.csv           (local, 8935 rows)
    xlsx       nafdac_greenbook.xlsx          (local, same rows)
```

## Files

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 3.5 KB | Re-exports the whole public API. `__version__ = "1.0.0"`. |
| `sources.py` | 24.3 KB | The three sources (postgres / csv / xlsx) behind one interface (662 lines). |
| `verify.py` | 37.3 KB | `verify_nafdac_product()` / `find_products()` / `search_products()` (927 lines). |
| `records.py` | 16.6 KB | `ProductRecord`, verdicts, ranking, rendering (466 lines). |
| `extract.py` | 17.5 KB | `extract_nrn()` / `extract_search_terms()` / NRN normalisation (492 lines). |
| `formatting.py` | 36.5 KB | Terminal + prompt formatting, localised in 4 languages (710 lines). |
| `load_postgres.py` | 19.1 KB | Loads the greenbook into PostgreSQL. |
| `verify_postgres.py` | 11.6 KB | Postgres-specific verification. |
| `schema.sql` | 5.8 KB | Database schema (3 tables). |
| `tests.py` | 42.2 KB | **232 offline checks**, no network, no database. |
| `requirements.txt` | 1.4 KB | Optional: `python-dotenv`, `psycopg[binary]`, `openpyxl`. |
| `.env` | 4.8 KB | Config (gitignored). |
| `README.md` | 14.1 KB | The source's own detailed README. |

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

## The four verdicts

`authorised` is the accurate technical answer, but it answers the wrong question: a user asking "is this genuine?" does not want True/False, they want to know whether they can trust the pack. These four values say that directly:

| Verdict | Meaning |
|---|---|
| `authentic` | found, currently Active, and not marked as a duplicate |
| `flagged` | found, but NAFDAC has withdrawn/suspended/recalled it, or the greenbook marks the entry itself as a duplicate |
| `not_registered` | no source has it at all |
| `unknown` | found, but the status is not one this code recognises, or the number is genuinely ambiguous |

There is deliberately **no "fake" verdict**. The greenbook records registrations, not counterfeiting: a miss means this export does not list it, which is not the same claim.

`_ACTION_STATUSES` (the statuses that mean NAFDAC has taken the product off the register, as opposed to it simply having lapsed): `suspended`, `withdrawn`, `recalled`, `cancelled`, `canceled`, `revoked`, `expired`, `superseded`. Both "flagged" for the user, but the reason is worth naming.

## Confidence levels

| Level | Meaning |
|---|---|
| `verified` | PostgreSQL and at least one file agree |
| `single-source` | only one source had it |
| `local-file` | files only, no database configured |
| `superseded` | only in the history table |
| `none` | not found anywhere |

## NRN is not unique

`Product ID` is the greenbook's own primary key and is always unique. **NRN is *not* unique: 257 of the 8935 rows share an NRN with another row** (the scraper manifest records 256 collisions), so a lookup can legitimately return several records. Callers must handle `len(matches) > 1` rather than assuming one row. Several candidates means the data has not answered the question about the pack in the user's hand, so **no verdict is claimed** and the block tells the model to ask which product they mean. It never picks one.

## Verifying by name, not just by number

A user rarely has the number to hand. `find_products()` searches product name, active ingredients and applicant, ranked: an exact NRN or product-ID hit first, then a name prefix, then a substring, then a match in the ingredients or the applicant.

Terms are matched independently rather than as one joined string, because the match is a substring test and no product name contains the exact phrase "paracetamol syrup". `extract_search_terms()` produces them: an NRN in the text wins outright, otherwise stop words are dropped and the longest remaining terms are kept.

## Every message is searched

There is no test for "is this about a product". Every message goes through the greenbook, because a hit is cheap and a miss costs nothing:

```
    hit   -> the block is attached, and the model decides whether it is relevant
    miss  -> nothing is attached, and the model answers conversationally
```

The alternative — deciding up front whether a message is about a product — does not survive contact with the data. A greenbook name is ordinary English, so `good morning` matches "Good Morning Cough Syrup", `I love my family` matches 16 products containing *Family*, `what is the weather` matches 4 products via applicant names, and `I need water` matches "Water Guard Plus Powder".

## NRN extraction

The greenbook is messy: NRNs look like `A4-100160`, `B4-6238`, `04-3877`, `C4-1142`, sometimes with stray spaces (`A4 - 5180`), sometimes with an en dash (`04-2045`), a handful written as a fraction (`4/1/9086`) or carrying a suffix letter (`A1-4924L`). Two things are kept separate:

- `normalize_nrn()` — the display form, kept close to what NAFDAC published.
- `normalize_nrn_key()` — the lookup key, so `a4 - 5180` finds `A4-5180`.

`normalize_nrn_key` is what the sources index on, so a lookup only succeeds if the digits agree. Everything else is cosmetic. `EXTRACTION_PATTERN` handles the common two-numeric-group form with a trailing lookahead that stops `B4-9999-0000` being silently truncated to `B4-9999`; `FRACTION_PATTERN` covers the three-group year/month/serial rows; `_BARE_DIGITS` is a last resort for pure digits like `100160`.

## Localisation

`format_*` is terminal output, so it takes a `lang` and renders its labels and prose in that language (`en`, `ha`, `yo`, `ig`). Two rules keep that honest: (1) **values are never translated** — a product name, an applicant, a strength and an approval date are printed exactly as the greenbook holds them, only labels/verdicts/sentences change; (2) `VerificationResult.render()` and `render_record()` are untouched and always English, because that text is fed to the model as verified data. `_t` falls back to English one key at a time, so a missing translation degrades to a readable English label rather than printing a key name at someone checking whether a medicine is genuine.

## PostgreSQL schema

Two tables, because a verification answer has to distinguish "this product is not registered" from "this product used to be registered and has since been superseded or withdrawn":

- **`nafdac_products`** — the current greenbook, one row per Product ID (34 columns). `nrn_key` is the lookup key and is **not** unique; the uniqueness constraint is on `product_id`.
- **`nafdac_product_history`** — superseded, withdrawn, suspended or recalled rows, with `superseded_at`, `superseded_reason`, `superseded_by_product_id`. One history row per (product, supersession event), so re-running the loader for the same event updates it instead of piling up duplicates.
- **`nafdac_load_runs`** — a record of what was loaded and when, so an operator can tell whether a stale answer came from a stale database.

Indexes on `nrn_key`, `source_url`, and `lower()` of product name / ingredients / applicant (the last needs `pg_trgm`; without it the index is simply skipped).

## The Supabase connection

`NAFDAC_DATABASE_URL` must be the **session** pooler (port **5432**), not the transaction pooler (port 6543) and not the direct connection:

- **Direct** (`db.<ref>.supabase.co`) publishes only an AAAA record — no IPv4 address unless the paid IPv4 add-on is enabled, so it is unreachable from a Kaggle kernel.
- **Transaction mode** (6543) is PgBouncer in transaction mode, which does not manage server-side prepared statements. psycopg3 prepares after the fifth execution of identical SQL, and `nafdac_source` reissues the same lookup query on every message. Measured on this project: session mode prepared cleanly, while transaction mode raised `DuplicatePreparedStatement: prepared statement "_pg3_0" already exists`.
- `nafdac_source/sources.py` keeps one connection open for the life of the process, so session affinity is what the workload wants.

The password contains `@` and `&`, so it **must stay percent-encoded** (`%40`, `%26`): the driver is handed the whole string as a DSN, and an unencoded `@` would truncate the netloc and produce an unresolvable hostname.

---

# 4. `nafdac_scraper/` — Greenbook scraper

Extracts every field the NAFDAC Greenbook site publishes into a structured dataset, exported as CSV plus a multi-sheet Excel workbook.

Last verified against the live site: **8,935 products, 34 columns, plus 5 reference sheets** (1,427 manufacturers, 1,661 applicants, 2,393 ingredients, 6 product categories, 902 ATC codes). All ten originally requested columns are 100% populated except NRN, which is blank on 3 products because the site has no NRN for them.

## Files

| File | Size | Purpose |
|---|---|---|
| `scraper.py` | 18.7 KB | The scraper itself — three phases, export (476 lines). |
| `client.py` | 17.3 KB | HTTP layer: bulk JSON, detail pages, reference indexes (464 lines). |
| `mapping.py` | 10.1 KB | JSON row → dataset columns, plus detail-page merging (276 lines). |
| `config.py` | 5.1 KB | Every tunable value: URLs, headers, retries, pagination, columns (146 lines). |
| `reference.py` | 3.1 KB | Reference-index page scraping. |
| `nafdac_greenbook.csv` | 6.9 MB | The exported dataset (feeds `nafdac_source/`). |
| `nafdac_greenbook.xlsx` | 3.1 MB | Multi-sheet workbook (Products + 5 reference sheets). |
| `nafdac_greenbook.xlsx.nafdac_rows.csv` | 6.9 MB | Parsed index of the xlsx (gitignored, rebuilt automatically). |
| `nafdac_greenbook.xlsx.nafdac_rows.csv.stamp` | 108 B | Cache stamp (gitignored). |
| `detail_cache.jsonl` | 6.9 MB | Cached detail pages (resume support). |
| `detail_dead_ids.json` | 168 B | Product ids whose detail page returns HTTP 500 unconditionally. |
| `nafdac_scraper.log` | 96.6 KB | Run log. |
| `scrape_manifest.json` | 889 B | Run metadata (see below). |
| `requirements.txt` | 44 B | `requests>=2.31,<3`, `pandas>=2.0`, `openpyxl>=3.1`. |
| `README.md` | 15.2 KB | The scraper's own detailed README. |

## How the site works

The Greenbook is a Laravel application. There is **no public REST API** — the data lives behind jQuery DataTables (`serverSide: true`) and plain server-rendered HTML pages. Three access patterns are needed:

1. **Bulk product JSON** — `GET https://greenbook.nafdac.gov.ng/` with header `X-Requested-With: XMLHttpRequest` (the only real trap: without it the server returns the HTML home page instead of JSON). A `csrf-token` meta tag exists but is not needed for GETs. Pagination uses the DataTables protocol (`draw`, `start`, `length`); the server honours large `length` values but the scraper pages at 500 so an interrupted run costs one page, not the whole crawl. The query is filtered server-side to `status IN ('Active','Inactive')` and `deleted_at IS NULL`.
2. **Product detail pages** — `https://greenbook.nafdac.gov.ng/products/details/{product_id}`, server-rendered HTML, no JS needed. Each field is an `<h1>` label followed by a `<p>` value. Three fields exist **only** here: **Manufacturer Name**, **Manufacturer Country**, **Marketing Category**.
3. **Reference index pages** — server-rendered, paginated with `?page=N`, 50 cards per page (`/manufacturers` 29 pages, `/applicants` 34 pages, `/ingredients` 48 pages, …).

## Reliability tuning (from `config.py`)

- `REQUEST_TIMEOUT = 60` s, `MAX_RETRIES = 5`, `BACKOFF_FACTOR = 2.0`, `BACKOFF_CAP = 60.0`, `REQUEST_DELAY = 1.0` s between bulk pages.
- `RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}`.
- **Detail crawl** is the expensive phase (~8.9k requests). Detail pages are far more fragile than the JSON endpoint: at 4 workers the site starts returning HTTP 500 and dropping connections. **2 workers with a 0.4 s pause** sustains a stable ~4 requests/second and never trips the errors.
- `DEAD_FILE` remembers product ids whose detail page returns HTTP 500 unconditionally — a server-side bug, not throttling. Retrying them costs minutes and never succeeds, so they are skipped on later runs. Delete the file to force a fresh attempt if NAFDAC fixes the bug.

## `scrape_manifest.json` (last run, 2026-10-01 14:17:03)

```json
{
  "raw_rows": 8935, "detail_products": 8909, "detail_merged": 8909,
  "manufacturer_name": 25, "marketing_category": 24, "unresolved": 1,
  "total_records": 8935, "elapsed_seconds": 254.0,
  "details_included": true, "indexes_included": true, "nrn_collisions": 256,
  "reference_counts": {
    "Manufacturers": 1427, "Applicants": 1661, "Ingredients": 2393,
    "Product Categories": 6, "ATC Classification": 902
  },
  "detail_products_unreachable": ["11500","11881","12242","12379","179","4554", …]
}
```

26 product ids were unreachable on the last run (HTTP 500 server-side bug). `manufacturer_name: 25` and `marketing_category: 24` are the counts that came from the detail pages; `unresolved: 1` is a product whose detail page could not be matched.

## Output columns

The ten originally requested columns are kept first and in this exact order so the export stays a drop-in replacement for the earlier CSV: `Product Name`, `Active Ingredients`, `Product Category`, `NRN`, `Form`, `ROA`, `Strengths`, `Applicant Name`, `Approval Date`, `Status`. Then 24 more: `Product ID`, `Product Category ID`, `ATC Code`, `Ingredient ID`, `Ingredient Synonym`, `Form ID`, `Route ID`, `Product Description`, `Composition`, `Pack Size`, `Applicant ID`, `Applicant Address`, `Manufacturer ID`, `Manufacturer Name`, `Manufacturer Country`, `Country ID`, `Marketing Category ID`, `Marketing Category`, `Expiry Date`, `SMPC URL`, `Biosimilar`, `Created At`, `Updated At`, `Source URL`.

`STRIP_PRODUCT_NAME_MARKERS = True` because the site's own JavaScript strips `#` and `*` markers from product names before display; keep True to match what NAFDAC shows, set False for raw API values.

## Run

```bash
python scraper.py --limit 20      # quick smoke test
python scraper.py                 # full crawl, all sources
python scraper.py --no-details    # skip the ~8.9k detail requests
```

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

| File | Size | Role |
|---|---|---|
| `train_notebook.py` | 32.8 KB | **Source of truth.** Plain sequential Python with `# %%` cell markers (774 lines). |
| `confam_train_kaggle.ipynb` | 42.5 KB | **Build artefact** — upload this to Kaggle. Never hand-edit it. |
| `build_train_notebook.py` | 3.9 KB | `train_notebook.py` → `confam_train_kaggle.ipynb`. |
| `validate_train_notebook.py` | 3.9 KB | Compiles every cell, checks nbformat line encoding, proves the notebook reassembles exactly. |
| `confam_training_dataset.jsonl` | 2.1 MB | The **643-record** training set Notebook A consumes. |
| `prepare_training_data.py` | 51.4 KB | Rebuilds that dataset from a raw source, or validates it in place (1204 lines). |
| `check_transcript_coverage.py` | 3.2 KB | Compares real saved conversations against what the dataset covers. |
| `qlora_finetuning_guide.md` | 9.1 KB | The original cell-by-cell walkthrough (reference; **superseded** by the README). |
| `time.md` | 11.8 KB | The training-time estimate: arithmetic, research anchors, Kaggle limits, and the measurement that replaced the prediction. |
| `README.md` | 13.9 KB | The training folder's own detailed README. |

Mirrors `n_atlas_server/` on purpose: one editable source, one generated upload, one validator. It reuses the server's `notebook_cells.py` rather than copying the cell-splitting logic, because a drifting second copy of that logic is how "it validates locally but breaks on Kaggle" happens.

## Running it

**Before the first run**

1. Upload `confam_training_dataset.jsonl` to Kaggle as a dataset named `confam-training`.
2. Create a Kaggle secret named `HF_TOKEN` with a Hugging Face token that has **write** permission (Cell 8 pushes the adapter). The account on that token is the first half of `ADAPTER_ID`.
3. Accelerator: **GPU T4 x2**. The second T4 is deliberately left idle (see "Why one GPU" below).
4. **Accept N-ATLaS's licence on Hugging Face first** — `NCAIR1/N-ATLaS` is gated: `from_pretrained` fails with a 401 until your account has agreed to share contact information.
5. Budget **~1.5 hours** for the default 200 steps on one T4 — measured, see below.

**Then**

```bash
python build_train_notebook.py
python validate_train_notebook.py   # must print VALID
# upload confam_train_kaggle.ipynb to Kaggle, attach confam-training + HF_TOKEN, run all cells
```

## The nine cells

| Cell | Job | Fails loudly if |
|---|---|---|
| 1 | config, finds the dataset in `/kaggle/input/**`, **pins the process to ONE GPU** | the JSONL is not attached |
| 2 | installs libraries, `trl` pinned to 0.12.2 | no GPU, or more than one is visible |
| 3 | Hugging Face login | `HF_TOKEN` is missing |
| 4 | **dataset integrity checks** | personas are mixed, a verdict has no block, or a record claims counterfeit / safety |
| 5 | base model in 4-bit NF4 + LoRA | the base model exposes no chat template |
| 6 | renders records with the model's own template, **measures token lengths** | any record exceeds `MAX_SEQ_LENGTH` |
| 7 | trains 200 steps | — |
| 8 | pushes the adapter to the Hub | the pushed repo is implausibly large |
| 9 | three behavioural checks | prints PASS/CHECK; `identity` not saying ConfamAI means the adapter did not take |

Cell 4 runs **before** the 5-minute model load, so a bad upload fails in seconds. Cell 6's length measurement matters because all 643 records repeat the same ~850-token system prompt, so a too-small `max_seq_length` silently truncates the assistant turn — the only part of a record that teaches anything.

### Why one GPU (Cell 1)

It is not about fitting the model: a 4-bit 8B plus LoRA ranks fits one T4 with room to spare. It is about the **trainer**. With more than one GPU visible in a plain-notebook launch, transformers wraps the model in `torch.nn.DataParallel` (whenever `args.n_gpu > 1`), and `Trainer.compute_loss` adds `num_items_in_batch` — a 0-dim tensor — to the batch before calling the model. `DataParallel.scatter` cannot chunk a 0-dim tensor across replicas, so the FIRST training step dies with `RuntimeError: chunk expects at least a 1-dimensional tensor` (huggingface/transformers#37151, huggingface/trl#2338). `device_map` in Cell 5 does not save this run: it puts the weights on cuda:0, but it is `args.n_gpu` — a count of VISIBLE devices, not of devices in use — that decides whether the model gets wrapped at all. The cost is the second T4 sitting idle.

### Why `trl` is pinned to 0.12.2

The `SFTTrainer` call in Cell 5 uses the 0.12 argument names (`max_seq_length=`, `tokenizer=`). TRL renamed them to `max_length=` and `processing_class=` in later releases, and the old names are then forwarded to `TrainingArguments`, which rejects them with an unhelpful error about an unknown argument. Pinning keeps this notebook working.

### A T4 has no bfloat16

Cell 5 pins the compute dtype to **float16**, because a Tesla T4 does not support bfloat16.

## Training time (measured, from `time.md`)

**Answer: ~1.5 hours on Kaggle GPU T4 x2**, of which ~1.3 h is training and 10–20 minutes is setup. Only one of the two T4s is used.

Configuration: `max_steps=200`, effective batch 8 (per-device 1 × accumulation 8), `MAX_SEQ_LENGTH=2048`, LoRA r=16 on `q_proj, k_proj, v_proj, o_proj`, 643 records, stock HF Trainer + bitsandbytes + PEFT (no Unsloth).

From the first run, 5 October 2026: **12 optimizer steps in 3 m 54 s**, i.e. **19.5 s/step** at effective batch 8, which puts the full 200 steps at **1.3 h**.

| Quantity | Measured | How |
|---|---|---|
| Seconds per optimizer step | **19.5** | 234 s ÷ 12 steps, from the progress bar |
| Training time, 200 steps | **1.3 h** | 200 × 19.5 s |
| Throughput, counting padding | **~705 tok/s** | 1600 × 2048 padded positions ÷ 4650 s |
| Throughput, real tokens | **~430 tok/s** | the ~2.0M of the prediction ÷ 4650 s |
| VRAM after the fix | **7.3 of 14.6 GiB** | Cell 7's own line, batch 1 |
| Training loss at step 10 | **2.27** | healthy for LoRA SFT on a base model |

**The prediction was 2.2× pessimistic** (it predicted 3.2 h), and the bandwidth argument in the prediction is what failed: the 695 tok/s RTX 3090 figure is for full-finetune-shaped work — all linear layers adapted, dense 4-bit dequantisation on every matmul — whereas this run adapts only four attention projections.

## The training dataset

`confam_training_dataset.jsonl` holds **643 records**, rebuilt from an original 866 by `prepare_training_data.py`. What was wrong with the file as shipped, all found by reading it:

1. **73 records asserted a verdict with no verified data attached**, three of them "CRITICAL ALERT: This product has been FLAGGED as a known counterfeit". Training on those teaches the model to invent verdicts and to call products counterfeit — the one behaviour this project exists to prevent.
2. Every record carried **invented registry facts** ("Orange Drugs Ampiclox | A4-2045 | Orange Drugs Ltd"). Fine-tuning on them turns the model into a second, wrong database.
3. The answers claimed **"It is safe for use"** and **"NAFDAC approved"**. The greenbook records registrations; it never says a pack is safe.
4. The verdict words were not the project's: the headings in use were `TABBATACCE`, `VERIFIED`, `KO SI NIPA AKOSILE NAFDAC`, `BABU A RIJISTA` and `NO RECORDS FOUND` instead of `AUTHENTIC` / `FLAGGED` / `NOT REGISTERED`.
5. Every system message was a different persona: "Confam AI … medicines and consumer products … NAFDAC approved". The served prompt is ConfamAI, consumer products only, and forbids claiming approval.
6. Igbo was 13 of 866 records against 703 English.

What `prepare_training_data.py` does: the system message becomes `confam_prompt.txt` verbatim in every record; a verified block becomes its own system message in the shape the server actually sends; the values inside that block are **synthetic and unique per record**, so the lesson learned is "copy the value out of the block", not "this product is this number" — no real registry value survives into the output; the answer is regenerated from templates so every verdict word is canonical; the 73 verdict-without-data records become **negatives** (no block, so no verdict — the assistant asks for the number); negatives are added for all four languages and the Igbo/Hausa/Yoruba deficit is made up with generated records; multi-turn records are split into one record per turn, because each API call in production is a single prompt with one block.

**What fine-tuning still cannot fix:** the loaded dataset has **no consumer products** in it, so the assistant will still find nothing for food, drinks or cosmetics. That is a data problem, not a weights problem.

`check_transcript_coverage.py` reads the saved conversations from the client (gitignored, so genuinely held-out), classifies every user turn with the same classifier the dataset builder uses, and reports the shape of real traffic against what the cleaned dataset covers.

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

## Files

| File | Size | Purpose |
|---|---|---|
| `__main__.py` | 10.7 KB | Entry point: `--status`, `--test`, `--ask`, `--assay`, `--encode-dsn`, `--version`, serve (301 lines). |
| `__init__.py` | 1.7 KB | Package init. `__version__ = "1.0.0"`. |
| `config.py` | 14.5 KB | `Config` dataclass, `load_config`, `validate`, `describe` (337 lines). |
| `webhook.py` | 9.9 KB | The FastAPI app: one GET, one POST, one health endpoint (246 lines). |
| `router.py` | 16.0 KB | The orchestration: one inbound message in, one reply out (368 lines). |
| `sessions.py` | 18.9 KB | Per-phone conversation history, sqlite + supabase backends (492 lines). |
| `whatsapp.py` | 11.6 KB | The WhatsApp Cloud API sender (302 lines). |
| `render.py` | 10.5 KB | Turning a model reply into text WhatsApp renders cleanly (278 lines). |
| `natlas.py` | 12.9 KB | The N-ATLaS server client. One method, one endpoint (351 lines). |
| `queue.py` | 5.4 KB | Two levels of concurrency control (130 lines). |
| `framing.py` | 5.8 KB | Rephrasing aid for messages that carry a NAFDAC number but no question (130 lines). |
| `languages.py` | 11.3 KB | Language modes and the opt-out keyword, handled without the model (268 lines). |
| `assay.py` | 11.1 KB | A live battery of real questions against the N-ATLaS server (308 lines). |
| `tests.py` | 50.8 KB | The offline test suite. |
| `requirements.txt` | 1.4 KB | `fastapi`, `uvicorn[standard]`, `httpx`, `python-dotenv`, `psycopg[binary]`. |
| `.env` | 3.1 KB | Config (gitignored). |
| `.env.example` | 7.3 KB | Example config with every variable documented. |
| `README.md` | 13.3 KB | The bot's own detailed README. |

## What it owns — and what it deliberately does not

**Owns:** transport (the webhook and the sender), sessions (per-phone history), formatting (WhatsApp-safe text), and honesty about failure.

**Does not own, because the server already does and owns it better:** the NAFDAC greenbook lookup, the intent gate, the verdict, the system prompt. So this package queries no greenbook and holds no product data. It imports exactly one thing from `nafdac_source` — `extract_nrn`, a pure text function that reads no database and no file — for the framing workaround in `framing.py`. Nothing else.

That restraint is the design, not laziness. A second retrieval authority on this side would run *before* the server has seen the message, and the concrete failure is recorded in `n_atlas_client/api.py:499-510`: a greeting that matched a product locally talked the server into searching as well.

The cost is explicit: **when the N-ATLaS server is down, this bot cannot verify anything and says so.** There is no local greenbook to fall back on, by design — the alternative would be answering from the model's memory, which is exactly how a cough syrup ends up described with ingredients nobody put in the database.

## The webhook (three routes and nothing else)

| Route | Purpose |
|---|---|
| `GET {webhook_path}` | Meta's subscription handshake. |
| `POST {webhook_path}` | Inbound messages. |
| `GET /healthz` | For the operator, never for Meta. |

Two things are load-bearing:

1. **The POST returns 200 immediately.** Meta requires a 200 within twenty seconds or it treats the webhook as failed and retries, which produces duplicate replies and duplicate billed messages. A T4 turn takes about thirteen seconds on its own, so doing the work inline would guarantee a retry on nearly every message. The request path therefore validates, deduplicates, and hands off to a background task. Meta gets its 200 in milliseconds.
2. **The signature is checked before the body is parsed.** `X-Hub-Signature-256` is an HMAC-SHA256 of the raw request body with the app secret. Checking it after parsing, or skipping it, means anyone who learns the webhook URL can inject messages that look like they came from a customer. `hmac.compare_digest` is used rather than `==` so a wrong signature cannot be discovered one character at a time.

## The router (seven steps, deliberately shallow)

1. is this number allowed at all? → refuse locally
2. has it opted out? → refuse locally
3. is it START after an opt-out? → clear locally
4. is it STOP? → delete the session locally
5. is it `/hausa`, `/new`, `/lang`? → answer locally, no model
6. is it a photo or a document? → answer locally, no model
7. otherwise → hand it to the server

Steps 1 to 6 never call the model. That is not only cheaper, it is better: a `/hausa` answered by the model would take about thirteen seconds on a T4 to say something this package can say instantly, and would consume a slot that a real verification question is waiting for.

Step 7 is the whole product, and it is one HTTP request. The message goes to the server exactly as the user typed it. No NRN extraction, no product-name search, no pre-attached greenbook block, no verdict text on this side. The server's `_nafdac_enrich` decides whether the question is about a product at all, and `_intent_permits_retrieval` decides whether to query the greenbook.

## Sessions

The N-ATLaS server is stateless per request: it does not remember the previous turn, because the client sends the history every time. For a laptop that is convenient. For a WhatsApp bot it is not — there is no terminal holding a conversation object — so history has to live somewhere durable here.

Stored per phone (normalised E.164, the primary key): `history` (a JSON list of `{role, content}`, user and assistant only), `language` (the pinned reply language, `''` for auto), `updated_at` (unix seconds, for TTL). **Never stored:** the N-ATLaS API key, the Meta access token, the verify token, anything from `.env`. `to_dict()` on a row returns messages only, so there is no path by which a secret reaches this table.

Two backends, one interface: **sqlite** (stdlib only, one local file — fine for one process on one host) and **supabase** (the Postgres you already run for `nafdac_source` — makes the bot fully stateless, so it can be redeployed, scaled out, or moved between hosts without losing anyone's conversation). Pick `supabase` when there is any chance of running more than one instance or restarts on ephemeral disk; pick `sqlite` for local development.

The same SQLite file holds a second table, `wa_seen`, which records inbound message ids. Meta retries a webhook that does not answer 200 within twenty seconds, so the same user message can arrive two or three times; without this table a retry produces duplicate replies and duplicate billed messages. `DEDUPE_TTL = 3600` s.

## Concurrency (two levels, both necessary)

The N-ATLaS server handles requests in a threadpool and generates on one shared 4-bit GPU, so concurrency there is a queue, not a pool. `chat_completions` is a sync `def`, which means FastAPI will happily run dozens of them at once, and uvicorn is configured with `limit_concurrency=32`. Thirty-two simultaneous generations on one Tesla T4 is not thirty-two answers arriving together: it is a wall of 507s, 429s, and very slow replies.

- **A per-user lock.** One person sending three messages in quick succession must get three answers in order. Without a lock the turns race, and the history is written in completion order rather than conversation order — so the model is shown a conversation that never happened.
- **A global semaphore** (`WHATSAPP_MAX_CONCURRENCY`, default **3**). This keeps the bot inside what one GPU can serve. Three is the default because a 4-bit model on a T4 takes about thirteen seconds per turn; three in flight keeps the pipeline full without piling up. Raising it past ~6 mostly buys 507s.

Waiting on either is bounded by `per_user_timeout` (default 120 s). Without that bound a stuck request holds a semaphore slot forever and the bot degrades to zero throughput with no error and no log line — the worst possible failure mode, because nothing looks broken.

## Rendering (three jobs, in order of damage when wrong)

1. **Length.** WhatsApp rejects a text body over **4096 characters**. The chunker exists so that never happens. `_SOFT_LIMIT = 4000` keeps a reply one character over the edge to a single message on the first pass.
2. **Cost.** Since Meta's 1 October 2026 rate change, a service message inside the customer's 24-hour window is charged per message delivered — about **$0.0101 (₦14)** to a Nigerian number. So every chunk is a separate bill. `chunk()` works hard to return exactly one message for a normal reply and treats splitting as the failure case, not the default.
3. **Legibility.** WhatsApp gives no formatting language of its own; it interprets WhatsApp markup (`*bold*`, `_italic_`, code fences). The model is instructed to answer in plain text, so anything that looks like a code fence is an accident and is flattened rather than left to render as a grey box.

Nothing rewrites, translates or summarises the reply. The model's wording is the product: the safety rules about never calling a product counterfeit live in the server's NAFDAC instruction block, and every transformation is proved in `tests.py` not to touch a product name, a number, a date or a verdict.

## The WhatsApp sender (Meta error codes)

`POST https://graph.facebook.com/{version}/{phone_number_id}/messages` with `Authorization: Bearer {system user token}`. Error codes worth naming, because they need different handling and lumping them together is how a bot silently stops answering people:

| Code | Meaning | Handling |
|---|---|---|
| `131047` | rate limit hit | back off hard (4 s floor), retry |
| `131056` | pair rate limit hit | back off harder (15 s floor), retry |
| `130429` | rate limit hit (older) | back off hard (4 s floor), retry |
| `130472` | quality or blocklisting | stop and investigate (30 s floor) |
| `131026` | the token cannot send to this number | do **NOT** retry, drop |
| `100` | invalid parameter / bad phone number | do **NOT** retry, drop |
| `131009` | parameter value is not valid | do **NOT** retry, drop |

The retry rule is: retry 5xx and the rate limits, never retry anything else. A retried `131026` is a token that will never work, and a bot that retries it forever is a bot that hides its own misconfiguration. Access tokens are never logged, never echoed in an exception, and never sent anywhere but `graph.facebook.com`.

## Language modes and the opt-out

A WhatsApp user has no slash-command help text in front of them, so the commands have to work by guessable spelling in four languages. The table is a table and not four separate checks: someone who types `/harshe` and gets a reply in Hausa immediately has no way to know `/hausa` was the canonical word, and no reason to care.

The mode reaches the server two ways, both from `natlas.py`: (1) the `language` field of the request, which the server treats as authoritative and skips its own detector for this turn; (2) the server's own directive, which the model reads. `/auto` sends neither and hands the choice back to the server's detector. These commands are answered locally, without a model call.

**The opt-out keyword is not optional: NDPA 2023 (Nigeria's Data Protection Act) requires an easy way to stop.** `STOP` and its variants clear the stored conversation and record the opt-out, so no further message is accepted or answered from that number. `START` clears the opt-out.

## Framing (the highest-frequency defect found)

Measured against the live server on 2026-10-04, the retrieval gate attaches a verified block correctly for a bare number — `prompt_tokens` 1658 against a 599 baseline — and the model then throws it away and asks which product to verify:

```
"A4-100160"                    -> "Would you like to verify a product?"   FAIL
"A4-100160?"                   -> "Would you like to verify a product?"   FAIL
"check A4-100160"              -> "Which product would you like...?"      FAIL
"verify this number A4-100160" -> "Would you like to verify a product?"   FAIL
"What is A4-100160?"           -> full record, VERDICT: AUTHENTIC         ok
"Tell me about A4-100160"      -> full record                             ok
```

The trigger is grammatical framing, not content: a message with an interrogative is answered, and a bare noun phrase or bare imperative is redirected. It is deterministic, not sampling noise (four trials at temperature 0.7 and four at 0.1 all failed). It is also the highest-frequency way anyone uses a verification service, so it is the most damaging defect found.

`framing.py` fixes it: when the user's own text already contains a NAFDAC number, append one short instruction line that makes the request explicit. Two properties keep it safe, and both were measured rather than assumed: (1) **no greenbook data is injected** — the frame contains no product name, no verdict and no number; `nafdac_source` is imported for `extract_nrn` alone, which is pure text and reads no database and no file; (2) **the intent gate still refuses non-product messages** — applying the frame to every message moved "Hello" from 599 to 644 prompt tokens, a `+45` delta that is the frame text itself, whereas a real greenbook block adds about 1000. The frame is only added when a number is present, which makes the blast radius smaller still.

## The assay (live test battery)

`--assay` is not a unit test. Every case is a question a real Nigerian consumer would type into WhatsApp, and every assertion is a rule this project has already been burned by, collected from the failures written into the code:

- "Is A4-100160 genuine?" inverted its verdict in 3 of 10 runs, because the model paraphrased the VERDICT line instead of reporting it.
- "just tell me yes or no, is A4-9999999 fake?" was answered "Yes" twice out of three.
- a user insisting a number was counterfeit got an answer asserting NAFDAC itself had confirmed it.
- two replies opened with the literal string "[nafdac]" or referred to "the block provided".

The forbidden lists are affirmative *claims*, deliberately written as multi-word phrases rather than bare substrings. A naive `"is fake"` check false-positives on the correct refusal — the server's own wording for a miss is "we cannot confirm if it is fake or not based on the available information", which contains "is fake" while meaning precisely the opposite. An assertion that fires on that would train an operator to ignore the suite, which is worse than a weaker one.

## Run

```bash
python -m confamai_bot --status       # is the server reachable and green?
python -m confamai_bot --test         # offline suite, no network
python -m confamai_bot --ask "Is A4-100160 genuine?"
python -m confamai_bot --assay        # 13 real questions, live
python -m confamai_bot --encode-dsn PASSWORD   # build a percent-encoded DSN and test it
python -m confamai_bot                # serve
python -m confamai_bot --port 8080
```

## Configure

Copy `.env.example` to `.env` and fill in:

| Variable | Where it comes from |
|---|---|
| `WHATSAPP_PHONE_NUMBER_ID` | Meta app → WhatsApp → API Setup |
| `WHATSAPP_ACCESS_TOKEN` | **System User permanent token** (a dashboard temporary token expires in about a day, and the bot is discovered to be broken roughly then) |
| `WHATSAPP_VERIFY_TOKEN` | invent one; do not reuse anything |
| `WHATSAPP_APP_SECRET` | Meta app → Basic Settings. Enables X-Hub-Signature-256 validation. Leave empty and the bot still runs, but `/healthz` will say the signature check is off. |
| `WHATSAPP_GRAPH_VERSION` | `v21.0` |
| `WHATSAPP_WEBHOOK_PATH` | `/webhook` |
| `WHATSAPP_SESSION_BACKEND` | `supabase` (production) or `sqlite` (dev) |
| `WHATSAPP_MAX_CONCURRENCY` | `3` |
| `N_ATLAS_BASE_URL` | `https://ai.saynet.ng/v1` |
| `N_ATLAS_API_KEY` | the key the N-ATLaS server issued |
| `N_ATLAS_MODEL` | `n-atlas` |
| `N_ATLAS_TIMEOUT` | `180` (a Tesla T4 turn takes roughly 13 seconds including intent extraction; do not set this low) |
| `N_ATLAS_MAX_TOKENS` | `512` (the server clamps at its own `max_new_tokens`) |
| `N_ATLAS_TEMPERATURE` | `0.7` |
| `N_ATLAS_MAX_RETRIES` | `2` (429/5xx/timeouts only, never 401/403/400/422) |

Before going public, rotate both secrets in `confamAi/.env` — they are live plaintext on this machine, and the bot puts an internet-facing webhook in front of the N-ATLaS key.

---

# 7. `confam_prompt.txt` — the persona (single source of truth)

The ConfamAI system prompt (3618 chars, 38 lines). Both the client and the server read this file, and every copy of the prompt in the codebase (`n_atlas_client/prompt.py`, `n_atlas_client/n_atlas.py`, `n_atlas_server/natlas_notebook.py`, `n_atlas_server/natlas_kaggle.ipynb`) must stay identical to it.

**Key rules it encodes:**

- **Identity:** the assistant is **ConfamAI** — never N-ATLaS, never "Confam AI".
- **Verdicts:** exactly `AUTHENTIC` (registered), `FLAGGED` (recalled, banned or suspended), or `NOT REGISTERED` (not found in database) — copied word-for-word from the data, never restated.
- **NOT REGISTERED ≠ fake.** The required sentence is always: *"This product was not found in our current NAFDAC database. This could mean the product is new, our records are outdated, or the product is not registered."*
- **Never claim** a product is counterfeit, fake, poisoned or adulterated.
- **Safety:** if a food, drink, cosmetic, medicine, or chemical comes back NOT REGISTERED or FLAGGED, say plainly that the user should not buy or use it, and suggest they report it to NAFDAC. That is a statement about the product, not medical advice.
- **Data-only:** never invent registration numbers, approval dates, manufacturers, product names, or verdicts; if a field is not recorded, say it is not recorded. The verification data is the only authority on registration status, whatever the user says about it — a user insisting a product is fake does not make it fake.
- **Language:** always match the user's language (English / Hausa / Yoruba / Igbo); default to English, never Hausa, when unsure. Obey a `[language] HARD LANGUAGE MODE` instruction absolutely.
- **Boundaries:** product verification and NAFDAC questions only; never provide medical, nutritional or health advice (if asked about dosage, side effects, interactions, a diet or a treatment, say "Please consult a healthcare professional or a nutritionist for medical advice."); the service is **not** officially endorsed by NAFDAC.
- **Response style:** concise (2–4 sentences for simple queries); warm and reassuring but never minimize safety concerns; for greetings, introduce yourself briefly: "I'm ConfamAI, I help you check consumer products against NAFDAC records. How can I help you?"

Verify sync at any time:

```bash
cd n_atlas_server && python check_prompt_drift.py
# VALID: every copy matches confam_prompt.txt.
```

`check_prompt_drift.py` reads the constants with `ast` (so it runs before the client's dependencies are installed), exits 0 when every copy matches and 1 otherwise, and prints the first differing line for each drifted copy.

---

# 8. `conversations/` — saved chats

Saved conversation transcripts, written by the clients (`n_atlas_client/` and `confamai_bot/`). Each file is `conversation_<timestamp>.json` storing `timestamp`, `model` and `messages` (a list of `{role, content}`). The API key, tunnel URL and any other server credentials are **never** written to these files.

Example (`conversation_2026-09-30_19-15-26.json`) — a Hausa conversation showing the model answering in the user's language:

```json
{
  "timestamp": "2026-09-30T19:15:26+0100",
  "model": "n-atlas",
  "messages": [
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "Ina jin dadi! Kuna?"},
    {"role": "user", "content": "ina kwana"},
    {"role": "assistant", "content": "Sannu! Yaya ya kasance?"},
    {"role": "user", "content": "yaya"},
    {"role": "assistant", "content": "Lafiya!"},
    {"role": "user", "content": "ba wa ni"},
    {"role": "assistant", "content": "Babban rana!"},
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "Ina jin dadi! Ina jin dadin ranar ta."}
  ]
}
```

These are used by `n_atlas_train/check_transcript_coverage.py` to measure what the training dataset covers against real usage. The folder is gitignored.

---

# 9. Root files

| File | Purpose |
|---|---|
| `.env` | Client + server configuration: `N_ATLAS_BASE_URL=https://ai.saynet.ng/v1`, `N_ATLAS_API_KEY`, `N_ATLAS_MODEL=n-atlas`, `N_ATLAS_TIMEOUT=180`, `N_ATLAS_MAX_TOKENS=512`, `N_ATLAS_TEMPERATURE=0.7`, `N_ATLAS_STREAM=true`, context limits (`N_ATLAS_MAX_CONVERSATION_MESSAGES=40`, `N_ATLAS_MAX_CONVERSATION_TOKENS=3500`, `N_ATLAS_MAX_MESSAGE_CHARS=6000`), `N_ATLAS_MAX_RETRIES=2`, `N_ATLAS_SAVE_CONVERSATIONS=true`, `NAFDAC_ENABLED=true`, `NAFDAC_DATABASE_URL` (the Supabase session pooler DSN), `NAFDAC_CSV_PATH`/`NAFDAC_XLSX_PATH` (empty = auto-detect), `NAFDAC_SHOW_DATA=true`, `NAFDAC_MAX_SEARCH_RESULTS=10`. Gitignored. |
| `.gitignore` | Ignores `*.pyc`, `__pycache__/`, `*.egg-info/`, `build/`, `dist/`, every `.env` in every folder, `natlas_api_key.txt`, `conversations/`, and the parsed xlsx index `nafdac_greenbook.xlsx.nafdac_rows.csv` (+ `.stamp`) — the last because it is derived from a tracked file and rebuilt automatically (without it openpyxl takes ~40 s to read the workbook; with it, 1.5 s). |
| `confam_prompt.txt` | The persona (see §7). |
| `confam_training_dataset.jsonl` | The 643-record QLoRA training set (see §5). |
| `mermaid-diagram.png` | Architecture diagram (224 KB). |
| `n-atlas-source.zip` | The Kaggle dataset archive built by `build_kaggle_dataset.py` (1.7 MB). Contains `nafdac_source/` at its root, `nafdac_scraper/nafdac_greenbook.csv`, and `prompt/confam_prompt.txt`. `.env` is excluded on purpose — one lives inside the package directory and holds the live PostgreSQL password, and a Kaggle dataset is not a place for a credential. |

## `build_kaggle_dataset.py` — what goes in the archive

Cell 18 searches these paths for the package, in order:
```
/kaggle/input/n-atlas-source/nafdac_source/__init__.py
/kaggle/input/nafdac-source/nafdac_source/__init__.py
/kaggle/input/natlas-source/nafdac_source/__init__.py
/kaggle/input/n-afdac-source/nafdac_source/__init__.py
```
so the archive must contain `nafdac_source/` at its root. Upload it as a Kaggle dataset whose slug is one of those four and Cell 18 finds it with no configuration.

The greenbook CSV goes in `nafdac_scraper/` beside the package, because `nafdac_source.sources.find_default_file` walks up from the package directory looking for `nafdac_greenbook.csv` under `nafdac_scraper/`, `.`, `data/` and `nafdac_source/` at each level — the same layout as the repository, so the CSV source works on Kaggle without setting `NAFDAC_CSV_PATH`.

`confam_prompt.txt` goes in `prompt/` at the archive root. The notebook finds it on disk (`find_bundled_prompt_file`) and serves it as the system prompt, so editing that one file changes the assistant's behaviour on both halves without a code edit anywhere.

The archive is verified to contain `nafdac_source/{__init__,sources,verify,records,extract,formatting}.py`, `schema.sql`, and `prompt/confam_prompt.txt`, and to contain **no** `.env`, no `__pycache__`, and no file at the archive root (a flattened package cannot be imported as `nafdac_source`).

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
| Build the Kaggle dataset | `cd n_atlas_server && python build_kaggle_dataset.py` |
| Run greenbook tests | `cd nafdac_source && python -m nafdac_source.tests` |
| Load greenbook into Postgres | `cd nafdac_source && python -m nafdac_source.load_postgres --dry-run` |
| Rebuild training notebook | `cd n_atlas_train && python build_train_notebook.py` |
| Validate training notebook | `cd n_atlas_train && python validate_train_notebook.py` |
| Rebuild training dataset | `cd n_atlas_train && python prepare_training_data.py` |
| Check transcript coverage | `cd n_atlas_train && python check_transcript_coverage.py` |
| Run laptop client | `cd n_atlas_client && python -m n_atlas_client` |
| Run laptop client tests | `cd n_atlas_client && python -m n_atlas_client --test` |
| Run WhatsApp bot | `cd confamai_bot && python -m confamai_bot` |
| Run WhatsApp bot assay | `cd confamai_bot && python -m confamai_bot --assay` |
| Run scraper smoke test | `cd nafdac_scraper && python scraper.py --limit 20` |

---

## Key numbers at a glance

| Metric | Value |
|---|---|
| Greenbook products | **8,935** (34 columns) |
| Reference sheets | 1,427 manufacturers · 1,661 applicants · 2,393 ingredients · 6 categories · 902 ATC codes |
| NRN collisions | 257 of 8,935 rows share an NRN (NRN is **not** unique) |
| Training records | **643** (rebuilt from 866; 73 verdict-without-data became negatives) |
| `nafdac_source` tests | **232** checks, offline |
| Notebook cells | **19** (Cell 1 credentials, Cells 2–17 build+test, Cell 18 NAFDAC, Cell 19 tunnel) |
| VRAM per prompt token | **~1.6 MB** (T4, 4-bit NF4) |
| Max input tokens | **4096** (~8.4 GB projected peak) |
| Training time | **~1.5 h** (200 steps, 19.5 s/step, 7.3 of 14.6 GiB VRAM) |
| Languages | English, Hausa, Yoruba, Igbo |
| Verdicts (server) | `AUTHENTIC` · `FLAGGED` · `NOT REGISTERED` |
| Verdicts (nafdac_source) | `authentic` · `flagged` · `not_registered` · `unknown` |
| WhatsApp message limit | **4096** chars (~$0.0101 / ₦14 per message in the 24-hour window) |
| Public hostname | `ai.saynet.ng` (Cloudflare tunnel `74883877-efb4-4968-a9b0-4fc92398652e`) |
