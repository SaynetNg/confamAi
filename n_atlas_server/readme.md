# N-ATLaS — Stage 1: Kaggle GPU Server

`NCAIR1/N-ATLaS` (4-bit NF4) on a Tesla T4, exposed as an authenticated
OpenAI-compatible FastAPI server. Multilingual: English, Hausa, Igbo, Yoruba.

This folder (`n_atlas_server/`) is the **server half** of the project. It runs
on Kaggle. The **client half** runs on the laptop and lives one level up:

```
confamAi/
├── n_atlas_server/     <- this folder: notebook, engine, FastAPI server
├── n_atlas_client/     <- laptop clients: `python -m n_atlas_client`
│                        and the standalone `n_atlas_client/n_atlas.py`
├── nafdac_source/      <- the greenbook lookup, shared with the client
└── conversations/      <- saved chats, written by the clients
```

**Stage 1 is the Kaggle server only.** Do not start Stage 2 until every check in
Cell 17 prints `PASS` with zero skips on the real T4. The API key and the
Hugging Face login are Cell 1; the tunnel is Cell 19 and comes **after** the
Cell 17 gate, never before it. Both are described under
[Kaggle setup cells](#kaggle-setup-cells).

---

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Full pinned dependency set. Safe to `pip install -r` on Kaggle. |
| `natlas_notebook.py` | Source of truth. 19 `# %%`-delimited cells, plain sequential Python. |
| `natlas_kaggle.ipynb` | Built from the above by `build_notebook.py`. **Upload this to Kaggle.** |
| `build_notebook.py` | Regenerates the `.ipynb`. Run after editing `natlas_notebook.py`. |
| `validate_notebook.py` | Checks the `.ipynb` parses, every cell compiles, no content is lost. |
| `verify_offline.py` | Runs Cells 1–18 plus all three scripts, with a mock model + real HTTP. Cell 19 is skipped. |
| `verify_nafdac.py` | Checks Cell 18 (the greenbook) on its own. No GPU needed. |
| `mock_backend.py` | Test doubles for the tokenizer/model (used only by the verifier). |
| `notebook_cells.py` | Loads/parses the cells. Shared by the scripts and the builder. |
| `kaggleserver.py` | Script entry point: the **generation engine** (Cells 2–8), no HTTP. |
| `apiserver.py` | Script entry point: **engine + FastAPI server + greenbook** (Cells 1–11 plus Cell 18). |
| `kaggle.py` | Runs the notebook cells as a plain script (`--list`, `--cells 2-11`). |

The scripts contain **no duplicated logic** — they execute the very same cells
the notebook does, so there is nothing to drift out of sync. Editing
`natlas_notebook.py` updates the notebook *and* every script at once.

---

## Dependencies

`requirements.txt` lists everything: model runtime (`transformers`,
`accelerate`, `safetensors`, `bitsandbytes`), web server (`fastapi`,
`uvicorn[standard]`, `pydantic`, `requests`, `nest-asyncio`), and test tooling
(`openai`, `httpx`, `nbformat`). Two caveats:

- **`torch` is deliberately not listed.** It must match your CUDA build.
  Install it with `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
  Kaggle already ships `torch 2.10.0+cu128` for the T4; reinstalling it is the
  most common way to break the notebook.
- **`kaggle_secrets` is not on PyPI.** Kaggle injects it into the image.

`pip install -r requirements.txt` will not upgrade anything already installed,
so it is safe to run on Kaggle as-is.

**`bitsandbytes` is the one hard requirement Kaggle does not ship.** Without it
Cell 5 cannot build the 4-bit NF4 `BitsAndBytesConfig`.

---

## Running on Kaggle

1. Create a notebook, enable the **GPU T4 x2** accelerator.
2. Set `Internet` on (needed once to download the model).
3. Upload `natlas_kaggle.ipynb`.
4. Run Cell 1 first. It **is** the credentials step: it sets
   `N_ATLAS_API_KEY` with `os.environ.setdefault` and imports the Hugging Face
   login. A Kaggle Secret attached as `N_ATLAS_API_KEY` still wins, because
   `setdefault` only fills the variable when it is not already set. With no
   secret attached the cell falls back to the key it carries as a literal,
   which is what keeps the notebook self-contained on Kaggle.
5. Run Cells 2 → 18 **in order, once each**.
6. Read the Cell 17 summary. Stage 1 is done only when it says
   `ALL CHECKS PASSED` with **no skipped checks**.
7. Only then run Cell 19, the public tunnel. It is the last cell and it blocks
   until you interrupt the kernel.

Cell 3 only pip-installs what is genuinely missing, in two groups:

- **server** — `fastapi`, `uvicorn`, `pydantic`, `requests`, `nest_asyncio`
- **model runtime** — `bitsandbytes`, `accelerate`, `safetensors`, installed
  only when a CUDA GPU is present

It never touches torch or transformers. Set `N_ATLAS_AUTO_INSTALL=0` to make a
missing package a hard error instead.

---

## Kaggle setup cells

Two separate cells, and they do **not** sit together:

| Block | Cells | Where they run |
|---|---|---|
| **Before the model** | Cell 1 | First cell in the notebook, before anything loads. |
| **After everything passes** | Cell 19 | At the very bottom, only once Cell 17 prints `ALL CHECKS PASSED`. |

```
Cell 1  credentials ─→ Cells 2 … 18 ─→ ALL CHECKS PASSED ─→ Cell 19  tunnel
```

Cells 12–17 test `http://127.0.0.1:8000` directly. The tunnel is not involved in
any of them, so nothing about Stage 1 depends on it being up.

### Cell 1 — Hugging Face login

`NCAIR1/N-ATLaS` is a gated repo, so the token is required before any download.
Cell 1 imports the login call itself:

```python
from huggingface_hub import login
```

`login()` is then called in Cell 19, inside a `try`/`except`, so the tunnel does
not depend on it. Accept the model licence at
<https://huggingface.co/NCAIR1/N-ATLaS> first, or every download fails with `403`
regardless of token validity.

### Cell 1 — API key

Cell 1 sets the key with `setdefault`, so a Kaggle Secret attached as
`N_ATLAS_API_KEY` wins over the literal the cell carries:

```python
import os
os.environ.setdefault("N_ATLAS_API_KEY", "<your key>")
```

The key is the **only** thing standing between the public tunnel URL and your
GPU. Generate your own rather than leaving the notebook default:

```python
import os, secrets
os.environ["N_ATLAS_API_KEY"] = "natlas_" + secrets.token_urlsafe(32)
print("N_ATLAS_API_KEY loaded:", bool(os.environ.get("N_ATLAS_API_KEY")))
```

Set `N_ATLAS_ALLOW_EPHEMERAL_KEY=0` before Cell 2 to make a missing key a hard
failure instead of silently falling back to an ephemeral one.

### Cell 19 — Download the Cloudflare tunnel client

Already done for you: Cell 19 runs the download itself, through `subprocess`, and
`chmod +x`es the result. The equivalent, if you ever need it by hand:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
```

Lands in `/kaggle/working`. Verified with `!./cloudflared --version`. Needs
Internet on, and re-downloads every session because the working directory is
ephemeral.

### Cell 19 — Expose the server

**Stage 2. Do not run this until Cell 17 prints `ALL CHECKS PASSED` with no
skipped checks.** It is the last cell in the notebook, after the Cell 17 gate,
never in the run above it.

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb
cloudflared tunnel run --token "<tunnel token>"
```

Uvicorn is already bound by Cell 11 and already proven by Cells 12–16, so
`127.0.0.1:8000` is known good before this runs. If Stage 1 has not passed,
there is nothing worth exposing.

All three commands above are already implemented in Cell 19: it installs
cloudflared from the `.deb` and runs the named tunnel with the token pasted
into the cell, so there is nothing left to paste. It **blocks until
interrupted**, which is why it is last — there is no `nohup` and no background
job to set up. Interrupt the kernel when you are done.

This is a **named** tunnel, not a quick tunnel, so there is no
`trycloudflare.com` URL in the output. The hostname is the Public Hostname
configured for the tunnel in the Cloudflare Zero Trust dashboard, and it must
point at `http://localhost:8000` to match `CONFIG.port`. A named tunnel learns
where to serve from its token, so a port mismatch shows up as a `502` from the
edge rather than as an error in the cell. Paste the dashboard hostname into the
client as `https://<that-hostname>/v1`. Only now is Stage 2 reachable.

| Concern | Handling |
|---|---|
| Token is a live credential | It is hard coded in Cell 19 so the cell runs as-is, which means the notebook file contains it. Rotate it in the dashboard if it leaks. |
| Hostname fixed by the dashboard | One hostname for every run, but a change there needs no notebook edit. |
| Public and unauthenticated to probe | The bearer key is the only control. No IP allowlist, no TLS you control. |
| Process dies with the kernel | The tunnel is gone when the session ends. Re-run Cell 19. |
| Notebook autosave | Cell 19 runs in the foreground and blocks. Save before you start it, and expect the cell output to grow for as long as it runs. |
| Binding | Uvicorn binds `0.0.0.0:8000`, the dashboard Public Hostname serves `http://localhost:8000` — keep both on the default port or change both. |

---

## Configuration

Everything is an environment variable, so the core server never needs editing.

The one thing that is *not* an environment variable is the persona: it is
`confam_prompt.txt` at the project root, uploaded with the `n-atlas-source`
dataset as `prompt/confam_prompt.txt` and found on disk by
`find_bundled_prompt_file()`. The precedence is

```
N_ATLAS_SYSTEM_PROMPT_FILE > N_ATLAS_SYSTEM_PROMPT > confam_prompt.txt on disk
                            > DEFAULT_SYSTEM_PROMPT in this notebook
```

Cell 3 prints which one won after `System prompt:`. `bundled:` means your file is
being served; `builtin` means the dataset was not attached and the notebook fell
back to its own copy, so an edit to the file is not live yet. Rebuild the dataset
with `python build_kaggle_dataset.py`, re-upload it, and re-run the cells.

| Variable | Default | Notes |
|---|---|---|
| `N_ATLAS_API_KEY` | random ephemeral dev key | **Set this before exposing the server.** |
| `N_ATLAS_DEV_KEY` | unset | Known temporary key, dev only. |
| `N_ATLAS_ALLOW_EPHEMERAL_KEY` | `1` | Set `0` to hard-fail when no key is configured. |
| `N_ATLAS_SYSTEM_PROMPT` | unset — `confam_prompt.txt` is served | Inline system prompt override. |
| `N_ATLAS_SYSTEM_PROMPT_FILE` | unset — `confam_prompt.txt` is found on disk | Path to a prompt file; wins over inline. |
| `N_ATLAS_ADAPTER_ID` | unset — base model only | QLoRA adapter repo served on top of `model_id`, e.g. `Saynetng/confam-ai-adapter-v1.1`; see `../n_atlas_train/qlora_finetuning_guide.md`. A configured adapter that fails to load stops the server rather than falling back to the base model, because a silent fallback would serve un-fine-tuned weights while the config claimed otherwise. `served_model_name` stays `n-atlas`, so no client changes. |
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

`PYTORCH_ALLOC_CONF` is set to `expandable_segments:True` in Cell 2 before CUDA
initialises. It only takes effect in a fresh kernel, so re-run Cell 2 if you
restart mid-notebook.

---

## Measured on the Tesla T4

From a real run of this notebook (T4 x2, 4-bit NF4, 14.56 GB usable per card):

| Prompt tokens | Peak VRAM | Result |
|---|---|---|
| ~1,350 | 1.6 GB steady | 200 OK |
| ~6,700 | 12.7 GB peak, wanted 4.3 GB more | **507 CUDA OOM** |

That is ~1.6 MB of VRAM **per prompt token**, from the large-vocab logits tensor
plus the KV cache. Weights alone are only ~1.6 GB, so a long prompt costs
several times the model's own size.

Three changes follow from that measurement:

1. `N_ATLAS_MAX_INPUT_TOKENS` defaults to **4096**, which projects to ~8.4 GB
   peak — real headroom on a 16 GB card. Raise it only deliberately.
2. The prompt budget is now checked against the **real templated prompt**, not a
   content-only estimate. Chat-template overhead (~50 tokens per message with
   this tokenizer) was previously invisible, and `usage.prompt_tokens` under-
   reported the true length.
3. On CUDA OOM the server empties the cache and retries once at half the budget
   before returning `507`.

---

## Language handling

Two layers work together:

**1. Language tagging (server, deterministic).** Before each generation the last
user message is classified into `en` / `ha` / `yo` / `ig` and an explicit
directive is appended to the system prompt:

```
[language] The user's last message is in Yoruba.
Write the entire reply in Yoruba. Do not use English, Hausa or Igbo.
```

- Diacritics are folded first (`Báwo ni` → `bawo ni`, `ị` → `i`, `ṣ` → `s`), so
  missing diacritics classify identically.
- Weighted marker scoring; distinctive words (`bawo`, `kedu`, `kana`) outweigh
  function words (`i`, `o`, `na`).
- An explicit request ("answer in Igbo") always wins over the score.
- English is the fallback when nothing is clear.
- **Deliberate anti-English bias:** English is this model's default fallback, so
  when a message mixes English with a well-attested Nigerian language, the
  Nigerian language wins. `How are you? Yaya kake?` targets Hausa, not English.
- Set `N_ATLAS_LANGUAGE_TAGGING=0` to turn it off.

### Pinned language mode

A client that wants one language for every reply says so per request, in the
top-level `language` field:

```json
{ "model": "n-atlas", "messages": [...], "language": "ha" }
```

`"en"`, `"ha"`, `"yo"` or `"ig"` pin the reply; `"auto"`, `null` or an absent
field keeps detection. Anything else is rejected with `400`.

When a pin is present the detector **does not run** for that request, and a
harder directive replaces the detected one:

```
[language] HARD LANGUAGE MODE: every reply must be written in Hausa.
The user's message may be written in another language; that does not change
this mode. Write the entire reply in Hausa and do not use English, Yoruba, Igbo.
```

That is the whole point: a pin is a statement about the reply, so it has to hold
when the message disagrees with it. The built-in system prompt's language rules
yield to it — they describe a default, this is a choice.

Two properties worth relying on:

- **Additive.** `ChatCompletionRequest` is `extra="allow"`, so an older server
  ignores the field rather than rejecting the request, and a client that never
  sends it is unaffected. A client also sends a directive on the turn, so a pin
  still works against a build that predates this field.
- **Reported back.** The response echoes `"language"` with what was applied. It
  is informational: an older SDK may drop an unknown response field, so do not
  depend on reading it.

The other two call paths — `generate_response()` and the Cell 14 self-test —
omit the field and stay on auto, which is why the existing language-adherence
checks are unchanged.

**2. Model adherence.** The server relays whatever the model produces. On the
T4, N-ATLaS ignored the marker hints in the system prompt and answered in the
wrong language, which is what motivated the deterministic tagger above.

| Prompt | Reply before tagging | Language |
|---|---|---|
| `How are you?` | `Ina lafiya! Yaya kake ji?` | Hausa, not English |
| `Báwo ni?` | `I'm good, thanks for asking. How about you?` | English, not Yoruba |
| `Kedu ka ị mere?` | `Ina lafiya.` | Hausa, not Igbo |
| `Kana lafiya?` | `Eh, na gode! Kuma ku?` | correct Hausa |

It never refuses — it answers, in the wrong language. Cell 14 now checks the
tagger and the model separately, so the report distinguishes "my code chose the
wrong language" from "the model ignored a correct instruction":

```
[PASS] [Yoruba] tagger targeted the right language :: detected=yo confidence=medium injected=True
[FAIL] [Yoruba] generates a reply :: ... :: "I'm good, thanks for asking."
[FAIL] replies stay in the requested language (5/8) :: missed=[Yoruba (answered in en), ...]
```

If the tagger passes and adherence still fails, the limit is the model's
Yoruba/Igbo generation quality, not the API. Decide that before pointing any
consumer at this server: budget for a stronger multilingual model, or add
fine-tuning on Yoruba/Igbo dialogue. A `N_ATLAS_LANGUAGE_TAGGING=0` rollback
costs nothing.



---

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

### NAFDAC verification (Cell 18)

A chat message carrying a NAFDAC number is verified before generation, and the
verified record is injected as a system message. The model then writes the
reply in the user's language from those values, so a product name, status,
applicant or approval date can only come from the data.

Cell 18 loads the shared `nafdac_source` package and reads, in order:

1. **PostgreSQL** via `NAFDAC_DATABASE_URL`. The only source reachable from
   Kaggle, and the one carrying superseded, withdrawn and recalled products.
2. **CSV** via `NAFDAC_CSV_PATH`, or auto-detected.
3. **XLSX** via `NAFDAC_XLSX_PATH`, or auto-detected. Needs `openpyxl`.

Upload `nafdac_source/` as a Kaggle dataset to make it importable, or set
`NAFDAC_SOURCE_DIR` to a directory containing it. If it cannot be found the
cell degrades: the server starts, `/health` reports `nafdac.enabled=false`, and
chat still works. A missing greenbook never takes the model offline.

| Variable | Default | Effect |
|---|---|---|
| `N_ATLAS_NAFDAC_ENABLED` | `true` | master switch for the cell |
| `NAFDAC_DATABASE_URL` | unset | enables the PostgreSQL source |
| `NAFDAC_SOURCE_DIR` | unset | where to find `nafdac_source/` |
| `NAFDAC_CSV_PATH` | auto-detected | greenbook CSV |
| `NAFDAC_XLSX_PATH` | auto-detected | greenbook workbook |
| `N_ATLAS_NAFDAC_MAX_CONTEXT_CHARS` | `1800` | cap on the injected block |
| `N_ATLAS_NAFDAC_MAX_SEARCH_RESULTS` | `10` | cap on `/v1/nafdac/search` |

A number that is not found is a `200` with `found=false`, not a `404`: not
being in the export is a normal answer, not a client error. A request with no
number in it at all is a `400`.

Check Cell 18 on its own, without a GPU:

```bash
python verify_nafdac.py
```

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

### Response cleaning

Decoding covers only the newly generated tokens. Then, in order:

1. Cut at the earliest turn boundary — a chat-template turn marker
   (`<|im_start|>`, `[INST]`, `<<SYS>>`, `</s>`, …), a line-anchored speaker
   label (`Question:`, `User:`, …), or a bare role line.
2. If the output *starts* with a speaker label, recover the `Answer:` segment
   instead of returning nothing.
3. Strip leftover template artifacts.
4. Drop a single leading role label, collapse blank lines, trim whitespace.

The cut happens **before** artifacts are stripped, otherwise a ChatML
continuation (`<|im_end|><|im_start|>user`) loses the evidence of the new turn
and leaves a dangling `user`. Mid-sentence uses of `user:` and numbered lists
(`Question 1:`) are preserved.

---

## Verifying on your laptop

```
python verify_offline.py
```

This runs all 18 cells with `mock_backend.py` substituted for the real model,
boots a real uvicorn server on a free port, and exercises every assertion over
real HTTP, plus extra assertions the notebook does not make: forced CUDA OOM →
`507`, forced generation failure → `500`, and a live OpenAI SDK round-trip.

It then launches `kaggleserver.py`, `apiserver.py` and `kaggle.py` as real
subprocesses and checks them too — `apiserver.py` gets its own port and is
polled until `/health` answers, then served `/v1/models`, a chat completion
and an unauthenticated request before being terminated.

The mock deliberately emits template artifacts and a runaway `Question:`
continuation so the cleaner is genuinely exercised.

**This proves the server, not the model.** Language quality, real 4-bit VRAM
usage, and tokens/second can only be confirmed on the T4.

---

## Stage 1 gate

Cells 12–17 cover all of the following:

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
- [ ] replies actually in the requested language — **model-dependent, see above**

On the T4, the two GPU checks must report `PASS`, not `SKIP`.

Cell 14 reports language adherence as one aggregate line
(`replies stay in the requested language (N/8)`) so a single model-quality
number is visible without burying it in eight near-identical failures. It is
still a gate: any miss fails the run, because your spec lists Hausa, Yoruba and
Igbo generation as separate stop conditions.

---

## Script entry points

The three original files now work. They contain **no duplicated logic** — each
executes the same cells the notebook does, via `notebook_cells.py`, so they
cannot drift out of sync with `natlas_notebook.py`.

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

What changed from the earlier drafts:

- `apiserver.py` no longer hard-codes `n-atlas-kaggle-2026`. The key comes from
  `N_ATLAS_API_KEY`; with none set, a random ephemeral key is generated for the
  process and written to `<workdir>/natlas_api_key.txt` with mode 600.
- The `NameError` on `torch` and `generate_response` is gone — both now come
  from the engine that `apiserver.py` actually loads.
- `apiserver.py` now blocks on shutdown instead of exiting immediately and
  taking the daemon thread's server with it.
- `kaggleserver.py` no longer references undefined `tokenizer` / `model`
  globals; it loads and owns them.
- `kaggle.py` is no longer empty.

Because these scripts run the notebook cells, importing `kaggleserver.py` loads
the model. That is intended — there is no cheap path around it.


---

## Next stage (the tunnel)

Only after Stage 1 fully passes:

```
Kaggle server → secure external tunnel → laptop client
```

The tunnel is Cell 19, at the bottom of the notebook, already implemented
there: it installs cloudflared from the `.deb` and runs
`cloudflared tunnel run --token "<tunnel token>"` against the named tunnel.
It is not part of the server code, and no tunnel hostname is assumed anywhere in
the server.

The laptop client exists already — `../n_atlas_client/` (`python -m
n_atlas_client`, or the standalone `../n_atlas_client/n_atlas.py`) — and needs
only `N_ATLAS_BASE_URL` and `N_ATLAS_API_KEY` in a `.env`. Everything else it
speaks is covered above: `/health`, `/v1/models`, `/v1/chat/completions`.

The client also verifies NAFDAC numbers **on the laptop**, against the local
greenbook files and its own database connection, before a message is sent here.
That is deliberate: the data is on the laptop, and it means a product question
still gets an answer when this server is offline. When the client is configured
with a hosted `NAFDAC_DATABASE_URL` the same source is reachable from Kaggle,
so both halves resolve a number identically.
