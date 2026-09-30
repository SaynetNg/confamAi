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
└── conversations/      <- saved chats, written by the clients
```

**Stage 1 is the Kaggle server only.** Do not start Stage 2 until every check in
Cell 16 prints `PASS` with zero skips on the real T4. The Hugging Face login,
API key and Cloudflare download commands are documented under
[Kaggle setup cells](#kaggle-setup-cells); the tunnel cell (D) comes **after**
the Cell 16 gate, never before it.

---

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Full pinned dependency set. Safe to `pip install -r` on Kaggle. |
| `natlas_notebook.py` | Source of truth. 16 `# %%`-delimited cells, plain sequential Python. |
| `natlas_kaggle.ipynb` | Built from the above by `build_notebook.py`. **Upload this to Kaggle.** |
| `build_notebook.py` | Regenerates the `.ipynb`. Run after editing `natlas_notebook.py`. |
| `validate_notebook.py` | Checks the `.ipynb` parses, every cell compiles, no content is lost. |
| `verify_offline.py` | Runs Cells 1–16 plus all three scripts, with a mock model + real HTTP. |
| `mock_backend.py` | Test doubles for the tokenizer/model (used only by the verifier). |
| `notebook_cells.py` | Loads/parses the cells. Shared by the scripts and the builder. |
| `kaggleserver.py` | Script entry point: the **generation engine** (Cells 1–7), no HTTP. |
| `apiserver.py` | Script entry point: **engine + FastAPI server** (Cells 1–10). |
| `kaggle.py` | Runs the notebook cells as a plain script (`--list`, `--cells 1-10`). |

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
Cell 4 cannot build the 4-bit NF4 `BitsAndBytesConfig`.

---

## Running on Kaggle

1. Create a notebook, enable the **GPU T4 x2** accelerator.
2. Set `Internet` on (needed once to download the model).
3. Upload `natlas_kaggle.ipynb`.
4. Add your API key as a Kaggle Secret named `N_ATLAS_API_KEY`, or paste this
   above Cell 1:

   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ["N_ATLAS_API_KEY"] = UserSecretsClient().get_secret("N_ATLAS_API_KEY")
   ```

5. Run Cells 1 → 16 **in order, once each**.
6. Read the Cell 16 summary. Stage 1 is done only when it says
   `ALL CHECKS PASSED` with **no skipped checks**.

Cell 2 only pip-installs what is genuinely missing, in two groups:

- **server** — `fastapi`, `uvicorn`, `pydantic`, `requests`, `nest_asyncio`
- **model runtime** — `bitsandbytes`, `accelerate`, `safetensors`, installed
  only when a CUDA GPU is present

It never touches torch or transformers. Set `N_ATLAS_AUTO_INSTALL=0` to make a
missing package a hard error instead.

---

## Kaggle setup cells

Two separate blocks, and they do **not** sit together:

| Block | Cells | Where they run |
|---|---|---|
| **Before the model** | A, B, C | Above Cell 1, before anything loads. |
| **After everything passes** | D | At the very bottom, only once Cell 16 prints `ALL CHECKS PASSED`. |

```
A  HF login  ─┐
B  API key   ─┼─ above Cell 1 ─→ Cells 1 … 16 ─→ ALL CHECKS PASSED ─→ D  tunnel
C  cloudflared ┘
```

Cells 11–16 test `http://127.0.0.1:8000` directly. The tunnel is not involved in
any of them, so nothing about Stage 1 depends on it being up.

### Cell A — Hugging Face login

`NCAIR1/N-ATLaS` is a gated repo, so the token is required before any download.

```python
from huggingface_hub import login

login()
```

`login()` with no argument reads `HF_TOKEN` from the Kaggle secret environment.
Accept the model licence at <https://huggingface.co/NCAIR1/N-ATLaS> first, or
every download fails with `403` regardless of token validity.

### Cell B — API key

```python
import os
os.environ["N_ATLAS_API_KEY"] = "<your key>"
```

The key is the **only** thing standing between the public tunnel URL and your
GPU. Generate one rather than pasting the placeholder:

```python
import os, secrets
os.environ["N_ATLAS_API_KEY"] = "natlas_" + secrets.token_urlsafe(32)
print("N_ATLAS_API_KEY loaded:", bool(os.environ.get("N_ATLAS_API_KEY")))
```

Set `N_ATLAS_ALLOW_EPHEMERAL_KEY=0` in this cell to make a missing key a hard
failure instead of silently falling back to an ephemeral one.

### Cell C — Download the Cloudflare tunnel client

Can be run with A and B above Cell 1, or later. Either way it is inert until
Cell D starts it.

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
```

Lands in `/kaggle/working`. Verified with `!./cloudflared --version`. Needs
Internet on, and re-downloads every session because the working directory is
ephemeral.

### Cell D — Expose the server

**Stage 2. Do not run this until Cell 16 prints `ALL CHECKS PASSED` with no
skipped checks.** It is the last cell in the notebook, after the Cell 16 gate,
never in the run above it.

```python
!./cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

Uvicorn is already bound by Cell 10 and already proven by Cells 11–15, so
`127.0.0.1:8000` is known good before this runs. If Stage 1 has not passed,
there is nothing worth exposing.

`nohup ... &` is required — a bare `!./cloudflared tunnel` blocks the cell
forever and the notebook can never be saved. `&` alone is not enough in Kaggle;
the process gets killed when the cell's shell exits.

Then read the URL out of the log, once, after the tunnel is up:

```python
!grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' tunnel.log | head -1
```

Point the client at `https://<that-url>/v1`. Only now is Stage 2 reachable.

| Concern | Handling |
|---|---|
| URL changes every session | Quick tunnels have no fixed hostname. Re-read it after each restart. |
| Public and unauthenticated to probe | The bearer key is the only control. No IP allowlist, no TLS you control. |
| Process dies with the kernel | The tunnel is gone when the session ends. Re-run Cell D. |
| Notebook autosave | `nohup` plus a background job keeps it saving; do not run the tunnel in the foreground. |
| Binding | Uvicorn binds `0.0.0.0:8000`, cloudflared dials `127.0.0.1:8000` — keep both on the default port or change both. |

---

## Configuration

Everything is an environment variable, so the core server never needs editing.

| Variable | Default | Notes |
|---|---|---|
| `N_ATLAS_API_KEY` | random ephemeral dev key | **Set this before exposing the server.** |
| `N_ATLAS_DEV_KEY` | unset | Known temporary key, dev only. |
| `N_ATLAS_ALLOW_EPHEMERAL_KEY` | `1` | Set `0` to hard-fail when no key is configured. |
| `N_ATLAS_SYSTEM_PROMPT` | built-in prompt | Inline system prompt override. |
| `N_ATLAS_SYSTEM_PROMPT_FILE` | unset | Path to a prompt file; wins over inline. |
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

`PYTORCH_ALLOC_CONF` is set to `expandable_segments:True` in Cell 1 before CUDA
initialises. It only takes effect in a fresh kernel, so re-run Cell 1 if you
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

**2. Model adherence.** The server relays whatever the model produces. On the
T4, N-ATLaS ignored the marker hints in the system prompt and answered in the
wrong language, which is what motivated the deterministic tagger above.

| Prompt | Reply before tagging | Language |
|---|---|---|
| `How are you?` | `Ina lafiya! Yaya kake ji?` | Hausa, not English |
| `Báwo ni?` | `I'm good, thanks for asking. How about you?` | English, not Yoruba |
| `Kedu ka ị mere?` | `Ina lafiya.` | Hausa, not Igbo |
| `Kana lafiya?` | `Eh, na gode! Kuma ku?` | correct Hausa |

It never refuses — it answers, in the wrong language. Cell 13 now checks the
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
| `GET /health` | none | Readiness. GPU, quantization, load time, model_loaded. |
| `GET /v1/models` | bearer | OpenAI-compatible model list. |
| `POST /v1/chat/completions` | bearer | Chat completion. Full message history. |

Streaming (`stream=true`) is **not** supported and returns `400`.

### Error mapping

| Code | Cause |
|---|---|
| `400` | Malformed JSON, empty/whitespace messages, no user turn, too many messages, stream requested, legacy `/v1/completions`. |
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

This runs all 16 cells with `mock_backend.py` substituted for the real model,
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

Cells 11–16 cover all of the following:

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

Cell 13 reports language adherence as one aggregate line
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
| `kaggleserver.py` | Cells 1–7 | Engine only, no HTTP. Smoke-tests all four languages. |
| `apiserver.py` | Cells 1–10 | Engine + FastAPI on `0.0.0.0:8000`, blocks until Ctrl+C. |
| `kaggle.py` | any range | `python kaggle.py --list`, `--cells 1-7`, `--cells 1-10`. |

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

The tunnel is documented as manual cells under
[Kaggle setup cells](#kaggle-setup-cells) (Cell D above). It is not wired into
the server code, and no tunnel URL is assumed anywhere in it.

The laptop client exists already — `../n_atlas_client/` (`python -m
n_atlas_client`, or the standalone `../n_atlas_client/n_atlas.py`) — and needs
only `N_ATLAS_BASE_URL` and `N_ATLAS_API_KEY` in a `.env`. Everything else it
speaks is covered above: `/health`, `/v1/models`, `/v1/chat/completions`.
