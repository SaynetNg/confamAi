# Confam AI — WhatsApp bot

A WhatsApp front end for the N-ATLaS NAFDAC verification API. It receives a
message, hands it to the server, and sends back the reply.

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

## 1. What it does and does not own

**Owns:** transport (the webhook and the sender), sessions (per-phone history),
formatting (WhatsApp-safe text), and honesty about failure.

**Does not own, because the server already does and owns it better:**

- the NAFDAC greenbook lookup
- the intent gate that decides whether to look anything up
- the verdict
- the system prompt

So this package queries no greenbook and holds no product data. It imports
exactly one thing from `nafdac_source` — `extract_nrn`, a pure text function
that reads no database and no file — for the framing workaround in section 7.
Nothing else.

That restraint is the design, not laziness. A second retrieval authority on this
side would run *before* the server has seen the message, and the concrete failure
is recorded in `n_atlas_client/api.py:499-510`: a greeting that matched a product
locally talked the server into searching as well.

The cost is explicit: **when the N-ATLaS server is down, this bot cannot verify
anything and says so.** There is no local greenbook to fall back on, by design.
The alternative would be answering from the model's memory, which is exactly how
a cough syrup ends up described with ingredients nobody put in the database.

## 2. Install

```bash
pip install -r confamai_bot/requirements.txt
```

No torch, no transformers, no bitsandbytes, no GPU. This process is a webhook
receiver and an HTTP client.

`psycopg[binary]` is in `requirements.txt` unconditionally, because a deployment
that must not lose conversations on a restart should not depend on which
requirements file someone remembered to build with.

## 3. Configure

```bash
copy confamai_bot\.env.example confamai_bot\.env     # Windows
cp  confamai_bot/.env.example confamai_bot/.env      # Linux / macOS
```

Fill in:

| Variable | Where it comes from |
|---|---|
| `WHATSAPP_PHONE_NUMBER_ID` | Meta app → WhatsApp → API Setup |
| `WHATSAPP_ACCESS_TOKEN` | **System User permanent token** |
| `WHATSAPP_VERIFY_TOKEN` | invent one; do not reuse anything |
| `WHATSAPP_APP_SECRET` | Meta app → Basic Settings |
| `N_ATLAS_BASE_URL` | `https://ai.saynet.ng/v1` |
| `N_ATLAS_API_KEY` | the key the N-ATLaS server issued |

A dashboard temporary token expires in about a day, and the bot is discovered to
be broken roughly then.

### Before going public

Rotate both secrets in `confamAi/.env`. They are live plaintext on this machine,
and the bot puts an internet-facing webhook in front of the N-ATLaS key.

## 4. Run

```bash
python -m confamai_bot --status       # is the server reachable and green?
python -m confamai_bot --test         # offline suite, no network
python -m confamai_bot --ask "Is A4-100160 genuine?"
python -m confamai_bot --assay        # 13 real questions, live
python -m confamai_bot                # serve
python -m confamai_bot --port 8080
```

`--status` prints the greenbook source list, which is the first thing to check
when verification returns nothing. A top-level `status: ok` does **not** imply
the greenbook is usable — `/health` carries a separate `nafdac` block, and a
server with `nafdac.ready: false` answers product questions from memory. The bot
refuses to run in that state and says so at boot.

## 5. Commands a user can type

No slash-command help in a chat bubble, so these are guessable spellings in four
languages.

| Command | Effect | Model call |
|---|---|---|
| `/hausa` `/yoruba` `/igbo` `/english` | pin the reply language | no |
| `/auto` | follow each message's own language | no |
| `/lang` `/language` | show the current mode | no |
| `/new` `/clear` `/reset` | forget the conversation | no |
| `STOP` `QUIT` `UNSUBSCRIBE` | opt out and delete the conversation | no |
| `START` | resume after opting out | no |

Aliases work too: `/harshe`, `/sauraya` (Hausa), `/yebu`, `/ede` (Yoruba),
`/ahukeri` (Igbo), plus `/sabisha`/`/tso` for clear and `/ohuru`/`/tọ ọ`.

Local commands never reach the model. A `/hausa` answered by the model would
take ~13 seconds on a T4 to say something this package says instantly, and would
consume a GPU slot a real question is waiting for.

**Opt-out is enforced, not documented.** `STOP` deletes the stored conversation
and registers the number; afterwards every message from it is dropped **in
silence**, with no reply at all, because honouring an opt-out means sending
nothing. Only `START` revokes it. NDPA 2023 requires this.

## 6. Voice notes

A voice note is downloaded from Meta and transcribed to text, then takes the
ordinary text path — so a spoken "stop" opts out and a spoken NAFDAC number is
verified exactly as if it had been typed.

By default the note is transcribed by the **server**, POSTed to
`/v1/audio/transcriptions` — the same OpenAI-compatible endpoint the website's
microphone uses. The four NCAIR1 models already run there, preloaded on the
server's GPU, so this bot needs no torch, no model download and no GPU. That is
the whole point of the server endpoint.

`WHATSAPP_VOICE_TRANSCRIPTION` selects the path:

| Value | Behaviour |
|---|---|
| `auto` (default) | try the server, fall back to the local model if it is unreachable |
| `server` | server only — no local fallback, so a server outage is reported rather than silently served by a local model |
| `local` | load the NCAIR1 models in this process (needs `pip install torch transformers`) |

Both paths handle the two traps a WhatsApp voice note hits: the clip is passed
as **raw bytes** (never a file object, which the pipeline rejects), and a note
over **30 seconds** is retried with long-form chunking, because Whisper refuses
a single pass that long and a real voice note usually is.

## 7. A server defect this bot works around

Found by `--assay` on 2026-10-04. The retrieval gate attaches a verified block
correctly for a bare number — `prompt_tokens` 1658 against a 599 baseline — and
the model then discards it and asks which product to verify:

| Message | Result |
|---|---|
| `A4-100160` | "Would you like to verify a product?" — **FAIL** |
| `A4-100160?` | "Would you like to verify a product?" — **FAIL** |
| `check A4-100160` | "Which product would you like...?" — **FAIL** |
| `verify this number A4-100160` | "Would you like to verify a product?" — **FAIL** |
| `What is A4-100160?` | full record, `VERDICT: AUTHENTIC` |
| `Tell me about A4-100160` | full record |
| `Is A4-100160 genuine?` | full record |

The trigger is grammatical framing, not content: an interrogative gets answered,
a bare noun phrase or bare imperative gets redirected. The Confam AI persona's
`BOUNDARIES` rule wins over the server's `NAFDAC_INSTRUCTION` when nothing looks
like a question. It is deterministic — 4/4 failures at temperature 0.7 and 4/4 at
0.1 — and it is the highest-frequency way anyone uses a verification service.

**The proper fix is server-side**, a clause in `NAFDAC_INSTRUCTION` alongside the
others added after real failures:

> A message that is only a NAFDAC number, or an instruction like "check
> A4-100160", IS a verification request. Answer it with that number's record and
> verdict. Never ask which product the user means when a number was given.

Until then `framing.py` appends one instruction line to such a turn, which fixes
all four cases. It is safe by two measured properties:

1. **No greenbook data is injected.** The frame names no product, no verdict and
   no number. `nafdac_source` is imported for `extract_nrn` alone, which reads
   no database and no file and does not load psycopg.
2. **The intent gate still refuses non-product messages.** Applying the frame to
   *every* message moved "Hello" from 599 to 644 prompt tokens — a `+45` delta,
   which is the frame text itself. A real greenbook block adds about 1000. No
   greeting starts searching.

The frame is added only when the user's own text already holds a number, so a
greeting gets nothing appended at all, and the **unframed** text is what goes
into the stored history — keeping the frame would append another copy every turn
and hand the model a conversation the user never had.

Set `WHATSAPP_VERIFICATION_FRAME=false` once the server covers bare numbers.

## 8. Design notes worth knowing

**No system message is ever sent.** The server drops client-supplied system
messages and logs a warning, because a client that could send
`[nafdac] Verdict: AUTHENTIC` as a system turn could assert any verdict it liked
(`n_atlas_server/natlas_notebook.py:1564-1569`). The Confam AI persona is
`DEFAULT_SYSTEM_PROMPT` on the server. There is a test that asserts no system
role ever reaches the wire.

**Replies are never split unless they must be.** Since Meta's 1 October 2026
change, a service message inside the 24-hour customer-service window is charged
per message delivered — about $0.0101 (₦14) to a Nigerian number, with a free
tier of 1,000 per month per number. Each chunk `render.chunk()` produces is a
separate bill, so a normal reply is one message and splitting is the failure
case.

**Two levels of concurrency control.** A per-user lock, so one person's turns
serialise and history stays in conversation order. And a global semaphore,
default 3, because the server handles requests in a threadpool and generates on
one shared GPU — 32 concurrent turns is a wall of 507s, not 32 answers. Both
waits are bounded, so a stuck request cannot silently hold every slot.

**Deduplication is mandatory.** Meta retries a webhook that has not answered 200
within twenty seconds, and a T4 turn takes about thirteen — so doing the work
inline would guarantee a retry on nearly every message. The POST returns 200 in
milliseconds and the work runs in a background task; `wa_seen` records inbound
message ids for an hour so a retry sends nothing.

**The acknowledgement is one free message.** A short "checking the NAFDAC
records — one moment" goes out before the request, because an unexplained
13-second pause reads as a broken bot. It is free inside the customer's 24-hour
window. Set `WHATSAPP_ACK_TEXT=none` to switch it off.

**Session storage.** `sqlite` is one local file and needs no extra dependency.
`supabase` makes the bot stateless, which matters as soon as there is more than
one instance or the host disk is ephemeral — two instances with separate SQLite
files would each hold a different half of one user's conversation.

The bot's database is **dedicated**, not the one holding the NAFDAC greenbook.
The greenbook is public regulatory data; `wa_sessions` holds what people typed,
which is personal data under NDPA 2023. Separate databases mean a backup of
either is safe to share, and `nafdac_source`'s table-swapping migrations cannot
reach the bot's tables. The `confamai_bot` schema is kept on top of that as
belt and braces, so the same code still works if you ever consolidate.

Use the **session pooler** (`:5432`), not the direct connection — the direct
host needs a static outbound IPv4 that several hosting platforms lack or charge
extra for. Percent-encode the password: Supabase passwords contain `@`, `:`,
`/` and `#`, which are structural in a `postgresql://` URL, and an unescaped one
reports as a misleading `missing '/' after host`.

The schema and both tables are created automatically at startup, so there is no
DDL to run by hand. The 24-hour TTL is enforced by a sweeper, not just
documented.

## 9. Tests

```bash
python -m confamai_bot --test
```

324 checks, no network, no WhatsApp account, no GPU. Three halves:

- **Payload replay.** Realistic webhook bodies through the router with a stubbed
  N-ATLaS. Asserts a retry sends nothing, a system message is never built,
  `language` travels only when pinned, `/hausa` never reaches the model, an
  opted-out number gets silence, a photo costs no model call, a voice note is
  transcribed then answered as text, and a server failure is reported rather
  than guessed.
- **Voice request shape.** The multipart `file` upload to
  `/v1/audio/transcriptions`, the pinned language, and the error split: an empty
  transcript and an undecodable clip are told apart. Plus the transcriber being
  server-first and only falling back to the local NCAIR1 model, and the local
  path retrying long-form for a clip over 30 seconds.
- **Reply contract.** The safety rules are not stylistic. A reply calling an
  unregistered product counterfeit is the failure this project exists to
  prevent, so the same assertions the live assay uses are applied to fixtures
  offline.

Four bugs were caught by these tests during the build, each of which would
have shipped:

1. **The webhook handshake was broken.** FastAPI mapped a `mode` parameter to
   `?mode=`, but Meta sends `?hub.mode=`, so the subscription returned 403 and
   no messages would ever arrive.
2. **`/auto` never worked.** The empty string was used for both "auto mode" and
   "read-only command", so `/auto` matched the read-only branch and left the
   language pin in place while appearing to succeed.
3. **Every SQLite call leaked a file handle.** `with sqlite3.connect(...)` is a
   transaction context manager, not a resource one; it commits and leaves the
   handle open.
4. **The queue log reported the wrong number.** It timed the whole turn instead
   of the slot wait, so every healthy 13-second T4 turn was logged as a queue
   problem, sending an operator to look in the wrong place.

## 10. Files

| File | Purpose |
|---|---|
| `webhook.py` | FastAPI app: handshake, inbound, `/healthz` (liveness), `/readyz` (readiness) |
| `router.py` | orchestration — local commands vs the model |
| `natlas.py` | the N-ATLaS client; the only module that talks upstream (chat and voice transcription) |
| `whatsapp.py` | Graph API sender and webhook parsing |
| `audio.py` | speech-to-text for voice notes — server-first, local NCAIR1 fallback |
| `sessions.py` | per-phone history, SQLite or Supabase, TTL sweeper |
| `languages.py` | language commands and the opt-out keyword |
| `render.py` | WhatsApp-safe text, chunking, acknowledgement language |
| `queue.py` | per-user locks and the global semaphore |
| `framing.py` | the bare-number workaround in section 7 |
| `assay.py` | the live question battery |
| `tests.py` | the offline suite |
| `config.py` | configuration, with `describe()` that never prints a secret |

## 11. Deploying on Render

The service is a plain Python web service. Settings:

| Setting | Value |
|---|---|
| **Root Directory** | **leave empty** (the repository root) |
| **Build command** | `pip install -r requirements.txt` (the default) |
| **Start command** | `python -m confamai_bot --host 0.0.0.0 --port $PORT` |

The root `requirements.txt` installs `confamai_bot/requirements.txt` and
then the package itself (`pip install -e .` via the root `pyproject.toml`),
so `confamai_bot` is importable from the virtualenv no matter which
directory the start command runs from. That is the fix for the deploy
failure `No module named confamai_bot`: the module must be **installed**,
not merely present beside the working directory.

**Root directory must be empty.** If the service has a Root Directory set
to anything (commonly `confamai_bot`), the start command runs from inside
the package and cannot find `confamai_bot` as a package — this produces the
exact `No module named confamai_bot` error. Clear that setting.

## What to do if you are STILL getting "No module named confamai_bot"

The repository now ships `render.yaml`, which declares the exact build and
start commands plus the env vars the bot needs. Importing it overwrites a
manually-typed configuration (which is usually the source of the mismatch).

1. In the Render dashboard, open the service → **Settings** (gear icon).
2. Click **Sync Infrastructure** and choose **Import from render.yaml**.
3. When prompted, copy the entire content of `render.yaml` and choose
   **Replace** to overwrite the existing build/start commands and env vars.
4. If you prefer, you can also delete this service and create a new one from
   the repository: on creation Render auto-imports `render.yaml`.
5. After the config is applied, **trigger a fresh deploy** (Settings →
   "Deploy" / "Redeploy", not a cached deployment).

If you do not import the file, make sure the dashboard matches the table
above exactly: build command `pip install -r requirements.txt`, start command
`python -m confamai_bot --host 0.0.0.0 --port $PORT`, Root Directory empty,
and the service building from the `main` branch. Then redeploy.

**Environment variables** (set in the Render dashboard, not in a file —
`.env` is gitignored and real env vars always win over any `.env` on
disk). Copy the values from `confamai_bot/.env.example`:

- `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN` (System User
  **permanent** token), `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`
- `N_ATLAS_BASE_URL` (`https://ai.saynet.ng/v1`), `N_ATLAS_API_KEY`
- `WHATSAPP_SESSION_BACKEND` and, for `supabase`, `WHATSAPP_SESSION_DSN`
  (session pooler, `:5432`, password percent-encoded)

The bot **refuses to start** (exit code 2, naming every missing variable)
without the WhatsApp and N-ATLaS variables, so a deploy with an empty
dashboard fails fast instead of serving silently. With the default
`sqlite` session backend one instance is fine; use `supabase` if the
service ever scales past one instance or the disk is ephemeral.

After pushing, redeploy from the Render dashboard and check
`https://<service>.onrender.com/healthz` — `"bot": "up"` with
`"natlas": "ready"` means the whole chain works.

`/healthz` (aliased as `/health`) is **liveness**: it always answers 200 while
the process is up, from memory, and never calls N-ATLaS. Point Render's health
check and any keep-alive pinger (e.g. cron-job.org, UptimeRobot) at it — that
is what keeps a free-tier service from sleeping, and it will never be slow or
flap because the GPU server behind the bot is down or still loading.

`/readyz` is **readiness**: a live N-ATLaS probe that returns 503 while the
model or the greenbook is not ready. Use it for a monitor that wants to know
whether *answers* are possible, not for keeping the service awake.