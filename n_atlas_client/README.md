# N-ATLaS laptop client — README
#
# This is a CLIENT. The N-ATLaS model runs on Kaggle (Tesla T4). Nothing
# model-related is installed or runs on this laptop.
#
# This folder holds both client forms: the modular package (the normal way)
# and `n_atlas.py`, a self-contained single-file copy you can drop onto any
# machine on its own.
#
# The matching server code lives in `../n_atlas_server/` — see
# `../n_atlas_server/readme.md` for the notebook, the engine and the API.

## 1. What this is

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

The laptop only needs Python 3.10+ plus a few small libraries.
It does NOT need torch, transformers, bitsandbytes, CUDA, an NVIDIA GPU,
or any Hugging Face model download. The entire model stays on Kaggle.

NAFDAC verification is the one part that runs locally. The greenbook data and
the database connection are on this machine, so the lookup happens here, and
it keeps working when the Kaggle session has ended. See section 13.

## 2. Installation

```bash
cd n_atlas_client
pip install -r requirements.txt
# which installs just:  openai  python-dotenv  requests
```

## 3. Configuration

Both entry points read the first `.env` they find, in this order:

1. the current working directory
2. the project root (`../.env`)
3. this folder (`./.env`)

So `cp ../.env.example .env` — or just edit `n_atlas_client/.env`, which is
already there and already gitignored.

```
N_ATLAS_BASE_URL=https://REAL-TUNNEL-URL/v1
N_ATLAS_API_KEY=YOUR_SECRET_KEY
```

- Replace the placeholders with the REAL tunnel URL and API key you receive
  after the Kaggle server is running.
- Never use `localhost:8000`. localhost on your laptop is not the Kaggle
  server.
- Do not invent a URL. If the server isn't running yet, the client will report
  the server as offline — that is correct behaviour.
- The API key is read from the environment / .env and is NEVER printed or
  written into conversation files.

### System prompt (Confam AI)

The client sends a system message on every request, so the model answers as
**Confam AI**, the CONFAM verification assistant for saynet.ng, instead of the
general-purpose N-ATLaS assistant the server was built for. The prompt lives in
`prompt.py` (`CONFAM_AI_SYSTEM_PROMPT`) and is used when nothing overrides it.

```
N_ATLAS_SYSTEM_PROMPT=inline text      # wins over everything
N_ATLAS_SYSTEM_PROMPT_FILE=path.txt    # used when the inline value is empty
N_ATLAS_SYSTEM_PROMPT=none             # send no system message at all
```

Empty values count as unset, so the blank lines in `.env` mean "use the built-in
Confam AI prompt". `none` (also `off`, `0`, `false`, `no`, `disabled`) switches
the system message off and lets the server's own prompt take over.

The Kaggle server **drops client-supplied system messages** — only server code
may set them (`n_atlas_server/natlas_notebook.py`, the chat-completions handler).
So the persona has to live on the server, and it does: `DEFAULT_SYSTEM_PROMPT`
there is the same Confam AI prompt, byte for byte. The client's copy is what a
server that *does* honour client system messages sees, and it stays useful for
testing a prompt without redeploying the notebook.

## 4. Startup

Run from the project root:

```bash
python -m n_atlas_client                 # modular version (preferred)
python -m n_atlas_client --debug         # show HTTP status/durations
python -m n_atlas_client --test          # run the automated test suite
python -m n_atlas_client --version       # version info
```

The single-file version lives in this folder and takes the same flags:

```bash
python n_atlas_client/n_atlas.py
python n_atlas_client/n_atlas.py --debug
python n_atlas_client/n_atlas.py --test
python n_atlas_client/n_atlas.py --version
```

Both write conversations to `conversations/` **relative to the directory you
run from**, so run them from the project root to keep one history folder.

## 5. Commands (handled locally, never sent to the model)

| Command    | Action                                        |
|------------|-----------------------------------------------|
| `/help`    | Show available commands                       |
| `/new`     | Start a new conversation (clears local memory)|
| `/clear`   | Clear conversation history                    |
| `/history` | Display the local conversation history        |
| `/status`  | Re-check the remote server (health / models)  |
| `/nafdac`  | Show which NAFDAC data sources are reachable  |
| `/verify`  | Check one NAFDAC number (`/verify A4-100160`) |
| `/search`  | Search products (`/search malaria`)           |
| `/hausa`   | Pin every reply to Hausa                      |
| `/yoruba`  | Pin every reply to Yoruba                     |
| `/igbo`    | Pin every reply to Igbo                       |
| `/english` | Pin every reply to English                    |
| `/auto`    | Drop the pin; follow each message's language  |
| `/langueage` | Show the current language mode              |
| `/exit`    | Quit the client                               |

`/verify`, `/search` and `/nafdac` are answered entirely on the laptop. They
never reach the model, so they work with the server offline.

## 5b. Language modes

Each language is its own command, so the mode is one keystroke away:

```
/hausa        Language mode: Hausa. N-ATLaS will reply in Hausa until you switch...
Báwo ni?      -> a reply in Hausa, not Yoruba
/yoruba       Language mode: Yoruba...
Sannu da yawa?  -> a reply in Yoruba, not Hausa
/english      Language mode: English...
/auto         Language mode: Auto. N-ATLaS will reply in the language of each message.
```

**The mode is a hard pin.** It is a choice about the *reply*, so it holds even
when the next message is in another language: in Yoruba mode a Hausa message is
still answered in Yoruba. The mode changes when you run another language
command, never because of what you typed.

`/auto` is the default and sends nothing at all, which is how the client has
always behaved — the server reads each message and answers in the language it
finds there.

The mode reaches the server two ways at once: as the `language` field of the
request, and as one directive line on the outgoing turn. A server that
understands the field uses it as authoritative and skips its own detector; a
server that does not ignore the unknown field (its request model is
`extra="allow"`) and still reads the directive. Either build works, and
switching modes never resets the conversation.

The mode also sets the language of the local NAFDAC tables, so `/verify` prints
Hausa labels when you are in Hausa mode. **Product data is never translated** —
a product name, an applicant and an approval date are printed exactly as the
greenbook holds them. Only labels, verdicts and sentences change.

| Setting            | Default | Meaning                                    |
|--------------------|---------|--------------------------------------------|
| `N_ATLAS_LANGUAGE` | `auto`  | Starting mode: `auto`, `en`, `ha`, `yo`, `ig` |

In `auto`, local NAFDAC output is English, since there is no mode to localise
to. The model's own reply still follows the message.

## 6. Example multilingual conversation

```
You: How are you?
Assistant: I'm doing well. How can I help?

You: Kana lafiya?
Assistant: Ina lafiya! Kai fa?

You: What's your name?
Assistant: I'm N-ATLaS...

You: Bawo ni?
Assistant: Mo wà dáadáa! ...

You: Now explain artificial intelligence in English.
Assistant: (English reply)
```

The client never resets history when the language switches — you can switch
between English, Hausa, Igbo and Yoruba freely in one conversation, and you can
pin a mode with `/hausa` and change your mind later without losing anything.

In `auto` (the default) the model follows the language of each message, so
"explain this in English" is not needed. `/hausa` and friends are for when you
want every reply in one language regardless of what you type.

## 7. Conversation memory and context

- The laptop OWNS the conversation history. The server is stateless per
  request; each request carries the relevant history.
- The client trims old turns when history exceeds
  `N_ATLAS_MAX_CONVERSATION_MESSAGES` / `N_ATLAS_MAX_CONVERSATION_TOKENS`,
  keeping the opening exchange and the most recent messages. It never
  silently wipes the whole conversation.
- Optional JSON saving (default on) writes to `conversations/` files:
  `conversation_2026-09-30_15-30-00.json` containing timestamp, model and
  messages only. The path is relative to the directory you run from, so run
  from the project root. API keys and credentials are never stored there.

## 8. Streaming

Streaming is enabled by default (`N_ATLAS_STREAM=true`). If the Kaggle
server rejects streaming (some deployments do not implement SSE), the client
automatically falls back to normal non-streaming responses. You do not need
to configure anything.

## 9. Error handling

The client translates failures into friendly messages:

| Situation                    | Message shown                                 |
|------------------------------|-----------------------------------------------|
| Server / network offline     | Unable to reach the N-ATLaS server...         |
| Wrong API key                | Authentication failed. Check N_ATLAS_API_KEY. |
| GPU OOM (server returns 507) | The remote N-ATLaS GPU ran out of memory...   |
| Rate limit (429)             | Rate limit reached...                         |
| Server error (5xx)           | Server error...                               |
| Empty model response         | The model returned an empty response...       |

Technical details (HTTP status, durations, exceptions) are only shown with
`--debug`. No stack traces are shown to normal users, and the API key is
never displayed.

## 10. Test mode

```bash
python -m n_atlas_client --test
# or the single-file version
python n_atlas_client/n_atlas.py --test
```

Runs (against the live server):
- connection
- `GET /health`
- `GET /v1/models` and that `n-atlas` is listed
- English, Hausa, Yoruba and Igbo requests — each only passes if the response
  is non-empty (a 200 status alone is not enough)
- conversation memory ("My name is Ahmed." then "What is my name?")
- language switching ("Answer in Hausa." then "Now answer in English.")

It also runs 22 NAFDAC checks first, before the connection test, because those
need no server. They cover a known product, a known-inactive product, an unknown
number, which messages get intercepted, and the rule that a miss never claims a
product is counterfeit.

## 11. Troubleshooting

| Symptom                                    | Fix                                                    |
|--------------------------------------------|--------------------------------------------------------|
| `Missing required configuration: N_ATLAS_BASE_URL, N_ATLAS_API_KEY` | Edit `n_atlas_client/.env` (or the project-root `.env`) and fill in the real values. |
| `Unable to reach the N-ATLaS server...`    | Server is offline or the tunnel is down. Check Kaggle, the tunnel, your network. |
| `Authentication failed...`                 | The key in `.env` doesn't match the server's key.       |
| `The remote N-ATLaS GPU ran out of memory` | Shorten the conversation (`/new`), lower `N_ATLAS_MAX_TOKENS`, or ask for a shorter reply. |
| Works on `--test` but errors in chat       | Check you're sending `/`-free text; test commands are local only. |
| Picked up the wrong URL after moving files | Two `.env` files exist (project root and this folder). Run from the project root, or delete one. |
| `/nafdac` shows `csv` as not found | Run from the project root, or set `NAFDAC_CSV_PATH` to the full path. |
| `/nafdac` shows `postgres not configured` | Expected without a database. Set `NAFDAC_DATABASE_URL` to get superseded products. |
| `/nafdac` shows `xlsx (openpyxl not installed)` | Expected. `pip install openpyxl`. The CSV holds the same rows. |
| A product is reported from local files only | The database was unreachable, so a superseding update would not show. Check `NAFDAC_DATABASE_URL`. |
| `/verify` says a real product is not found | The export may predate the product. Try `/search <product name>`. |

## 12. Security notes

- The client never executes code returned by the model. Code blocks are
  only displayed.
- Chat input is never interpreted as a URL to connect to. The only endpoint
  used is `N_ATLAS_BASE_URL` from configuration.
- Conversation history is local. Only the messages needed for the current
  request leave the laptop.

## 13. NAFDAC product verification

Ask about a product in your own words. The client extracts the NAFDAC number,
checks it against the greenbook, and hands N-ATLaS the verified values so the
reply is built from data rather than memory.

```
You: Is A4-100160 genuine?
Assistant: A4-100160 is AG Spironolactone, and it is currently authorised.
  It was approved on 2021-12-17 and expires on 2026-12-16. The applicant is
  AnnyGod Pharmaceutical Company Ltd ...

You: Ka A4-100160 babu?
Assistant: (Yoruba, same verified values)
```

### Where the answer comes from

Three sources, consulted in this order and **merged**, not raced. The first
source that has a value wins, later sources only fill gaps:

| Source      | What it adds                                                     |
|-------------|------------------------------------------------------------------|
| PostgreSQL  | Current state, plus superseded/withdrawn products. The only source the Kaggle server can reach. |
| CSV         | `nafdac_scraper/nafdac_greenbook.csv`, 8930 products. No dependency needed. |
| XLSX        | `nafdac_scraper/nafdac_greenbook.xlsx`, same rows. Needs `openpyxl`. |

Every source is optional. With none of them the client still runs and says the
number was not found. Check what is reachable with `/nafdac`.

### What it will and will not tell you

- **A hit names its sources.** A result found in both the database and the CSV
  reports `verified`; one found only in a local file reports `local-file` and
  warns that a superseding update would not show.
- **A miss is never a verdict.** Not being in the export is not the same as
  being counterfeit, and the client says so. The export may be out of date, or
  the number may be mistyped.
- **A shared NRN is not hidden.** 257 of the 8930 rows share a number with
  another row. When that happens you get all of them and a request to say which
  product you meant.
- **An inactive product is not a valid one.** The greenbook only publishes
  `Active` and `Inactive`, and both are shown, so a withdrawn product is never
  presented as authorised.
- **Odd formats resolve.** `A4 - 5180`, `a4-5180`, `04-2045` with an en dash,
  `4/1/9086`, `A1-4924L` and `NA` are all handled. `NA` and `Not available yet`
  are recognised as "no number was recorded" and never looked up.

### Setting up the database

```bash
pip install "psycopg[binary]"
# set NAFDAC_DATABASE_URL in .env, then:
python -m nafdac_source.load_postgres --dry-run   # read the CSV, write nothing
python -m nafdac_source.load_postgres             # create the schema and load
python -m nafdac_source.verify_postgres           # check the round trip
```

The loader creates `nafdac_products` (current), `nafdac_product_history`
(superseded/withdrawn) and `nafdac_load_runs` (provenance). Rows that leave the
CSV are moved to history rather than deleted, so "this used to be registered"
stays answerable. Re-running is safe: the schema is `IF NOT EXISTS` throughout
and a row already in history is not re-archived.

Never commit the connection string. It is gitignored in every `.env`, and the
tools log only the host and database name.

### Configuration

```
NAFDAC_ENABLED=true                # master switch
NAFDAC_DATABASE_URL=               # the remote PostgreSQL URL
NAFDAC_CSV_PATH=                   # defaults to nafdac_scraper/nafdac_greenbook.csv
NAFDAC_XLSX_PATH=                  # defaults to nafdac_scraper/nafdac_greenbook.xlsx
NAFDAC_SHOW_DATA=true              # /verify shows the full table
NAFDAC_MAX_SEARCH_RESULTS=10
```

### Which messages get verified

A message is verified when it mentions NAFDAC, or when it contains a number the
greenbook actually knows. So "Is A4-100160 genuine?" and a bare `A4-100160` are
checked, while "what is the weather", "I take 500mg of ibuprofen" and ordinary
Hausa/Yoruba/Igbo conversation are left alone. The number is only tested
against the data, which is what stops a date or a dose from being treated as a
registration number.

The model is told to report the verified values and to say a field is not
recorded rather than filling the gap. It is never asked whether a product is
real, because that is a question about data, not language.

## 14. Files

```
confamAi/
├── .env                             # shared client configuration
├── .env.example                     # the same, with placeholders
├── n_atlas_client/                  # THIS FOLDER — the laptop client
│   ├── __init__.py
│   ├── __main__.py                  # python -m n_atlas_client
│   ├── main.py                      # entry point
│   ├── config.py                    # configuration
│   ├── prompt.py                    # the Confam AI system prompt
│   ├── api.py                       # NAtlasClient (API client)
│   ├── language.py                  # /hausa /yoruba /igbo /english /auto modes
│   ├── nafdac.py                    # NAFDAC interception for chat
│   ├── nafdac_source.py             # locates the shared nafdac_source
│   ├── conversation.py              # local memory + context trimming
│   ├── storage.py                   # JSON saving/loading
│   ├── terminal.py                  # banner, REPL, commands
│   ├── tests.py                     # --test suite
│   ├── n_atlas.py                   # standalone single-file client
│   ├── requirements.txt
│   ├── README.md
│   └── .env                         # gitignored
├── nafdac_source/                   # the greenbook lookup, shared
│   ├── __init__.py                  # public API
│   ├── extract.py                   # NRN extraction and normalisation
│   ├── records.py                   # the ProductRecord model
│   ├── sources.py                   # CSV, XLSX and PostgreSQL sources
│   ├── verify.py                    # verify_nafdac_product()
│   ├── formatting.py                # terminal and prompt rendering
│   ├── schema.sql                   # the PostgreSQL schema
│   ├── load_postgres.py             # CSV -> PostgreSQL loader
│   ├── verify_postgres.py           # checks the database path
│   ├── tests.py                     # 222 checks, no network needed
│   └── requirements.txt
├── n_atlas_server/                  # the Kaggle half
│   ├── readme.md
│   ├── natlas_notebook.py           # single source of truth (17 cells)
│   ├── natlas_kaggle.ipynb          # build artefact — upload this to Kaggle
│   ├── verify_nafdac.py             # checks Cell 18 without a GPU
│   ├── requirements.txt
│   └── kaggleserver.py, apiserver.py, kaggle.py, ...
├── nafdac_scraper/                  # where the greenbook data comes from
│   ├── nafdac_greenbook.csv         # 8930 products
│   └── nafdac_greenbook.xlsx
└── conversations/                   # saved chats (gitignored)
```

`nafdac_source/` sits at the project root rather than inside the client because
the Kaggle server imports the same package. Keeping one implementation is what
stops the two from disagreeing about whether a product is authorised.

`n_atlas.py` is a **standalone script, not a package module**. It duplicates
the client in one file on purpose, imports nothing from this package, and is
never loaded by `python -m n_atlas_client`. It does import `nafdac_source`
from the project root, and degrades to a plain chat client if that is missing.
Edit one or the other — they do not share code and can drift.

The modular layout keeps the API client, conversation manager, terminal UI,
configuration and storage separate so a GUI can be added later without
touching the API layer. `n_atlas_server/` shares no code with the client apart
from `nafdac_source/`, and must never import anything else from the laptop.

## 15. Development rule

This client only talks to the endpoints the Kaggle server actually exposes:
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. It assumes
nothing else. If the tunnel URL hasn't been provided yet, the expected
result is a clean "server not reachable" message — not a fabricated URL.