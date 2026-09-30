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
            | HTTPS (OpenAI-compatible API + Bearer key)
            |
            ▼
   Secure Tunnel URL     <- the real URL goes in .env (N_ATLAS_BASE_URL)
            |
            ▼
      KAGGLE SERVER (FastAPI)
            |
            ▼
         N-ATLaS
            |
            ▼
        Tesla T4 GPU
```

The laptop only needs Python 3.10+ plus a few small libraries.
It does NOT need torch, transformers, bitsandbytes, CUDA, an NVIDIA GPU,
or any Hugging Face model download. The entire model stays on Kaggle.

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
| `/exit`    | Quit the client                               |

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
between English, Hausa, Igbo and Yoruba freely in one conversation.

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

## 11. Troubleshooting

| Symptom                                    | Fix                                                    |
|--------------------------------------------|--------------------------------------------------------|
| `Missing required configuration: N_ATLAS_BASE_URL, N_ATLAS_API_KEY` | Edit `n_atlas_client/.env` (or the project-root `.env`) and fill in the real values. |
| `Unable to reach the N-ATLaS server...`    | Server is offline or the tunnel is down. Check Kaggle, the tunnel, your network. |
| `Authentication failed...`                 | The key in `.env` doesn't match the server's key.       |
| `The remote N-ATLaS GPU ran out of memory` | Shorten the conversation (`/new`), lower `N_ATLAS_MAX_TOKENS`, or ask for a shorter reply. |
| Works on `--test` but errors in chat       | Check you're sending `/`-free text; test commands are local only. |
| Picked up the wrong URL after moving files | Two `.env` files exist (project root and this folder). Run from the project root, or delete one. |

## 12. Security notes

- The client never executes code returned by the model. Code blocks are
  only displayed.
- Chat input is never interpreted as a URL to connect to. The only endpoint
  used is `N_ATLAS_BASE_URL` from configuration.
- Conversation history is local. Only the messages needed for the current
  request leave the laptop.

## 13. Files

```
confamAi/
├── .env                             # shared client configuration
├── n_atlas_client/                  # THIS FOLDER — the laptop client
│   ├── __init__.py
│   ├── __main__.py                  # python -m n_atlas_client
│   ├── main.py                      # entry point
│   ├── config.py                    # configuration
│   ├── api.py                       # NAtlasClient (API client)
│   ├── conversation.py              # local memory + context trimming
│   ├── storage.py                   # JSON saving/loading
│   ├── terminal.py                  # banner, REPL, commands
│   ├── tests.py                     # --test suite
│   ├── n_atlas.py                   # standalone single-file client
│   ├── requirements.txt
│   ├── README.md
│   └── .env                         # gitignored
├── n_atlas_server/                  # the Kaggle half — not imported here
│   ├── readme.md
│   ├── natlas_notebook.py           # single source of truth (16 cells)
│   ├── natlas_kaggle.ipynb          # build artefact — upload this to Kaggle
│   ├── requirements.txt
│   └── kaggleserver.py, apiserver.py, kaggle.py, ...
└── conversations/                   # saved chats (gitignored)
```

`n_atlas.py` is a **standalone script, not a package module**. It duplicates
the client in one file on purpose, imports nothing from this package, and is
never loaded by `python -m n_atlas_client`. Edit one or the other — they do
not share code and can drift.

The modular layout keeps the API client, conversation manager, terminal UI,
configuration and storage separate so a GUI can be added later without
touching the API layer. `n_atlas_server/` is self-contained: it shares no code
with the client and must never be imported from the laptop.

## 14. Development rule

This client only talks to the endpoints the Kaggle server actually exposes:
`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. It assumes
nothing else. If the tunnel URL hasn't been provided yet, the expected
result is a clean "server not reachable" message — not a fabricated URL.