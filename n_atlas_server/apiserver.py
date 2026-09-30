"""N-ATLaS OpenAI-compatible API server — standalone / Kaggle script.

Starts the full stack: the N-ATLaS 4-bit engine from kaggleserver.py plus the
FastAPI app, authentication and endpoints (Cells 8-10 of natlas_notebook.py),
bound to 0.0.0.0:8000 in a background thread.

Run it:

    python apiserver.py

    # or point it somewhere else
    N_ATLAS_PORT=9000 python apiserver.py

Configure the bearer key before exposing it:

    N_ATLAS_API_KEY='<strong secret>' python apiserver.py

Health:  GET  http://127.0.0.1:8000/health
Chat:    POST http://127.0.0.1:8000/v1/chat/completions
Docs:    GET  http://127.0.0.1:8000/docs

Note: no API key is hard-coded here. If N_ATLAS_API_KEY is unset, a random
ephemeral key is generated for the lifetime of the process and written to
<workdir>/natlas_api_key.txt with mode 600. That is for local development only.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggleserver
import notebook_cells

_API = notebook_cells.exec_range(8, 10, vars(kaggleserver))

app = _API["app"]
start_server = _API["start_server"]
stop_server = _API["stop_server"]
is_server_running = _API["is_server_running"]
serve_forever = _API["serve_forever"]
CONFIG = _API["CONFIG"]

__all__ = [
    "app",
    "start_server",
    "stop_server",
    "is_server_running",
    "serve_forever",
    "CONFIG",
    "main",
]


def main(host: Optional[str] = None, port: Optional[int] = None) -> int:
    start_server(host=host, port=port)
    print(
        f"[apiserver] serving {CONFIG.served_model_name} on "
        f"http://{host or CONFIG.host}:{port or CONFIG.port} — Ctrl+C to stop."
    )
    try:
        serve_forever()
    except KeyboardInterrupt:
        print("\n[apiserver] shutting down.")
    finally:
        stop_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
