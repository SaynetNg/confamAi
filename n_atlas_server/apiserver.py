"""N-ATLaS OpenAI-compatible API server — standalone / Kaggle script.

Starts the full stack: the N-ATLaS 4-bit engine from kaggleserver.py plus the
FastAPI app, authentication and endpoints (Cells 8-10 of natlas_notebook.py),
the NAFDAC greenbook lookup (Cell 17), and uvicorn bound to 0.0.0.0:8000 in a
background thread.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# nafdac_source lives at the project root, one level up. Cell 17 imports it, so
# the root has to be importable before the cells are executed.
sys.path.insert(0, str(HERE.parent))

import kaggleserver
import notebook_cells

_ENGINE = vars(kaggleserver)
_API = notebook_cells.exec_range(8, 10, _ENGINE)

app = _API["app"]
start_server = _API["start_server"]
stop_server = _API["stop_server"]
is_server_running = _API["is_server_running"]
serve_forever = _API["serve_forever"]
CONFIG = _API["CONFIG"]

# Cell 17 registers the NAFDAC routes and fills NAFDAC_HOOKS. It is executed
# after Cells 8-10 because it needs the app those cells build, and it must run
# before the server starts so the routes exist when the first request arrives.
_NAFDAC = notebook_cells.exec_range(17, 17, _API)

nafdac_status = _NAFDAC.get("_nafdac_status")
nafdac_available = bool(_NAFDAC.get("NAFDAC_AVAILABLE"))

__all__ = [
    "app",
    "start_server",
    "stop_server",
    "is_server_running",
    "serve_forever",
    "CONFIG",
    "nafdac_available",
    "nafdac_status",
    "main",
]


def main(host: Optional[str] = None, port: Optional[int] = None) -> int:
    start_server(host=host, port=port)
    print(
        f"[apiserver] serving {CONFIG.served_model_name} on "
        f"http://{host or CONFIG.host}:{port or CONFIG.port} — Ctrl+C to stop."
    )
    if nafdac_available:
        _status = nafdac_status() or {}
        _usable = [s["name"] for s in _status.get("sources", []) if s.get("usable")]
        print(
            "[apiserver] nafdac: on, sources reachable: %s"
            % (", ".join(_usable) or "none")
        )
    else:
        print("[apiserver] nafdac: unavailable (chat still works)")
    try:
        serve_forever()
    except KeyboardInterrupt:
        print("\n[apiserver] shutting down.")
    finally:
        stop_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
