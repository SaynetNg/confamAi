"""N-ATLaS laptop client package.

A thin OpenAI-compatible remote client for the N-ATLaS server running on
Kaggle (Tesla T4). The laptop never runs the model — it only talks to the
remote OpenAI-compatible API over an HTTPS tunnel.

Modules:
    config       - environment / .env configuration
    conversation - local conversation memory and context trimming
    storage      - optional local JSON conversation saving
    api          - NAtlasClient API wrapper (health, models, chat, stream)
    nafdac       - NAFDAC greenbook verification and the /verify, /search,
                   /nafdac commands
    nafdac_source - locates the shared nafdac_source package at the project root
    tests        - --test suite (connection, health, models, languages, memory,
                   NAFDAC verification)
    terminal     - terminal banner, REPL and local commands
    main         - entrypoint with --debug and --test flags

`n_atlas.py` in this folder is a standalone single-file client. It imports
nothing from this package and is not loaded by `python -m n_atlas_client`.
"""

from .api import (  # noqa: F401
    NAtlasClient,
    NAtlasError,
    AuthenticationFailure,
    ConnectionFailure,
    EmptyResponse,
    ForbiddenFailure,
    GPUExhaustedFailure,
    NotFoundFailure,
    RateLimitFailure,
    ServerFailure,
    StreamingUnsupported,
    TimeoutFailure,
    ValidationFailure,
)
from .config import Config, ConfigError, load_config  # noqa: F401
from .conversation import Conversation, estimate_tokens  # noqa: F401
from .nafdac import NafdacInterceptor  # noqa: F401

__version__ = "1.0.0"