"""Configuration loading and validation for the N-ATLaS laptop client.

Everything comes from environment variables or a local .env file. The real
tunnel URL and API key are supplied by the user — they are never hard-coded
in this codebase.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .prompt import resolve_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _env_str(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(key, default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(str(os.environ.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- connection -----------------------------------------------------
    base_url: str = ""            # N_ATLAS_BASE_URL   (real tunnel URL, /v1)
    api_key: str = ""             # N_ATLAS_API_KEY    (never printed/logged)
    model: str = "n-atlas"        # N_ATLAS_MODEL
    timeout: float = 180.0        # N_ATLAS_TIMEOUT    (seconds)
    max_tokens: int = 512         # N_ATLAS_MAX_TOKENS
    temperature: float = 0.1      # N_ATLAS_TEMPERATURE (server default; low by design)
    stream: bool = True           # N_ATLAS_STREAM     (auto-fallback)

    # --- language --------------------------------------------------------
    # One of auto, en, ha, yo, ig. `auto` sends nothing and lets the server
    # detect the language of each message, which is how the client has always
    # behaved. A code pins every reply to that language until /hausa, /yoruba,
    # /igbo, /english or /auto changes it. See language.py.
    language: str = "auto"       # N_ATLAS_LANGUAGE

    # --- client-side context management ---------------------------------
    max_conversation_messages: int = 40   # N_ATLAS_MAX_CONVERSATION_MESSAGES
    max_conversation_tokens: int = 3500   # N_ATLAS_MAX_CONVERSATION_TOKENS
    max_message_chars: int = 6000         # N_ATLAS_MAX_MESSAGE_CHARS

    # --- retries ---------------------------------------------------------
    max_retries: int = 2          # N_ATLAS_MAX_RETRIES

    # --- local storage ---------------------------------------------------
    save_conversations: bool = True            # N_ATLAS_SAVE_CONVERSATIONS
    conversations_dir: str = "conversations"   # N_ATLAS_CONVERSATIONS_DIR

    # --- system prompt ---------------------------------------------------
    # The ConfamAI prompt from confam_prompt.txt (or the constant in
    # prompt.py when that file is absent) is sent unless
    # N_ATLAS_SYSTEM_PROMPT / ..._FILE say otherwise, or the prompt is
    # switched off with N_ATLAS_SYSTEM_PROMPT=none.
    system_prompt: str = ""       # N_ATLAS_SYSTEM_PROMPT / ..._FILE

    # --- NAFDAC verification ---------------------------------------------
    # When on, a message that carries a NAFDAC number is answered from the
    # greenbook data before N-ATLaS is asked, and the verified facts are passed
    # to the model so it cannot invent a product name or an approval date.
    nafdac_enabled: bool = True          # NAFDAC_ENABLED
    nafdac_database_url: str = ""        # NAFDAC_DATABASE_URL
    nafdac_csv_path: str = ""            # NAFDAC_CSV_PATH
    nafdac_xlsx_path: str = ""           # NAFDAC_XLSX_PATH
    nafdac_show_data: bool = True        # NAFDAC_SHOW_DATA
    nafdac_max_search_results: int = 10  # NAFDAC_MAX_SEARCH_RESULTS

    # The server is the authoritative NAFDAC layer: it extracts intent, applies
    # the intent gate and only then queries the greenbook. When this is on, the
    # client never searches locally on its own initiative, because a second
    # retrieval authority would decide for itself that a message was about a
    # product — the exact failure the server's gate exists to prevent. Local data
    # is still used as a fallback when no server is reachable, and on explicit
    # commands like /verify, which are the user asking directly.
    nafdac_server_authoritative: bool = True  # NAFDAC_SERVER_AUTHORITATIVE

    # --- diagnostics ------------------------------------------------------
    debug: bool = False           # --debug flag

    def server_root(self) -> str:
        """Derive the server root from a base URL that normally ends in /v1."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base[: -len("/v1")].rstrip("/")
        return base

    def health_url(self) -> str:
        return self.server_root() + "/health"


def load_config(env_file: str | None = None, debug: bool = False) -> Config:
    """Load configuration from environment variables and a .env file.

    Real environment variables always win over .env entries.
    Search order for the .env file:
      1. the explicit env_file argument
      2. the current working directory
      3. the folder containing this package / the single-file client
    """
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(Path.cwd() / ".env", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    system_prompt = resolve_system_prompt(
        _env_str("N_ATLAS_SYSTEM_PROMPT"),
        _env_str("N_ATLAS_SYSTEM_PROMPT_FILE"),
    )

    return Config(
        base_url=_env_str("N_ATLAS_BASE_URL"),
        api_key=_env_str("N_ATLAS_API_KEY"),
        model=_env_str("N_ATLAS_MODEL", "n-atlas"),
        timeout=max(1.0, _env_float("N_ATLAS_TIMEOUT", 180.0)),
        max_tokens=max(1, _env_int("N_ATLAS_MAX_TOKENS", 512)),
        temperature=_env_float("N_ATLAS_TEMPERATURE", 0.1),
        stream=_env_bool("N_ATLAS_STREAM", True),
        language=_env_str("N_ATLAS_LANGUAGE", "auto") or "auto",
        max_conversation_messages=max(4, _env_int("N_ATLAS_MAX_CONVERSATION_MESSAGES", 40)),
        max_conversation_tokens=max(256, _env_int("N_ATLAS_MAX_CONVERSATION_TOKENS", 3500)),
        max_message_chars=max(256, _env_int("N_ATLAS_MAX_MESSAGE_CHARS", 6000)),
        max_retries=max(0, _env_int("N_ATLAS_MAX_RETRIES", 2)),
        save_conversations=_env_bool("N_ATLAS_SAVE_CONVERSATIONS", True),
        conversations_dir=_env_str("N_ATLAS_CONVERSATIONS_DIR", "conversations"),
        system_prompt=system_prompt,
        nafdac_enabled=_env_bool("NAFDAC_ENABLED", True),
        nafdac_database_url=_env_str("NAFDAC_DATABASE_URL"),
        nafdac_csv_path=_env_str("NAFDAC_CSV_PATH"),
        nafdac_xlsx_path=_env_str("NAFDAC_XLSX_PATH"),
        nafdac_show_data=_env_bool("NAFDAC_SHOW_DATA", True),
        nafdac_max_search_results=max(1, _env_int("NAFDAC_MAX_SEARCH_RESULTS", 10)),
        nafdac_server_authoritative=_env_bool(
            "NAFDAC_SERVER_AUTHORITATIVE", True
        ),
        debug=debug,
    )


def apply_nafdac_env(cfg: Config) -> None:
    """Push the NAFDAC settings in `cfg` into the environment.

    `nafdac_source` reads its configuration from the environment, so the same
    package works identically on the laptop and on Kaggle. Calling this before
    the first lookup is what lets one set of variables drive both.
    """
    if cfg.nafdac_database_url:
        os.environ["NAFDAC_DATABASE_URL"] = cfg.nafdac_database_url
    if cfg.nafdac_csv_path:
        os.environ["NAFDAC_CSV_PATH"] = cfg.nafdac_csv_path
    if cfg.nafdac_xlsx_path:
        os.environ["NAFDAC_XLSX_PATH"] = cfg.nafdac_xlsx_path


def validate(cfg: Config) -> None:
    """Fail fast when required configuration is missing."""
    missing = []
    if not cfg.base_url:
        missing.append("N_ATLAS_BASE_URL")
    if not cfg.api_key:
        missing.append("N_ATLAS_API_KEY")
    if missing:
        raise ConfigError(
            "Missing required configuration: %s.\n"
            "Copy .env.example to .env and fill in the real tunnel URL and API "
            "key, then start the client again."
            % ", ".join(missing)
        )
    if "http://" not in cfg.base_url and "https://" not in cfg.base_url:
        raise ConfigError(
            "N_ATLAS_BASE_URL must be a full URL such as "
            "https://REAL-TUNNEL-DOMAIN/v1 "
            "(got %r). Use the real tunnel URL, not localhost." % cfg.base_url
        )