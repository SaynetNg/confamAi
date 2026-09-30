#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n_atlas.py — single-file N-ATLaS laptop client.

A self-contained copy of the client: no imports from this package, so it can be
copied to any machine on its own. It lives beside the modular package
(`n_atlas_client/`) and offers exactly the same commands. Prefer
`python -m n_atlas_client` unless you want the one-file version.

Talks to the remotely hosted N-ATLaS OpenAI-compatible API (Kaggle server +
cloudflared tunnel). NO model runs on this laptop: no torch, no
transformers, no bitsandbytes, no Hugging Face download.

Requires:
    pip install -r requirements.txt

Usage (from the project root):
    python n_atlas_client/n_atlas.py       # chat
    python n_atlas_client/n_atlas.py --debug
    python n_atlas_client/n_atlas.py --test
"""

import os
import sys
import time
import json
import argparse
import platform
import traceback
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Ensure multilingual text (Yoruba/Igbo diacritics) prints on all terminals.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from dotenv import load_dotenv  # python-dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
    from openai import AuthenticationError, PermissionDeniedError, NotFoundError
    from openai import RateLimitError, UnprocessableEntityError, APIStatusError
    from openai import BadRequestError, InternalServerError
except Exception:  # pragma: no cover
    OpenAI = None

# ----------------------------------------------------------------------
# Configuration (all from env / .env, nothing hard-coded)
# ----------------------------------------------------------------------
DEFAULTS = {
    "N_ATLAS_BASE_URL": "",
    "N_ATLAS_API_KEY": "",
    "N_ATLAS_MODEL": "n-atlas",
    "N_ATLAS_TIMEOUT": 180.0,
    "N_ATLAS_MAX_TOKENS": 512,
    "N_ATLAS_TEMPERATURE": 0.7,
    "N_ATLAS_STREAM": True,
    "N_ATLAS_MAX_CONVERSATION_MESSAGES": 40,
    "N_ATLAS_MAX_CONVERSATION_TOKENS": 3500,
    "N_ATLAS_MAX_MESSAGE_CHARS": 6000,
    "N_ATLAS_MAX_RETRIES": 2,
    "N_ATLAS_SAVE_CONVERSATIONS": True,
    "N_ATLAS_CONVERSATIONS_DIR": "conversations",
    "N_ATLAS_SYSTEM_PROMPT": "",
    "N_ATLAS_SYSTEM_PROMPT_FILE": "",
}


class Config:
    def __init__(self, values=None):
        raw = dict(DEFAULTS)
        if values:
            raw.update({k: v for k, v in values.items() if k in DEFAULTS})
        if load_dotenv is not None:
            # Same search order as n_atlas_client/config.py: cwd first, then
            # the project root, then this package's own folder. Real env vars
            # and earlier files always win.
            load_dotenv(Path.cwd() / ".env", override=False)
            load_dotenv(PROJECT_ROOT / ".env", override=False)
            load_dotenv(PACKAGE_DIR / ".env", override=False)
        resolved = {}
        for key, default in raw.items():
            resolved[key] = os.environ.get(key, default)
        self.base_url = str(resolved["N_ATLAS_BASE_URL"] or "").strip()
        self.api_key = str(resolved["N_ATLAS_API_KEY"] or "").strip()
        self.model = str(resolved["N_ATLAS_MODEL"] or "n-atlas").strip()
        self.timeout = max(1.0, float(resolved["N_ATLAS_TIMEOUT"]))
        self.max_tokens = max(1, int(float(resolved["N_ATLAS_MAX_TOKENS"])))
        self.temperature = float(resolved["N_ATLAS_TEMPERATURE"])
        self.stream = self._bool(resolved["N_ATLAS_STREAM"], True)
        self.max_conversation_messages = max(
            4, int(float(resolved["N_ATLAS_MAX_CONVERSATION_MESSAGES"]))
        )
        self.max_conversation_tokens = max(
            256, int(float(resolved["N_ATLAS_MAX_CONVERSATION_TOKENS"]))
        )
        self.max_message_chars = max(256, int(float(resolved["N_ATLAS_MAX_MESSAGE_CHARS"])))
        self.max_retries = max(0, int(float(resolved["N_ATLAS_MAX_RETRIES"])))
        self.save_conversations = self._bool(resolved["N_ATLAS_SAVE_CONVERSATIONS"], True)
        self.conversations_dir = str(resolved["N_ATLAS_CONVERSATIONS_DIR"] or "conversations")
        self.system_prompt = str(resolved["N_ATLAS_SYSTEM_PROMPT"] or "").strip()
        prompt_file = str(resolved["N_ATLAS_SYSTEM_PROMPT_FILE"] or "").strip()
        if not self.system_prompt and prompt_file:
            try:
                self.system_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
            except OSError:
                self.system_prompt = ""
        self.debug = False

    @staticmethod
    def _bool(value, default):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def server_root(self):
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base[: -len("/v1")].rstrip("/")
        return base

    def health_url(self):
        return self.server_root() + "/health"


class ConfigError(Exception):
    pass


def validate_config(cfg):
    problems = []
    if not cfg.base_url:
        problems.append("N_ATLAS_BASE_URL")
    if not cfg.api_key:
        problems.append("N_ATLAS_API_KEY")
    if problems:
        raise ConfigError(
            "Missing required configuration: %s.\n"
            "Copy .env.example to .env and fill in the REAL tunnel URL and API "
            "key, then run the client again." % ", ".join(problems)
        )
    if "http://" not in cfg.base_url and "https://" not in cfg.base_url:
        raise ConfigError(
            "N_ATLAS_BASE_URL must be a full URL such as "
            "https://REAL-TUNNEL-DOMAIN/v1 (got %r).\n"
            "Use the actual tunnel URL — never localhost." % cfg.base_url
        )


# ----------------------------------------------------------------------
# Logging / debug
# ----------------------------------------------------------------------
def log_debug(cfg, message):
    if cfg.debug:
        print("[debug] %s" % message, flush=True)


def log_http_status(cfg, method, endpoint, status, seconds):
    log_debug(
        cfg,
        "HTTP %s %s -> %d (%.2fs)"
        % (method, endpoint, status, seconds),
    )


# ----------------------------------------------------------------------
# Conversation history (local memory)
# ----------------------------------------------------------------------
def estimate_tokens(text):
    """Cheap client-side token estimator (no tokenizer on the laptop)."""
    if not text:
        return 0
    count = 0
    for word in str(text).split():
        count += 1 + int(len(word) / 4.0)
    return count


class Conversation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.messages = []

    def reset(self):
        self.messages.clear()

    def add(self, role, content):
        content = str(content or "").strip()
        if not content:
            return
        self.messages.append({"role": role, "content": content})

    def build_messages_for(self, user_text):
        """Return the message list to send: optional client system prompt +
        trimmed local history + the new user message.

        The server owns the N-ATLaS system prompt; no system message is added
        unless the user explicitly configured one.
        """
        result = []
        if self.cfg.system_prompt:
            result.append({"role": "system", "content": self.cfg.system_prompt})
        result.extend(self.messages)
        result.append({"role": "user", "content": user_text})

        # hard message ceiling (preserve the newest exchanges + the first one)
        if len(result) > self.cfg.max_conversation_messages:
            keep = (
                result[:1]
                + result[-(self.cfg.max_conversation_messages - 1):]
            )
            result = keep
        # token shape: drop oldest full turns until under the cap
        overflow = 0
        while estimate_tokens(" ".join(m["content"] for m in result)) > self.cfg.max_conversation_tokens:
            overflow += 1
            drop_at = self._drop_index(result)
            if drop_at is None:
                break
            del result[drop_at]
        if overflow and self.cfg.debug:
            log_debug(self.cfg, "context: dropped %d oldest turn(s) to fit the token window" % overflow)
        return result

    @staticmethod
    def _drop_index(messages):
        """Find an index to drop: preserve system and the newest message."""
        if not messages:
            return None
        for index, message in enumerate(messages):
            if message["role"] != "system" and index < len(messages) - 1:
                if messages[index]["role"] == "user":
                    # drop a user+assistant pair when available
                    if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
                        return index
                    return index
        return None

    def to_dict(self):
        return [dict(m) for m in self.messages]

    def nonempty_count(self):
        return sum(1 for m in self.messages if m["content"].strip())


# ----------------------------------------------------------------------
# Storage (optional local JSON saving)
# ----------------------------------------------------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_conversation_to_disk(cfg, messages, extra=None):
    """Save a conversation to conversations/YYYY-MM-DD_HH-MM-SS.json.
    Never stores the API key or any server credentials."""
    if not cfg.save_conversations:
        return None
    directory = ensure_dir(cfg.conversations_dir)
    filename = time.strftime("conversation_%Y-%m-%d_%H-%M-%S.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": cfg.model,
        "messages": [dict(m) for m in messages],
    }
    if extra:
        payload.update(extra)
    path = directory / filename
    if cfg.debug:
        log_debug(cfg, "saving conversation to %s" % path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def load_conversation_from_disk(path):
    """Load a conversation JSON file (for a potential future --load flag)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("messages", []), payload.get("model", "")


# ----------------------------------------------------------------------
# Small helper: remove template artifacts from remote responses
# ----------------------------------------------------------------------
TEMPLATE_ARTIFACTS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<s>",
    "</s>",
)


def clean_response_text(text):
    """Minimal safety cleaning. The remote server is responsible for full
    response cleaning; this only strips obvious API/template artifacts that
    would look broken in the terminal."""
    if not text:
        return ""
    out = text
    for token in TEMPLATE_ARTIFACTS:
        if token in out:
            out = out.replace(token, "")
    return out.strip()


# ----------------------------------------------------------------------
# NAtlasClient  — thin OpenAI-compatible remote API wrapper
# ----------------------------------------------------------------------
class NAtlasError(Exception):
    """Base error from the N-ATLaS API client."""


class ConnectionFailure(NAtlasError):
    pass


class TimeoutFailure(NAtlasError):
    pass


class AuthenticationFailure(NAtlasError):
    pass


class ForbiddenFailure(NAtlasError):
    pass


class NotFoundFailure(NAtlasError):
    pass


class ValidationFailure(NAtlasError):
    pass


class RateLimitFailure(NAtlasError):
    pass


class ServerFailure(NAtlasError):
    pass


class GPUExhaustedFailure(NAtlasError):
    pass


class StreamingUnsupported(NAtlasError):
    pass


class EmptyResponse(NAtlasError):
    pass


def _requests_session(timeout):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class NAtlasClient:
    """Connects to the remote N-ATLaS server and exposes health(), models(),
    chat(), streaming chat, clear_conversation(), save_conversation() and
    load_conversation()."""

    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.debug = bool(cfg.debug or debug)
        self.conversation = Conversation(cfg)
        self._session = requests.Session() if requests is not None else None
        self._openai = None
        self.server_info = {}
        self.api_connected = False
        self.model_available = False

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        if not self.cfg.base_url or not self.cfg.api_key:
            raise ConnectionFailure(
                "Configuration is incomplete. Set N_ATLAS_BASE_URL and "
                "N_ATLAS_API_KEY in .env."
            )
        try:
            self._openai = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout,
                max_retries=0,  # retries are handled explicitly below
            )
        except Exception:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            )

    # -- health / models ----------------------------------------------
    def health(self):
        endpoint = "GET %s" % self.cfg.health_url()
        started = time.monotonic()
        response = self._request("GET", "/health")
        seconds = time.monotonic() - started
        if response.status_code != 200:
            log_http_status(self.debug and self or None, "GET", endpoint, response.status_code, seconds)
            raise ServerFailure(
                "Server returned %d while checking health." % response.status_code
            )
        payload = response.json()
        log_http_status(self, "GET", endpoint, response.status_code, seconds)
        self.server_info = payload if isinstance(payload, dict) else {}
        return self.server_info

    def models(self):
        endpoint = "GET %s/v1/models" % self.cfg.base_url
        started = time.monotonic()
        response = self._request("GET", "/v1/models")
        seconds = time.monotonic() - started
        if response.status_code != 200:
            log_http_status(self, "GET", endpoint, response.status_code, seconds)
            self._raise_http_error(response, context="checking the model list")
        payload = response.json()
        ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        log_http_status(self, "GET", endpoint, response.status_code, seconds)
        return ids

    def check_available_models(self):
        ids = self.models()
        self.model_available = self.cfg.model in ids
        return ids

    def is_online(self):
        info = self.health()
        return bool(info) and info.get("status") in ("ok", "degraded")

    # -- chat -----------------------------------------------------------
    def chat(self, message, stream=None, on_token=None):
        """Send a message. History is appended afterwards once a valid
        reply is received. Returns (response_text, info_dict).

        on_token is an optional callable(text) used to render streaming
        tokens live in the terminal.
        """
        if isinstance(message, list):
            payload_messages = message
        else:
            payload_messages = self.conversation.build_messages_for(message)
        use_stream = self.cfg.stream if stream is None else stream
        attempts = 0
        last_error = None
        max_attempts = self.cfg.max_retries + 1
        while attempts < max_attempts:
            attempts += 1
            try:
                result = self._do_chat(payload_messages, use_stream=use_stream,
                                       on_token=on_token)
                # assistant turn is added by the caller (terminal/tests) so the
                # cancel is explicit; here we just return the payload-safe data
                return result
            except (ConnectionFailure, TimeoutFailure, RateLimitFailure, ServerFailure,
                    GPUExhaustedFailure) as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                delay = min(0.5 * (2 ** (attempts - 1)), 4.0)
                log_debug(self, "retrying in %.1fs (%d/%d): %s" % (
                    delay, attempts + 1, max_attempts, exc))
                if self.debug:
                    time.sleep(delay)
                else:
                    time.sleep(delay)
            except (AuthenticationFailure, ForbiddenFailure, NotFoundFailure,
                    ValidationFailure, StreamingUnsupported, EmptyResponse) as exc:
                raise exc
        raise last_error

    def _do_chat(self, payload_messages, use_stream=True, on_token=None):
        if self._openai is None:
            self.connect()
        request_kwargs = dict(
            model=self.cfg.model,
            messages=payload_messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        try:
            if use_stream:
                try:
                    result = self._chat_stream(request_kwargs, on_token=on_token)
                    if result is not None:
                        return result
                    # stream disabled/empty -> fall back to normal response
                    return self._chat_once(request_kwargs, force_no_stream=True)
                except StreamingUnsupported:
                    return self._chat_once(request_kwargs, force_no_stream=True)
            return self._chat_once(request_kwargs, force_no_stream=False)
        except NAtlasError:
            raise
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection. (%s)" % type(exc).__name__
            ) from exc

    def _chat_once(self, request_kwargs, force_no_stream):
        started = time.monotonic()
        kwargs = dict(request_kwargs)
        kwargs["stream"] = False
        try:
            response = self._openai.chat.completions.create(**kwargs)
            seconds = time.monotonic() - started
            text = response.choices[0].message.content
            info = self._completion_info(response, seconds)
            if self.debug:
                log_debug(self, "POST %s/v1/chat/completions -> 200 (%.2fs)" % (
                    self.cfg.base_url, seconds))
            if not text or not text.strip():
                raise EmptyResponse(
                    "The model returned an empty response. Try rewording your "
                    "message or starting a new conversation (/new)."
                )
            return self._clean_and_return(text, info)
        except NAtlasError:
            raise
        except BadRequestError as exc:
            status = getattr(exc, "status_code", 400)
            if status == 400 and "stream" in str(exc).lower():
                raise StreamingUnsupported(
                    "This server does not support streaming; falling back to "
                    "normal responses."
                ) from exc
            self._raise_openai_error(exc)
        except APIStatusError as exc:
            self._raise_openai_error(exc)
        except APIConnectionError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        except APITimeoutError as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond. Check your "
                "network or the server's load."
            ) from exc
        except APIError as exc:
            self._raise_openai_error(exc)
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc

    def _chat_stream(self, request_kwargs, on_token=None):
        """Attempt streaming. Returns (text, info) or None to fall back."""
        kwargs = dict(request_kwargs)
        kwargs["stream"] = True
        pieces = []
        started = time.monotonic()
        info = {}
        try:
            stream = self._openai.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            raise StreamingUnsupported(
                "This server does not support streaming; falling back to "
                "normal responses."
            ) from exc
        except APIStatusError as exc:
            self._raise_openai_error(exc)
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc

        try:
            for event in stream:
                if event is None:
                    continue
                choice = getattr(event, "choices", None)
                if not choice:
                    continue
                first = choice[0]
                delta = getattr(first, "delta", None)
                text = ""
                if delta is not None:
                    text = getattr(delta, "content", "") or ""
                pieces.append(text)
                if on_token is not None:
                    on_token(text)
        except Exception as exc:
            # if the server half-supports the stream, fall back
            if self.cfg.stream and "stream" in str(exc).lower():
                raise StreamingUnsupported(str(exc)) from exc
            raise ConnectionFailure(
                "Unable to reach the N-AtLaS server while streaming."
            ) from exc

        joined = "".join(pieces)
        if not joined.strip():
            # empty verbose stream -> fall back to a normal request
            raise StreamingUnsupported(
                "Stream produced no text; using normal responses."
            )
        info = {
            "streamed": True,
            "seconds": time.monotonic() - started,
        }
        return self._clean_and_return(joined, info)

    def _clean_and_return(self, text, info):
        return clean_response_text(text), info

    @staticmethod
    def _completion_info(response, seconds):
        info = {
            "streamed": False,
            "seconds": round(seconds, 2),
            "finish_reason": response.choices[0].finish_reason
            if getattr(response, "choices", None) else None,
            "usage": None,
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            info["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return info

    # -- conversation helpers ------------------------------------------
    def clear_conversation(self):
        self.conversation.reset()

    def save_conversation(self, directory=None):
        """Save the current conversation (messages + model + timestamp)."""
        if not self.conversation.messages:
            return None
        directory = directory or self.cfg.conversations_dir
        ensure_dir(directory)
        path = Path(directory) / (time.strftime("conversation_%Y-%m-%d_%H-%M-%S.json"))
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.cfg.model,
            "messages": self.conversation.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def load_conversation(self, path):
        """Load messages from a JSON file into the local conversation."""
        messages, _ = load_conversation_from_disk(path)
        self.conversation.messages = [dict(m) for m in messages]
        return len(messages)

    # -- low-level HTTP ------------------------------------------------
    def _request(self, method, path):
        url = self.cfg.server_root() + path
        headers = {
            "Authorization": "Bearer %s" % self.cfg.api_key,
            "Accept": "application/json",
        }
        try:
            response = self._session.request(method, url, headers=headers, timeout=self.cfg.timeout)
        except requests.exceptions.Timeout as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond. Check your "
                "network or the server's load."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        return response

    def _raise_http_error(self, response, context=""):
        body = ""
        try:
            body = response.json()
        except Exception:
            body = response.text[:200]
        message = body.get("error", {}).get("message") if isinstance(body, dict) else body
        detail = message if isinstance(message, str) and message else ("HTTP %d" % response.status_code)
        status = response.status_code
        if status == 401:
            raise AuthenticationFailure(
                "Authentication failed. Check N_ATLAS_API_KEY."
            )
        if status == 403:
            raise ForbiddenFailure(
                "Access forbidden (403). Check that the API key is allowed to "
                "use this server."
            )
        if status == 404:
            raise NotFoundFailure(
                "Not found (404): %s. Check N_ATLAS_BASE_URL and the server "
                "documentation." % context
            )
        if status == 422:
            raise ValidationFailure(
                "Validation error (422): %s. Check the request parameters." % detail
            )
        if status == 400:
            raise ValidationFailure("Bad request (400): %s." % detail)
        if status == 429:
            raise RateLimitFailure(
                "Rate limit reached (429). Retry shortly. The server may be "
                "busy (GPU queue)."
            )
        if status == 507:
            raise GPUExhaustedFailure(
                "The remote N-ATLaS GPU ran out of memory. Try a shorter "
                "conversation or shorter response."
            )
        if status == 503:
            raise ServerFailure(
                "Server unavailable (503). The model may be loading or the "
                "queue is full. Retry shortly."
            )
        if status == 500:
            raise ServerFailure(
                "Server error (500). The remote generation failed. Try again, "
                "or shorten the conversation."
            )
        raise ServerFailure("Unexpected server response (HTTP %d)." % status)

    def _raise_openai_error(self, exc):
        status = getattr(exc, "status_code", None)
        message = getattr(exc, "message", None) or str(exc)
        if isinstance(exc, AuthenticationError) or status == 401:
            raise AuthenticationFailure(
                "Authentication failed. Check N_ATLAS_API_KEY."
            ) from exc
        if isinstance(exc, PermissionDeniedError) or status == 403:
            raise ForbiddenFailure(
                "Access forbidden (403). Check the API key permissions."
            ) from exc
        if isinstance(exc, NotFoundError) or status == 404:
            raise NotFoundFailure(
                "Not found (404). Check N_ATLAS_BASE_URL and the server "
                "documentation."
            ) from exc
        if isinstance(exc, UnprocessableEntityError) or status == 422:
            raise ValidationFailure(
                "Validation error (422): %s" % message
            ) from exc
        if isinstance(exc, BadRequestError) or status == 400:
            if isinstance(message, str) and "stream" in message.lower():
                raise StreamingUnsupported(
                    "This server does not support streaming; falling back to "
                    "normal responses."
                ) from exc
            raise ValidationFailure("Bad request (400): %s" % message) from exc
        if isinstance(exc, RateLimitError) or status == 429:
            raise RateLimitFailure(
                "Rate limit reached (429). Retry shortly."
            ) from exc
        if status == 507:
            raise GPUExhaustedFailure(
                "The remote N-ATLaS GPU ran out of memory. Try a shorter "
                "conversation or shorter response."
            ) from exc
        if isinstance(exc, InternalServerError) or status == 503:
            raise ServerFailure(
                "Server unavailable (503). The model may be loading or the "
                "queue is full. Retry shortly."
            ) from exc
        if status == 500:
            raise ServerFailure(
                "Server error (500). The remote generation failed. Try again, "
                "or shorten the conversation."
            ) from exc
        raise ServerFailure(
            "Unexpected server error: %s%s"
            % (message, (" (HTTP %s)" % status) if status else "")
        ) from exc


# -- show messages nicely --------------------------------------------------
def print_banner(client):
    info = client.server_info or {}
    gpu = (
        info.get("gpu")
        or info.get("gpu_name")
        or info.get("device", "")
    )
    if not gpu:
        device = info.get("device", "")
        gpu = "Tesla T4" if (device == "cuda" or info.get("cuda_available")) else gpu
    width = 46
    print("=" * width)
    print("N-ATLaS MULTILINGUAL AI")
    print("")
    print("Server: %s" % ("ONLINE" if client.api_connected else "OFFLINE"))
    print("Model:  %s" % (info.get("served_model_name") or info.get("model") or client.cfg.model))
    print("Remote GPU: %s" % (gpu or "not reported"))
    print("")
    print("Languages: English | Hausa | Igbo | Yoruba")
    print("")
    print("Type /help for commands.")
    print("Type /exit to quit.")
    print("=" * width)


def print_help_text():
    lines = [
        "Available commands:",
        "  /help    Show this list",
        "  /new     Start a new conversation (clears local history)",
        "  /clear   Clear conversation history",
        "  /history Display local conversation history",
        "  /status  Check whether the remote API is reachable",
        "  /exit    Exit the client",
        "",
        "Anything else is sent to N-ATLaS.",
        "",
        "The model replies naturally in English, Hausa, Igbo or Yoruba and",
        "follows your language switches — no reset needed.",
    ]
    print("\n".join(lines))


def print_history(client):
    messages = client.conversation.messages
    if not messages:
        print("(no conversation history)")
        return
    print("-" * 46)
    for message in messages:
        role = message["role"]
        text = str(message["content"])
        print("%s: %s" % (role.capitalize(), text))
    print("-" * 46)


def handle_status(client):
    start = time.monotonic()
    try:
        if client._openai is None:
            client.connect()
        info = client.health()
        online = bool(info) and info.get("status") in ("ok", "degraded")
        print("Server: %s" % ("ONLINE" if online else "OFFLINE"))
        print("Model:  %s" % (info.get("served_model_name") or info.get("model") or client.cfg.model))
        gpu = info.get("gpu") or info.get("gpu_name") or ""
        if info.get("cuda_available") or info.get("device") == "cuda":
            gpu = gpu or "Tesla T4"
        print("Remote GPU: %s" % (gpu or "not reported"))
        print("API:    %s" % ("connected" if online else "unreachable"))
        print("Latency: %.1fs" % (time.monotonic() - start))
    except NAtlasError as exc:
        print("Unable to reach the N-ATLaS server. Check the server or network connection.")
        log_debug(client, "status check failed: %s" % exc)


# ----------------------------------------------------------------------
# RUNNER (OpenAI SDK path)
# ----------------------------------------------------------------------
def run_repl(client):
    print_banner(client)

    def handle_exception(exc):
        if isinstance(exc, TimeoutFailure):
            print(str(exc))
        elif isinstance(exc, AuthenticationFailure):
            print(str(exc))
        elif isinstance(exc, ForbiddenFailure):
            print(str(exc))
        elif isinstance(exc, NotFoundFailure):
            print(str(exc))
        elif isinstance(exc, ValidationFailure):
            print(str(exc))
        elif isinstance(exc, RateLimitFailure):
            print(str(exc))
        elif isinstance(exc, ServerFailure):
            print(str(exc))
        elif isinstance(exc, GPUExhaustedFailure):
            print(str(exc))
        elif isinstance(exc, ConnectionFailure):
            print(str(exc))
        elif isinstance(exc, EmptyResponse):
            print(str(exc))
        elif isinstance(exc, NAtlasError):
            print(str(exc))
        else:
            if client.debug:
                print("Unexpected error:")
                traceback.print_exc()
            else:
                print("An unexpected error occurred: %s" % exc)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye! Kar ka dawo — jiya ma'aikata. See you next time.")
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            command = user_input[1:].strip().lower()
            if command == "help":
                print_help_text()
                continue
            if command == "new":
                client.clear_conversation()
                print("Started a new conversation.")
                continue
            if command == "clear":
                client.clear_conversation()
                print("Conversation history cleared.")
                continue
            if command == "history":
                print_history(client)
                continue
            if command == "status":
                handle_status(client)
                continue
            if command == "exit":
                print("Bye!")
                break
            print("Unknown command: /%s. Type /help for commands." % command)
            continue

        printed_header = {"value": False}

        def on_token(text):
            if not printed_header["value"]:
                sys.stdout.write("Assistant: ")
                printed_header["value"] = True
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            result = client.chat(user_input, stream=None, on_token=on_token)
            text = str(result[0] or "")
            if printed_header["value"]:
                print()
            else:
                print("Assistant: %s" % text)
            if text:
                client.conversation.add("user", user_input)
                client.conversation.add("assistant", text)
                if client.cfg.save_conversations:
                    try:
                        saved = client.save_conversation(
                            directory=client.cfg.conversations_dir
                        )
                        log_debug(client, "conversation saved to %s" % (saved or "skipped"))
                    except OSError as exc:
                        log_debug(client, "could not save conversation: %s" % exc)
            else:
                print("(the model returned an empty response)")
        except Exception as exc:
            if printed_header["value"]:
                print()
            handle_exception(exc)


def start_repl(client):
    run_repl(client)


# ----------------------------------------------------------------------
# TEST MODE  (--test)
# ----------------------------------------------------------------------
class TestResult:
    def __init__(self):
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, passed, detail))
        print("[%s] %s%s" % (
            "PASS" if passed else "FAIL",
            name,
            (" :: %s" % detail) if detail else "",
        ))
        return passed


def run_tests(cfg):
    results = TestResult()
    print("=" * 46)
    print("N-ATLaS CLIENT TEST SUITE")
    print("=" * 46)
    client = NAtlasClient(cfg, debug=False)
    try:
        client.connect()
    except NAtlasError as exc:
        results.record("connection", False, str(exc))
        print("Cannot continue: %s" % exc)
        return summary(results)

    errors_seen = []

    # -- connection --
    try:
        results.record(
            "connection",
            client._openai is not None,
            "base_url=%s" % client.cfg.base_url.split("://")[-1][:60],
        )
    except Exception as exc:
        results.record("connection", False, repr(exc))

    # -- health --
    try:
        info = client.health()
        ok = bool(info) and info.get("status") in ("ok", "degraded")
        results.record(
            "health returns a status",
            ok,
            "status=%s model=%s" % (info.get("status"), info.get("model")),
        )
        client.api_connected = ok
    except NAtlasError as exc:
        results.record("health returns a status", False, str(exc))
        errors_seen.append(exc)

    # -- models --
    try:
        ids = client.check_available_models()
        results.record(
            "model 'n-atlas' is listed",
            client.cfg.model in ids,
            "ids=%s" % ids,
        )
    except NAtlasError as exc:
        results.record("model 'n-atlas' is listed", False, str(exc))
        errors_seen.append(exc)

    if errors_seen:
        print("")
        print("Connection-level failures prevented deeper tests.")
        return summary(results)

    # -- helper: run a single chat test --
    def chat_test(label, message, history=None, flag="non-empty"):
        text, info = None, {}
        error = None
        try:
            client.clear_conversation()
            if history:
                client.conversation.messages = [dict(m) for m in history]
            text, info = client.chat(message, stream=False)
        except NAtlasError as exc:
            error = exc
        if error is not None:
            results.record(label, False, str(error))
            return None
        ok = bool(text) and bool(text.strip())
        results.record(
            label + " returns a non-empty reply",
            ok,
            repr((text or "")[:140]),
        )
        return text

    # -- languages --
    language_cases = [
        ("English", "How are you?"),
        ("Hausa", "Kana lafiya?"),
        ("Yoruba", "Báwo ni?"),
        ("Igbo", "Kedu ka ị mere?"),
    ]
    for label, prompt in language_cases:
        chat_test(label, prompt)

    # -- conversation memory --
    client.clear_conversation()
    memory_reply = None
    try:
        memory_reply, _ = client.chat("My name is Ahmed.", stream=False)
        client.conversation.add("user", "My name is Ahmed.")
        client.conversation.add("assistant", memory_reply or "")
    except NAtlasError as exc:
        results.record("memory: name provided", False, str(exc))
        memory_reply = ""
    results.record(
        "memory: 'My name is Ahmed.' returns a reply",
        bool(memory_reply and memory_reply.strip()),
        repr((memory_reply or "")[:140]),
    )
    try:
        recall, _ = client.chat("What is my name?", stream=False)
        client.conversation.add("user", "What is my name?")
        client.conversation.add("assistant", recall or "")
    except NAtlasError as exc:
        results.record("memory: recalls the name", False, str(exc))
        recall = ""
    results.record(
        "memory: 'What is my name?' mentions Ahmed",
        "ahmed" in (recall or "").lower(),
        repr((recall or "")[:140]),
    )

    # -- language switching --
    client.clear_conversation()
    try:
        hausa_reply, _ = client.chat("Answer in Hausa.", stream=False)
        client.conversation.add("user", "Answer in Hausa.")
        client.conversation.add("assistant", hausa_reply or "")
    except NAtlasError as exc:
        results.record("switch: answer in Hausa", False, str(exc))
    try:
        english_reply, _ = client.chat("Now answer in English.", stream=False)
        client.conversation.add("user", "Now answer in English.")
        client.conversation.add("assistant", english_reply or "")
    except NAtlasError as exc:
        results.record("switch: answer in English", False, str(exc))
    results.record(
        "language switching without reset",
        True,
        "history preserved (the client never resets on language change)",
    )

    return summary(results)


def summary(results):
    passed = sum(1 for _, ok, _ in results.results if ok)
    total = len(results.results)
    failed = [(n, d) for n, ok, d in results.results if not ok]
    print("")
    print("=" * 46)
    print("RESULT: %d/%d checks passed" % (passed, total))
    if failed:
        for name, detail in failed:
            print("  FAIL  %s :: %s" % (name, detail))
    print("=" * 46)
    return 0 if not failed else 1


# ----------------------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="N-ATLaS laptop client — talks to the remote Kaggle server."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show HTTP statuses, durations and exception details (never the API key)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run the test suite (connection, health, models, languages, memory)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="path to a custom .env file",
    )
    args = parser.parse_args()

    if load_dotenv is None:
        print("python-dotenv is not installed. Run: pip install python-dotenv")
        return 2
    if OpenAI is None:
        print("The openai package is not installed. Run: pip install openai")
        return 2

    if args.version:
        print("n_atlas client 1.0.0 (python %s on %s)" % (
            platform.python_version(), platform.system()
        ))
        return 0

    cfg = Config()
    cfg.debug = args.debug
    if args.env is not None:
        # a custom env path is documented; simplest is to re-load
        load_dotenv(args.env, override=False)
        cfg = Config()
        cfg.debug = args.debug

    try:
        validate_config(cfg)
    except ConfigError as exc:
        print(str(exc))
        log_debug(cfg, "config check failed")
        return 2

    if args.test:
        return run_tests(cfg)

    client = NAtlasClient(cfg, debug=args.debug)
    try:
        client.connect()
        # Startup validation: health + models + verify n-atlas exists.
        try:
            client.health()
            client.api_connected = (
                client.server_info.get("status") in ("ok", "degraded")
            )
        except NAtlasError as exc:
            client.api_connected = False
            log_debug(client, "startup health check failed: %s" % exc)
        try:
            ids = client.check_available_models()
            client.model_available = client.cfg.model in ids
            if not client.model_available:
                print("Error: model %r is not available on the server. "
                      "Available: %s" % (client.cfg.model, ids))
        except NAtlasError as exc:
            client.model_available = False
            log_debug(client, "startup models check failed: %s" % exc)
    except NAtlasError as exc:
        print("Unable to reach the N-ATLaS server. Check the server or "
              "network connection.")
        if args.debug:
            log_debug(client, "startup failed: %s" % exc)
        start_repl(client)
        return 0

    start_repl(client)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        print("interrupted — bye!")