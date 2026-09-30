"""NAtlasClient: thin OpenAI-compatible wrapper for the remote N-ATLaS API.

The laptop is only a client. All model computation happens on the Kaggle
server (Tesla T4). This module talks to exactly three endpoints:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions

Streaming is attempted when configured (N_ATLAS_STREAM=true) and falls back
to a normal response when the server rejects streaming.
"""

import json
import time
from pathlib import Path

from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from .config import Config
from .conversation import Conversation

# Template/lint artifacts that occasionally leak out of remote chat templates.
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


class NAtlasError(Exception):
    """Base class for client-level errors (never leaves tracebacks visible)."""


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


def clean_response_text(text: str) -> str:
    """Minimal safety cleaning of a remote response.

    The N-ATLaS server is responsible for full response cleaning. This client
    only strips obvious chat-template artifacts so the terminal looks clean.
    It never translates or rewrites the model's language.
    """
    if not text:
        return ""
    out = text
    for token in TEMPLATE_ARTIFACTS:
        if token in out:
            out = out.replace(token, "")
    return out.strip()


class NAtlasClient:
    """Connects to the remote N-ATLaS API and manages local conversation."""

    def __init__(self, config: Config, debug: bool = False):
        self.cfg = config
        self.debug = bool(config.debug or debug)
        self.conversation = Conversation(config)
        self._openai: OpenAI | None = None
        self.server_info: dict = {}
        self.api_connected = False
        self.model_available = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if not self.cfg.base_url or not self.cfg.api_key:
            raise ConnectionFailure(
                "Configuration is incomplete. Set N_ATLAS_BASE_URL and "
                "N_ATLAS_API_KEY in .env."
            )
        self._openai = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout,
            max_retries=0,  # retries are handled explicitly in chat()
        )

    def initialize(self) -> None:
        """Startup validation: health then models. Fills server_info."""
        errors = []
        try:
            self.health()
            self.api_connected = (
                self.server_info.get("status") in ("ok", "degraded")
            )
        except NAtlasError as exc:
            self.api_connected = False
            errors.append(str(exc))

        try:
            ids = self.check_available_models()
            self.model_available = self.cfg.model in ids
        except NAtlasError as exc:
            self.model_available = False
            errors.append(str(exc))

        if errors and self.debug:
            for error in errors:
                self._log_debug("startup check: %s" % error)

    # ------------------------------------------------------------------
    # health / models
    # ------------------------------------------------------------------
    def health(self) -> dict:
        response = self._request("GET", "/health")
        status = response.status_code
        if status != 200:
            raise ServerFailure(
                "Server returned %d while checking health." % status
            )
        self.server_info = response.json()
        return self.server_info

    def models(self) -> list[str]:
        response = self._request("GET", "/v1/models")
        status = response.status_code
        if status != 200:
            self._raise_error_from_http(status, response, "checking the model list")
        ids = [
            item.get("id")
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        ]
        return ids

    def check_available_models(self) -> list[str]:
        ids = self.models()
        self.model_available = self.cfg.model in ids
        return ids

    def is_online(self) -> bool:
        info = self.health()
        return bool(info) and info.get("status") in ("ok", "degraded")

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def chat(self, message, stream: bool | None = None, on_token=None):
        """Send a message and return (response_text, info).

        `message` is the new user text, appended to the local history by the
        caller after a valid reply. `on_token` is an optional callable used to
        render streaming tokens live (e.g. a terminal). Streaming is tried
        only when configured and falls back to non-streaming automatically.
        """
        payload_messages = self.conversation.build_messages_for(message)
        use_stream = self.cfg.stream if stream is None else stream

        last_error = None
        attempts = 0
        max_attempts = self.cfg.max_retries + 1
        while attempts < max_attempts:
            attempts += 1
            try:
                return self._do_chat(payload_messages, use_stream, on_token)
            except (ConnectionFailure, TimeoutFailure, RateLimitFailure,
                    ServerFailure, GPUExhaustedFailure) as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                delay = min(0.5 * (2 ** (attempts - 1)), 4.0)
                self._log_debug("retry %d/%d in %.1fs: %s"
                                % (attempts + 1, max_attempts, delay, exc))
                time.sleep(delay)
            except (AuthenticationFailure, ForbiddenFailure, NotFoundFailure,
                    ValidationFailure, EmptyResponse) as exc:
                raise exc
        raise last_error

    def _do_chat(self, payload_messages, use_stream, on_token):
        if self._openai is None:
            self.connect()
        kwargs = dict(
            model=self.cfg.model,
            messages=payload_messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        if use_stream:
            result = self._try_stream(kwargs, on_token)
            if result is not None:
                return result
        return self._chat_once(kwargs)

    def _try_stream(self, kwargs, on_token):
        """Attempt a streaming request. Returns (text, info) or None to
        trigger the non-streaming fallback. Raises when streaming failed in a
        hard way (auth/validation/connection)."""
        request_kwargs = dict(kwargs)
        request_kwargs["stream"] = True
        started = time.monotonic()
        pieces = []
        try:
            stream = self._openai.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            if "stream" in str(exc).lower():
                self._log_debug("server rejected streaming; using normal response")
                return None
            raise
        except NotFoundError as exc:
            self._raise_openai_error(exc)
            return None
        except APIStatusError as exc:
            # the server may 400 any stream=true payload — treat as
            # "streaming unsupported" so we fall back.
            status = getattr(exc, "status_code", None)
            if status in (400, 422):
                self._log_debug("server rejected streaming (%s); fallback" % status)
                return None
            self._raise_openai_error(exc)
            return None
        except APIConnectionError as exc:
            self._raise_openai_error(exc)
            return None

        header_printed = False
        try:
            for event in stream:
                if event is None:
                    continue
                choice = getattr(event, "choices", None)
                if not choice:
                    continue
                delta = getattr(choice[0], "delta", None)
                text = getattr(delta, "content", None) or ""
                if not text:
                    continue
                pieces.append(text)
                if on_token is not None:
                    if not header_printed and on_token.__name__ == "stream_printer":
                        pass
                    on_token(text)
                    header_printed = True
        except (APIConnectionError, APITimeoutError) as exc:
            raise ConnectionFailure(
                "The connection to the N-ATLaS server dropped while streaming. "
                "Try again."
            ) from exc
        except APIError as exc:
            self._raise_openai_error(exc)
        except Exception as exc:
            raise ConnectionFailure(
                "Unable to reach the N-AtLaS server while streaming."
            ) from exc

        joined = "".join(pieces)
        if not joined.strip():
            self._log_debug("stream produced no text; falling back")
            return None
        info = {"streamed": True, "seconds": round(time.monotonic() - started, 2)}
        return clean_response_text(joined), info

    def _chat_once(self, kwargs):
        request_kwargs = dict(kwargs)
        request_kwargs["stream"] = False
        started = time.monotonic()
        try:
            response = self._openai.chat.completions.create(**request_kwargs)
        except (AuthenticationError,) as exc:
            self._raise_openai_error(exc)
        except PermissionDeniedError as exc:
            self._raise_openai_error(exc)
        except NotFoundError as exc:
            self._raise_openai_error(exc)
        except UnprocessableEntityError as exc:
            self._raise_openai_error(exc)
        except BadRequestError as exc:
            self._raise_openai_error(exc)
        except RateLimitError as exc:
            self._raise_openai_error(exc)
        except InternalServerError as exc:
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

        seconds = round(time.monotonic() - started, 2)
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise EmptyResponse(
                "The model returned an empty response. Try rewording your "
                "message or starting a new conversation (/new)."
            )
        info = {
            "streamed": False,
            "seconds": seconds,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            info["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return clean_response_text(text), info

    # ------------------------------------------------------------------
    # conversation / storage conveniences
    # ------------------------------------------------------------------
    def clear_conversation(self) -> None:
        self.conversation.reset()

    def save_conversation(self, directory: str | None = None):
        """Save the current local conversation to a JSON file."""
        return self.conversation.save(self.cfg, directory)

    def load_conversation(self, path: str) -> int:
        """Replace the local conversation with messages from a JSON file."""
        self.conversation.load(path)
        return len(self.conversation.messages)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _request(self, method, path):
        """requests-based GET against the server root (health, models)."""
        import requests

        root = self.cfg.server_root()
        url = root + path
        headers = {
            "Authorization": "Bearer %s" % self.cfg.api_key,
            "Accept": "application/json",
        }
        started = time.monotonic()
        try:
            response = requests.request(
                method, url, headers=headers, timeout=self.cfg.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or "
                "network connection."
            ) from exc
        seconds = round(time.monotonic() - started, 2)
        self._log_debug("HTTP %s %s -> %d (%.2fs)"
                        % (method, path, response.status_code, seconds))
        return response

    def _raise_error_from_http(self, status, response, context=""):
        body = {}
        try:
            body = response.json()
        except Exception:
            pass
        message = ""
        if isinstance(body, dict):
            error = body.get("error") or {}
            if isinstance(error, dict):
                message = error.get("message") or ""
        detail = message if message else ("HTTP %d" % status)
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
                "Not found (404): %s. Check N_ATLAS_BASE_URL." % context
            )
        if status in (400, 422):
            raise ValidationFailure(
                "Validation error (%d): %s." % (status, detail)
            )
        if status == 429:
            raise RateLimitFailure(
                "Rate limit reached (429). The GPU queue may be full. Retry shortly."
            )
        if status == 507:
            raise GPUExhaustedFailure(
                "The remote N-ATLaS GPU ran out of memory. Try a shorter "
                "conversation or shorter response."
            )
        if status == 503:
            raise ServerFailure(
                "Server unavailable (503). The model may still be loading. "
                "Retry shortly."
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
            raise ValidationFailure("Validation error (422): %s" % message) from exc
        if isinstance(exc, BadRequestError) or status == 400:
            if isinstance(message, str) and "stream" in message.lower():
                raise StreamingUnsupported(
                    "This server does not support streaming; using normal responses."
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
                "Server unavailable (%s). The model may be loading or the "
                "queue full. Retry shortly." % (status or 503)
            ) from exc
        if status == 500:
            raise ServerFailure(
                "Server error (500). The remote generation failed. Try again, "
                "or shorten the conversation."
            ) from exc
        raise ServerFailure("Unexpected server error: %s" % message) from exc

    def _log_debug(self, message: str) -> None:
        if self.debug:
            print("[debug] %s" % message, flush=True)


def info_as_json(info: dict) -> str:
    return json.dumps(info, ensure_ascii=False)