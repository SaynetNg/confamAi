"""The N-ATLaS server client. One method, one endpoint.

Everything this bot knows about product verification arrives through
`chat()`. It POSTs to `/v1/chat/completions` and returns text. There is no
greenbook query here, no NRN extraction, no intent check, because all three
already exist on the server and are better there:

    * `_intent_permits_retrieval()` decides whether a lookup happens at all
      (n_atlas_server/natlas_notebook.py:3614-3635)
    * `extract_nrn_candidates()` is the only thing allowed to decide what the
      number is, deliberately not the model
    * `NAFDAC_INSTRUCTION` holds the rules that stop a reply inventing a
      product name or calling a product counterfeit

Two server behaviours shape this module and are easy to get wrong:

**No system message.** `build_prompt_messages()` drops every client-supplied
system message and logs a warning, because a client that could send
`[nafdac] Verdict: AUTHENTIC` as a system turn could assert any verdict it
liked (natlas_notebook.py:1564-1569). This client therefore sends only user
and assistant turns. The Confam AI persona is `DEFAULT_SYSTEM_PROMPT` on the
server and needs nothing from here.

**No streaming.** `STREAMING_SUPPORTED = False`; a `stream=true` request is
answered with a 400. Asking anyway would spend a full round trip being refused
on every message, so `stream` is always false.

The error taxonomy mirrors n_atlas_client/api.py so the retry policy and the
message a user sees are the same ones the laptop client already has.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from .config import Config

# Named so callers can retry the right ones and fail fast on the wrong ones.
# Mirrors the split in n_atlas_client/api.py:322-338.
TRANSIENT = "transient"
FATAL = "fatal"


class NAtlasError(Exception):
    """Base class. Carries a `kind` so the router knows whether to retry."""

    kind = FATAL


class ConnectionFailure(NAtlasError):
    kind = TRANSIENT


class TimeoutFailure(NAtlasError):
    kind = TRANSIENT


class RateLimitFailure(NAtlasError):
    kind = TRANSIENT


class ServerFailure(NAtlasError):
    kind = TRANSIENT


class GPUExhaustedFailure(NAtlasError):
    kind = TRANSIENT


class AuthenticationFailure(NAtlasError):
    kind = FATAL


class ForbiddenFailure(NAtlasError):
    kind = FATAL


class NotFoundFailure(NAtlasError):
    kind = FATAL


class ValidationFailure(NAtlasError):
    kind = FATAL


class EmptyResponse(NAtlasError):
    kind = FATAL


class NAtlasClient:
    """Async client for the remote N-ATLaS API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.debug = bool(getattr(cfg, "debug", False))
        # The bot serves many users from one process, so the client is long
        # lived and connection pooling is what keeps a turn from paying a TLS
        # handshake. Limits are sized for the concurrency semaphore.
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "NAtlasClient":
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Open the connection pool. Idempotent."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.cfg.timeout, connect=15.0),
                limits=httpx.Limits(
                    max_connections=max(4, self.cfg.max_concurrency * 2),
                    max_keepalive_connections=max(2, self.cfg.max_concurrency),
                ),
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict:
        return {
            "Authorization": "Bearer %s" % self.cfg.api_key,
            "Content-Type": "application/json",
            # Named explicitly because the server sits behind Cloudflare, whose
            # browser-integrity check answers default library agents with
            # "error code: 1010" and a 403. A default httpx agent is refused
            # here exactly as Python-urllib was; naming the client is immune.
            "User-Agent": "confamai-bot/1.0 (+https://ai.saynet.ng)",
        }

    def _client_or_start(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ConnectionFailure(
                "The N-ATLaS client is not started. Call start() first."
            )
        return self._client

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    async def health(self) -> dict:
        """GET /health. Also the bot's own readiness signal.

        The `nafdac` block in the payload is the one that matters operationally:
        `ready: true` means the server can actually verify something, and a
        green-looking top-level `status` does not imply that. See
        natlas_notebook.py:4300-4312, where a server that answers from memory
        instead of the greenbook is called out as the failure worth shouting
        about.
        """
        client = self._client_or_start()
        try:
            response = await client.get(
                self.cfg.health_url(), headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or the "
                "tunnel."
            ) from exc
        if response.status_code != 200:
            raise ServerFailure(
                "The N-ATLaS server returned %d while checking health."
                % response.status_code
            )
        return response.json()

    async def verify_ready(self) -> tuple[bool, str]:
        """(can_verify, reason) — for /healthz and the status command."""
        try:
            info = await self.health()
        except NAtlasError as exc:
            return False, str(exc)
        if not info.get("model_loaded"):
            return False, "the model is not loaded yet"
        nafdac = info.get("nafdac") or {}
        if not nafdac.get("enabled"):
            return False, nafdac.get("reason") or "the greenbook data is not loaded"
        if not nafdac.get("ready"):
            return False, nafdac.get("hint") or nafdac.get("reason") or "no usable source"
        return True, "ready"

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        language: str | None = None,
    ) -> str:
        """One turn. Returns the assistant's text.

        `messages` must contain only user and assistant turns. `language` is
        the pinned reply language ("en" | "ha" | "yo" | "ig"); pass None for
        auto, where the server detects each message on its own.

        The pinned language travels two ways at once, because either one alone
        has a failure mode: as the `language` field, which a patched server
        treats as authoritative and skips its detector for, and — via the
        server's own directive, not this module's — as an instruction the model
        reads. Sending the field is all this client has to do.
        """
        if not messages:
            raise ValidationFailure("No messages to send.")

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            # Always false: the server refuses streaming with a 400 and /health
            # advertises `streaming: false`. Asking would waste a round trip.
            "stream": False,
        }
        if language:
            payload["language"] = language

        return await self._chat_once(payload)

    async def _chat_once(self, payload: dict) -> str:
        client = self._client_or_start()
        started = time.monotonic()
        try:
            response = await client.post(
                self.cfg.chat_url(), headers=self._headers(), json=payload
            )
        except httpx.TimeoutException as exc:
            raise TimeoutFailure(
                "The N-ATLaS server is taking too long to respond. It may still "
                "be loading the model."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionFailure(
                "Unable to reach the N-ATLaS server. Check the server or the "
                "tunnel."
            ) from exc

        self._log(
            "POST /v1/chat/completions -> %d (%.1fs)"
            % (response.status_code, time.monotonic() - started)
        )

        if response.status_code != 200:
            raise self._raise_from_http(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ServerFailure("The server sent a reply that was not JSON.") from exc

        choices = body.get("choices") or []
        if not choices:
            raise EmptyResponse("The model returned no choices.")

        content = (choices[0].get("message") or {}).get("content") or ""
        if not str(content).strip():
            raise EmptyResponse("The model returned an empty response.")

        from .render import strip_template_artifacts

        return strip_template_artifacts(content)

    async def chat_with_retry(
        self,
        messages: list[dict],
        language: str | None = None,
    ) -> str:
        """`chat()` plus bounded retry on transient failures only.

        401/403/404/400/422 are never retried: a wrong API key or a malformed
        request will fail identically every time, and retrying only delays the
        honest error the user needs to see. Backoff is capped at 4s because a
        WhatsApp user is watching a bubble, not a queue position.
        """
        attempts = 0
        max_attempts = self.cfg.max_retries + 1
        last_error: NAtlasError | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                return await self.chat(messages, language=language)
            except NAtlasError as exc:
                last_error = exc
                if exc.kind != TRANSIENT or attempts >= max_attempts:
                    raise
                delay = min(0.5 * (2 ** (attempts - 1)), 4.0)
                self._log("retry %d/%d in %.1fs: %s" % (attempts + 1, max_attempts, delay, exc))
                await asyncio.sleep(delay)

        raise last_error or ConnectionFailure("The N-ATLaS request failed.")

    # ------------------------------------------------------------------
    def _raise_from_http(self, response: httpx.Response) -> NAtlasError:
        """Translate a status code into the exception the router understands."""
        status = response.status_code
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error") or {}
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
        except Exception:
            detail = ""

        if status == 401:
            return AuthenticationFailure(
                "Authentication failed. Check N_ATLAS_API_KEY."
            )
        if status == 403:
            return ForbiddenFailure(
                "Access forbidden (403). The API key is not allowed to use this "
                "server."
            )
        if status == 404:
            return NotFoundFailure(
                "Not found (404). Check N_ATLAS_BASE_URL."
            )
        if status in (400, 422):
            return ValidationFailure("Validation error (%d): %s" % (status, detail))
        if status == 429:
            return RateLimitFailure(
                "The server's GPU queue is full. Retry shortly."
            )
        if status == 507:
            return GPUExhaustedFailure(
                "The remote GPU ran out of memory. Try a shorter conversation."
            )
        if status == 503:
            return ServerFailure(
                "The server is unavailable. The model may still be loading."
            )
        if status == 500:
            return ServerFailure("The remote generation failed.")
        return ServerFailure("Unexpected server response (HTTP %d)." % status)

    def _log(self, message: str) -> None:
        if self.debug:
            print("[natlas] %s" % message, flush=True)