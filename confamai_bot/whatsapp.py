"""The WhatsApp Cloud API sender.

One method that matters, `send_text`, plus the two optional ones Meta expects
around it: marking a message read, and reporting a typing indicator.

    POST https://graph.facebook.com/{version}/{phone_number_id}/messages
    Authorization: Bearer {system user token}

Error codes worth naming, because they need different handling and lumping them
together is how a bot silently stops answering people:

    131047  rate limit hit          -> back off hard, retry
    131056  pair rate limit hit     -> back off harder, retry
    130429  rate limit hit (older) -> back off hard, retry
    131026  the token cannot send to this number -> do NOT retry, drop
    100     invalid parameter / bad phone number -> do NOT retry, drop
    131009  parameter value is not valid       -> do NOT retry, drop
    131047/130472  quality or blocklisting    -> stop and investigate

The retry rule is: retry 5xx and the rate limits, never retry anything else. A
retried `131026` is a token that will never work, and a bot that retries it
forever is a bot that hides its own misconfiguration.

Access tokens are never logged, never echoed in an exception, and never sent
anywhere but graph.facebook.com. `Config.describe()` reports only whether one
is set.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .config import Config

# Meta error codes that mean "try again later", with the backoff floor for each.
_RETRY_CODES = {
    131047: 4.0,   # rate limit
    130429: 4.0,   # rate limit (older code)
    131056: 15.0,  # pair rate limit — deliberately much slower
    130472: 30.0,  # partially blocked for quality
}

# Codes that mean this will never succeed. A retried 131026 is a token that
# will never work; retrying it forever just hides the misconfiguration. Kept for
# the log line an operator needs, not for the retry decision, which is driven
# entirely by _RETRY_CODES membership so the two lists cannot disagree.
_NEVER_WORKS = frozenset(
    {
        100,      # invalid parameter / bad phone number
        131008,   # required parameter is missing
        131009,   # parameter value is not valid
        131021,   # sender cannot be used to send this message
        131026,   # undeliverable: the token cannot reach this number
        131045,   # certificate / app not authorised for this number
    }
)

_TYPING_ON = "typing_on"
_TYPING_OFF = "typing_off"


class WhatsAppError(Exception):
    """Any failure to send. Carries whether retrying could help."""

    def __init__(self, message: str, code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class WhatsAppClient:
    """Async sender for the Cloud API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.debug = bool(getattr(cfg, "debug", False))
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=10.0)
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict:
        return {
            "Authorization": "Bearer %s" % self.cfg.access_token,
            "Content-Type": "application/json",
            "User-Agent": "confamai-bot/1.0 (+https://ai.saynet.ng)",
        }

    async def _post(self, payload: dict) -> dict:
        if self._client is None:
            await self.start()
        try:
            response = await self._client.post(
                self.cfg.graph_base(), headers=self._headers(), json=payload
            )
        except httpx.TimeoutException as exc:
            raise WhatsAppError(
                "WhatsApp did not respond in time.", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise WhatsAppError(
                "Could not reach the WhatsApp API.", retryable=True
            ) from exc

        try:
            body: Any = response.json()
        except ValueError:
            body = {}

        if response.status_code >= 500:
            raise WhatsAppError(
                "The WhatsApp API returned %d." % response.status_code,
                retryable=True,
            )

        error = body.get("error") if isinstance(body, dict) else None
        if error:
            # The token itself is never part of this message: `body` can echo
            # request context, and this string reaches logs and /healthz.
            code = error.get("code")
            details = error.get("message") or "the request was rejected"
            retryable = code in _RETRY_CODES
            if code in _NEVER_WORKS:
                details = (
                    "%s — this will not succeed on retry; check the access "
                    "token and the phone number id" % details
                )
            raise WhatsAppError(
                "WhatsApp error %s: %s" % (code, details),
                code=code,
                retryable=retryable,
            )

        return body if isinstance(body, dict) else {}

    # ------------------------------------------------------------------
    async def send_text(self, to: str, body: str) -> dict:
        """Send one text message. Returns Meta's response including wamid.

        The body is sent exactly as given. `render.chunk()` decides the
        boundaries; this method does not trim, rewrite or truncate, because a
        silently shortened safety answer is worse than a second message.
        """
        if not body:
            raise WhatsAppError("Refusing to send an empty message.")
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body, "preview_url": False},
        }
        result = await self._post(payload)
        self._log("sent %d chars to %s" % (len(body), to))
        return result

    async def send_with_retry(self, to: str, body: str, attempts: int = 3) -> dict:
        """`send_text` plus bounded retry for rate limits and 5xx only."""
        last: WhatsAppError | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                return await self.send_text(to, body)
            except WhatsAppError as exc:
                if not exc.retryable:
                    raise
                last = exc
                if attempt >= attempts:
                    break
                delay = _RETRY_CODES.get(exc.code or 0, 2.0) * (2 ** (attempt - 1))
                delay = min(delay, 30.0)
                self._log(
                    "WhatsApp %s, retrying in %.1fs (attempt %d)"
                    % (exc.code, delay, attempt)
                )
                await asyncio.sleep(delay)
        raise last or WhatsAppError("The message could not be sent.")

    async def mark_read(self, message_id: str) -> None:
        """Show the blue ticks. Never fatal if it fails."""
        if not message_id:
            return
        try:
            await self._post(
                {
                    "messaging_product": "whatsapp",
                    "status": "read",
                    "message_id": message_id,
                }
            )
        except WhatsAppError as exc:
            # Read receipts are cosmetic. A failure here must not fail a reply
            # the user is waiting for.
            self._log("mark_read failed (ignored): %s" % exc)

    async def set_typing(self, message_id: str, on: bool = True) -> None:
        """Show the typing indicator, where the deployment supports it.

        Best effort by design. Some Cloud API configurations reject the typing
        action, and that must never affect the reply path, so every failure is
        swallowed. The acknowledgement line in render.py is the real signal.
        """
        if not message_id:
            return
        try:
            await self._post(
                {
                    "messaging_product": "whatsapp",
                    "status": _TYPING_ON if on else _TYPING_OFF,
                    "message_id": message_id,
                }
            )
        except WhatsAppError:
            pass

    def _log(self, message: str) -> None:
        if self.debug:
            print("[whatsapp] %s" % message, flush=True)


def parse_inbound(payload: dict) -> list[dict]:
    """Pull the user messages out of a webhook body.

    Three rules, all of them learned the hard way:

    * `entry` and `changes` are ARRAYS. Taking `[0]` on the wrong level is the
      single most common webhook bug, and it looks fine until two numbers share
      a payload.
    * `statuses` lives in `value` alongside `messages`, and a status callback
      has no `from` field. A handler that assumes `value.messages` exists will
      raise KeyError on every delivery receipt — which is several callbacks per
      reply, so it fails constantly.
    * Only `type == "text"` is handled here. Image and document messages are
      collected by `parse_unsupported` so they can be logged rather than
      silently dropped.

    Returns a list of {id, from, text, timestamp, type}.
    """
    messages: list[dict] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}

            for message in value.get("messages") or []:
                kind = message.get("type")
                if kind != "text":
                    messages.append(
                        {
                            "id": message.get("id", ""),
                            "from": message.get("from", ""),
                            "text": "",
                            "timestamp": message.get("timestamp", ""),
                            "type": kind or "unknown",
                        }
                    )
                    continue
                body = ((message.get("text") or {}).get("body")) or ""
                messages.append(
                    {
                        "id": message.get("id", ""),
                        "from": message.get("from", ""),
                        "text": body,
                        "timestamp": message.get("timestamp", ""),
                        "type": "text",
                    }
                )
    return messages


def parse_unsupported(payload: dict) -> list[dict]:
    """Non-text messages, so they can be answered rather than dropped.

    A user photographing a medicine pack is the single most likely thing a real
    person does with a verification bot, and `image` is what arrives. OCR is a
    separate feature; this only makes the bot say something useful instead of
    going silent, which reads as broken.
    """
    unsupported: list[dict] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for message in value.get("messages") or []:
                if message.get("type") == "text":
                    continue
                unsupported.append(
                    {
                        "id": message.get("id", ""),
                        "from": message.get("from", ""),
                        "type": message.get("type", "unknown"),
                    }
                )
    return unsupported