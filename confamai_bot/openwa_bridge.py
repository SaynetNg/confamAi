"""Bridge between open-wa (Easy API) and confamai_bot.

This replaces the Meta Cloud API entirely. open-wa sends incoming WhatsApp
messages as webhooks; this file translates them into the Meta payload format
that router.py already understands, and replaces the WhatsApp sender to reply
through open-wa's REST API instead of graph.facebook.com.

Usage:
    1. Start open-wa:
       npx @open-wa/wa-automate -p 8082 -w http://localhost:8000/openwa

    2. Start this bot:
       python -m confamai_bot.openwa_bridge

    The bot listens on port 8000 with a POST /openwa endpoint that receives
    open-wa events, and sends replies back via open-wa's API at localhost:8082.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config
from .natlas import NAtlasClient
from .queue import Concurrency
from .router import Router
from .sessions import Sweeper, build_store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENWA_API_URL = os.getenv("OPENWA_API_URL", "http://localhost:8082")
OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")  # set if you started open-wa with -k


# ---------------------------------------------------------------------------
# OpenWA WhatsApp Client (replaces Meta's Graph API sender)
# ---------------------------------------------------------------------------
class OpenWAClient:
    """Sends messages through open-wa's REST API instead of Meta Cloud API.

    Matches the interface that router.py expects from WhatsAppClient:
      - send_text(to, body)
      - send_with_retry(to, body)
      - download_media(media_id)
      - mark_read(message_id)
      - set_typing(message_id, on)
      - start() / aclose()
    """

    def __init__(self, base_url: str = OPENWA_API_URL, api_key: str = OPENWA_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self.debug = True
        # Router expects this attribute from WhatsAppClient
        self.cfg = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api_key"] = self.api_key
        return h

    async def _post_openwa(self, endpoint: str, data: dict) -> dict:
        """POST to open-wa's API."""
        if self._client is None:
            await self.start()
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = await self._client.post(url, json=data, headers=self._headers())
            return resp.json() if resp.status_code < 400 else {}
        except Exception as exc:
            print(f"[openwa] API call to {endpoint} failed: {exc}", flush=True)
            return {}

    async def send_text(self, to: str, body: str) -> dict:
        """Send a text message via open-wa."""
        if not body:
            return {}
        # open-wa expects the chat id format: number@c.us
        chat_id = _to_chat_id(to)
        result = await self._post_openwa("sendText", {
            "args": {
                "to": chat_id,
                "content": body,
            }
        })
        print(f"[openwa] sent {len(body)} chars to {to}", flush=True)
        return result

    async def send_with_retry(self, to: str, body: str, attempts: int = 3) -> dict:
        """Send with basic retry logic."""
        for attempt in range(1, attempts + 1):
            try:
                return await self.send_text(to, body)
            except Exception as exc:
                if attempt >= attempts:
                    raise
                print(f"[openwa] retry {attempt}: {exc}", flush=True)
                await asyncio.sleep(2.0)
        return {}

    async def download_media(self, media_id: str) -> bytes:
        """Download media (voice notes, images) from open-wa."""
        if self._client is None:
            await self.start()
        # open-wa provides a decryptMedia endpoint
        result = await self._post_openwa("decryptMedia", {
            "args": {"message": media_id}
        })
        if isinstance(result, dict) and "result" in result:
            import base64
            return base64.b64decode(result["result"])
        raise Exception(f"Could not download media {media_id}")

    async def mark_read(self, message_id: str) -> None:
        """Mark message as read. Best effort."""
        pass  # open-wa handles this differently; not critical

    async def set_typing(self, message_id: str, on: bool = True) -> None:
        """Show typing indicator. Best effort."""
        pass  # open-wa handles this differently; not critical


# ---------------------------------------------------------------------------
# Payload translator: open-wa event → Meta webhook format
# ---------------------------------------------------------------------------
def _to_chat_id(phone: str) -> str:
    """Convert a phone number to open-wa chat id format (number@c.us)."""
    digits = "".join(c for c in phone if c.isdigit())
    if not digits.endswith("@c.us"):
        return f"{digits}@c.us"
    return digits


def _from_chat_id(chat_id: str) -> str:
    """Extract phone number from open-wa chat id (number@c.us → number)."""
    return chat_id.replace("@c.us", "").replace("@s.whatsapp.net", "")


def openwa_to_meta_payload(event: dict) -> dict:
    """Convert an open-wa webhook event into the Meta webhook format
    that parse_inbound() and parse_unsupported() in whatsapp.py expect.

    Meta format:
    {
      "entry": [{
        "changes": [{
          "value": {
            "messages": [{
              "id": "...",
              "from": "2348012345678",
              "type": "text",
              "text": {"body": "hello"},
              "timestamp": "..."
            }]
          }
        }]
      }]
    }
    """
    # open-wa sends different event shapes depending on the event type.
    # The most common is an "onMessage" event with these fields:
    #   id, from (chat id), body, type, isMedia, mimetype, t (timestamp)
    #
    # Handle both single-event and wrapped formats
    if isinstance(event, list):
        # Some open-wa versions send arrays
        event = event[0] if event else {}

    # Extract the sender phone number
    sender = event.get("from", "") or event.get("sender", {}).get("id", "")
    phone = _from_chat_id(str(sender))

    # Extract message body
    body = event.get("body", "") or event.get("content", "") or ""
    msg_id = event.get("id", str(uuid.uuid4()))
    timestamp = str(event.get("t", int(time.time())))

    # Determine message type
    msg_type = event.get("type", "chat")
    is_media = event.get("isMedia", False)
    mimetype = event.get("mimetype", "")

    if msg_type in ("ptt", "audio") or (is_media and "audio" in mimetype):
        # Voice note
        meta_message = {
            "id": msg_id,
            "from": phone,
            "type": "audio",
            "timestamp": timestamp,
            "audio": {"id": msg_id, "mime_type": mimetype or "audio/ogg; codecs=opus"},
        }
    elif msg_type == "image" or (is_media and "image" in mimetype):
        meta_message = {
            "id": msg_id,
            "from": phone,
            "type": "image",
            "timestamp": timestamp,
        }
    elif msg_type in ("chat", "text") or body:
        # Text message
        meta_message = {
            "id": msg_id,
            "from": phone,
            "type": "text",
            "text": {"body": body},
            "timestamp": timestamp,
        }
    else:
        # Unknown type — pass through as unsupported
        meta_message = {
            "id": msg_id,
            "from": phone,
            "type": msg_type or "unknown",
            "timestamp": timestamp,
        }

    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [meta_message]
                }
            }]
        }]
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_bridge_app() -> FastAPI:
    """Build a FastAPI app that receives open-wa webhooks and routes them
    through the existing confamai_bot router."""

    cfg = load_config()
    openwa_client = OpenWAClient()
    natlas = NAtlasClient(cfg)
    store = build_store(cfg)
    concurrency = Concurrency(cfg.max_concurrency, cfg.per_user_timeout)
    router = Router(cfg, natlas, openwa_client, store, concurrency)
    sweeper = Sweeper(store, cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await natlas.start()
        await openwa_client.start()
        await store.setup()
        sweeper.start()
        print("[bridge] Confam AI ↔ open-wa bridge ready", flush=True)
        print(f"[bridge] open-wa API: {OPENWA_API_URL}", flush=True)
        print("[bridge] Listening for webhooks on POST /openwa", flush=True)
        try:
            yield
        finally:
            await sweeper.stop()
            await natlas.aclose()
            await openwa_client.aclose()
            await store.close()

    app = FastAPI(
        title="Confam AI ↔ open-wa Bridge",
        description="Bridges open-wa webhooks to confamai_bot router",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.post("/openwa")
    async def receive_openwa(request: Request):
        """Receive a webhook from open-wa, translate it, and route it."""
        try:
            event = await request.json()
        except Exception:
            return JSONResponse({"status": "bad request"}, status_code=400)

        # Skip group messages and status updates
        if isinstance(event, dict):
            # open-wa "onMessage" events from groups have isGroupMsg=True
            if event.get("isGroupMsg", False):
                return JSONResponse({"status": "skipped (group)"})
            # Skip if it's from the bot itself
            if event.get("fromMe", False):
                return JSONResponse({"status": "skipped (self)"})

        # Convert to Meta format and hand to the router
        meta_payload = openwa_to_meta_payload(event)
        asyncio.create_task(_safe_handle(router, meta_payload))

        return JSONResponse({"status": "ok"})

    @app.get("/healthz")
    async def health():
        return JSONResponse({
            "bridge": "up",
            "openwa": OPENWA_API_URL,
            "stats": router.snapshot(),
        })

    @app.get("/")
    async def root():
        return JSONResponse({
            "service": "confamai-openwa-bridge",
            "webhook": "/openwa",
            "health": "/healthz",
        })

    return app


async def _safe_handle(router: Router, payload: dict) -> None:
    """Run the router in the background, swallowing exceptions."""
    try:
        await router.handle_payload(payload)
    except Exception as exc:
        print(f"[bridge] handler failed: {exc!r}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BRIDGE_PORT", "8000"))
    print(f"[bridge] Starting Confam AI ↔ open-wa bridge on port {port}", flush=True)
    bridge_app = create_bridge_app()
    uvicorn.run(bridge_app, host="0.0.0.0", port=port)
