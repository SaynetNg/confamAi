"""The FastAPI app: one GET, one POST, and two health endpoints.

This is deliberately the thinnest possible surface, because everything
interesting is a liability on a public endpoint:

    GET  {webhook_path}    Meta's subscription handshake
    POST {webhook_path}    inbound messages
    GET  /healthz          liveness — always 200, never touches upstream
    GET  /readyz           readiness — live N-ATLaS probe, 200 or 503

`/healthz` (aliased as `/health`) is the one Render's health check and any
keep-alive pinger should hit. It answers instantly, from memory, and is 200
whenever the process is up. It deliberately does not call N-ATLaS: the bot
must stay alive and accepting the handshake even when the GPU server behind
it is down or still loading, and a health check that blocks on an upstream
with a 180-second timeout is a health check that gets the service restarted.
The last-known N-ATLaS state is still reported, from the startup probe.

`/readyz` carries the old behaviour — a real N-ATLaS round trip, 503 while
the model or the greenbook is not ready — for an operator or a monitor that
wants to know whether *answers* are possible, as opposed to whether the
service is *running*.

Two things are load-bearing here.

**The POST returns 200 immediately.** Meta requires a 200 within twenty seconds
or it treats the webhook as failed and retries, which produces duplicate replies
and duplicate billed messages. A T4 turn takes about thirteen seconds on its own,
so doing the work inline would guarantee a retry on nearly every message. The
request path therefore validates, deduplicates, and hands off to a background
task. Meta gets its 200 in milliseconds.

**The signature is checked before the body is parsed.** `X-Hub-Signature-256` is
an HMAC-SHA256 of the raw request body with the app secret. Checking it after
parsing, or skipping it, means anyone who learns the webhook URL can inject
messages that look like they came from a customer. `hmac.compare_digest` is used
rather than `==` so a wrong signature cannot be discovered one character at a
time.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import Config, ConfigError, load_config, normalize_phone, validate
from .natlas import NAtlasClient
from .queue import Concurrency
from .router import Router
from .sessions import Sweeper, build_store
from .whatsapp import WhatsAppClient


def _signature_ok(raw_body: bytes, header: str | None, secret: str) -> bool:
    """Validate X-Hub-Signature-256, constant time.

    `sha256=` prefixed hex HMAC of the raw body. Returns True when no secret is
    configured: an operator without an app secret gets a working bot, but is
    told about it by /healthz and by the startup banner, because an unverified
    webhook on a public number is not a safe default to ship silently.
    """
    if not secret:
        return True
    if not header:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header.strip())


class State:
    """Everything the routes need, assembled once at startup."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.natlas = NAtlasClient(cfg)
        self.whatsapp = WhatsAppClient(cfg)
        self.store = build_store(cfg)
        self.concurrency = Concurrency(cfg.max_concurrency, cfg.per_user_timeout)
        self.router = Router(cfg, self.natlas, self.whatsapp, self.store, self.concurrency)
        self.sweeper = Sweeper(self.store, cfg)
        self.ready = False
        self.readiness = "starting"
        self._probe: asyncio.Task | None = None

    async def startup(self) -> None:
        await self.natlas.start()
        await self.whatsapp.start()
        await self.store.setup()
        self.sweeper.start()

        # Probe the server once at boot, but do NOT block startup on it. The
        # bot must bind and answer /healthz within milliseconds even when
        # N-ATLaS is down or still loading the model. Awaiting here would hold
        # the port closed for up to a connect timeout, during which Render's
        # health check and any keep-alive pinger see a dead service — and a
        # slow deploy can be failed outright.
        self._probe = asyncio.create_task(self._probe_ready())

    async def _probe_ready(self) -> None:
        can_verify, reason = await self.natlas.verify_ready()
        self.ready = can_verify
        self.readiness = reason
        if can_verify:
            print("[bot] N-ATLaS: %s" % reason, flush=True)
        else:
            # Loud on purpose. A server that answers from memory instead of the
            # greenbook is the failure that damages trust fastest, so it is
            # named at boot rather than discovered by a customer.
            print(
                "[bot] N-ATLaS NOT READY: %s — replies will say verification is "
                "temporarily unavailable" % reason,
                flush=True,
            )

    async def shutdown(self) -> None:
        await self.sweeper.stop()
        probe = self._probe
        if probe is not None and not probe.done():
            probe.cancel()
            try:
                await probe
            except asyncio.CancelledError:
                pass
        await self.natlas.aclose()
        await self.whatsapp.aclose()
        await self.store.close()


def create_app(cfg: Config | None = None, env_file: str | None = None) -> FastAPI:
    """Build the app. `cfg` is injectable so tests can drive it directly."""
    config = cfg or load_config(env_file=env_file)
    # Meta credentials are not required to construct the app, only to receive
    # real messages. Tests and the live-assay tool use this to talk to N-ATLaS
    # without a WhatsApp number.
    try:
        validate(config, require_whatsapp=False)
    except ConfigError:
        pass

    state = State(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.startup()
        print("[bot] Confam AI WhatsApp bot ready", flush=True)
        print("[bot] webhook path: %s" % config.webhook_path, flush=True)
        print("[bot] signature check: %s"
              % ("ON" if config.app_secret else "OFF (no app secret)"), flush=True)
        # N-ATLaS readiness is probed in the background (state._probe_ready) and
        # announced there, because blocking here would keep the port closed.
        print("[bot] N-ATLaS: probing in background; see /readyz for a live "
              "answer", flush=True)
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(
        title="Confam AI WhatsApp Bot",
        description="WhatsApp front end for the N-ATLaS NAFDAC verification API.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.bot = state

    # -- handshake ------------------------------------------------------
    @app.get(config.webhook_path)
    async def verify_webhook(request: Request):
        """Meta's subscription check.

        The query parameters are `hub.mode`, `hub.verify_token` and
        `hub.challenge`. They are read from the raw query string rather than
        declared as typed parameters, because FastAPI maps a parameter named
        `mode` to `?mode=` and would silently miss `?hub.mode=`. That produces
        a 403 on every subscription attempt, which then looks like "webhooks
        work but no messages arrive" — the handshake failed and Meta never
        registered the callback URL at all.

        `hub.challenge` must come back as the entire response body. Any
        wrapper, any other status, any JSON, and the subscription fails.
        """
        params = request.query_params
        mode = params.get("hub.mode") or params.get("mode")
        token = params.get("hub.verify_token") or params.get("verify_token")
        challenge = params.get("hub.challenge") or params.get("challenge")

        if mode == "subscribe" and token and hmac.compare_digest(
            str(token), str(config.verify_token)
        ):
            return PlainTextResponse(challenge or "", status_code=200)

        # 403 rather than 404: an unauthenticated caller learns nothing about
        # whether this endpoint exists or what token it expects.
        return PlainTextResponse("forbidden", status_code=403)

    # -- inbound --------------------------------------------------------
    @app.post(config.webhook_path)
    async def receive_webhook(request: Request):
        raw = await request.body()

        if not _signature_ok(raw, request.headers.get("X-Hub-Signature-256"), config.app_secret):
            print("[bot] rejected a webhook with a bad signature", flush=True)
            return JSONResponse({"status": "forbidden"}, status_code=403)

        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            # A 200 here rather than a 400: this is Meta's retry behaviour
            # working as intended, and an unparseable body will never become
            # parseable, so a 400 only invites more of them.
            print("[bot] unparseable webhook body", flush=True)
            return JSONResponse({"status": "ignored"})

        # Hand off. Everything below this line runs after Meta has its 200.
        asyncio.create_task(_safe_handle(state, payload))
        return JSONResponse({"status": "ok"})

    # -- health ---------------------------------------------------------
    @app.get("/healthz")
    @app.get("/health")
    async def healthz():
        """Liveness. Always 200 while the process is up.

        Never calls N-ATLaS. A keep-alive pinger and Render's health check
        both use this, and either one blocking on a 180-second upstream
        timeout would be worse than useless. The `natlas` field is the
        last-known state from the startup probe; hit `/readyz` for a live one.
        """
        return JSONResponse(
            {
                "bot": "up",
                "natlas": "ready" if state.ready else "not-ready",
                "reason": state.readiness,
                "config": config.describe(),
                **state.router.snapshot(),
            },
            status_code=200,
        )

    @app.get("/readyz")
    async def readyz():
        """Readiness. A live probe, so it can be slow — and can say 503."""
        can_verify, reason = await state.natlas.verify_ready()
        state.ready = can_verify
        state.readiness = reason
        return JSONResponse(
            {
                "bot": "up",
                "natlas": "ready" if can_verify else "not-ready",
                "reason": reason,
                "config": config.describe(),
                **state.router.snapshot(),
            },
            status_code=200 if can_verify else 503,
        )

    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "service": "confamai-bot",
                "webhook": config.webhook_path,
                "health": "/healthz",
                "ready": "/readyz",
            }
        )

    return app


async def _safe_handle(state: State, payload: dict) -> None:
    """Run the router in the background, swallowing everything.

    An exception escaping a background task is swallowed by asyncio and logged
    nowhere useful, and it would leave a user with no reply and no record of
    why. Catching it here and printing is the difference between a diagnosable
    outage and a silent one.
    """
    try:
        await state.router.handle_payload(payload)
    except Exception as exc:  # noqa: BLE001 - last line of defence
        print("[bot] background handler failed: %r" % (exc,), flush=True)


# The instance uvicorn serves. Created at import so `uvicorn confamai_bot.webhook:app`
# works, but only when N-ATLaS credentials are present — otherwise the import
# should not fail for a `--test` run or a status check.
try:
    app = create_app()
except Exception as _exc:  # pragma: no cover - import-time guard
    print("[bot] app not created at import: %r" % (_exc,), flush=True)
    app = None  # type: ignore[assignment]