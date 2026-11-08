#!/usr/bin/env python3
"""FastAPI web service that refreshes the NAFDAC Greenbook every week.

This is a long-running web service rather than a cron job: the process stays
up, answers health and status requests, and an in-process scheduler thread runs
the scrape + PostgreSQL load once a week. `POST /run` triggers the same job on
demand.

The actual work is delegated to ``weekly_nafdac_refresh.py``, which runs the
scraper and then ``nafdac_source.load_postgres``. That keeps a single tested
entry point for both the scheduled run and a manual run.

Run locally::

    uvicorn nafdac_refresh_service:app --app-dir nafdac_scraper --host 0.0.0.0 --port 8000

Environment:
    NAFDAC_DATABASE_URL          hosted PostgreSQL DSN (required for the load)
    NAFDAC_REFRESH_DOW           weekday: mon..sun (default: sun)
    NAFDAC_REFRESH_HOUR          hour 0-23 (default: 3)
    NAFDAC_REFRESH_MINUTE        minute 0-59 (default: 0)
    NAFDAC_REFRESH_TIMEZONE      IANA zone (default: UTC)
    NAFDAC_REFRESH_API_KEY       bearer token for POST /run and GET /logs
    NAFDAC_SCRAPER_INSECURE_TLS  set to 1 only while NAFDAC's cert is expired

Run a **single** uvicorn worker. The scheduler lives in the process, so two
workers would schedule the same weekly job twice. A free Render instance spins
down after inactivity and would miss the schedule, so use a paid instance type.
"""

from __future__ import annotations

import hmac
import logging
import os
import subprocess
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Optional

from fastapi import Depends, FastAPI, Header, HTTPException

HERE = Path(__file__).resolve().parent          # nafdac_scraper/
REPO_ROOT = HERE.parent                          # repository root
RUNNER = HERE / "weekly_nafdac_refresh.py"


def _load_dotenv() -> None:
    """Read `.env` files so a local run sees nafdac_scraper/.env.

    The host's real environment variables always win. python-dotenv is optional;
    without it this is a no-op. Loaded once at import, before any env is read.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(HERE / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


_load_dotenv()

log = logging.getLogger("nafdac.refresh")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Schedule:
    weekday: int
    hour: int
    minute: int
    tz: object
    tz_name: str

    def describe(self) -> dict:
        return {
            "day": _WEEKDAY_NAMES[self.weekday],
            "hour": self.hour,
            "minute": self.minute,
            "timezone": self.tz_name,
        }


def _parse_weekday(value: str) -> int:
    key = (value or "sun").strip().lower()
    if key not in _WEEKDAYS:
        log.warning("unknown NAFDAC_REFRESH_DOW=%r; using 'sun'", value)
        return 6
    return _WEEKDAYS[key]


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("invalid %s=%r; using %s", name, raw, default)
        return default
    if not low <= value <= high:
        log.warning("%s=%s out of range %s-%s; using %s", name, value, low, high, default)
        return default
    return value


def _load_timezone(name: str):
    name = (name or "UTC").strip() or "UTC"
    if name.upper() in {"UTC", "GMT", "Z"}:
        return timezone.utc, "UTC"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name), name
    except Exception as exc:  # missing tzdata on Windows, unknown zone, etc.
        log.warning("unknown timezone %r (%s); using UTC", name, exc)
        return timezone.utc, "UTC"


def _load_schedule() -> Schedule:
    tz, tz_name = _load_timezone(os.environ.get("NAFDAC_REFRESH_TIMEZONE", "UTC"))
    return Schedule(
        weekday=_parse_weekday(os.environ.get("NAFDAC_REFRESH_DOW", "sun")),
        hour=_env_int("NAFDAC_REFRESH_HOUR", 3, 0, 23),
        minute=_env_int("NAFDAC_REFRESH_MINUTE", 0, 0, 59),
        tz=tz,
        tz_name=tz_name,
    )


def _next_run(now: datetime, schedule: Schedule) -> datetime:
    """The next occurrence of the weekly slot, strictly after `now`."""
    candidate = now.replace(
        hour=schedule.hour, minute=schedule.minute, second=0, microsecond=0
    )
    candidate += timedelta(days=(schedule.weekday - candidate.weekday()) % 7)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
@dataclass
class _State:
    running: bool = False
    trigger: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    ok: Optional[bool] = None
    error: str = ""


_state = _State()
_state_lock = threading.Lock()
_run_lock = threading.Lock()
_recent: Deque[str] = deque(maxlen=300)
_stop_event = threading.Event()
_next_run_at: Optional[datetime] = None


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _append_log(line: str) -> None:
    _recent.append(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {line}")


def run_refresh(trigger: str = "manual") -> bool:
    """Run the scrape + load. Blocking, so callers use a thread."""
    if not _run_lock.acquire(blocking=False):
        log.warning("a refresh is already running; skipping %s trigger", trigger)
        return False

    started = datetime.now(timezone.utc)
    with _state_lock:
        _state.running = True
        _state.trigger = trigger
        _state.started_at = started
        _state.finished_at = None
        _state.exit_code = None
        _state.ok = None
        _state.error = ""

    log.info("refresh started (%s)", trigger)
    _append_log(f"--- refresh started ({trigger}) ---")

    try:
        try:
            process = subprocess.Popen(
                [sys.executable, str(RUNNER)],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            message = f"could not start the refresh runner: {exc}"
            log.error(message)
            _append_log(message)
            with _state_lock:
                _state.running = False
                _state.finished_at = datetime.now(timezone.utc)
                _state.ok = False
                _state.error = message
            return False

        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            if line:
                _append_log(line)
                log.info("%s", line)

        code = process.wait()
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        ok = code == 0
        with _state_lock:
            _state.running = False
            _state.finished_at = datetime.now(timezone.utc)
            _state.exit_code = code
            _state.ok = ok
            if not ok:
                _state.error = f"refresh exited with code {code}"
        log.info("refresh finished with exit %s in %.1f min", code, elapsed / 60)
        _append_log(f"--- refresh finished: exit {code} in {elapsed / 60:.1f} min ---")
        return ok
    finally:
        _run_lock.release()


def trigger_async(trigger: str = "manual") -> bool:
    """Start a refresh in the background. False when one is already running."""
    if _run_lock.locked():
        return False
    with _state_lock:
        if _state.running:
            return False
    threading.Thread(
        target=run_refresh,
        args=(trigger,),
        name=f"refresh-{trigger}",
        daemon=True,
    ).start()
    return True


# ---------------------------------------------------------------------------
# Weekly scheduler thread
# ---------------------------------------------------------------------------
def _scheduler_loop(schedule: Schedule) -> None:
    global _next_run_at
    while not _stop_event.is_set():
        next_dt = _next_run(datetime.now(schedule.tz), schedule)
        _next_run_at = next_dt.astimezone(timezone.utc)
        log.info("next scheduled refresh: %s", _next_run_at.isoformat())

        while not _stop_event.is_set():
            remaining = (next_dt - datetime.now(schedule.tz)).total_seconds()
            if remaining <= 0:
                break
            _stop_event.wait(timeout=min(remaining, 60.0))

        if _stop_event.is_set():
            break

        if not trigger_async("schedule"):
            log.warning("scheduled refresh skipped: another refresh is running")
        # Do not re-enter the same minute before recomputing the next slot.
        _stop_event.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    schedule = _load_schedule()
    app.state.schedule = schedule
    log.info("NAFDAC refresh service starting; schedule=%s", schedule.describe())
    if not (os.environ.get("NAFDAC_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        log.warning(
            "NAFDAC_DATABASE_URL is not set; the load step will fail and the "
            "database will not be updated"
        )
    if not os.environ.get("NAFDAC_REFRESH_API_KEY"):
        log.warning(
            "NAFDAC_REFRESH_API_KEY is not set; POST /run and GET /logs are "
            "disabled. Set it to enable manual triggers"
        )

    _stop_event.clear()
    scheduler = threading.Thread(
        target=_scheduler_loop,
        args=(schedule,),
        name="weekly-scheduler",
        daemon=True,
    )
    scheduler.start()
    try:
        yield
    finally:
        _stop_event.set()
        scheduler.join(timeout=5)


app = FastAPI(
    title="NAFDAC Greenbook weekly refresh",
    description=(
        "Runs the NAFDAC Greenbook scrape and PostgreSQL load on a weekly "
        "schedule, and exposes manual trigger, status and log endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> bool:
    expected = os.environ.get("NAFDAC_REFRESH_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail=(
                "NAFDAC_REFRESH_API_KEY is not configured; the trigger and log "
                "endpoints are disabled"
            ),
        )
    token = ""
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


def _schedule() -> Schedule:
    return getattr(app.state, "schedule", None) or _load_schedule()


@app.get("/")
def root() -> dict:
    return {
        "service": "nafdac-weekly-refresh",
        "status": "ok",
        "endpoints": ["/healthz", "/status", "/run", "/logs", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/status")
def refresh_status() -> dict:
    schedule = _schedule()
    with _state_lock:
        running = _state.running
        last_run = {
            "trigger": _state.trigger,
            "started_at": _iso(_state.started_at),
            "finished_at": _iso(_state.finished_at),
            "exit_code": _state.exit_code,
            "ok": _state.ok,
            "error": _state.error,
        }
    return {
        "service": "nafdac-weekly-refresh",
        "running": running,
        "database_configured": bool(
            os.environ.get("NAFDAC_DATABASE_URL") or os.environ.get("DATABASE_URL")
        ),
        "schedule": schedule.describe(),
        "next_scheduled_run": _iso(_next_run_at),
        "last_run": last_run,
    }


@app.post("/run", status_code=202)
def run_now(_: bool = Depends(require_api_key)) -> dict:
    if not trigger_async("manual"):
        raise HTTPException(status_code=409, detail="a refresh is already running")
    return {"status": "accepted", "message": "refresh started"}


@app.get("/logs")
def recent_logs(
    _: bool = Depends(require_api_key),
    limit: int = 100,
) -> dict:
    limit = max(1, min(limit, 300))
    return {"lines": list(_recent)[-limit:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "nafdac_refresh_service:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
    )