"""Two levels of concurrency control.

The N-ATLaS server handles requests in a threadpool and generates on one shared
4-bit GPU, so concurrency there is a queue, not a pool. `chat_completions` is a
sync `def`, which means FastAPI will happily run dozens of them at once, and
uvicorn is configured with `limit_concurrency=32`. Thirty-two simultaneous
generations on one Tesla T4 is not thirty-two answers arriving together: it is a
wall of 507s, a wall of 429s, and a wall of very slow replies.

So two primitives, and both are necessary:

**A per-user lock.** One person sending three messages in quick succession must
get three answers in order. Without a lock the turns race, and the history is
written in completion order rather than conversation order — so the model is
shown a conversation that never happened. Two locks for the same person
serialising against each other is exactly the intended behaviour.

**A global semaphore.** This is the one that keeps the bot inside what one GPU
can serve. Three is the default because a 4-bit model on a T4 takes about
thirteen seconds per turn; three in flight keeps the pipeline full without
piling up.

Waiting on either is bounded by `per_user_timeout`. Without that bound a stuck
request holds a semaphore slot forever and the bot degrades to zero throughput
with no error and no log line — the worst possible failure mode, because
nothing looks broken.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from .config import normalize_phone


class QueueTimeout(Exception):
    """A slot could not be acquired within the timeout."""


class UserBusy(Exception):
    """This user already has a turn in flight and would not finish in time."""


class Concurrency:
    """The per-user locks and the global semaphore."""

    def __init__(self, max_concurrency: int, per_user_timeout: float = 120.0):
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.per_user_timeout = max(5.0, float(per_user_timeout))
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.in_flight = 0
        self.max_seen = 0
        self.timeouts = 0

    def user_lock(self, phone: str) -> asyncio.Lock:
        return self._locks[normalize_phone(phone)]

    def forget(self, phone: str) -> None:
        """Drop a user's lock once they have been idle long enough.

        Without this the lock table grows by one entry per phone number that
        ever wrote a message, which on a public number is unbounded. Only
        called from the sweeper, and only for a user with nothing in flight, so
        it cannot delete a lock somebody is holding.
        """
        self._locks.pop(normalize_phone(phone), None)

    def snapshot(self) -> dict:
        return {
            "in_flight": self.in_flight,
            "max_seen": self.max_seen,
            "timeouts": self.timeouts,
            "known_users": len(self._locks),
        }

    async def run(self, phone: str, coro_factory):
        """Run `coro_factory()` for `phone` under both locks.

        `coro_factory` is a callable rather than a coroutine on purpose. A
        coroutine is created when this function is called, so a request that
        then waits thirty seconds for a semaphore slot has already built a
        coroutine nobody is running — and worse, if this call is ever cancelled
        while queued, that orphaned coroutine is never awaited and Python warns
        about it. Taking a factory means nothing is constructed until a slot is
        actually held.
        """
        lock = self.user_lock(phone)
        queued_at = time.monotonic()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.per_user_timeout)
        except asyncio.TimeoutError as exc:
            self.timeouts += 1
            raise UserBusy(
                "This user already has a reply in progress."
            ) from exc

        # Measured here, before the semaphore, so the number logged is the wait
        # to *start* and not the whole turn. A T4 turn takes ~13s on its own, so
        # timing after the model call would report every healthy turn as a
        # queue problem and send an operator looking in the wrong place.
        slot_wait = time.monotonic() - queued_at
        try:
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(), timeout=self.per_user_timeout
                )
            except asyncio.TimeoutError as exc:
                self.timeouts += 1
                raise QueueTimeout(
                    "The verification server is busy. Please try again shortly."
                ) from exc
            queued_at = time.monotonic()
            try:
                self.in_flight += 1
                self.max_seen = max(self.max_seen, self.in_flight)
                return await coro_factory()
            finally:
                self.in_flight -= 1
                self.semaphore.release()
        finally:
            lock.release()
            held = time.monotonic() - queued_at
            if slot_wait > 2.0 or held > 2.0:
                print(
                    "[queue] %s: waited %.1fs for the user lock, held a GPU slot "
                    "for %.1fs" % (normalize_phone(phone), slot_wait, held),
                    flush=True,
                )