"""Two-clock scheduler: a fast loop for sensing/trading and a slow loop for
learning. Both are plain asyncio so the whole agent runs in one process/container.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, fast_seconds: float, slow_seconds: float, on_critical=None):
        self.fast_seconds = fast_seconds
        self.slow_seconds = slow_seconds
        # Optional callback(label, consecutive_failures) fired at escalation
        # points so an operator alert can page a human, not just a log line.
        self.on_critical = on_critical
        self._stop = asyncio.Event()
        # Last completed iteration's wall time per loop label ("fast"/"slow"),
        # published to status.json so overruns are visible on the dashboard.
        self.last_duration: dict[str, float] = {}

    def stop(self) -> None:
        self._stop.set()

    async def run(
        self,
        fast: Callable[[], Awaitable[None]],
        slow: Callable[[], Awaitable[None]],
    ) -> None:
        await asyncio.gather(
            self._loop(fast, self.fast_seconds, "fast"),
            self._loop(slow, self.slow_seconds, "slow"),
        )

    async def _loop(self, fn: Callable[[], Awaitable[None]], interval: float, label: str) -> None:
        import time
        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                await fn()
                failures = 0
                # Loop duration is the #1 operational metric: a loop that takes
                # longer than its interval is trading on stale decisions.
                elapsed = time.monotonic() - started
                self.last_duration[label] = round(elapsed, 3)
                from . import metrics
                metrics.observe_loop(label, elapsed)
                if elapsed > interval:
                    log.warning("%s loop took %.1fs (interval %.0fs) — iterations "
                                "are overrunning their budget.", label, elapsed, interval)
                else:
                    log.debug("%s loop completed in %.2fs", label, elapsed)
            except Exception:  # noqa: BLE001 — one bad cycle must not kill the loop
                failures += 1
                log.exception("%s loop iteration failed (%d consecutive)", label, failures)
                # A persistent crash-loop is a dead bot wearing a heartbeat —
                # escalate so it can't hide in DEBUG-level noise (audit F-07).
                if failures == 5 or failures % 50 == 0:
                    log.critical(
                        "%s loop has failed %d times in a row — the agent is NOT "
                        "trading correctly. Investigate the traceback above.",
                        label, failures)
                    if self.on_critical is not None:
                        try:
                            self.on_critical(label, failures)
                        except Exception:  # noqa: BLE001 — alerting must not kill the loop
                            log.exception("on_critical callback failed")
                    # A loop that has failed five consecutive times is not
                    # healthy enough to continue making or evaluating trades.
                    # Stop all scheduler loops so the service fails closed rather
                    # than presenting a live heartbeat over a dead trading loop.
                    if failures == 5:
                        self._stop.set()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
