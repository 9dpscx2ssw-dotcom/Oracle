from __future__ import annotations

import asyncio

from gungnir.core.scheduler import Scheduler


def test_repeated_loop_failures_stop_scheduler_and_escalate():
    failures: list[tuple[str, int]] = []

    async def broken_loop() -> None:
        raise RuntimeError("invariant failure")

    async def run() -> None:
        scheduler = Scheduler(0.001, 0.001,
                               on_critical=lambda label, count: failures.append((label, count)))
        await asyncio.wait_for(scheduler._loop(broken_loop, 0.001, "fast"), timeout=1)

    asyncio.run(run())
    assert failures == [("fast", 5)]
