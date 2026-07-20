"""Loss-streak cooldown: the anti-revenge-trading guard.

In a sustained one-direction move, level-based strategies re-arm every time
their condition releases — so a mean-reversion strategy can buy a falling
market, get stopped out, and re-enter a bar later, all the way down (this is
exactly the loop a 16h incident log showed on the index symbols). Brackets cap
each loss; nothing capped the *sequence*.

This guard tracks consecutive losing round-trips per (strategy, symbol). After
``max_streak`` losses in a row, that strategy is benched on that symbol for
``cooldown_minutes``. Any winning close resets the streak (and lifts an active
bench — the market stopped disagreeing). Scoped per strategy+symbol so one
strategy's bad day doesn't bench the others.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class LossStreakGuard:
    def __init__(self, max_streak: int = 3, cooldown_minutes: float = 60.0):
        self.max_streak = int(max_streak or 0)
        self.cooldown_seconds = float(cooldown_minutes or 0.0) * 60.0
        self._streak: dict[tuple[str, str], int] = {}
        self._last_loss: dict[tuple[str, str], float] = {}

    @property
    def enabled(self) -> bool:
        return self.max_streak > 0 and self.cooldown_seconds > 0

    def record(self, strategy: str, symbol: str, pnl: float,
               now: float | None = None) -> None:
        """Feed every closed round-trip's PnL (wins reset, losses accumulate)."""
        if not self.enabled or not strategy or not symbol:
            return
        key = (strategy, symbol)
        if pnl < 0:
            streak = self._streak.get(key, 0) + 1
            self._streak[key] = streak
            self._last_loss[key] = time.monotonic() if now is None else now
            if streak == self.max_streak:
                log.warning(
                    "%s has lost %d in a row on %s — benched for %.0f minutes.",
                    strategy, streak, symbol, self.cooldown_seconds / 60.0)
        else:
            self._streak.pop(key, None)
            self._last_loss.pop(key, None)

    def blocked_seconds(self, strategy: str, symbol: str,
                        now: float | None = None) -> float:
        """Seconds of bench time remaining for this strategy on this symbol
        (0 when clear to trade)."""
        if not self.enabled:
            return 0.0
        key = (strategy, symbol)
        if self._streak.get(key, 0) < self.max_streak:
            return 0.0
        now = time.monotonic() if now is None else now
        remaining = self.cooldown_seconds - (now - self._last_loss.get(key, 0.0))
        if remaining <= 0:
            # Cooldown served: allow ONE re-entry — a further loss re-benches
            # immediately (streak stays above threshold), a win clears it.
            self._streak[key] = self.max_streak - 1
            return 0.0
        return remaining
