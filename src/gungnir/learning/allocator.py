"""Continuous capital allocator: one objective for all strategy-level learning.

Binary controls (demote, prune) answer "should this strategy exist?"; the
allocator answers the everyday question "how much capital does its next
signal deserve *right now*?" — continuously, from evidence, per regime.

Mechanics:
  • every closed trade (real + shadow) contributes to its strategy's score,
    exponentially decayed by recency, so last week matters more than last
    month and the weight adapts as conditions change;
  • score = decayed mean PnL / decayed mean |PnL|  ∈ [-1, 1] — a bounded,
    scale-free expectancy ratio (a strategy that wins and loses the same
    dollar amounts with 50/50 odds scores 0);
  • weight = clip(1 + score, floor, cap): neutral evidence → 1.0×, strong
    edge → up to 1.5×, demonstrated bleed → down to 0.1× — never zero, so a
    weak strategy keeps producing (tiny) shadow evidence and can earn its
    way back when the market changes. Hard on/off remains the job of
    auto-demotion and the kill switch;
  • when a (strategy, regime) pair has enough of its own trades, the weight
    conditions on the *current regime* — trend-followers sized up in trends
    and down in chop, mean-reverters the opposite — falling back to the
    strategy's overall record, then to neutral 1.0 while evidence is thin.

The multiplier applies AFTER the sizer and BEFORE the risk caps, so it can
never enlarge a position past what PortfolioRisk.vet() permits.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class CapitalAllocator:
    def __init__(self, *, floor: float = 0.1, cap: float = 1.5,
                 decay: float = 0.99, min_trades: int = 15,
                 lookback: int = 400):
        self.floor = floor
        self.cap = cap
        self.decay = decay            # per-trade recency decay (newest = 1.0)
        self.min_trades = min_trades  # evidence required before deviating from 1.0
        self.lookback = lookback
        # (strategy, regime|None) -> (weight, n_trades); refreshed each slow loop.
        self._weights: dict[tuple[str, str | None], tuple[float, int]] = {}

    # ── scoring ────────────────────────────────────────────────────────────────

    def _score(self, pnls: list[float]) -> float:
        """Decayed expectancy ratio in [-1, 1]. ``pnls`` newest-first."""
        num = den = w = 0.0
        for i, p in enumerate(pnls):
            w = self.decay ** i
            num += w * p
            den += w * abs(p)
        return (num / den) if den > 0 else 0.0

    def _weight_from(self, pnls: list[float]) -> float:
        return max(self.floor, min(self.cap, 1.0 + self._score(pnls)))

    def refresh(self, journal) -> None:
        """Recompute all weights from the journal (called each slow loop)."""
        weights: dict[tuple[str, str | None], tuple[float, int]] = {}
        try:
            by_strat: dict[str, list] = {}
            for t in journal.recent(limit=self.lookback * 4):
                if t.pnl is None or not t.strategy or t.mode == "learning":
                    continue
                by_strat.setdefault(t.strategy, []).append(t)

            for name, trades in by_strat.items():
                trades = trades[:self.lookback]           # newest-first from the DB
                pnls = [t.pnl for t in trades]
                if len(pnls) >= self.min_trades:
                    weights[(name, None)] = (self._weight_from(pnls), len(pnls))
                # Regime-conditional weights from the entry-regime stamp.
                by_regime: dict[str, list[float]] = {}
                for t in trades:
                    r = (t.context or {}).get("regime")
                    if r:
                        by_regime.setdefault(r, []).append(t.pnl)
                for r, rp in by_regime.items():
                    if len(rp) >= self.min_trades:
                        weights[(name, r)] = (self._weight_from(rp), len(rp))
            self._weights = weights
            if weights:
                worst = min(weights.items(), key=lambda kv: kv[1][0])
                best = max(weights.items(), key=lambda kv: kv[1][0])
                log.info("Allocator refreshed over %d strategies: best %s=%.2fx, "
                         "worst %s=%.2fx", len(by_strat), best[0], best[1][0],
                         worst[0], worst[1][0])
        except Exception as e:  # noqa: BLE001 — allocation must not break the loop
            log.warning("Allocator refresh failed: %s", e)

    # ── lookup ─────────────────────────────────────────────────────────────────

    def weight(self, strategy: str, regime: str | None = None) -> float:
        """Sizing multiplier for this strategy in this regime (1.0 = neutral)."""
        if regime is not None and (strategy, regime) in self._weights:
            return self._weights[(strategy, regime)][0]
        if (strategy, None) in self._weights:
            return self._weights[(strategy, None)][0]
        return 1.0

    def snapshot(self) -> dict:
        """JSON-safe view for the dashboard / status file."""
        out: dict[str, dict] = {}
        for (name, regime), (w, n) in self._weights.items():
            entry = out.setdefault(name, {"overall": None, "n": 0, "by_regime": {}})
            if regime is None:
                entry["overall"] = round(w, 3)
                entry["n"] = n
            else:
                entry["by_regime"][regime] = {"weight": round(w, 3), "n": n}
        return out
