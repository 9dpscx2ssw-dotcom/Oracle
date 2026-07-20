"""Transaction-cost model for backtests and paper fills.

Mid-price candles overstate performance: in reality you cross the bid/ask spread
on entry *and* exit, pay commission on notional, and eat slippage on top. Filling
both sides at the candle close (mid) makes every strategy look better than it is —
catastrophically so for high-frequency ones (scalp/FVG/M1) where the round-trip
cost can exceed the per-trade edge. This applies all three so backtest and shadow
numbers reflect tradeable performance.

Everything is in basis points (1 bp = 0.01%) of price/notional, so the model is
instrument-scale invariant (works for EURUSD at 1.1 and US30 at 40,000 alike).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.models import Side


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 0.0       # full bid/ask spread, bps of price
    commission_bps: float = 0.0   # per-side commission, bps of notional
    slippage_bps: float = 0.0     # extra adverse fill beyond half-spread, bps of price

    @classmethod
    def from_config(cls, cfg, **override) -> "CostModel":
        """Build from config `costs.*` with optional per-call overrides (e.g. a
        backtest request). Override values that are None are ignored."""
        def pick(key, default):
            v = override.get(key)
            if v is None:
                v = cfg.get("costs", key, default=default) if cfg else default
            return float(v)
        return cls(
            spread_bps=pick("spread_bps", 0.0),
            commission_bps=pick("commission_bps", 0.0),
            slippage_bps=pick("slippage_bps", 0.0),
        )

    @property
    def zero(self) -> bool:
        return self.spread_bps == 0 and self.commission_bps == 0 and self.slippage_bps == 0

    def fill_price(self, side: Side, mid: float, *, opening: bool) -> float:
        """Adverse fill: pay half the spread + slippage in the direction that hurts.

        Opening a BUY (or closing a SELL → you buy back) lifts the ask; opening a
        SELL (or closing a BUY → you sell out) hits the bid.
        """
        edge = (self.spread_bps / 2.0 + self.slippage_bps) / 10_000.0
        buying = (side == Side.BUY) if opening else (side == Side.SELL)
        return mid * (1.0 + edge) if buying else mid * (1.0 - edge)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000.0
