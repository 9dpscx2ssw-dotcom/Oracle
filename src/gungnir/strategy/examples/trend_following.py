"""EMA-crossover trend strategy, biased by LLM prediction and order-book pressure.

Demonstrates the intended pattern: deterministic core (EMA cross) modulated by
the soft signals (prediction confidence, order-book imbalance) rather than driven
by them. Conviction blends the technical edge with the LLM/microstructure context.
"""

from __future__ import annotations

from ...data.models import Side, Signal
from ...features.feature_store import FeatureSet
from ..base import Strategy


class TrendFollowing(Strategy):
    name = "trend_following"

    def generate(self, features: FeatureSet) -> list[Signal]:
        if not self.trades_symbol(features.symbol):
            return []

        min_conviction = self.params.get("min_conviction", 0.3)

        # Deterministic core: fast EMA above slow EMA = uptrend.
        trend_up = features.ema_fast > features.ema_slow
        base_side = Side.BUY if trend_up else Side.SELL
        # Strength of the cross, normalized by ATR so it's comparable across assets.
        gap = abs(features.ema_fast - features.ema_slow)
        edge = min(1.0, gap / features.atr) if features.atr else 0.3

        conviction = 0.6 * edge + 0.4 * self._soft_bias(features, trend_up)

        if conviction < min_conviction:
            return []

        return [
            Signal(
                strategy=self.name,
                symbol=features.symbol,
                side=base_side,
                conviction=round(conviction, 3),
                rationale=f"EMA {'up' if trend_up else 'down'} cross, edge={edge:.2f}",
            )
        ]

    def _soft_bias(self, f: FeatureSet, trend_up: bool) -> float:
        """0..1 agreement of soft signals (prediction + order book) with the trend."""
        score = 0.5
        if f.prediction:
            agree = (f.prediction.direction > 0) == trend_up
            score = f.prediction.confidence if agree else (1 - f.prediction.confidence)
        if f.orderbook is not None:
            ob_up = f.orderbook.imbalance > 0
            score = 0.5 * score + 0.5 * (0.5 + 0.5 * (1 if ob_up == trend_up else -1)
                                         * abs(f.orderbook.imbalance))
        return max(0.0, min(1.0, score))
