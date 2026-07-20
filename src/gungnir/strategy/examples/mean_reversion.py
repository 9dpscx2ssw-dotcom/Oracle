"""RSI + Bollinger mean-reversion strategy.

Fades extremes: buy when oversold and price pierces the lower band, sell when
overbought at the upper band. Sentiment acts as a veto — don't fade into a strong
news-driven move.
"""

from __future__ import annotations

from ...data.models import Side, Signal
from ...features.feature_store import FeatureSet
from ..base import Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"

    def generate(self, features: FeatureSet) -> list[Signal]:
        if not self.trades_symbol(features.symbol):
            return []

        oversold = self.params.get("rsi_oversold", 30)
        overbought = self.params.get("rsi_overbought", 70)
        price = features.last_price

        side: Side | None = None
        if features.rsi <= oversold and price <= features.bb_lower:
            side = Side.BUY
        elif features.rsi >= overbought and price >= features.bb_upper:
            side = Side.SELL
        if side is None:
            return []

        # Sentiment veto: skip fading a strongly-aligned-against-us news flow.
        if features.sentiment and features.sentiment.confidence > 0.6:
            if side == Side.BUY and features.sentiment.score < -0.5:
                return []
            if side == Side.SELL and features.sentiment.score > 0.5:
                return []

        # Conviction scales with how far RSI is past the threshold.
        if side == Side.BUY:
            conviction = min(1.0, (oversold - features.rsi) / oversold + 0.5)
        else:
            conviction = min(1.0, (features.rsi - overbought) / (100 - overbought) + 0.5)

        return [
            Signal(
                strategy=self.name,
                symbol=features.symbol,
                side=side,
                conviction=round(max(0.0, conviction), 3),
                rationale=f"RSI={features.rsi:.0f} at band extreme",
            )
        ]
