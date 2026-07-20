"""Position sizing: turn a conviction + volatility into a trade size.

Three sizers, selectable via config `risk.sizer`:
  • fixed_fractional — risk a fixed % of equity per trade (stop distance = k·ATR).
  • vol_target       — size so each position contributes a target volatility.
  • kelly            — fractional Kelly from the strategy's historical edge.

All return *volume* in the broker's units; the portfolio manager then applies
account-level caps on top.

Note: account leverage deliberately does NOT scale these sizes. Leverage is a
margin constraint, not a risk multiplier — it is enforced as a max-notional cap
in PortfolioRisk.vet(). (Audit F-04: multiplying sizer output by usable leverage
turned a 0.5%-risk trade into a ~90%-of-equity risk at 200x.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..data.models import Signal
from ..features.feature_store import FeatureSet


class PositionSizer(ABC):
    @abstractmethod
    def size(self, signal: Signal, features: FeatureSet, equity: float) -> float: ...


class FixedFractional(PositionSizer):
    def __init__(self, config: Config):
        self.risk_per_trade = config.get("risk", "account_risk_per_trade", default=0.005)
        self.stop_atr_mult = config.get("risk", "stop_atr_mult", default=2.0)

    def size(self, signal: Signal, features: FeatureSet, equity: float) -> float:
        stop_distance = self.stop_atr_mult * features.atr
        if stop_distance <= 0:
            return 0.0
        risk_cash = equity * self.risk_per_trade * signal.conviction
        return max(0.0, risk_cash / stop_distance)


_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240,
               "1d": 1440, "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60,
               "H4": 240, "D1": 1440}


class VolTarget(PositionSizer):
    """Size so the position's ANNUALIZED volatility contribution hits the target.

    The previous formula divided target cash vol by the raw per-bar ATR with no
    time scaling — on a 5m EURUSD bar that produced ~$1M notional on a $10k
    account, which the per-asset cap then silently truncated, erasing the whole
    sizing chain (conviction, allocator, sentiment scaling) behind it.

        volume = equity × target_vol × conviction / (ATR × √bars_per_year)

    ATR × √bars_per_year is the annualized price-unit volatility of one unit of
    the instrument; dividing target cash vol by it yields units.
    """

    def __init__(self, config: Config):
        self.target_vol = config.get("risk", "vol_target_annual", default=0.10)

    @staticmethod
    def _bars_per_year(features: FeatureSet) -> float:
        candles = getattr(features, "candles", None) or []
        tf = candles[0].timeframe if candles else "5m"
        minutes = _TF_MINUTES.get(str(tf), 5)
        return 365.0 * 24.0 * 60.0 / minutes

    def size(self, signal: Signal, features: FeatureSet, equity: float) -> float:
        atr = features.atr
        if atr <= 0:
            return 0.0
        ann_vol_per_unit = atr * self._bars_per_year(features) ** 0.5
        target_cash_vol = equity * self.target_vol * signal.conviction
        return max(0.0, target_cash_vol / ann_vol_per_unit)


class FractionalKelly(PositionSizer):
    def __init__(self, config: Config):
        self.fraction = config.get("risk", "kelly_fraction", default=0.25)

    def size(self, signal: Signal, features: FeatureSet, equity: float) -> float:
        # Expects the strategy/journal to supply win_prob & payoff in context.
        # Falls back to conviction-as-edge if stats are missing.
        ctx = getattr(features.prediction, "confidence", None)
        p = ctx if ctx is not None else signal.conviction
        b = 1.0  # assume 1:1 payoff until the evaluator provides a real one
        kelly = max(0.0, (p * (b + 1) - 1) / b)
        return equity * self.fraction * kelly


def build_sizer(config: Config) -> PositionSizer:
    name = config.get("risk", "sizer", default="vol_target")
    return {
        "fixed_fractional": FixedFractional,
        "vol_target": VolTarget,
        "kelly": FractionalKelly,
    }.get(name, VolTarget)(config)
