"""All 26 Kraken trading strategies.

Each strategy consumes a KrakenFeatureSet and emits zero or more Signals based
on its indicator logic. Strategies are deterministic and PARAMETERIZED: every
threshold is read through ``self.p(...)`` with the class default, and BOUNDS
declares the optimizer's search range — this is what makes the Bayesian /
walk-forward tuning loop able to act at all (previously every threshold was a
hardcoded literal and ``get_parameter_bounds()`` returned {} for all 26).

Conviction is graded where a natural strength measure exists, but never BELOW
``conviction_base`` — so at default parameters the firing behavior is exactly
the old one, while strong setups carry more weight through sizing and the RL
gate.
"""

from __future__ import annotations

from ..data.models import Signal, Side
from ..features.feature_store import KrakenFeatureSet
from .base import Strategy


def _sig(strategy: Strategy, features: KrakenFeatureSet, side: Side,
         conviction: float) -> list[Signal]:
    return [Signal(strategy=strategy.name, symbol=features.symbol, side=side,
                   conviction=max(0.0, min(1.0, conviction)))]


class CCIMACDStrategy(Strategy):
    """S1: CCI(14) + MACD(12,26,2) — M5 trend."""

    name = "cci_macd"
    family = "trend"
    DEFAULTS = {"cci_threshold": 100.0}
    BOUNDS = {"cci_threshold": (50.0, 200.0)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        thr = self.p("cci_threshold")
        cci = features.cci14
        macd = features.macd12_26
        conviction = min(abs(cci) / max(2 * thr, 1e-9), 1.0)
        if cci > thr and macd > 0:
            return _sig(self, features, Side.BUY, conviction)
        elif cci < -thr and macd < 0:
            return _sig(self, features, Side.SELL, conviction)
        return []


class ParSARCCIStrategy(Strategy):
    """S2: Parabolic SAR + CCI(45) + EMA(50) — M1."""

    name = "parsar_cci_ema"
    family = "trend"
    DEFAULTS = {"cci_threshold": 100.0, "conviction_base": 0.5}
    BOUNDS = {"cci_threshold": (50.0, 200.0), "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        thr = self.p("cci_threshold")
        base = self.p("conviction_base")
        cci = features.cci45
        price = features.last_price
        conviction = base + 0.3 * min(max(abs(cci) - thr, 0.0) / max(thr, 1e-9), 1.0)
        if features.sar_trend > 0 and price > features.ema50 and cci > thr:
            return _sig(self, features, Side.BUY, conviction)
        elif features.sar_trend < 0 and price < features.ema50 and cci < -thr:
            return _sig(self, features, Side.SELL, conviction)
        return []


class BBMACDStrategy(Strategy):
    """S3: Bollinger Bands(20,2) + MACD(11,27,4) + SMA(2) — M15."""

    name = "bb_macd_sma"
    family = "meanrev"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        macd = features.macd_11_27
        base = self.p("conviction_base")
        # Grade by how far price sits from the mid-band, in half-band units.
        half = max(features.bb_upper - features.bb_mid, 1e-9)
        depth = min(abs(price - features.bb_mid) / half, 1.0)
        conviction = base + 0.3 * depth
        # Momentum confirmation: long above the mid-band WITH positive MACD,
        # short below it WITH negative MACD. The previous version had the MACD
        # condition inverted (buy above mid on falling momentum), which showed
        # up live as a 1.8% win rate over 228 trades (audit F-16).
        if price > features.bb_mid and macd > 0:
            return _sig(self, features, Side.BUY, conviction)
        elif price < features.bb_mid and macd < 0:
            return _sig(self, features, Side.SELL, conviction)
        return []


class CCI200EMAStrategy(Strategy):
    """S4: CCI(200) + EMA(10,21,50) + Pivot Points — M5."""

    name = "cci200_ema_pivot"
    family = "trend"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        cci = features.cci200
        price = features.last_price
        conviction = self.p("conviction_base")
        if (features.ema10 > features.ema21 > features.ema50
                and cci > 0 and price > features.pivot):
            return _sig(self, features, Side.BUY, conviction)
        elif (features.ema10 < features.ema21 < features.ema50
                and cci < 0 and price < features.pivot):
            return _sig(self, features, Side.SELL, conviction)
        return []


class EMAStochRSIStrategy(Strategy):
    """S5: EMA(5,10) + Stochastic(14,3,3) + RSI(14) — H1."""

    name = "ema_stoch_rsi"
    family = "oscillator"
    DEFAULTS = {"rsi_mid": 50.0, "stoch_upper": 80.0, "stoch_lower": 20.0,
                "conviction_base": 0.5}
    BOUNDS = {"rsi_mid": (40.0, 60.0), "stoch_upper": (60.0, 95.0),
              "stoch_lower": (5.0, 40.0), "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        rsi_mid = self.p("rsi_mid")
        base = self.p("conviction_base")
        rsi = features.rsi
        stoch_k = features.stoch_k
        conviction = base + 0.3 * min(abs(rsi - 50.0) / 50.0, 1.0)
        if (features.ema5 > features.ema10 and rsi > rsi_mid
                and stoch_k < self.p("stoch_upper")):
            return _sig(self, features, Side.BUY, conviction)
        elif (features.ema5 < features.ema10 and rsi < rsi_mid
                and stoch_k > self.p("stoch_lower")):
            return _sig(self, features, Side.SELL, conviction)
        return []


class CCIReversalStrategy(Strategy):
    """S6: CCI(14) Reversal — H1."""

    name = "cci_reversal"
    family = "meanrev"
    DEFAULTS = {"cci_threshold": 100.0, "conviction_base": 0.5}
    BOUNDS = {"cci_threshold": (50.0, 250.0), "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        thr = self.p("cci_threshold")
        base = self.p("conviction_base")
        cci = features.cci14
        conviction = base + 0.3 * min(max(abs(cci) - thr, 0.0) / max(thr, 1e-9), 1.0)
        if cci < -thr:
            return _sig(self, features, Side.BUY, conviction)
        elif cci > thr:
            return _sig(self, features, Side.SELL, conviction)
        return []


class ADXMomentumStrategy(Strategy):
    """S7: ADX(14) + Momentum(14) + EMA(55) — M5."""

    name = "adx_momentum_ema"
    family = "trend"
    DEFAULTS = {"adx_threshold": 25.0, "momentum_mid": 100.0}
    BOUNDS = {"adx_threshold": (15.0, 40.0), "momentum_mid": (95.0, 105.0)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        adx_thr = self.p("adx_threshold")
        mom_mid = self.p("momentum_mid")
        adx = features.adx
        mom = features.momentum
        price = features.last_price
        if adx <= adx_thr:
            return []
        # Trend strength grades conviction: barely-trending 0.4 → strong 0.8.
        conviction = 0.4 + 0.4 * min((adx - adx_thr) / max(adx_thr, 1e-9), 1.0)
        if features.plus_di > features.minus_di and mom > mom_mid and price > features.ema55:
            return _sig(self, features, Side.BUY, conviction)
        elif features.plus_di < features.minus_di and mom < mom_mid and price < features.ema55:
            return _sig(self, features, Side.SELL, conviction)
        return []


class BBRSICuttingStrategy(Strategy):
    """S8: BB(20,2) + ADX(14) + RSI(7) — M5."""

    name = "bb_rsi_cutting"
    family = "meanrev"
    DEFAULTS = {"rsi_oversold": 30.0, "rsi_overbought": 70.0, "adx_max": 30.0,
                "conviction_base": 0.5}
    BOUNDS = {"rsi_oversold": (10.0, 40.0), "rsi_overbought": (60.0, 90.0),
              "adx_max": (20.0, 50.0), "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        rsi = features.rsi
        conviction = self.p("conviction_base")
        if features.adx >= self.p("adx_max"):
            return []
        if price <= features.bb_lower and rsi < self.p("rsi_oversold"):
            return _sig(self, features, Side.BUY, conviction)
        elif price >= features.bb_upper and rsi > self.p("rsi_overbought"):
            return _sig(self, features, Side.SELL, conviction)
        return []


class AwesomeOscillatorStrategy(Strategy):
    """S9: Awesome Oscillator + MACD(5,7,4) — H4."""

    name = "ao_macd"
    family = "oscillator"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        conviction = self.p("conviction_base")
        if features.ao > 0 and features.macd_5_7 > 0:
            return _sig(self, features, Side.BUY, conviction)
        elif features.ao < 0 and features.macd_5_7 < 0:
            return _sig(self, features, Side.SELL, conviction)
        return []


class BBRSIStrategy(Strategy):
    """S10: Bollinger Bands(20,2) + RSI(11) — M15."""

    name = "bb_rsi"
    family = "meanrev"
    DEFAULTS = {"rsi_overbought": 70.0, "rsi_oversold": 30.0, "conviction_base": 0.5}
    BOUNDS = {"rsi_overbought": (60.0, 90.0), "rsi_oversold": (10.0, 40.0),
              "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        rsi = features.rsi
        conviction = self.p("conviction_base")
        if rsi > self.p("rsi_overbought") and price > features.bb_upper:
            return _sig(self, features, Side.BUY, conviction)
        elif rsi < self.p("rsi_oversold") and price < features.bb_lower:
            return _sig(self, features, Side.SELL, conviction)
        return []


class IntelligentTradingStrategy(Strategy):
    """S11: SMMA(8,18) + ParSAR + Stoch + MACD — H1."""

    name = "intelligent_trading"
    family = "trend"
    DEFAULTS = {"stoch_mid": 50.0, "conviction_base": 0.6}
    BOUNDS = {"stoch_mid": (30.0, 70.0), "conviction_base": (0.3, 0.9)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        stoch_mid = self.p("stoch_mid")
        conviction = self.p("conviction_base")
        macd = features.macd12_26
        if (features.smma8 > features.smma18 and macd > 0
                and features.stoch_k < stoch_mid and features.sar_trend > 0):
            return _sig(self, features, Side.BUY, conviction)
        elif (features.smma8 < features.smma18 and macd < 0
                and features.stoch_k > stoch_mid and features.sar_trend < 0):
            return _sig(self, features, Side.SELL, conviction)
        return []


class MultiBBStrategy(Strategy):
    """S12: Multi-deviation BB(20; 2,3,4) — M1."""

    name = "multi_bb"
    family = "meanrev"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        base = self.p("conviction_base")
        # Deeper penetration past the band → higher conviction.
        half = max(features.bb_upper - features.bb_mid, 1e-9)
        if price <= features.bb_lower:
            depth = min((features.bb_lower - price) / half, 1.0)
            return _sig(self, features, Side.BUY, base + 0.3 * depth)
        elif price >= features.bb_upper:
            depth = min((price - features.bb_upper) / half, 1.0)
            return _sig(self, features, Side.SELL, base + 0.3 * depth)
        return []


class MACDStochStrategy(Strategy):
    """S13: MACD(13,26,9) + Stochastic(5,3,3) — M1."""

    name = "macd_stoch"
    family = "oscillator"
    DEFAULTS = {"stoch_lower": 20.0, "stoch_upper": 80.0, "conviction_base": 0.5}
    BOUNDS = {"stoch_lower": (5.0, 40.0), "stoch_upper": (60.0, 95.0),
              "conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        macd = features.macd_13_26
        stoch_k = features.stoch_k
        conviction = self.p("conviction_base")
        if macd > 0 and stoch_k < self.p("stoch_lower"):
            return _sig(self, features, Side.BUY, conviction)
        elif macd < 0 and stoch_k > self.p("stoch_upper"):
            return _sig(self, features, Side.SELL, conviction)
        return []


class AlligatorStrategy(Strategy):
    """S14: Williams Alligator(13/8/5) + SMA(144) — M15."""

    name = "alligator"
    family = "trend"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        conviction = self.p("conviction_base")
        if (features.alligator_lips > features.alligator_teeth > features.alligator_jaw
                and price > features.sma144):
            return _sig(self, features, Side.BUY, conviction)
        elif (features.alligator_lips < features.alligator_teeth < features.alligator_jaw
                and price < features.sma144):
            return _sig(self, features, Side.SELL, conviction)
        return []


class _HMADonchianBase(Strategy):
    """HMA(55) + Donchian(20) trend-following, shared across six timeframes."""

    family = "channel"
    DEFAULTS = {"conviction_base": 0.5}
    BOUNDS = {"conviction_base": (0.3, 0.8)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        base = self.p("conviction_base")
        # Distance from the HMA in ATR units grades trend maturity.
        atr = features.atr or 0.0
        stretch = min(abs(price - features.hma55) / atr, 1.0) if atr > 0 else 0.0
        conviction = base + 0.3 * stretch
        if price > features.hma55 and features.dc_trend > 0:
            return _sig(self, features, Side.BUY, conviction)
        elif price < features.hma55 and features.dc_trend < 0:
            return _sig(self, features, Side.SELL, conviction)
        return []


class HMADonchianM1Strategy(_HMADonchianBase):
    """S15: HMA(55) + Donchian(20) — M1."""
    name = "hma_dc_m1"


class HMADonchianM5Strategy(_HMADonchianBase):
    """S16: HMA(55) + Donchian(20) — M5."""
    name = "hma_dc_m5"


class HMADonchianM15Strategy(_HMADonchianBase):
    """S17: HMA(55) + Donchian(20) — M15."""
    name = "hma_dc_m15"


class HMADonchianH1Strategy(_HMADonchianBase):
    """S18: HMA(55) + Donchian(20) — H1."""
    name = "hma_dc_h1"


class HMADonchianH4Strategy(_HMADonchianBase):
    """S19: HMA(55) + Donchian(20) — H4."""
    name = "hma_dc_h4"


class HMADonchianD1Strategy(_HMADonchianBase):
    """S20: HMA(55) + Donchian(20) — D1."""
    name = "hma_dc_d1"


class _FVGBase(Strategy):
    """Fair Value Gap retracement entries, shared across four timeframes."""

    family = "structure"
    DEFAULTS = {"conviction_base": 0.6}
    BOUNDS = {"conviction_base": (0.3, 0.9)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price
        conviction = self.p("conviction_base")
        if (features.fvg_bull_bot > 0
                and features.fvg_bull_bot <= price <= features.fvg_bull_top):
            return _sig(self, features, Side.BUY, conviction)
        if (features.fvg_bear_bot > 0
                and features.fvg_bear_bot <= price <= features.fvg_bear_top):
            return _sig(self, features, Side.SELL, conviction)
        return []


class FVGM1Strategy(_FVGBase):
    """S21: Fair Value Gap — M1."""
    name = "fvg_m1"


class FVGM5Strategy(_FVGBase):
    """S22: Fair Value Gap — M5."""
    name = "fvg_m5"


class FVGM15Strategy(_FVGBase):
    """S23: Fair Value Gap — M15."""
    name = "fvg_m15"


class FVGM30Strategy(_FVGBase):
    """S24: Fair Value Gap — M30."""
    name = "fvg_m30"


class _ScalpEMAVWAPBase(Strategy):
    """9-EMA × VWAP scalp, shared across two timeframes.

    ``min_gap_bps`` (default 0 = old behavior) requires the EMA to clear the
    VWAP by a minimum distance — the raw crossover fires on every bar of noise.
    """

    family = "scalp"
    DEFAULTS = {"conviction_base": 0.6, "min_gap_bps": 0.0}
    BOUNDS = {"conviction_base": (0.3, 0.9), "min_gap_bps": (0.0, 20.0)}

    def generate(self, features: KrakenFeatureSet) -> list[Signal]:
        if not isinstance(features, KrakenFeatureSet):
            return []
        price = features.last_price or 0.0
        if price <= 0 or features.vwap_val <= 0:
            return []
        gap_bps = (features.ema5 - features.vwap_val) / price * 10_000.0
        conviction = self.p("conviction_base")
        min_gap = self.p("min_gap_bps")
        if gap_bps > min_gap:
            return _sig(self, features, Side.BUY, conviction)
        elif gap_bps < -min_gap:
            return _sig(self, features, Side.SELL, conviction)
        return []


class FashionablyLateScalpM1Strategy(_ScalpEMAVWAPBase):
    """S25: Fashionably Late Scalp — 9 EMA × VWAP — M1."""
    name = "scalp_ema_vwap_m1"


class FashionablyLateScalpM5Strategy(_ScalpEMAVWAPBase):
    """S26: Fashionably Late Scalp — 9 EMA × VWAP — M5."""
    name = "scalp_ema_vwap_m5"


# Registry of all 26 strategies
KRAKEN_STRATEGIES = [
    CCIMACDStrategy,
    ParSARCCIStrategy,
    BBMACDStrategy,
    CCI200EMAStrategy,
    EMAStochRSIStrategy,
    CCIReversalStrategy,
    ADXMomentumStrategy,
    BBRSICuttingStrategy,
    AwesomeOscillatorStrategy,
    BBRSIStrategy,
    IntelligentTradingStrategy,
    MultiBBStrategy,
    MACDStochStrategy,
    AlligatorStrategy,
    HMADonchianM1Strategy,
    HMADonchianM5Strategy,
    HMADonchianM15Strategy,
    HMADonchianH1Strategy,
    HMADonchianH4Strategy,
    HMADonchianD1Strategy,
    FVGM1Strategy,
    FVGM5Strategy,
    FVGM15Strategy,
    FVGM30Strategy,
    FashionablyLateScalpM1Strategy,
    FashionablyLateScalpM5Strategy,
]
