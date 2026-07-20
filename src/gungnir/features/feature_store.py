"""Assemble a per-asset FeatureSet from raw market data + external context.

The FeatureSet is the single object a Strategy consumes. It bundles deterministic
technical/microstructure features with the (optional) LLM-derived sentiment and
prediction so a strategy can weigh both.

KrakenFeatureSet extends this with all 26 Kraken strategy indicators for
comprehensive backtesting and parameter optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..data.models import Candle, MacroIndicator, OrderBook, Prediction, Sentiment
from . import indicators
from . import kraken_indicators
from .orderbook import OrderBookFeatures, analyze


class FeatureSet(BaseModel):
    symbol: str
    last_price: float
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    rsi: float = 50.0
    atr: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_upper: float = 0.0
    orderbook: OrderBookFeatures | None = None
    sentiment: Sentiment | None = None
    prediction: Prediction | None = None
    macro: dict[str, float] = Field(default_factory=dict)


def build(
    symbol: str,
    candles: list[Candle],
    book: OrderBook | None = None,
    sentiment: Sentiment | None = None,
    prediction: Prediction | None = None,
    macro: list[MacroIndicator] | None = None,
    fast_ema: int = 20,
    slow_ema: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> FeatureSet:
    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    last = float(closes[-1]) if len(closes) else 0.0

    ema_fast = float(indicators.ema(closes, fast_ema)[-1]) if len(closes) else last
    ema_slow = float(indicators.ema(closes, slow_ema)[-1]) if len(closes) else last
    bb_lower, bb_mid, bb_upper = indicators.bollinger(closes, bb_period, bb_std)

    return FeatureSet(
        symbol=symbol,
        last_price=last,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi=indicators.rsi(closes, rsi_period),
        atr=indicators.atr(highs, lows, closes, atr_period),
        bb_lower=bb_lower,
        bb_mid=bb_mid,
        bb_upper=bb_upper,
        orderbook=analyze(book) if book else None,
        sentiment=sentiment,
        prediction=prediction,
        macro={m.name: m.value for m in (macro or [])},
    )


def _at(arr, i: int, default: float = 0.0) -> float:
    """Safe scalar read from an indicator array at bar ``i`` (NaN/short → default)."""
    try:
        v = float(arr[i])
    except (IndexError, TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def build_kraken_series(symbol: str, candles: list[Candle]) -> list["KrakenFeatureSet"]:
    """Vectorized per-bar KrakenFeatureSet builder for backtests.

    Computes every indicator array *once* over the full series and then slices it
    per bar, instead of rebuilding all 26 indicators (and a pandas DataFrame) on a
    rolling window each step. This turns an O(bars × window) backtest into O(bars),
    which is the difference between seconds and many minutes for a UI request.
    """
    n = len(candles)
    if n == 0:
        return []

    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    opens = np.array([c.open for c in candles], dtype=float)
    vols = np.array([getattr(c, "volume", 1.0) for c in candles], dtype=float)

    # Array indicators (full series, one pass each).
    ema_fast_a = indicators.ema(closes, 20)
    ema_slow_a = indicators.ema(closes, 50)
    ema5_a, ema10_a = indicators.ema(closes, 5), indicators.ema(closes, 10)
    ema21_a, ema50_a, ema55_a = indicators.ema(closes, 21), indicators.ema(closes, 50), indicators.ema(closes, 55)
    sma2_a, sma144_a = kraken_indicators.sma(closes, 2), kraken_indicators.sma(closes, 144)
    smma8_a, smma18_a = kraken_indicators.smma(closes, 8), kraken_indicators.smma(closes, 18)
    cci14_a = kraken_indicators.cci(highs, lows, closes, 14)
    cci45_a = kraken_indicators.cci(highs, lows, closes, 45)
    cci200_a = kraken_indicators.cci(highs, lows, closes, 200)
    ml12_a, sl12_a, hist12_a = kraken_indicators.macd(closes, 12, 26, 2)
    ml11_a, _, _ = kraken_indicators.macd(closes, 11, 27, 4)
    ml13_a, _, _ = kraken_indicators.macd(closes, 13, 26, 9)
    ml5_a, _, _ = kraken_indicators.macd(closes, 5, 7, 4)
    sar_a, sar_trend_a = kraken_indicators.parabolic_sar(highs, lows)
    k_a, d_a = kraken_indicators.stochastic(highs, lows, closes, 14, 3, 3)
    adx_a, pdi_a, mdi_a = kraken_indicators.adx(highs, lows, closes, 14)
    mom_a = kraken_indicators.momentum(closes, 14)
    ao_a = kraken_indicators.awesome_oscillator(highs, lows)
    hma55_a = kraken_indicators.hma(closes, 55)
    dc_trend_a, dc_u_a, dc_l_a, dc_m_a = kraken_indicators.donchian_trend(highs, lows, closes, 20)
    jaw_a, teeth_a, lips_a = kraken_indicators.alligator(highs, lows)

    # VWAP is cumulative from the series start, so the full-series value at bar i is
    # the running VWAP up to i — correct without re-slicing.
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                       "volume": vols},
                      index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq="min"))
    vwap_a = kraken_indicators.vwap(df)

    # Bollinger bands per bar (rolling window, computed once as arrays).
    bb_u_a, bb_m_a, bb_l_a = kraken_indicators.bollinger(closes, 20, 2.0)

    out: list[KrakenFeatureSet] = []
    for i in range(n):
        last = float(closes[i])
        # Fair-value gaps as of bar i (cheap O(lookback) scan over recent bars).
        lb0 = max(0, i - 50)
        fvgs = kraken_indicators.fair_value_gaps(
            highs[lb0:i + 1], lows[lb0:i + 1], closes[lb0:i + 1], opens[lb0:i + 1], lookback=50)
        bull = next((g for g in fvgs if g["type"] == "BULL"), None)
        bear = next((g for g in fvgs if g["type"] == "BEAR"), None)
        # Pivots from the prior bar.
        if i >= 1:
            piv = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3
            piv_r1, piv_s1 = 2 * piv - lows[i - 1], 2 * piv - highs[i - 1]
        else:
            piv = piv_r1 = piv_s1 = last
        out.append(KrakenFeatureSet(
            symbol=symbol, last_price=last,
            ema_fast=_at(ema_fast_a, i, last), ema_slow=_at(ema_slow_a, i, last),
            rsi=_at_rsi(closes, i), atr=_at_atr(highs, lows, closes, i),
            bb_lower=_at(bb_l_a, i), bb_mid=_at(bb_m_a, i), bb_upper=_at(bb_u_a, i),
            cci14=_at(cci14_a, i), cci45=_at(cci45_a, i), cci200=_at(cci200_a, i),
            macd12_26=_at(ml12_a, i), macd_signal=_at(sl12_a, i), macd_hist=_at(hist12_a, i),
            macd_11_27=_at(ml11_a, i), macd_13_26=_at(ml13_a, i), macd_5_7=_at(ml5_a, i),
            sar=_at(sar_a, i), sar_trend=_at(sar_trend_a, i, 1.0),
            stoch_k=_at(k_a, i, 50.0), stoch_d=_at(d_a, i, 50.0),
            ema5=_at(ema5_a, i, last), ema10=_at(ema10_a, i, last), ema21=_at(ema21_a, i, last),
            ema50=_at(ema50_a, i, last), ema55=_at(ema55_a, i, last),
            sma2=_at(sma2_a, i, last), sma144=_at(sma144_a, i, last),
            smma8=_at(smma8_a, i, last), smma18=_at(smma18_a, i, last),
            adx=_at(adx_a, i), plus_di=_at(pdi_a, i), minus_di=_at(mdi_a, i),
            momentum=_at(mom_a, i, 100.0), ao=_at(ao_a, i),
            hma55=_at(hma55_a, i), dc_upper=_at(dc_u_a, i), dc_lower=_at(dc_l_a, i),
            dc_mid=_at(dc_m_a, i), dc_trend=_at(dc_trend_a, i, 1.0),
            vwap_val=_at(vwap_a, i),
            fvg_bull_top=bull["fvg_top"] if bull else 0.0,
            fvg_bull_bot=bull["fvg_bot"] if bull else 0.0,
            fvg_bear_top=bear["fvg_top"] if bear else 0.0,
            fvg_bear_bot=bear["fvg_bot"] if bear else 0.0,
            alligator_jaw=_at(jaw_a, i), alligator_teeth=_at(teeth_a, i), alligator_lips=_at(lips_a, i),
            pivot=float(piv), pivot_r1=float(piv_r1), pivot_s1=float(piv_s1),
        ))
    return out


def _at_rsi(closes: np.ndarray, i: int) -> float:
    seg = closes[: i + 1]
    return indicators.rsi(seg, 14) if len(seg) >= 15 else 50.0


def _at_atr(highs, lows, closes, i: int) -> float:
    s = slice(max(0, i - 30), i + 1)
    seg = closes[s]
    return indicators.atr(highs[s], lows[s], seg, 14) if len(seg) >= 15 else 0.0


class KrakenFeatureSet(FeatureSet):
    """Extended FeatureSet with all 26 Kraken strategy indicators."""

    # CCI indicators
    cci14: float = 0.0
    cci45: float = 0.0
    cci200: float = 0.0

    # MACD (various periods)
    macd12_26: float = 0.0  # MACD line
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_11_27: float = 0.0
    macd_13_26: float = 0.0
    macd_5_7: float = 0.0

    # Parabolic SAR
    sar: float = 0.0
    sar_trend: float = 1.0  # 1 or -1

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # Additional MAs
    ema5: float = 0.0
    ema10: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    ema55: float = 0.0
    sma2: float = 0.0
    sma144: float = 0.0
    smma8: float = 0.0
    smma18: float = 0.0

    # Advanced indicators
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    momentum: float = 100.0
    ao: float = 0.0  # Awesome Oscillator
    hma55: float = 0.0
    dc_upper: float = 0.0
    dc_lower: float = 0.0
    dc_mid: float = 0.0
    dc_trend: float = 1.0

    # VWAP
    vwap_val: float = 0.0

    # Fair Value Gaps (last 3)
    fvg_bull_top: float = 0.0
    fvg_bull_bot: float = 0.0
    fvg_bear_top: float = 0.0
    fvg_bear_bot: float = 0.0

    # Alligator
    alligator_jaw: float = 0.0
    alligator_teeth: float = 0.0
    alligator_lips: float = 0.0

    # Pivot Points
    pivot: float = 0.0
    pivot_r1: float = 0.0
    pivot_s1: float = 0.0

    # Candle history (for indicators that need it)
    candles: list[Candle] = Field(default_factory=list)


def build_kraken(
    symbol: str,
    candles: list[Candle],
    book: OrderBook | None = None,
    sentiment: Sentiment | None = None,
    prediction: Prediction | None = None,
    macro: list[MacroIndicator] | None = None,
) -> KrakenFeatureSet:
    """Build enriched KrakenFeatureSet with all 26 strategy indicators."""
    if not candles:
        return KrakenFeatureSet(
            symbol=symbol,
            last_price=0.0,
            candles=candles,
            orderbook=analyze(book) if book else None,
            sentiment=sentiment,
            prediction=prediction,
            macro={m.name: m.value for m in (macro or [])},
        )

    closes = np.array([c.close for c in candles], dtype=float)
    highs = np.array([c.high for c in candles], dtype=float)
    lows = np.array([c.low for c in candles], dtype=float)
    opens = np.array([c.open for c in candles], dtype=float)
    last = float(closes[-1])

    # Build dataframe for VWAP and other functions that need it
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.array([getattr(c, "volume", 1.0) for c in candles], dtype=float),
        },
        index=pd.date_range(end=pd.Timestamp.now(), periods=len(candles), freq="min"),
    )

    # Core indicators
    ema_fast = float(indicators.ema(closes, 20)[-1])
    ema_slow = float(indicators.ema(closes, 50)[-1])
    bb_lower, bb_mid, bb_upper = indicators.bollinger(closes, 20, 2.0)  # Returns tuple of 3 floats

    # CCI
    cci14_arr = kraken_indicators.cci(highs, lows, closes, 14)
    cci45_arr = kraken_indicators.cci(highs, lows, closes, 45)
    cci200_arr = kraken_indicators.cci(highs, lows, closes, 200)

    # MACD variants
    ml12, sl12, hist12 = kraken_indicators.macd(closes, 12, 26, 2)
    ml11, sl11, hist11 = kraken_indicators.macd(closes, 11, 27, 4)
    ml13, sl13, hist13 = kraken_indicators.macd(closes, 13, 26, 9)
    ml5, sl5, hist5 = kraken_indicators.macd(closes, 5, 7, 4)

    # SAR
    sar, sar_trend = kraken_indicators.parabolic_sar(highs, lows)

    # Stochastic
    k, d = kraken_indicators.stochastic(highs, lows, closes, 14, 3, 3)

    # EMAs
    ema5 = float(indicators.ema(closes, 5)[-1])
    ema10 = float(indicators.ema(closes, 10)[-1])
    ema21 = float(indicators.ema(closes, 21)[-1])
    ema50 = float(indicators.ema(closes, 50)[-1])
    ema55 = float(indicators.ema(closes, 55)[-1])

    # SMAs / SMMAs (used by alligator's SMA(144) filter and intelligent_trading).
    sma2 = float(kraken_indicators.sma(closes, 2)[-1])
    sma144 = float(kraken_indicators.sma(closes, 144)[-1])
    smma8 = float(kraken_indicators.smma(closes, 8)[-1])
    smma18 = float(kraken_indicators.smma(closes, 18)[-1])

    # ADX
    adx_arr, pdi, mdi = kraken_indicators.adx(highs, lows, closes, 14)

    # Momentum & AO
    mom = kraken_indicators.momentum(closes, 14)
    ao = kraken_indicators.awesome_oscillator(highs, lows)

    # HMA & Donchian
    hma55 = float(kraken_indicators.hma(closes, 55)[-1])
    dc_trend, dc_u, dc_l, dc_m = kraken_indicators.donchian_trend(highs, lows, closes, 20)

    # VWAP
    vwap = kraken_indicators.vwap(df)

    # FVG
    fvgs = kraken_indicators.fair_value_gaps(highs, lows, closes, opens, lookback=50)
    bull_gaps = [g for g in fvgs if g["type"] == "BULL"]
    bear_gaps = [g for g in fvgs if g["type"] == "BEAR"]

    # Alligator
    jaw, teeth, lips = kraken_indicators.alligator(highs, lows)

    # Pivot Points (from previous bar)
    if len(closes) >= 2:
        h_prev, l_prev, c_prev = highs[-2], lows[-2], closes[-2]
        piv = (h_prev + l_prev + c_prev) / 3
        piv_r1 = 2 * piv - l_prev
        piv_s1 = 2 * piv - h_prev
    else:
        piv = piv_r1 = piv_s1 = last

    return KrakenFeatureSet(
        symbol=symbol,
        last_price=last,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi=indicators.rsi(closes, 14),
        atr=indicators.atr(highs, lows, closes, 14),
        bb_lower=bb_lower,
        bb_mid=bb_mid,
        bb_upper=bb_upper,
        # CCI
        cci14=float(cci14_arr[-1]) if len(cci14_arr) else 0.0,
        cci45=float(cci45_arr[-1]) if len(cci45_arr) else 0.0,
        cci200=float(cci200_arr[-1]) if len(cci200_arr) else 0.0,
        # MACD
        macd12_26=float(ml12[-1]) if len(ml12) else 0.0,
        macd_signal=float(sl12[-1]) if len(sl12) else 0.0,
        macd_hist=float(hist12[-1]) if len(hist12) else 0.0,
        macd_11_27=float(ml11[-1]) if len(ml11) else 0.0,
        macd_13_26=float(ml13[-1]) if len(ml13) else 0.0,
        macd_5_7=float(ml5[-1]) if len(ml5) else 0.0,
        # SAR
        sar=float(sar[-1]) if len(sar) else 0.0,
        sar_trend=float(sar_trend[-1]) if len(sar_trend) else 1.0,
        # Stochastic
        stoch_k=float(k[-1]) if len(k) else 50.0,
        stoch_d=float(d[-1]) if len(d) else 50.0,
        # EMAs
        ema5=ema5,
        ema10=ema10,
        ema21=ema21,
        ema50=ema50,
        ema55=ema55,
        # SMAs / SMMAs
        sma2=sma2,
        sma144=sma144,
        smma8=smma8,
        smma18=smma18,
        # ADX
        adx=float(adx_arr[-1]) if len(adx_arr) else 0.0,
        plus_di=float(pdi[-1]) if len(pdi) else 0.0,
        minus_di=float(mdi[-1]) if len(mdi) else 0.0,
        # Momentum
        momentum=float(mom[-1]) if len(mom) else 100.0,
        ao=float(ao[-1]) if len(ao) else 0.0,
        # HMA & Donchian
        hma55=hma55,
        dc_upper=float(dc_u[-1]) if len(dc_u) else 0.0,
        dc_lower=float(dc_l[-1]) if len(dc_l) else 0.0,
        dc_mid=float(dc_m[-1]) if len(dc_m) else 0.0,
        dc_trend=float(dc_trend[-1]) if len(dc_trend) else 1.0,
        # VWAP
        vwap_val=float(vwap[-1]) if len(vwap) else 0.0,
        # FVG
        fvg_bull_top=float(bull_gaps[0]["fvg_top"]) if bull_gaps else 0.0,
        fvg_bull_bot=float(bull_gaps[0]["fvg_bot"]) if bull_gaps else 0.0,
        fvg_bear_top=float(bear_gaps[0]["fvg_top"]) if bear_gaps else 0.0,
        fvg_bear_bot=float(bear_gaps[0]["fvg_bot"]) if bear_gaps else 0.0,
        # Alligator
        alligator_jaw=float(jaw[-1]) if len(jaw) else 0.0,
        alligator_teeth=float(teeth[-1]) if len(teeth) else 0.0,
        alligator_lips=float(lips[-1]) if len(lips) else 0.0,
        # Pivot
        pivot=float(piv),
        pivot_r1=float(piv_r1),
        pivot_s1=float(piv_s1),
        # Context
        candles=candles,
        orderbook=analyze(book) if book else None,
        sentiment=sentiment,
        prediction=prediction,
        macro={m.name: m.value for m in (macro or [])},
    )
