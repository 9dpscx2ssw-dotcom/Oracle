"""Kraken strategy indicators — 26 trading strategies' technical foundations.

All indicators are pure numpy/pandas functions, suitable for backtesting and
parameter optimization. Ported from kraken_feed.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _s(v) -> float:
    """Safe float conversion with rounding."""
    try:
        return round(float(v), 6)
    except (ValueError, TypeError):
        return 0.0


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Commodity Channel Index."""
    s = pd.Series(close)
    h = pd.Series(high)
    lo = pd.Series(low)
    n = min(period, len(s) - 1)
    tp = (h + lo + s) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return ((tp - ma) / (0.015 * md)).values


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    s = pd.Series(close)
    ema_fast = s.ewm(span=min(fast, len(s) - 1), adjust=False).mean()
    ema_slow = s.ewm(span=min(slow, len(s) - 1), adjust=False).mean()
    ml = ema_fast - ema_slow
    sl = ml.ewm(span=min(signal, len(ml) - 1), adjust=False).mean()
    return ml.values, sl.values, (ml - sl).values


def parabolic_sar(high: np.ndarray, low: np.ndarray, af_start=0.02, af_max=0.2):
    """Parabolic SAR with trend."""
    sar = low.copy().astype(float)
    trend = np.ones(len(high), dtype=float)
    ep = high.copy().astype(float)
    af = np.full(len(high), af_start, dtype=float)

    for i in range(2, len(high)):
        p_sar, p_trend, p_ep, p_af = (
            sar[i - 1],
            trend[i - 1],
            ep[i - 1],
            af[i - 1],
        )
        if p_trend == 1:
            new_sar = min(
                p_sar + p_af * (p_ep - p_sar), low[i - 1], low[i - 2]
            )
            if low[i] < new_sar:
                trend[i], sar[i], ep[i], af[i] = (
                    -1,
                    p_ep,
                    low[i],
                    af_start,
                )
            else:
                trend[i] = 1
                sar[i] = new_sar
                ep[i] = high[i] if high[i] > p_ep else p_ep
                af[i] = (
                    min(p_af + af_start, af_max)
                    if high[i] > p_ep
                    else p_af
                )
        else:
            new_sar = max(
                p_sar + p_af * (p_ep - p_sar), high[i - 1], high[i - 2]
            )
            if high[i] > new_sar:
                trend[i], sar[i], ep[i], af[i] = (
                    1,
                    p_ep,
                    high[i],
                    af_start,
                )
            else:
                trend[i] = -1
                sar[i] = new_sar
                ep[i] = low[i] if low[i] < p_ep else p_ep
                af[i] = (
                    min(p_af + af_start, af_max)
                    if low[i] < p_ep
                    else p_af
                )
    return sar, trend


def bollinger(close: np.ndarray, period: int = 20, dev: float = 2.0):
    """Bollinger Bands upper, mid, lower."""
    s = pd.Series(close)
    n = min(period, len(s) - 1)
    ma = s.rolling(n).mean()
    std = s.rolling(n).std()
    upper = ma + dev * std
    lower = ma - dev * std
    return upper.values, ma.values, lower.values


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic oscillator %K and %D."""
    h = pd.Series(high)
    lo = pd.Series(low)
    c = pd.Series(close)
    n = min(period, len(c) - 1)
    ll = lo.rolling(n).min()
    hh = h.rolling(n).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, 1e-10)
    k_smooth_val = k.rolling(k_smooth).mean()
    d_smooth_val = k_smooth_val.rolling(d_smooth).mean()
    return k_smooth_val.values, d_smooth_val.values


def rsi(close: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index (last value)."""
    s = pd.Series(close)
    n = min(period, len(s) - 1)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-10)
    return float(100 - (100 / (1 + rs)).iloc[-1])


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """ADX with +DI and -DI."""
    h = pd.Series(high)
    lo = pd.Series(low)
    c = pd.Series(close)
    prev_close = c.shift(1)
    prev_high = h.shift(1)
    prev_low = lo.shift(1)
    tr = pd.concat(
        [
            h - lo,
            (h - prev_close).abs(),
            (lo - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up_move = h - prev_high
    down_move = prev_low - lo
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = (
        100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, 1e-10)
    )
    minus_di = (
        100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, 1e-10)
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val.values, plus_di.values, minus_di.values


def momentum(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Momentum: close / close[period] * 100."""
    s = pd.Series(close)
    return (s / s.shift(period).replace(0, 1e-10) * 100).values


def sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (min_periods=1 so it's defined before fully warmed)."""
    return pd.Series(close).rolling(period, min_periods=1).mean().values


def smma(close: np.ndarray, period: int) -> np.ndarray:
    """Smoothed moving average (Wilder), alpha = 1/period."""
    return pd.Series(close).ewm(alpha=1.0 / period, adjust=False).mean().values


def awesome_oscillator(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Awesome Oscillator: SMA(midpoint, 5) - SMA(midpoint, 34)."""
    h = pd.Series(high)
    lo = pd.Series(low)
    mid = (h + lo) / 2
    return (mid.rolling(5).mean() - mid.rolling(34).mean()).values


def hma(close: np.ndarray, period: int = 55) -> np.ndarray:
    """Hull Moving Average."""
    s = pd.Series(close)
    half = max(int(period / 2), 1)
    sqrtn = max(int(period ** 0.5), 1)

    def wma(series, p):
        w = np.arange(1, p + 1, dtype=float)
        return series.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    return wma(2 * wma(s, half) - wma(s, period), sqrtn).values


def donchian_trend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20):
    """Donchian Trend: +1 when close > midpoint, -1 otherwise."""
    h = pd.Series(high)
    lo = pd.Series(low)
    c = pd.Series(close)
    upper = h.rolling(period).max()
    lower = lo.rolling(period).min()
    mid = (upper + lower) / 2
    trend = np.where(c > mid, 1, -1)
    return trend, upper.values, lower.values, mid.values


def vwap(df: pd.DataFrame) -> np.ndarray:
    """Session-anchored VWAP (UTC calendar day reset for crypto)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    day = df.index.normalize()
    cum_pv = (tp * vol).groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).values


def fair_value_gaps(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    openp: np.ndarray,
    lookback: int = 50,
    min_body_ratio: float = 0.5,
) -> list[dict]:
    """Detect Fair Value Gaps (bullish/bearish imbalances)."""
    gaps = []
    start = max(2, len(high) - lookback)
    for i in range(start, len(high)):
        a_high, a_low = high[i - 2], low[i - 2]
        c_high, c_low = high[i], low[i]
        b_o, b_c = openp[i - 1], close[i - 1]
        b_h, b_l = high[i - 1], low[i - 1]
        rng = b_h - b_l
        big_candle = rng > 0 and (abs(b_c - b_o) / rng) >= min_body_ratio

        if a_high < c_low and big_candle:
            gaps.append(
                {
                    "type": "BULL",
                    "fvg_top": round(c_low, 6),
                    "fvg_bot": round(a_high, 6),
                    "mid": round((c_low + a_high) / 2, 6),
                }
            )
        elif a_low > c_high and big_candle:
            gaps.append(
                {
                    "type": "BEAR",
                    "fvg_top": round(a_low, 6),
                    "fvg_bot": round(c_high, 6),
                    "mid": round((a_low + c_high) / 2, 6),
                }
            )
    return list(reversed(gaps))


def alligator(high: np.ndarray, low: np.ndarray, pj=13, pt=8, pl=5, sj=8, st=5, sl=3):
    """Williams Alligator: SMMA lines shifted."""
    h = pd.Series(high)
    lo = pd.Series(low)
    med = (h + lo) / 2

    def smma(s, p):
        return s.ewm(alpha=1.0 / p, adjust=False).mean()

    jaw = smma(med, pj).shift(sj)
    teeth = smma(med, pt).shift(st)
    lips = smma(med, pl).shift(sl)
    return jaw.values, teeth.values, lips.values
