"""Technical indicators. Pure functions over close-price series (numpy arrays).

Kept dependency-light (numpy only) so the base Docker image stays small. Swap in
`ta`/`pandas-ta` later if you want a bigger battery of indicators.
"""

from __future__ import annotations

import numpy as np


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < period + 1:
        return 50.0
    deltas = np.diff(values)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < 2:
        return 0.0
    prev_close = close[:-1]
    tr = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ]
    )
    return float(tr[-period:].mean())


def bollinger(values: np.ndarray, period: int = 20, n_std: float = 2.0) -> tuple[float, float, float]:
    """Return (lower, mid, upper) bands for the latest point."""
    if len(values) < period:
        mid = float(values.mean()) if len(values) else 0.0
        return mid, mid, mid
    window = values[-period:]
    mid = float(window.mean())
    sd = float(window.std())
    return mid - n_std * sd, mid, mid + n_std * sd
