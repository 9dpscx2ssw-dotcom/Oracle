"""Encode a (Signal, FeatureSet) pair into a fixed-length, normalized state vector.

The RL policy is a *meta-decision* layer: the strategies do the chart analysis and
emit a hard signal; the policy looks at that signal together with the market
context and decides whether the signal is worth taking. To learn across many
different market conditions, every input has to land on a comparable scale — so
each raw feature is squashed into roughly [-1, 1] here (tanh / clip), which keeps
the small network well-conditioned without any batch-norm machinery.

`STATE_DIM` and `FEATURE_NAMES` are the contract: the network input width and the
human-readable label for each slot (handy for debugging and the dashboard).
"""

from __future__ import annotations

import numpy as np

from ...data.models import Side, Signal
from ...features.feature_store import FeatureSet

FEATURE_NAMES: tuple[str, ...] = (
    "conviction",       # strategy's own confidence in the signal, 0..1
    "side",             # +1 long / -1 short
    "rsi",              # (rsi-50)/50
    "trend",            # ema_fast vs ema_slow, tanh-scaled
    "price_vs_ema",     # last vs ema_fast, tanh-scaled
    "bb_position",      # where price sits in the Bollinger band, ~[-1,1]
    "volatility",       # atr / price, tanh-scaled
    "macd_hist",        # macd histogram / atr, tanh-scaled
    "stochastic",       # (stoch_k-50)/50
    "adx",              # trend strength, tanh(adx/50)
    "sentiment",        # LLM sentiment score, -1..1
    "sentiment_conf",   # LLM sentiment confidence, 0..1
    "prediction_agree", # LLM prediction agreement with the signal side, -1..1
    "portfolio_heat",   # fraction of the open-position budget already in use
    # v2 additions — without these the policy pooled every strategy and hour
    # into one undifferentiated stream (a cci_reversal signal and a donchian
    # signal looked identical except for conviction):
    "strat_id_0",       # ┐
    "strat_id_1",       # │ stable 4-dim hash embedding of the strategy name
    "strat_id_2",       # │ (deterministic across restarts)
    "strat_id_3",       # ┘
    "regime_trend_low",   # ┐
    "regime_trend_high",  # │ one-hot market regime at decision time
    "regime_range_low",   # │
    "regime_range_high",  # ┘
    "hour_sin",         # time-of-day (UTC), cyclically encoded
    "hour_cos",
)

STATE_DIM: int = len(FEATURE_NAMES)

_REGIME_SLOTS = ("trend_low", "trend_high", "range_low", "range_high")


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _strategy_embedding(name: str) -> tuple[float, float, float, float]:
    """Deterministic 4-dim embedding of a strategy name in [-1, 1]."""
    import hashlib
    h = hashlib.sha256((name or "").encode()).digest()
    return tuple(h[i] / 127.5 - 1.0 for i in range(4))  # type: ignore[return-value]


def encode(signal: Signal, features: FeatureSet, portfolio_heat: float = 0.0,
           regime: str | None = None, hour: float | None = None) -> np.ndarray:
    """Build the normalized state vector for a signal in its market context.

    Robust to the plain ``FeatureSet`` as well as ``KrakenFeatureSet`` — every
    extended field is read through ``getattr`` with a neutral default so a missing
    indicator simply contributes zero rather than raising.
    """
    last = features.last_price or 0.0
    atr = features.atr or 0.0
    # Use ATR as the natural price scale; fall back to a small fraction of price.
    scale = atr if atr > 0 else (abs(last) * 0.001 or 1.0)

    side_sign = 1.0 if signal.side == Side.BUY else (-1.0 if signal.side == Side.SELL else 0.0)

    # Bollinger position: -1 at lower band, +1 at upper band.
    half_band = (features.bb_upper - features.bb_lower) / 2.0
    bb_position = float(np.clip(_safe_div(last - features.bb_mid, half_band), -2.0, 2.0)) / 2.0

    macd_hist = float(getattr(features, "macd_hist", 0.0))
    stoch_k = float(getattr(features, "stoch_k", 50.0))
    adx = float(getattr(features, "adx", 0.0))

    sent_score = features.sentiment.score if features.sentiment else 0.0
    sent_conf = features.sentiment.confidence if features.sentiment else 0.0

    pred_agree = 0.0
    if features.prediction is not None:
        pred_agree = side_sign * float(features.prediction.direction) * features.prediction.confidence

    sid = _strategy_embedding(signal.strategy)
    regime_onehot = [1.0 if regime == r else 0.0 for r in _REGIME_SLOTS]
    if hour is None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
    angle = 2.0 * np.pi * (hour / 24.0)

    vec = np.array(
        [
            float(np.clip(signal.conviction, 0.0, 1.0)),
            side_sign,
            float(np.clip((features.rsi - 50.0) / 50.0, -1.0, 1.0)),
            float(np.tanh(_safe_div(features.ema_fast - features.ema_slow, scale))),
            float(np.tanh(_safe_div(last - features.ema_fast, scale))),
            bb_position,
            float(np.tanh(_safe_div(atr, last) * 100.0)),
            float(np.tanh(_safe_div(macd_hist, scale))),
            float(np.clip((stoch_k - 50.0) / 50.0, -1.0, 1.0)),
            float(np.tanh(adx / 50.0)),
            float(np.clip(sent_score, -1.0, 1.0)),
            float(np.clip(sent_conf, 0.0, 1.0)),
            float(np.clip(pred_agree, -1.0, 1.0)),
            float(np.clip(portfolio_heat, 0.0, 1.0)),
            sid[0], sid[1], sid[2], sid[3],
            *regime_onehot,
            float(np.sin(angle)),
            float(np.cos(angle)),
        ],
        dtype=np.float64,
    )
    # Defensive: never let a stray NaN/inf reach the network.
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
