"""A sequential trading MDP for *offline* reinforcement learning.

The live policy (`policy.py`) is a one-step contextual bandit that only vetoes
pre-formed signals. This environment reframes trading as a proper sequential
decision process so an offline learner can learn *position* decisions (and thus
holding and exits), with credit assigned across bars.

MDP
---
* **State**  : market features (from a KrakenFeatureSet) + position context
  (current position, holding time, unrealized PnL) + trading-context features
  (regime, session, intrabar-range/spread proxy) so the policy sees the same
  selectivity signals the pre-trade filters use. See ``STATE_DIM``.
* **Action** : target position — 0 = flat, 1 = long, 2 = short (discrete, for a
  safe/interpretable first cut; continuous sizing is a later upgrade).
* **Reward** : the **Differential Sharpe Ratio** (Moody & Saffell, 1998) of the
  step return *net of transaction costs*. DSR is the online increment to the
  Sharpe ratio, so maximizing cumulative DSR maximizes risk-adjusted, cost-aware
  return — exactly the objective a trader cares about, and the reason this beats a
  raw-PnL reward (which ignores volatility and churn).

This is deterministic given the candle series, which is what we want for offline
RL: train on logged/real history, never explore with live capital.
"""

from __future__ import annotations

import numpy as np

from ...backtest.costs import CostModel
from ...core.filters import classify_regime, in_session
from ...features.feature_store import KrakenFeatureSet

# Market features (10) + position context (3) + trading context (3).
_MARKET_FEATURES = 10
_POSITION_FEATURES = 3
_CONTEXT_FEATURES = 3
STATE_DIM = _MARKET_FEATURES + _POSITION_FEATURES + _CONTEXT_FEATURES
ACTIONS = (0, 1, 2)          # flat, long, short
_POS = {0: 0.0, 1: 1.0, 2: -1.0}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def market_vector(f: KrakenFeatureSet) -> np.ndarray:
    """10-dim normalized market snapshot (no signal/position terms)."""
    last = f.last_price or 0.0
    atr = f.atr or 0.0
    scale = atr if atr > 0 else (abs(last) * 0.001 or 1.0)
    half_band = (f.bb_upper - f.bb_lower) / 2.0
    v = np.array([
        float(np.clip((f.rsi - 50.0) / 50.0, -1.0, 1.0)),
        float(np.tanh(_safe_div(f.ema_fast - f.ema_slow, scale))),
        float(np.tanh(_safe_div(last - f.ema_fast, scale))),
        float(np.clip(_safe_div(last - f.bb_mid, half_band), -2.0, 2.0)) / 2.0,
        float(np.tanh(_safe_div(atr, last) * 100.0)),
        float(np.tanh(_safe_div(getattr(f, "macd_hist", 0.0), scale))),
        float(np.clip((getattr(f, "stoch_k", 50.0) - 50.0) / 50.0, -1.0, 1.0)),
        float(np.tanh(getattr(f, "adx", 0.0) / 50.0)),
        float(np.clip(f.sentiment.score if f.sentiment else 0.0, -1.0, 1.0)),
        float(getattr(f, "dc_trend", 0.0)),
    ], dtype=np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)


def context_vector(f: KrakenFeatureSet, candle, symbol: str) -> np.ndarray:
    """3-dim trading-context snapshot the pre-trade filters also key on.

    * regime  — +1 trend-up / −1 trend-down / 0 range (from ADX + EMA stack).
    * session — 1 inside the instrument's liquid hours, 0 outside (from the
      candle timestamp; neutral 1.0 when no timestamp is available).
    * spread  — tanh of the intrabar range (high−low)/price, a costless proxy for
      the quoted spread/liquidity that is *consistent across train and live*
      (both have candles, whereas order-book depth is only live).
    """
    reg = classify_regime(f)
    regime = 1.0 if reg == "trend_up" else (-1.0 if reg == "trend_down" else 0.0)

    session = 1.0
    ts = getattr(candle, "ts", None) if candle is not None else None
    if ts is not None:
        session = 1.0 if in_session(symbol, ts.hour) else 0.0

    last = f.last_price or 0.0
    rng = 0.0
    if candle is not None and last:
        rng = float(np.tanh(_safe_div((candle.high or 0.0) - (candle.low or 0.0), last) * 200.0))

    v = np.array([regime, session, rng], dtype=np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)


def build_state(f: KrakenFeatureSet, candle, position: float, holding: int,
                unrealized: float, symbol: str) -> np.ndarray:
    """Assemble the full ``STATE_DIM`` state: market + position + context.

    Single source of truth shared by the env (training) and the live advisory
    ``recommend`` helpers, so train- and inference-time states never drift."""
    pos_ctx = np.array([
        float(position),
        float(np.tanh(holding / 20.0)),
        float(np.clip(unrealized * 20.0, -1.0, 1.0)),
    ], dtype=np.float64)
    return np.concatenate([market_vector(f), pos_ctx, context_vector(f, candle, symbol)])


class TradingEnv:
    """Replay one instrument's candle/feature series as a sequential MDP."""

    def __init__(
        self,
        candles: list,
        feats: list[KrakenFeatureSet],
        cost: CostModel | None = None,
        warmup: int = 60,
        dsr_eta: float = 0.01,        # DSR EMA horizon (~1/eta steps)
        symbol: str | None = None,
    ):
        assert len(candles) == len(feats), "candles and feats must align"
        self.candles = candles
        self.feats = feats
        self.symbol = symbol or (
            getattr(candles[0], "symbol", None) if candles else None
        ) or (getattr(feats[0], "symbol", None) if feats else "") or ""
        self.cost = cost or CostModel()
        self.warmup = warmup
        self.eta = dsr_eta
        self.n = len(candles)
        self.reset()

    def reset(self) -> np.ndarray:
        self.i = self.warmup
        self.position = 0.0          # -1 / 0 / +1
        self.entry_price = 0.0
        self.holding = 0
        self._A = 0.0                # DSR first moment of returns
        self._B = 0.0               # DSR second moment
        return self._state()

    @property
    def done(self) -> bool:
        return self.i >= self.n - 1

    def _state(self) -> np.ndarray:
        f = self.feats[self.i]
        last = f.last_price or 0.0
        unreal = 0.0
        if self.position != 0.0 and self.entry_price:
            unreal = self.position * (last - self.entry_price) / self.entry_price
        return build_state(f, self.candles[self.i], self.position, self.holding,
                           unreal, self.symbol)

    def _dsr(self, r: float) -> float:
        """Differential Sharpe ratio increment for step return r."""
        dA = r - self._A
        dB = r * r - self._B
        denom = (self._B - self._A * self._A) ** 1.5
        dsr = _safe_div(self._B * dA - 0.5 * self._A * dB, denom) if denom > 1e-12 else 0.0
        self._A += self.eta * dA
        self._B += self.eta * dB
        return float(np.clip(dsr, -10.0, 10.0))

    def step(self, action: int):
        """Apply a target-position action; advance one bar. Returns
        (next_state, reward, done, info)."""
        target = _POS[int(action)]
        price = self.candles[self.i].close or 0.0
        nxt = self.candles[self.i + 1].close or price

        # Turnover cost when the target position differs from the current one.
        turnover = abs(target - self.position)
        cost_frac = 0.0
        if turnover > 0:
            edge = (self.cost.spread_bps / 2.0 + self.cost.slippage_bps) / 10_000.0
            cost_frac = turnover * (edge + self.cost.commission_bps / 10_000.0)
            self.entry_price = price if target != 0 else 0.0
            self.holding = 0

        # Step return: today's position earns tomorrow's price move, minus the
        # cost paid to get into that position.
        price_ret = _safe_div(nxt - price, price)
        r = target * price_ret - cost_frac

        self.position = target
        if target != 0:
            self.holding += 1
        self.i += 1
        reward = self._dsr(r)
        return self._state(), reward, self.done, {"step_return": r, "cost": cost_frac}
