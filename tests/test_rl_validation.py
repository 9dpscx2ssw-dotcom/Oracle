"""IQL learner + purged walk-forward validation + WIS off-policy evaluation."""

from __future__ import annotations

import numpy as np

from gungnir.backtest.costs import CostModel
from gungnir.data.models import Candle
from gungnir.features.feature_store import build_kraken_series
from gungnir.learning.rl.env import ACTIONS, TradingEnv, build_state
from gungnir.learning.rl.iql import IQL
from gungnir.learning.rl.offline import collect_transitions
from gungnir.learning.rl.validation import (
    collect_episodes, run_walk_forward, walk_forward_boundaries, wis_ope,
)


def _trend_series(n=320, step=0.002, start=100.0):
    candles, px = [], start
    for _ in range(n):
        o, c = px, px * (1 + step)
        candles.append(Candle(symbol="T", timeframe="bt", open=o, high=c, low=o, close=c))
        px = c
    return candles, build_kraken_series("T", candles)


def _train_iql(candles, feats, cost=None, epochs=40, passes=4, seed=0):
    env = TradingEnv(candles, feats, cost=cost or CostModel(), warmup=60)
    data = []
    for p in range(passes):
        data += collect_transitions(env, epsilon=1.0, seed=seed + p)
    agent = IQL(seed=seed)
    agent.train(data, epochs=epochs, seed=seed)
    return agent


def test_iql_learns_long_in_uptrend():
    candles, feats = _trend_series()
    agent = _train_iql(candles, feats)
    # Greedy IQL policy should choose LONG (action 1) mid-uptrend.
    s = build_state(feats[150], candles[150], 0.0, 0, 0.0, "T")
    assert agent.act(s) == 1


def test_iql_save_load_roundtrip(tmp_path):
    candles, feats = _trend_series(n=200)
    agent = _train_iql(candles, feats, epochs=5, passes=1)
    p = str(tmp_path / "iql.npz")
    agent.save(p)
    fresh = IQL(seed=9)
    assert fresh.load(p) is True
    s = TradingEnv(candles, feats, warmup=60).reset()
    assert fresh.act(s) in ACTIONS
    assert np.allclose(agent.pi.forward(s.reshape(1, -1))[0],
                       fresh.pi.forward(s.reshape(1, -1))[0])


def test_walk_forward_boundaries_are_ordered_and_embargoed():
    splits = walk_forward_boundaries(1000, warmup=60, n_splits=4, embargo=20)
    assert len(splits) == 4
    for (a, b), (c, d) in splits:
        assert a < b < c < d            # train strictly before test
        assert c - b >= 20              # embargo gap respected


def test_walk_forward_runs_out_of_sample():
    candles, _ = _trend_series(n=600)
    out = run_walk_forward(candles, cost=CostModel(), algo="dqn",
                           n_splits=3, epochs=12, passes=2)
    assert out["n_folds"] >= 1
    assert np.isfinite(out["mean_oos_return"])


def test_wis_ope_equals_empirical_mean_for_uniform_target():
    candles, feats = _trend_series(n=400)
    env = TradingEnv(candles, feats, warmup=60)
    eps = collect_episodes(env, length=40, n_episodes=8, seed=1)
    # Target == behavior (uniform) → importance ratios are 1 → WIS == plain mean.
    def uniform(s):
        return np.array([1 / 3, 1 / 3, 1 / 3])
    est = wis_ope(eps, uniform, behavior_prob=1 / 3)
    emp = float(np.mean([sum(r for _, _, r in ep) for ep in eps]))
    assert abs(est - emp) < 1e-9
