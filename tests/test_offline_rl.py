"""Offline RL: trading MDP + Double-DQN learner.

Covers the environment mechanics (state shape, cost-aware step return, DSR
reward) and an end-to-end learning check: on a clean uptrend the learner should
prefer 'long' and earn a positive cost-net return.
"""

from __future__ import annotations

import numpy as np

from gungnir.backtest import engine
from gungnir.backtest.costs import CostModel
from gungnir.data.models import Candle
from gungnir.features.feature_store import build_kraken_series
from gungnir.learning.rl.env import ACTIONS, STATE_DIM, TradingEnv
from gungnir.learning.rl.offline import OfflineDQN, evaluate_policy, train_offline


def _trend_series(n=300, step=0.002, start=100.0):
    """A clean monotonic uptrend → 'long' is the optimal constant action."""
    candles, px = [], start
    for _ in range(n):
        o = px
        c = px * (1 + step)
        candles.append(Candle(symbol="T", timeframe="bt", open=o, high=c, low=o, close=c))
        px = c
    return candles, build_kraken_series("T", candles)


def test_env_state_shape_and_done():
    candles = engine.synthetic_candles("X", n=200, seed=1)
    feats = build_kraken_series("X", candles)
    env = TradingEnv(candles, feats, warmup=60)
    s = env.reset()
    assert s.shape == (STATE_DIM,)
    s2, r, done, info = env.step(1)
    assert s2.shape == (STATE_DIM,)
    assert np.isfinite(r) and "step_return" in info and not done


def test_long_earns_in_uptrend_and_cost_hurts_turnover():
    candles, feats = _trend_series()
    # No cost: holding long through the uptrend yields a positive cumulative return.
    env = TradingEnv(candles, feats, warmup=60, cost=CostModel())
    env.reset()
    tot = 0.0
    while not env.done:
        _, _, _, info = env.step(1)        # always long
        tot += info["step_return"]
    assert tot > 0

    # Flipping every bar with costs pays turnover each flip → strictly worse.
    env2 = TradingEnv(candles, feats, warmup=60,
                      cost=CostModel(spread_bps=20.0, commission_bps=2.0))
    env2.reset()
    paid = 0.0
    a = 1
    while not env2.done:
        _, _, _, info = env2.step(a)
        paid += info["cost"]
        a = 2 if a == 1 else 1             # flip long/short every bar
    assert paid > 0


def test_offline_dqn_learns_to_go_long_in_uptrend():
    candles, feats = _trend_series(n=320)
    agent, diag = train_offline(candles, feats, cost=CostModel(),
                                hidden=64, epochs=40, exploration_passes=4, seed=0)
    assert diag["transitions"] > 0
    env = TradingEnv(candles, feats, warmup=60, cost=CostModel())
    res = evaluate_policy(env, agent)
    # Learned greedy policy should be net-positive and favor 'long' (action 1).
    assert res["total_return"] > 0
    assert res["action_counts"][1] == max(res["action_counts"].values())


def test_recommend_advisory_matches_learned_direction():
    from gungnir.learning.rl.offline import recommend
    candles, feats = _trend_series(n=320)
    agent, _ = train_offline(candles, feats, cost=CostModel(),
                             epochs=40, exploration_passes=4, seed=0)
    a, label = recommend(agent, feats[100])
    assert a in ACTIONS and label in ("FLAT", "LONG", "SHORT")
    # In a clean uptrend the advisory should be LONG.
    assert label == "LONG"


def test_offline_dqn_save_load_roundtrip(tmp_path):
    candles, feats = _trend_series(n=200)
    agent, _ = train_offline(candles, feats, epochs=5, exploration_passes=1, seed=0)
    p = str(tmp_path / "dqn.npz")
    agent.save(p)
    fresh = OfflineDQN(seed=99)
    assert fresh.load(p) is True
    s = TradingEnv(candles, feats, warmup=60).reset()
    assert fresh.act(s) in ACTIONS
    assert np.allclose(agent.online.predict(s.reshape(1, -1)),
                       fresh.online.predict(s.reshape(1, -1)))
