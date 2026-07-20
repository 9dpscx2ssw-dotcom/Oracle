"""Tests for the RL decision layer: state encoding, the actor-critic network's
ability to learn, and the policy's decide → grade → learn loop.
"""

from __future__ import annotations

import numpy as np

from gungnir.data.models import Sentiment, Side, Signal
from gungnir.features.feature_store import FeatureSet
from gungnir.learning.rl import SKIP, TAKE, RLPolicy
from gungnir.learning.rl.network import ActorCritic
from gungnir.learning.rl.state import STATE_DIM, encode


def _features(**over) -> FeatureSet:
    base = dict(symbol="X", last_price=100.0, ema_fast=101.0, ema_slow=99.0,
                rsi=60.0, atr=1.0, bb_lower=98.0, bb_mid=100.0, bb_upper=102.0)
    base.update(over)
    return FeatureSet(**base)


def _signal(side=Side.BUY, conviction=0.7) -> Signal:
    return Signal(strategy="s1", symbol="X", side=side, conviction=conviction)


# ── state encoding ──────────────────────────────────────────────────────────────
def test_encode_shape_and_range():
    s = encode(_signal(), _features(), portfolio_heat=0.5)
    assert s.shape == (STATE_DIM,)
    assert np.isfinite(s).all()
    # tanh/clip features must stay bounded.
    assert (s >= -2.0).all() and (s <= 2.0).all()


def test_encode_is_robust_to_missing_indicators():
    # A bare FeatureSet (no Kraken fields, no sentiment/prediction) must not raise.
    s = encode(_signal(side=Side.SELL), FeatureSet(symbol="X", last_price=0.0))
    assert s.shape == (STATE_DIM,)
    assert np.isfinite(s).all()


def test_encode_reflects_side_and_sentiment():
    buy = encode(_signal(side=Side.BUY), _features(
        sentiment=Sentiment(symbol="X", score=0.8, confidence=0.9)))
    sell = encode(_signal(side=Side.SELL), _features(
        sentiment=Sentiment(symbol="X", score=0.8, confidence=0.9)))
    assert buy[1] == 1.0 and sell[1] == -1.0      # side slot
    assert abs(buy[10] - 0.8) < 1e-9              # sentiment slot


# ── network learning ────────────────────────────────────────────────────────────
def test_network_learns_a_separable_mapping():
    """Two state clusters with opposite rewards: the critic must separate them."""
    net = ActorCritic(STATE_DIM, hidden=16, lr=0.01, seed=1)
    rng = np.random.default_rng(0)
    good = np.zeros(STATE_DIM)
    good[0] = 1.0       # cluster A → reward +1
    bad = np.zeros(STATE_DIM)
    bad[0] = -1.0       # cluster B → reward -1

    for _ in range(400):
        X = np.stack([good if rng.random() < 0.5 else bad for _ in range(16)])
        actions = np.ones(16, dtype=np.int64)      # always "take"
        targets = np.array([1.0 if x[0] > 0 else -1.0 for x in X])
        net.update(X, actions, targets)

    _, v_good = net.predict(good)
    _, v_bad = net.predict(bad)
    assert v_good > 0.5 and v_bad < -0.5           # values track the rewards
    assert v_good - v_bad > 1.0


def test_policy_prefers_the_profitable_action():
    """A state where taking pays (+1) and skipping costs (-1): P(take) must rise.

    This mirrors how the agent trains the policy — every signal is graded against
    its counterfactual (taken trades by realized PnL, skipped ones by the PnL they
    avoided). The advantage actor-critic learns from that *contrast*: with only one
    action ever observed and a converged critic the advantage vanishes, so the
    take/skip pairing is what actually moves the policy.
    """
    pol = RLPolicy(hidden=16, lr=0.02, warmup_trades=0, batch_size=8,
                   epsilon_start=0.0, seed=2)
    sig, feat = _signal(), _features()
    state = encode(sig, feat)
    for _ in range(300):
        pol.learn(state, TAKE, reward=1.0)    # taking this signal pays off
        pol.learn(state, SKIP, reward=-1.0)   # skipping it would have cost us
    d = pol.decide(sig, feat, explore=False)
    assert d.confidence > 0.6
    assert d.take is True


def test_skip_reward_is_inverted_pnl():
    pol = RLPolicy(warmup_trades=0, seed=3)
    # Skipping a trade that would have LOST is good → positive reward.
    assert pol.reward_for(SKIP, pnl=-50.0, risk_amount=25.0) > 0
    # Skipping a winner is bad → negative reward.
    assert pol.reward_for(SKIP, pnl=50.0, risk_amount=25.0) < 0
    # Taking a winner is good.
    assert pol.reward_for(TAKE, pnl=50.0, risk_amount=25.0) > 0
    # Reward is clipped to ±reward_clip.
    assert abs(pol.reward_for(TAKE, pnl=1e9, risk_amount=1.0)) <= pol.reward_clip


def test_warmup_takes_everything():
    pol = RLPolicy(warmup_trades=10, seed=4)
    sig, feat = _signal(), _features()
    # During warm-up every decision is TAKE regardless of the (untrained) policy.
    assert all(pol.decide(sig, feat).take for _ in range(5))
    assert pol.warming_up is True


def test_snapshot_is_json_safe():
    import json
    pol = RLPolicy(warmup_trades=0, batch_size=4, seed=5)
    s = encode(_signal(), _features())
    for _ in range(10):
        pol.learn(s, TAKE, reward=0.5)
    snap = pol.snapshot()
    json.dumps(snap)        # must not raise
    assert snap["enabled"] is True
    assert snap["action_counts"]["take"] == 10
    assert isinstance(snap["reward_history"], list)


def test_policy_persists_across_reload(tmp_path):
    path = str(tmp_path / "rl.npz")
    pol = RLPolicy(hidden=16, warmup_trades=0, batch_size=8, epsilon_start=0.0, seed=6)
    sig, feat = _signal(), _features()
    state = encode(sig, feat)
    for _ in range(200):
        pol.learn(state, TAKE, reward=1.0)
    pol.save(path)
    before = pol.decide(sig, feat, explore=False).confidence

    fresh = RLPolicy(hidden=16, warmup_trades=0, epsilon_start=0.0, seed=99)
    assert fresh.load(path) is True
    after = fresh.decide(sig, feat, explore=False).confidence
    assert abs(before - after) < 1e-6      # reloaded weights reproduce the decision


def test_full_training_state_persists(tmp_path):
    """Counters, epsilon, action mix and the replay buffer survive a reload."""
    path = str(tmp_path / "rl.npz")
    pol = RLPolicy(hidden=16, warmup_trades=2, batch_size=4, buffer_size=100, seed=1)
    rng = np.random.default_rng(0)
    for i in range(40):
        pol.learn(rng.standard_normal(STATE_DIM), TAKE if i % 2 else SKIP,
                  reward=float(rng.standard_normal()))
    before = pol.snapshot()
    pol.save(path)

    fresh = RLPolicy(hidden=16, warmup_trades=2, batch_size=4, buffer_size=100, seed=2)
    assert fresh.load(path) is True
    after = fresh.snapshot()
    assert after["updates"] == before["updates"]
    assert after["states_learned"] == before["states_learned"]   # buffer restored
    assert after["action_counts"] == before["action_counts"]
    assert abs(after["epsilon"] - before["epsilon"]) < 1e-9
    assert abs(after["cumulative_reward"] - before["cumulative_reward"]) < 1e-6


def test_load_legacy_weights_without_sidecar(tmp_path):
    """A pre-upgrade weights-only file still loads; counters reset cleanly."""
    path = str(tmp_path / "rl.npz")
    pol = RLPolicy(hidden=16, seed=1)
    pol.net.save(path)                       # weights only, no .state.npz sidecar
    fresh = RLPolicy(hidden=16, seed=2)
    assert fresh.load(path) is True
    assert fresh.updates == 0 and fresh.snapshot()["states_learned"] == 0
