"""Phase-4 learning system: parameterized strategies, real Bayesian fitness,
RL v2 state/replay/calibration, and shadow-only exploration."""

from __future__ import annotations

import numpy as np

from gungnir.data.models import Side, Signal, Trade
from gungnir.features.feature_store import KrakenFeatureSet
from gungnir.learning.rl import state as rl_state
from gungnir.learning.rl.policy import SKIP, TAKE, RLPolicy
from gungnir.strategy.kraken_strategies import (
    KRAKEN_STRATEGIES, CCIMACDStrategy, CCIReversalStrategy,
)


def _features(**kw) -> KrakenFeatureSet:
    base = dict(symbol="EURUSD", last_price=1.10, atr=0.001,
                bb_lower=1.09, bb_mid=1.10, bb_upper=1.11)
    base.update(kw)
    return KrakenFeatureSet(**base)


# ── strategy parameterization ────────────────────────────────────────────────

def test_every_strategy_declares_bounds():
    """The optimizer is blind to a strategy without BOUNDS — all 26 must have
    at least one tunable parameter (previously all returned {})."""
    for cls in KRAKEN_STRATEGIES:
        s = cls()
        bounds = s.get_parameter_bounds()
        assert bounds, f"{cls.name} has no tunable parameters"
        for k, (lo, hi) in bounds.items():
            assert k in s.DEFAULTS, f"{cls.name}: bound {k} missing a default"
            assert lo < hi
            assert lo <= s.DEFAULTS[k] <= hi, \
                f"{cls.name}: default {k}={s.DEFAULTS[k]} outside bounds"


def test_params_change_behavior():
    """A tuned threshold actually changes what fires."""
    f = _features(cci14=120.0, macd12_26=1.0)
    default = CCIMACDStrategy()
    assert default.generate(f)                       # 120 > 100 → fires
    strict = CCIMACDStrategy(params={"cci_threshold": 150.0})
    assert not strict.generate(f)                    # 120 < 150 → silent


def test_default_params_preserve_old_firing_behavior():
    f = _features(cci14=-130.0)
    sigs = CCIReversalStrategy().generate(f)
    assert sigs and sigs[0].side == Side.BUY
    assert sigs[0].conviction >= 0.5                 # never below the old base


def test_graded_conviction_rises_with_signal_strength():
    weak = CCIReversalStrategy().generate(_features(cci14=-105.0))[0]
    strong = CCIReversalStrategy().generate(_features(cci14=-250.0))[0]
    assert strong.conviction > weak.conviction


# ── Bayesian fitness is a real backtest now ──────────────────────────────────

def test_bayesian_skips_without_history(tmp_path):
    """No stored candles → no proposal (never a placeholder-fit one)."""
    from gungnir.learning.bayesian_reflection import optimize_strategy
    from gungnir.learning.journal import Journal
    from gungnir.persistence.db import Database
    db = Database(tmp_path / "t.db")
    journal = Journal(db)
    strat = CCIMACDStrategy(timeframe="5m")
    for i in range(25):
        db.record_trade(Trade(symbol="EURUSD", side=Side.BUY, volume=1.0,
                              entry_price=1.1, exit_price=1.101, pnl=1.0,
                              strategy=strat.name))
    assert optimize_strategy(strat, journal) == {}
    db.close()


# ── RL v2 ────────────────────────────────────────────────────────────────────

def test_state_encodes_strategy_identity_and_regime():
    f = _features(cci14=0.0)
    s1 = Signal(strategy="cci_macd", symbol="EURUSD", side=Side.BUY, conviction=0.5)
    s2 = Signal(strategy="donchian", symbol="EURUSD", side=Side.BUY, conviction=0.5)
    v1 = rl_state.encode(s1, f, regime="trend_low", hour=12.0)
    v2 = rl_state.encode(s2, f, regime="trend_low", hour=12.0)
    assert v1.shape == (rl_state.STATE_DIM,)
    assert not np.allclose(v1, v2)                   # strategies distinguishable
    v3 = rl_state.encode(s1, f, regime="range_high", hour=12.0)
    assert not np.allclose(v1, v3)                   # regimes distinguishable
    # Regime one-hot occupies exactly one slot.
    names = rl_state.FEATURE_NAMES
    onehot = [v1[names.index(f"regime_{r}")] for r in
              ("trend_low", "trend_high", "range_low", "range_high")]
    assert sum(onehot) == 1.0


def test_explore_false_never_coin_flips():
    """Live signals must be greedy — exploration only spends shadow money."""
    pol = RLPolicy(warmup_trades=0, epsilon_start=1.0, epsilon_min=1.0)
    pol.updates = 100                                # out of warmup
    sig = Signal(strategy="s", symbol="EURUSD", side=Side.BUY, conviction=0.5)
    f = _features()
    for _ in range(50):
        d = pol.decide(sig, f, explore=False)
        assert not d.explored


def test_calibration_tracked_and_scored():
    pol = RLPolicy(warmup_trades=0)
    for i in range(30):
        t = Trade(symbol="EURUSD", side=Side.BUY, volume=1.0, entry_price=1.1,
                  pnl=1.0 if i % 2 == 0 else -1.0)
        t.context = {"rl_state": [0.0] * rl_state.STATE_DIM, "rl_action": TAKE,
                     "rl_risk": 1.0, "rl_p": 0.5}
        pol.learn_from_trade(t)
    # p=0.5 on a 50% win rate is perfectly calibrated: Brier = 0.25.
    assert abs(pol.brier_score - 0.25) < 1e-9
    assert pol.snapshot()["brier"] == 0.25


def test_stale_state_dim_is_skipped_not_crashed():
    pol = RLPolicy()
    t = Trade(symbol="EURUSD", side=Side.BUY, volume=1.0, entry_price=1.1, pnl=1.0)
    t.context = {"rl_state": [0.0] * 14, "rl_action": SKIP, "rl_risk": 1.0}
    assert pol.learn_from_trade(t) is False          # old 14-dim stamp ignored


def test_recency_weighted_replay_prefers_new_samples():
    pol = RLPolicy(batch_size=8, buffer_size=100, warmup_trades=0)
    dim = rl_state.STATE_DIM
    # Old regime: reward for SKIP; new regime: reward for TAKE.
    for _ in range(90):
        pol.learn(np.zeros(dim), SKIP, 1.0)
    for _ in range(10):
        pol.learn(np.ones(dim), TAKE, 1.0)
    n = len(pol.buffer)
    weights = [pol._replay_decay ** (n - 1 - i) for i in range(n)]
    # The newest 10% of the buffer must carry more than 10% of sampling mass.
    newest_mass = sum(weights[-10:]) / sum(weights)
    assert newest_mass > 0.15
