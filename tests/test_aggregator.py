"""Consensus SignalAggregator: weighted vote, family capping, 35%-opposing
veto, EMA smoothing, and enter/exit hysteresis."""

from __future__ import annotations

from gungnir.core.aggregator import SignalAggregator
from gungnir.data.models import Side


def _agg(**kw) -> SignalAggregator:
    defaults = dict(veto_opposing=0.35, ema_alpha=0.6, enter_threshold=0.25,
                    exit_threshold=0.10, min_hold_bars=0, family_cap=0.4)
    defaults.update(kw)
    return SignalAggregator(**defaults)


def _converge(a: SignalAggregator, symbol: str, pos=None, n: int = 12):
    """Run decide() until the EMA has effectively converged to the raw tally."""
    d = None
    for _ in range(n):
        d = a.decide(symbol, pos)
    return d


def test_unanimous_book_enters_in_consensus_direction():
    a = _agg()
    a.set_stance("US100", "s1", Side.BUY, 0.8, family="trend")
    a.set_stance("US100", "s2", Side.BUY, 0.6, family="meanrev")
    d = _converge(a, "US100")
    assert d.action == "enter" and d.side == Side.BUY
    assert d.opposing == 0.0 and d.n_stances == 2


def test_empty_book_decides_nothing():
    a = _agg()
    d = a.decide("US100", None)
    assert d.action == "none" and d.n_stances == 0


def test_opposing_35pct_vetoes_entry():
    a = _agg()
    # BUY 64% / SELL 36% of the vote: consensus +0.28 clears the entry band,
    # but the opposing share is over the 35% conflict line — stand aside.
    a.set_stance("US100", "b1", Side.BUY, 3.2, family="trend")
    a.set_stance("US100", "b2", Side.BUY, 3.2, family="meanrev")
    a.set_stance("US100", "s1", Side.SELL, 3.6, family="oscillator")
    d = _converge(a, "US100")
    assert d.action == "veto"
    assert d.opposing >= 0.35


def test_family_cap_deflates_clone_consensus():
    # Eight 'channel' clones BUY vs three independent SELL dissenters. Uncapped
    # the clones drown the dissent (opp 27% → enter); with the 40% family cap
    # the clone bloc counts as at most 40% of the vote and the entry dies.
    capped = _agg()
    uncapped = _agg(family_cap=0.0)
    for a in (capped, uncapped):
        for i in range(8):
            a.set_stance("US100", f"clone{i}", Side.BUY, 1.0, family="channel")
        for i in range(3):
            a.set_stance("US100", f"solo{i}", Side.SELL, 1.0, family=f"f{i}")
    assert _converge(uncapped, "US100").action == "enter"
    d = _converge(capped, "US100")
    assert d.action != "enter"                 # consensus deflated below entry


def test_ema_smoothing_blocks_single_bar_blip():
    a = _agg(ema_alpha=0.3)
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    d = a.decide("US100", None)                # first bar: EMA = 0.3 < 0.25? no —
    assert d.consensus < 1.0                   # smoothed below the raw +1 tally


def test_hysteresis_holds_between_bands_and_exits_below_floor():
    a = _agg(family_cap=0.0)                   # isolate the hysteresis bands
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    d = _converge(a, "US100")
    assert d.action == "enter"
    # In position now. Weaken the book into the dead zone (0.10 < |S| < 0.25):
    # BUY 1.0 / SELL 0.667 → S = +0.2 → hold, no churn.
    a.set_stance("US100", "s2", Side.SELL, 0.667, family="meanrev")
    d = _converge(a, "US100", pos=Side.BUY)
    assert d.action == "hold"
    # Drain conviction below the exit floor (S = +0.08) → exit, same sign.
    a.set_stance("US100", "s2", Side.SELL, 0.85, family="meanrev")
    d = _converge(a, "US100", pos=Side.BUY)
    assert d.action == "exit"


def test_sign_flip_exits():
    a = _agg()
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    _converge(a, "US100")
    a.clear_stance("US100", "s1")
    a.set_stance("US100", "s2", Side.SELL, 1.0, family="meanrev")
    d = _converge(a, "US100", pos=Side.BUY)
    assert d.action == "exit" and d.side == Side.BUY


def test_min_hold_blocks_immediate_exit():
    a = _agg(min_hold_bars=3)
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    _converge(a, "US100")
    a.clear_stance("US100", "s1")              # book empties → EMA decays
    # First few in-position bars: exit suppressed by min hold.
    assert a.decide("US100", Side.BUY).action == "hold"
    assert a.decide("US100", Side.BUY).action == "hold"
    assert a.decide("US100", Side.BUY).action == "hold"
    assert a.decide("US100", Side.BUY).action == "exit"


def test_cleared_stances_decay_to_exit():
    a = _agg()
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    _converge(a, "US100")
    a.clear_stance("US100", "s1")
    d = _converge(a, "US100", pos=Side.BUY)
    assert d.action == "exit"                  # nobody votes → conviction drains


def test_symbols_are_independent():
    a = _agg()
    a.set_stance("US100", "s1", Side.BUY, 1.0, family="trend")
    a.set_stance("GOLD", "s1", Side.SELL, 1.0, family="trend")
    d1 = _converge(a, "US100")
    d2 = _converge(a, "GOLD")
    assert d1.side == Side.BUY and d2.side == Side.SELL


def test_all_strategies_carry_a_family():
    from gungnir.strategy.kraken_strategies import KRAKEN_STRATEGIES
    fams = {s.family for s in KRAKEN_STRATEGIES}
    assert all(s.family for s in KRAKEN_STRATEGIES)
    assert len(fams) >= 5                      # genuinely diverse taxonomy


def test_horizon_weighting_is_exposed_in_diagnostics():
    a = _agg(family_cap=0.0)
    a.set_stance("US100", "fast", Side.BUY, 1.0, horizon="M1")
    a.set_stance("US100", "slow", Side.SELL, 1.0, horizon="H1")
    d = a.decide("US100", None)
    assert d.diagnostics["buy_weight"] == 0.8
    assert d.diagnostics["sell_weight"] == 1.15
    assert d.diagnostics["horizons"] == {"M1": 0.8, "H1": 1.15}


def test_short_lane_can_enter_against_long_horizon_context():
    a = _agg(family_cap=0.0)
    a.set_stance("US100", "scalp", Side.BUY, 1.0, horizon="5M")
    a.set_stance("US100", "trend", Side.SELL, 1.0, horizon="1H")
    d = _converge(a, "US100")
    assert d.action == "enter" and d.side == Side.BUY
    lane = d.diagnostics["short_lane"]
    assert lane["n_stances"] == 1 and lane["ema_score"] > 0.25
