"""Best-signal-per-symbol-per-bar gate: when several strategies fire on the
same symbol on the same bar, only the strongest may open (the rest net onto the
same position). Exercises Agent._select_opens in isolation with fakes."""

from __future__ import annotations

from types import SimpleNamespace

from gungnir.core.agent import Agent
from gungnir.data.models import Side


class _Sig:
    def __init__(self, side, conviction):
        self.side = side
        self.conviction = conviction


class _Strat:
    def __init__(self, name, side, conviction, *, timeframe="1h"):
        self.name = name
        self.timeframe = timeframe
        self._sig = _Sig(side, conviction) if side is not None else None

    def trades_symbol(self, symbol):
        return True

    def generate(self, features):
        return [self._sig] if self._sig is not None else []


def _fake_agent(cap=1, last_emit=None):
    return SimpleNamespace(
        config=SimpleNamespace(get=lambda *a, default=None: (
            cap if a[-1] == "max_opens_per_symbol_per_bar" else default)),
        tf="1h",
        _last_emit=last_emit or {},
    )


def _select(agent, strats):
    return Agent._select_opens(agent, strats, "US100",
                               features_by_tf={}, primary_features=object(),
                               bar_ts_by_tf={}, fresh_tfs=set())


def test_best_conviction_wins_the_slot():
    strats = [_Strat("weak", Side.BUY, 0.4),
              _Strat("strong", Side.BUY, 0.9),
              _Strat("mid", Side.BUY, 0.6)]
    assert _select(_fake_agent(cap=1), strats) == {"strong"}


def test_cap_of_two_keeps_top_two():
    strats = [_Strat("a", Side.BUY, 0.4),
              _Strat("b", Side.BUY, 0.9),
              _Strat("c", Side.BUY, 0.6)]
    assert _select(_fake_agent(cap=2), strats) == {"b", "c"}


def test_no_restriction_when_candidates_fit_under_cap():
    strats = [_Strat("only", Side.BUY, 0.5)]
    assert _select(_fake_agent(cap=1), strats) is None      # nothing to trim


def test_cap_zero_disables_the_gate():
    strats = [_Strat("a", Side.BUY, 0.4), _Strat("b", Side.BUY, 0.9)]
    assert _select(_fake_agent(cap=0), strats) is None


def test_non_firing_and_edge_suppressed_strategies_do_not_count():
    strats = [_Strat("firing", Side.BUY, 0.5),
              _Strat("silent", None, 0.0),               # emits nothing
              _Strat("held", Side.SELL, 0.9)]            # same side as last bar
    # 'held' already emitted SELL last bar → edge-suppressed, not a candidate.
    agent = _fake_agent(cap=1, last_emit={("held", "US100"): Side.SELL})
    # only 'firing' remains as a fresh edge → under cap → no restriction.
    assert _select(agent, strats) is None
