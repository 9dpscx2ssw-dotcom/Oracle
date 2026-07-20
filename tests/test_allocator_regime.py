"""Regime classification, the capital allocator, and the system scoreboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gungnir.core import regime
from gungnir.data.models import Side, Trade
from gungnir.learning.allocator import CapitalAllocator
from gungnir.learning import scoreboard


# ── regime ─────────────────────────────────────────────────────────────────────

def test_regime_quadrants():
    assert regime.classify(adx=30, vol_percentile=0.2) == "trend_low"
    assert regime.classify(adx=30, vol_percentile=0.9) == "trend_high"
    assert regime.classify(adx=10, vol_percentile=0.2) == "range_low"
    assert regime.classify(adx=10, vol_percentile=0.9) == "range_high"


def test_vol_percentile_relative_to_own_history():
    hist = [0.001, 0.002, 0.003, 0.004]
    assert regime.vol_percentile(hist, 0.0035) == 0.75   # above 3 of 4
    assert regime.vol_percentile(hist, 0.0005) == 0.0
    assert regime.vol_percentile([], 0.01) == 0.5        # no history → neutral


# ── allocator ──────────────────────────────────────────────────────────────────

class _FakeJournal:
    def __init__(self, trades):
        self._trades = trades

    def recent(self, limit=100):
        return self._trades[:limit]


def _trade(strategy, pnl, regime_label=None, mode="shadow"):
    ctx = {"regime": regime_label} if regime_label else {}
    return Trade(symbol="US500", side=Side.BUY, volume=1.0, entry_price=100.0,
                 exit_price=100.0 + pnl, pnl=pnl, strategy=strategy,
                 mode=mode, context=ctx)


def test_allocator_upweights_winners_downweights_losers():
    trades = ([_trade("winner", 1.0)] * 20 +
              [_trade("loser", -1.0)] * 20 +
              [_trade("coinflip", 1.0), _trade("coinflip", -1.0)] * 10)
    a = CapitalAllocator()
    a.refresh(_FakeJournal(trades))
    assert a.weight("winner") == 1.5           # pure edge → cap
    assert a.weight("loser") == 0.1            # pure bleed → floor, never 0
    assert abs(a.weight("coinflip") - 1.0) < 0.15   # no edge → ~neutral
    assert a.weight("unknown") == 1.0          # no evidence → neutral


def test_allocator_conditions_on_regime():
    # Wins only in trends, loses only in chop — with enough evidence of each.
    trades = ([_trade("tf", 1.0, "trend_low")] * 20 +
              [_trade("tf", -1.0, "range_high")] * 20)
    a = CapitalAllocator()
    a.refresh(_FakeJournal(trades))
    assert a.weight("tf", "trend_low") == 1.5
    assert a.weight("tf", "range_high") == 0.1
    # Unseen regime falls back to the strategy's overall (mixed ≈ neutral).
    assert 0.5 <= a.weight("tf", "trend_high") <= 1.5


def test_allocator_ignores_learning_fills_and_thin_evidence():
    trades = ([_trade("ghost", 5.0, mode="learning")] * 50 +   # counterfactuals
              [_trade("thin", 5.0)] * 5)                        # < min_trades
    a = CapitalAllocator()
    a.refresh(_FakeJournal(trades))
    assert a.weight("ghost") == 1.0
    assert a.weight("thin") == 1.0


# ── scoreboard ─────────────────────────────────────────────────────────────────

def test_scoreboard_verdicts():
    now = datetime.now(timezone.utc)

    def _closed(pnl, days_ago):
        t = _trade("s", pnl)
        t.closed_at = now - timedelta(days=days_ago)
        return t

    # Previous window losing, current window winning → improving.
    trades = [_closed(-1.0, 45) for _ in range(15)] + \
             [_closed(1.0, 5) for _ in range(15)]
    out = scoreboard.compute(_FakeJournal(trades))
    assert out["verdict"] == "improving"
    assert out["current"]["n"] == 15 and out["previous"]["n"] == 15

    # Too little data → says so instead of guessing.
    out2 = scoreboard.compute(_FakeJournal(trades[:5]))
    assert out2["verdict"] == "insufficient data"
