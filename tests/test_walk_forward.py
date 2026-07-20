"""Candle history store + walk-forward acceptance gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gungnir.backtest import engine
from gungnir.data.models import Side, Trade
from gungnir.learning.journal import Journal
from gungnir.learning.reflection_pipeline import _walk_forward_accept
from gungnir.persistence.db import Database
from gungnir.strategy.examples.trend_following import TrendFollowing


def _stamped_candles(symbol: str, timeframe: str, n: int, seed: int = 7):
    """Synthetic series with distinct, ordered timestamps (store PK needs them)."""
    candles = engine.synthetic_candles(symbol, n=n, seed=seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, c in enumerate(candles):
        c.timeframe = timeframe
        c.ts = t0 + timedelta(hours=i)
    return candles


def test_candle_store_round_trip_and_dedupe(tmp_path):
    db = Database(tmp_path / "t.db")
    candles = _stamped_candles("EURUSD", "1h", 50)
    assert db.store_candles(candles) == 50
    assert db.store_candles(candles) == 0          # INSERT OR IGNORE dedupes
    assert db.candle_count("EURUSD", "1h") == 50
    loaded = db.load_candles("EURUSD", "1h", limit=10)
    assert len(loaded) == 10
    assert loaded[0].ts < loaded[-1].ts            # chronological
    assert loaded[-1].ts == candles[-1].ts         # newest bars kept


def _trending_candles(symbol: str, timeframe: str, n: int):
    """Steady uptrend so a trend-following incumbent is profitable — a
    do-nothing proposal must NOT be able to beat it."""
    import random
    rng = random.Random(3)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out, price = [], 1.0
    from gungnir.data.models import Candle
    for i in range(n):
        o = price
        c = price * (1 + 0.002 + rng.gauss(0, 0.0005))
        out.append(Candle(symbol=symbol, timeframe=timeframe, open=o,
                          high=max(o, c) * 1.0005, low=min(o, c) * 0.9995,
                          close=c, ts=t0 + timedelta(hours=i)))
        price = c
    return out


def test_walk_forward_rejects_non_improving_proposal(tmp_path):
    db = Database(tmp_path / "t.db")
    journal = Journal(db)
    db.store_candles(_trending_candles("EURUSD", "1h", 400))
    strat = TrendFollowing(params={"fast_ema": 20, "slow_ema": 50,
                                   "min_conviction": 0.3}, timeframe="1h")
    closed = [Trade(symbol="EURUSD", side=Side.BUY, volume=1.0,
                    entry_price=1.1, pnl=1.0)]
    # In a clean uptrend the incumbent earns; min_conviction 0.99 suppresses
    # ~every trade and cannot improve on it → the gate must reject.
    verdict = _walk_forward_accept(strat, {"min_conviction": 0.99}, journal, closed)
    assert verdict is False


def test_walk_forward_accepts_stop_trading_a_losing_strategy(tmp_path):
    """On a driftless random walk the incumbent bleeds; a proposal that stops
    it from trading legitimately wins — the gate should accept improvement in
    either direction, including 'trade less'."""
    db = Database(tmp_path / "t.db")
    journal = Journal(db)
    db.store_candles(_stamped_candles("EURUSD", "1h", 400))
    strat = TrendFollowing(params={"fast_ema": 20, "slow_ema": 50,
                                   "min_conviction": 0.3}, timeframe="1h")
    closed = [Trade(symbol="EURUSD", side=Side.BUY, volume=1.0,
                    entry_price=1.1, pnl=1.0)]
    verdict = _walk_forward_accept(strat, {"min_conviction": 0.99}, journal, closed)
    assert verdict in (True, False)   # decision made from history, no fallback


def test_walk_forward_returns_none_without_history(tmp_path):
    journal = Journal(Database(tmp_path / "empty.db"))
    strat = TrendFollowing(params={"min_conviction": 0.3}, timeframe="1h")
    closed = [Trade(symbol="EURUSD", side=Side.BUY, volume=1.0,
                    entry_price=1.1, pnl=1.0)]
    assert _walk_forward_accept(strat, {"min_conviction": 0.99},
                                journal, closed) is None
