"""Daily/weekly performance report: aggregation, summary text, Telegram render,
and the /api/performance endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gungnir.data.models import Side, Trade
from gungnir.learning import reports
from gungnir.learning.journal import Journal
from gungnir.persistence.db import Database


def _trade(symbol, strategy, pnl, *, side=Side.BUY, mode="real", conf=0.6,
           closed=None) -> Trade:
    now = datetime.now(timezone.utc)
    return Trade(symbol=symbol, side=side, volume=1.0, entry_price=100.0,
                 exit_price=100.0 + pnl, pnl=pnl, strategy=strategy, mode=mode,
                 opened_at=(closed or now) - timedelta(minutes=5),
                 closed_at=closed or now, context={"confidence": conf})


class _FakeJournal:
    """Minimal journal stand-in returning a fixed closed-trade list."""
    def __init__(self, trades):
        self._t = trades

    def closed(self, limit=100):
        return self._t[:limit]

    def recent(self, strategy=None, mode=None, limit=100):
        # Mirrors Journal.recent: recent trades (closed or provisional), with
        # the same optional strategy/mode filters reports.build() relies on.
        out = self._t
        if strategy is not None:
            out = [t for t in out if t.strategy == strategy]
        if mode is not None:
            out = [t for t in out if t.mode == mode]
        return out[:limit]


def test_window_aggregates_by_group():
    now = datetime.now(timezone.utc)
    trades = [
        _trade("US500", "trend", 100.0, conf=0.7),
        _trade("US500", "trend", -40.0, conf=0.5),
        _trade("EURUSD", "revert", 20.0, side=Side.SELL, conf=0.9),
    ]
    w = reports.window(trades, "Today", now - timedelta(hours=1), now)
    assert w["trades"] == 3
    assert w["wins"] == 2 and w["losses"] == 1
    assert w["pnl"] == 80.0
    assert w["win_rate"] == 0.667                        # 2/3 rounded to 3dp
    # strategy rows sorted by pnl desc; trend nets +60
    by_s = {r["name"]: r for r in w["by_strategy"]}
    assert by_s["trend"]["pnl"] == 60.0 and by_s["trend"]["trades"] == 2
    assert by_s["revert"]["pnl"] == 20.0
    assert w["by_strategy"][0]["name"] == "trend"        # sorted
    # instrument breakdown
    by_i = {r["name"]: r for r in w["by_instrument"]}
    assert by_i["US500"]["pnl"] == 60.0
    # side breakdown
    assert w["by_side"]["sell"]["trades"] == 1
    assert w["by_side"]["buy"]["trades"] == 2
    # extremes
    assert w["best_trade"]["pnl"] == 100.0 and w["best_trade"]["symbol"] == "US500"
    assert w["worst_trade"]["pnl"] == -40.0
    # avg confidence over all three: (0.7+0.5+0.9)/3
    assert abs(w["avg_confidence"] - 0.7) < 1e-9


def test_profit_factor_infinite_serializes_as_none():
    now = datetime.now(timezone.utc)
    w = reports.window([_trade("US500", "trend", 50.0)], "x",
                       now - timedelta(hours=1), now)
    assert w["profit_factor"] is None                    # no losses → inf → None (JSON-safe)


def test_real_vs_shadow_split():
    now = datetime.now(timezone.utc)
    trades = [_trade("US500", "s", 30.0, mode="real"),
              _trade("US500", "s", -10.0, mode="shadow")]
    w = reports.window(trades, "x", now - timedelta(hours=1), now)
    assert w["real_pnl"] == 30.0 and w["shadow_pnl"] == -10.0


def test_build_splits_daily_and_weekly():
    now = datetime.now(timezone.utc)
    today = _trade("US500", "trend", 100.0, closed=now)
    old = _trade("US500", "trend", -50.0, closed=now - timedelta(days=3))
    ancient = _trade("US500", "trend", 999.0, closed=now - timedelta(days=20))
    j = _FakeJournal([today, old, ancient])
    r = reports.build(j, now=now)
    assert r["daily"]["trades"] == 1 and r["daily"]["pnl"] == 100.0
    assert r["weekly"]["trades"] == 2                    # today + 3d ago, not 20d
    assert r["weekly"]["pnl"] == 50.0
    assert "summary" in r and r["summary"]


def test_learning_trades_excluded():
    now = datetime.now(timezone.utc)
    real = _trade("US500", "s", 10.0, closed=now)
    learn = _trade("US500", "s", 500.0, mode="learning", closed=now)
    r = reports.build(_FakeJournal([real, learn]), now=now)
    assert r["daily"]["trades"] == 1 and r["daily"]["pnl"] == 10.0


def test_summary_text_reads_naturally():
    now = datetime.now(timezone.utc)
    r = reports.build(_FakeJournal([
        _trade("US500", "trend", 100.0, closed=now),
        _trade("EURUSD", "revert", -30.0, closed=now),
    ]), now=now)
    s = r["summary"]
    assert "trades closed" in s and "won" in s
    assert "Best strategy trend" in s


def test_summary_handles_empty_day():
    now = datetime.now(timezone.utc)
    r = reports.build(_FakeJournal([]), now=now)
    assert r["summary"] == "No trades closed today."


def test_format_telegram_compact():
    now = datetime.now(timezone.utc)
    r = reports.build(_FakeJournal([
        _trade("US500", "trend", 100.0, closed=now),
        _trade("GOLD", "revert", -20.0, closed=now),
    ]), now=now)
    msg = reports.format_telegram(r, equity=10_100.0, verdict="improving")
    assert "📊 Daily report" in msg
    assert "Equity 10,100.00" in msg
    assert "Today:" in msg and "7d:" in msg
    assert "Book verdict: improving" in msg
    assert "best US500/trend" in msg


def test_extended_metrics_block():
    now = datetime.now(timezone.utc)
    trades = [
        _trade("US500", "trend", 60.0, closed=now - timedelta(minutes=40)),
        _trade("US500", "trend", 40.0, closed=now - timedelta(minutes=30)),
        _trade("US500", "trend", -30.0, closed=now - timedelta(minutes=20)),
        _trade("US500", "trend", -10.0, closed=now - timedelta(minutes=10)),
    ]
    w = reports.window(trades, "x", now - timedelta(hours=1), now)
    m = w["metrics"]
    assert m["avg_win"] == 50.0                          # (60+40)/2
    assert m["avg_loss"] == 20.0                          # (30+10)/2
    assert m["payoff_ratio"] == 2.5                       # 50/20
    # kelly = win_rate - (1-win_rate)/payoff = 0.5 - 0.5/2.5
    assert abs(m["kelly"] - 0.3) < 1e-6
    assert m["max_win_streak"] == 2 and m["max_loss_streak"] == 2
    assert m["current_streak"] == -2                      # last two were losers
    assert m["recovery_factor"] is not None
    assert m["avg_hold_min"] == 5.0                       # every trade held 5 min


def test_equity_curve_is_cumulative_and_ordered():
    now = datetime.now(timezone.utc)
    trades = [
        _trade("US500", "trend", -20.0, closed=now - timedelta(minutes=5)),
        _trade("US500", "trend", 50.0, closed=now - timedelta(minutes=30)),
    ]
    w = reports.window(trades, "x", now - timedelta(hours=1), now)
    cum = [p["cum"] for p in w["equity_curve"]]
    assert cum == [50.0, 30.0]                            # chronological, cumulative


def test_matrix_and_combos():
    now = datetime.now(timezone.utc)
    trades = [
        _trade("US100", "trend", 100.0, closed=now),
        _trade("US100", "revert", -20.0, closed=now),
        _trade("GOLD", "trend", 30.0, closed=now),
    ]
    w = reports.window(trades, "x", now - timedelta(hours=1), now)
    mx = w["matrix"]
    assert mx["instruments"][0] == "US100"               # highest total edge first
    assert "trend" in mx["strategies"] and "revert" in mx["strategies"]
    ti = mx["instruments"].index("US100")
    ts = mx["strategies"].index("trend")
    assert mx["cells"][ti][ts] == 100.0
    combos = {c["name"]: c for c in w["combos"]}
    assert combos["US100 × trend"]["pnl"] == 100.0
    assert w["combos"][0]["pnl"] >= w["combos"][-1]["pnl"]  # sorted desc


def test_signal_dist_counts():
    now = datetime.now(timezone.utc)
    trades = [_trade("US500", "s", 30.0, mode="real", closed=now),
              _trade("US500", "s", 10.0, mode="real", closed=now),
              _trade("US500", "s", -5.0, mode="shadow", closed=now)]
    w = reports.window(trades, "x", now - timedelta(hours=1), now)
    assert w["signal_dist"]["real"]["trades"] == 2
    assert w["signal_dist"]["shadow"]["trades"] == 1
    assert w["signal_dist"]["real"]["pnl"] == 40.0


def test_deltas_vs_previous_window():
    now = datetime.now(timezone.utc)
    today = _trade("US500", "trend", 100.0, closed=now)
    yesterday = _trade("US500", "trend", 40.0, closed=now - timedelta(days=1, hours=1))
    r = reports.build(_FakeJournal([today, yesterday]), now=now)
    d = r["deltas"]["daily"]
    assert d["pnl"] == 60.0                               # 100 today vs 40 yesterday
    assert d["prev_trades"] == 1


def test_performance_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from gungnir.dashboard.server import create_app
    db_path = tmp_path / "perf.db"
    db = Database(db_path)
    j = Journal(db)
    now = datetime.now(timezone.utc)
    for t in [_trade("US500", "trend", 40.0, closed=now),
              _trade("US500", "trend", -15.0, closed=now),
              _trade("EURUSD", "revert", 22.0, side=Side.SELL, closed=now)]:
        j.record(t)
    db.close()
    monkeypatch.setenv("GUNGNIR_DB_PATH", str(db_path))
    client = TestClient(create_app())
    r = client.get("/api/performance")
    assert r.status_code == 200
    body = r.json()
    assert body["daily"]["trades"] == 3
    assert body["daily"]["pnl"] == 47.0
    assert any(row["name"] == "trend" for row in body["daily"]["by_strategy"])
    assert body["summary"]
