"""Tests for the new feature set: backtesting, control channel, persistence,
strategy modes, and the dashboard API."""

from __future__ import annotations

import json

from gungnir.backtest import engine
from gungnir.core.control import Control
from gungnir.data.models import Side, Signal
from gungnir.persistence.db import Database
from gungnir.strategy.examples.trend_following import TrendFollowing


def test_backtest_runs(tmp_path):
    candles = engine.synthetic_candles("X", n=300, seed=3)
    res = engine.run(TrendFollowing(params={"min_conviction": 0.2}), candles, "X")
    assert res.metrics.n_trades >= 0
    assert len(res.equity_curve) > 0
    # PnL of closed backtest trades must equal the metric total.
    total = sum(t.pnl for t in res.trades if t.pnl is not None)
    assert abs(total - res.metrics.total_pnl) < 1e-6


def test_backtest_is_reproducible():
    a = engine.synthetic_candles("X", n=200, seed=42)
    b = engine.synthetic_candles("X", n=200, seed=42)
    assert [c.close for c in a] == [c.close for c in b]


def test_control_roundtrip(tmp_path):
    ctrl = Control(tmp_path / "control.json")
    assert ctrl.read()["strategies"] == {}
    ctrl.set_strategy_mode("trend_following", "live")
    ctrl.set_paused(True)
    data = ctrl.read()
    assert data["strategies"]["trend_following"] == "live"
    assert data["paused"] is True


def test_control_instrument_toggle(tmp_path):
    ctrl = Control(tmp_path / "control.json")
    assert ctrl.read()["instruments"] == {}
    ctrl.set_instrument_enabled("US100", False)
    ctrl.set_instrument_enabled("BTCUSD", True)
    data = ctrl.read()
    assert data["instruments"] == {"US100": False, "BTCUSD": True}
    # The agent derives its skip-set from the disabled entries.
    disabled = {s for s, en in data["instruments"].items() if not en}
    assert disabled == {"US100"}


def test_strategy_modes():
    s = TrendFollowing(mode="off")
    assert not s.enabled
    s.mode = "live"
    assert s.enabled


def test_db_signals_and_learning(tmp_path):
    db = Database(tmp_path / "t.db")
    sig = Signal(strategy="trend_following", symbol="EURUSD", side=Side.BUY, conviction=0.7)
    db.record_signal(sig, "shadow", 1.10)
    db.record_signal(sig, "rejected_risk", 1.10)
    sigs = db.recent_signals()
    assert len(sigs) == 2
    assert {s["disposition"] for s in sigs} == {"shadow", "rejected_risk"}

    db.record_learning_event("trend_following", "tighten stop", {"fast_ema": 15},
                             accepted=True, sharpe_before=0.5, sharpe_after=0.8)
    events = db.recent_learning_events()
    assert len(events) == 1
    assert events[0]["accepted"] is True
    assert events[0]["param_updates"] == {"fast_ema": 15}
    db.close()


def test_dashboard_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    db_path = tmp_path / "g.db"
    status_path = tmp_path / "status.json"
    Database(db_path).close()
    status_path.write_text(json.dumps({
        "mode": "dry-run", "balance": 10000, "equity": 10000, "running_pl": 0,
        "closed_pl": 0, "trade_counts": {"real": 0, "shadow": 0}, "views": {},
        "strategies": [], "macro": [], "news": [],
    }))
    monkeypatch.setenv("GUNGNIR_DB_PATH", str(db_path))
    monkeypatch.setenv("GUNGNIR_STATUS_PATH", str(status_path))
    monkeypatch.setenv("GUNGNIR_CONTROL_PATH", str(tmp_path / "control.json"))
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-dashboard-token")

    from gungnir.dashboard.server import create_app

    c = TestClient(create_app())
    for ep in ["/api/overview", "/api/instruments", "/api/intelligence",
               "/api/strategies", "/api/metrics", "/api/signals", "/api/trades",
               "/api/learning", "/api/settings"]:
        assert c.get(ep).status_code == 200, ep

    # control writes
    headers = {"X-Dashboard-Token": "test-dashboard-token"}
    assert c.post("/api/strategies/trend_following/mode", json={"mode": "live"},
                  headers=headers).json()["ok"]
    assert c.post("/api/strategies/trend_following/mode", json={"mode": "bogus"},
                  headers=headers).status_code == 400
    assert c.post("/api/pause", json={"paused": True}, headers=headers).json()["ok"]

    # backtest
    r = c.post("/api/backtest", json={"strategy": "mean_reversion", "n_bars": 200, "seed": 1},
               headers=headers)
    assert r.status_code == 200
    assert "metrics" in r.json() and "equity_curve" in r.json()
