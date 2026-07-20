"""Control-plane protections: token auth, set_risk validation, DB timestamps."""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from gungnir.data.models import Side, Trade
from gungnir.persistence.db import Database


def _client(token: str | None = None):
    if token is None:
        os.environ.pop("DASHBOARD_TOKEN", None)
    else:
        os.environ["DASHBOARD_TOKEN"] = token
    import gungnir.dashboard.server as srv
    importlib.reload(srv)
    return TestClient(srv.create_app())


def test_writes_require_token_when_configured():
    c = _client(token="secret")
    # No token → 401 on a write; GET still works.
    assert c.get("/api/status").status_code == 200
    assert c.post("/api/offline_gate/toggle", json={"enabled": True}).status_code == 401
    ok = c.post("/api/offline_gate/toggle", json={"enabled": False},
                headers={"X-Dashboard-Token": "secret"})
    assert ok.status_code == 200
    os.environ.pop("DASHBOARD_TOKEN", None)


def test_writes_fail_closed_when_token_is_missing():
    c = _client(token=None)
    response = c.post("/api/offline_gate/toggle", json={"enabled": True})
    assert response.status_code == 503
    assert "DASHBOARD_TOKEN" in response.json()["error"]
    os.environ.pop("DASHBOARD_TOKEN", None)


def test_set_risk_rejects_unknown_and_invalid_keys():
    c = _client(token="secret")
    r = c.post("/api/risk", json={"account_risk_per_trade": 0.01, "evil_key": 1,
                                  "daily_loss_limit": 5.0},
               headers={"X-Dashboard-Token": "secret"})
    body = r.json()
    assert r.status_code == 200
    assert "account_risk_per_trade" in body["applied"]
    assert "evil_key" in body["rejected"] and "daily_loss_limit" in body["rejected"]

    r2 = c.post("/api/risk", json={"nonsense": True},
                headers={"X-Dashboard-Token": "secret"})
    assert r2.status_code == 400


def test_db_loads_real_trade_timestamps(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    opened = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    closed = datetime(2026, 1, 2, 4, 0, 0, tzinfo=timezone.utc)
    db.record_trade(Trade(symbol="EURUSD", side=Side.BUY, volume=1.0, entry_price=1.1,
                          exit_price=1.2, pnl=10.0, strategy="s", mode="real",
                          opened_at=opened, closed_at=closed))
    t = db.recent_trades(limit=1)[0]
    assert t.opened_at == opened and t.closed_at == closed
