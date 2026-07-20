"""Markets tab: most-profitable-strategy (MPS) aggregation per instrument."""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from gungnir.data.models import Side, Trade
from gungnir.persistence.db import Database


def _trade(sym, strat, pnl):
    return Trade(symbol=sym, side=Side.BUY, volume=1, entry_price=100.0,
                 exit_price=101.0, pnl=pnl, strategy=strat,
                 opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                 closed_at=datetime(2026, 1, 2, tzinfo=timezone.utc))


def _markets(tmp_path):
    os.environ["GUNGNIR_DB_PATH"] = str(tmp_path / "g.db")
    os.environ["GUNGNIR_STATUS_PATH"] = str(tmp_path / "status.json")
    os.environ.pop("DASHBOARD_TOKEN", None)
    db = Database(os.environ["GUNGNIR_DB_PATH"])
    # US100: trend_following wins every trade (no losses → ∞ profit factor),
    # mean_reversion loses → trend_following is the MPS.
    for p in (10.0, 10.0, 10.0):
        db.record_trade(_trade("US100", "trend_following", p))
    db.record_trade(_trade("US100", "mean_reversion", -5.0))
    # EURUSD: bb_rsi +20,+10,-5,-5 → pnl 20, 50% win rate, PF = 30/10 = 3.0.
    for p in (20.0, 10.0, -5.0, -5.0):
        db.record_trade(_trade("EURUSD", "bb_rsi", p))
    db.close()
    json.dump({"universe": ["US100", "EURUSD"], "views": {}, "per_symbol": {}},
              open(os.environ["GUNGNIR_STATUS_PATH"], "w"))
    import gungnir.dashboard.server as srv
    importlib.reload(srv)
    out = {m["symbol"]: m for m in TestClient(srv.create_app()).get("/api/markets").json()["markets"]}
    os.environ.pop("GUNGNIR_DB_PATH", None)
    os.environ.pop("GUNGNIR_STATUS_PATH", None)
    return out


def test_mps_picks_highest_pl_strategy_and_metrics(tmp_path):
    m = _markets(tmp_path)

    us = m["US100"]["mps"]
    assert us["strategy_name"] == "trend_following"     # highest total P/L
    assert us["pnl"] == 30.0
    assert us["win_rate"] == 100.0
    assert us["profit_factor"] is None                  # no losses → ∞ (rendered client-side)

    eu = m["EURUSD"]["mps"]
    assert eu["strategy_name"] == "bb_rsi"
    assert eu["pnl"] == 20.0
    assert eu["win_rate"] == 50.0
    assert eu["profit_factor"] == 3.0


def test_mps_absent_without_closed_trades(tmp_path):
    os.environ["GUNGNIR_DB_PATH"] = str(tmp_path / "empty.db")
    os.environ["GUNGNIR_STATUS_PATH"] = str(tmp_path / "status.json")
    os.environ.pop("DASHBOARD_TOKEN", None)
    Database(os.environ["GUNGNIR_DB_PATH"]).close()
    json.dump({"universe": ["US100"], "views": {}, "per_symbol": {}},
              open(os.environ["GUNGNIR_STATUS_PATH"], "w"))
    import gungnir.dashboard.server as srv
    importlib.reload(srv)
    m = {x["symbol"]: x for x in TestClient(srv.create_app()).get("/api/markets").json()["markets"]}
    assert m["US100"]["mps"] is None
    os.environ.pop("GUNGNIR_DB_PATH", None)
    os.environ.pop("GUNGNIR_STATUS_PATH", None)
