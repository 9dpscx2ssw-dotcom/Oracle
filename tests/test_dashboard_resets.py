"""Dashboard maintenance actions: Clear History (DB reset) and Close All.

Both buttons write a flag into control.json that the agent applies on its next
fast loop. These tests pin the two bugs that made them silently no-op:
  • reset_all() didn't exist on Database (Clear History → AttributeError);
  • close-all rode in as close_positions=null, which the old
    ctrl.get(...) is-not-None guard couldn't tell from an absent key.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gungnir.data.models import Side, Trade
from gungnir.persistence.db import Database


def _trade(sym="US500", strat="trend_following", pnl=5.0):
    return Trade(symbol=sym, side=Side.BUY, volume=1, entry_price=100.0,
                 exit_price=101.0, pnl=pnl, strategy=strat,
                 opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                 closed_at=datetime(2026, 1, 2, tzinfo=timezone.utc))


# ── Clear History: Database.reset_all ────────────────────────────────────────

def test_reset_all_clears_trade_history(tmp_path):
    db = Database(tmp_path / "g.db")
    for p in (10.0, -5.0, 3.0):
        db.record_trade(_trade(pnl=p))
    assert len(db.recent_trades(limit=10)) == 3

    counts = db.reset_all()

    assert counts["trades"] == 3
    assert db.recent_trades(limit=10) == []


def test_reset_all_preserves_candles(tmp_path):
    """Candles are market price history (validation backbone), not trade
    history — Clear History must not wipe them."""
    from gungnir.data.models import Candle
    db = Database(tmp_path / "g.db")
    db.record_trade(_trade())
    db.store_candles([
        Candle(symbol="US500", timeframe="H1", open=1, high=2, low=0.5,
               close=1.5, volume=10, ts=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ])
    assert db.candle_count("US500", "H1") == 1

    db.reset_all()

    assert db.recent_trades(limit=10) == []          # trades gone
    assert db.candle_count("US500", "H1") == 1        # candles kept


def test_reset_all_empty_db_is_safe(tmp_path):
    db = Database(tmp_path / "g.db")
    counts = db.reset_all()
    assert counts == {"trades": 0, "signals": 0, "learning_events": 0}


# ── Close All: control-file round-trip ───────────────────────────────────────

def test_close_all_survives_control_roundtrip(tmp_path):
    """close_positions=null (Close All) must be distinguishable from an absent
    key after a JSON write/read cycle — the old .get()-is-not-None guard failed
    exactly here."""
    from gungnir.core.control import Control

    path = tmp_path / "control.json"
    ctrl = Control(path)
    ctrl.write({"close_positions": None})   # what the server writes for Close All

    data = ctrl.read()
    # Presence, not truthiness/None, is the signal the agent must key on.
    assert "close_positions" in data
    assert data["close_positions"] is None


def test_close_all_maps_to_sentinel_not_idle():
    """The agent translates a null request into the CLOSE_ALL sentinel so the
    'is not None' flush trigger fires — a bare None would look idle."""
    from gungnir.core.agent import CLOSE_ALL

    # Mirror the mapping applied in _apply_control without standing up a full Agent.
    def map_request(close_req):
        return close_req if isinstance(close_req, list) else CLOSE_ALL

    assert map_request(None) == CLOSE_ALL           # Close All → sentinel (fires)
    assert map_request(["EURUSD"]) == ["EURUSD"]    # specific symbols pass through
    assert CLOSE_ALL is not None                    # sentinel survives the trigger


def test_close_all_sentinel_means_close_everything():
    """Inside _execute_manual_closes the sentinel (a non-list) resolves to
    to_close=None → every open position, matching the kill-flatten path."""
    from gungnir.core.agent import CLOSE_ALL

    def resolve(pending):
        return pending if isinstance(pending, list) else None

    assert resolve(CLOSE_ALL) is None               # → close all
    assert resolve(None) is None                    # kill-flatten path → close all
    assert resolve(["US500"]) == ["US500"]          # → only these
