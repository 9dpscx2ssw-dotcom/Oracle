"""Signals carry recommended lot/TP/SL and get graded WIN/LOSS via client_id."""

from __future__ import annotations

import sqlite3

from gungnir.data.models import Side, Signal
from gungnir.persistence.db import Database


def _sig(symbol="EURUSD"):
    return Signal(strategy="cci_macd", symbol=symbol, side=Side.BUY, conviction=0.7)


def test_record_signal_stores_lot_tp_sl_and_grades_outcome(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    cid = "cci_macd:EURUSD:123"
    db.record_signal(_sig(), "real", price=1.10, lot=0.5, take_profit=1.13,
                     stop_loss=1.08, client_id=cid)
    row = db.recent_signals(limit=1)[0]
    assert row["lot"] == 0.5 and row["take_profit"] == 1.13 and row["stop_loss"] == 1.08
    assert row["pnl"] is None                      # not graded yet

    db.update_signal_outcome(cid, 12.5)            # the trade closed a winner
    row = db.recent_signals(limit=1)[0]
    assert row["pnl"] == 12.5

    # A second close for the same id must not re-grade (pnl already set).
    db.update_signal_outcome(cid, -99.0)
    assert db.recent_signals(limit=1)[0]["pnl"] == 12.5


def test_migration_adds_signal_columns_to_legacy_db(tmp_path):
    # Build a pre-migration signals table (no lot/tp/sl/pnl/client_id).
    p = str(tmp_path / "legacy.db")
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, strategy TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
        conviction REAL NOT NULL, price REAL, disposition TEXT NOT NULL, rationale TEXT)""")
    con.commit()
    con.close()
    db = Database(p)                               # _migrate should add the columns
    cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(signals)")}
    assert {"lot", "take_profit", "stop_loss", "pnl", "client_id"} <= cols
    # And it's usable.
    db.record_signal(_sig(), "real", 1.1, lot=0.2, client_id="x:y:1")
    assert db.recent_signals(limit=1)[0]["lot"] == 0.2
