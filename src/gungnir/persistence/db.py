"""SQLite persistence for trades, signals, and learning events.

SQLite keeps the homelab footprint near zero. Swap the connection for
TimescaleDB/Postgres later if you start storing tick-level history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..data.models import Side, Signal, Trade

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    volume      REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL,
    pnl         REAL,
    strategy    TEXT,
    mode        TEXT DEFAULT 'real',   -- real | shadow
    opened_at   TEXT NOT NULL,
    closed_at   TEXT,
    context     TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    conviction  REAL NOT NULL,
    price       REAL,
    disposition TEXT NOT NULL,          -- real | shadow | rejected_risk | rejected_off
    rationale   TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);

CREATE TABLE IF NOT EXISTS learning_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    hypothesis    TEXT,
    param_updates TEXT,
    accepted      INTEGER NOT NULL,     -- 0/1
    sharpe_before REAL,
    sharpe_after  REAL
);
CREATE INDEX IF NOT EXISTS idx_learning_strategy ON learning_events(strategy);

-- Closed-bar price history, accumulated as the agent runs. This is the
-- validation backbone: walk-forward gates replay proposals against it
-- instead of re-scoring the same journal rows they were fitted on.
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts        TEXT NOT NULL,
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, ts)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: don't fail instantly if another connection (dashboard,
        # reflection worker) briefly holds the write lock. WAL lets readers
        # and one writer coexist without blocking each other.
        self.conn = sqlite3.connect(str(path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            # WAL + NORMAL is the standard durability/latency trade: fsync on
            # checkpoint, not on every commit. The journal is rebuilt from the
            # broker on restart, so losing the last few ms of writes on power
            # cut is acceptable; blocking the trading loop on fsync is not.
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass   # e.g. read-only or network filesystem; default mode still works
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)")}
        if "mode" not in cols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN mode TEXT DEFAULT 'real'")
        # Signals gained recommended sizing/brackets + a linkable id + graded
        # outcome so the dashboard can show lot/TP/SL and WIN/LOSS per signal.
        scols = {r["name"] for r in self.conn.execute("PRAGMA table_info(signals)")}
        for col, ddl in (("lot", "REAL"), ("take_profit", "REAL"),
                         ("stop_loss", "REAL"), ("pnl", "REAL"), ("client_id", "TEXT"),
                         # Soft context at decision time, for the Signals tab:
                         # LLM sentiment score and the RL policy's P(take).
                         ("sentiment", "REAL"), ("rl_p", "REAL")):
            if col not in scols:
                self.conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {ddl}")

    # ── candles (validation history) ─────────────────────────────────────────

    def store_candles(self, candles: list) -> int:
        """Insert closed bars, ignoring ones already stored. Returns rows added."""
        if not candles:
            return 0
        rows = [(c.symbol, c.timeframe, c.ts.isoformat(), c.open, c.high, c.low,
                 c.close, c.volume) for c in candles]
        cur = self.conn.executemany(
            """INSERT OR IGNORE INTO candles
               (symbol, timeframe, ts, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?,?)""", rows)
        self.conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def load_candles(self, symbol: str, timeframe: str, limit: int = 2000) -> list:
        """Chronological (oldest-first) stored bars for a symbol/timeframe."""
        from ..data.models import Candle
        rows = self.conn.execute(
            """SELECT * FROM (SELECT * FROM candles WHERE symbol=? AND timeframe=?
               ORDER BY ts DESC LIMIT ?) ORDER BY ts ASC""",
            (symbol, timeframe, limit)).fetchall()
        return [Candle(symbol=r["symbol"], timeframe=r["timeframe"],
                       open=r["open"], high=r["high"], low=r["low"],
                       close=r["close"], volume=r["volume"] or 0.0,
                       ts=datetime.fromisoformat(r["ts"])) for r in rows]

    def candle_count(self, symbol: str, timeframe: str) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) AS n FROM candles WHERE symbol=? AND timeframe=?",
            (symbol, timeframe)).fetchone()
        return int(r["n"]) if r else 0

    # ── trades ───────────────────────────────────────────────────────────────

    def record_trade(self, trade: Trade) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (symbol, side, volume, entry_price, exit_price, pnl, strategy,
                mode, opened_at, closed_at, context)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade.symbol,
                trade.side.value,
                trade.volume,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.strategy,
                trade.mode,
                trade.opened_at.isoformat(),
                trade.closed_at.isoformat() if trade.closed_at else None,
                json.dumps(trade.context),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_trades(
        self, strategy: str | None = None, mode: str | None = None, limit: int = 100,
        closed_only: bool = False,
    ) -> list[Trade]:
        where, args = [], []
        if strategy:
            where.append("strategy=?")
            args.append(strategy)
        if mode:
            where.append("mode=?")
            args.append(mode)
        if closed_only:
            # Filter in SQL, not post-limit in Python — otherwise a window of
            # "the last N trades" silently shrinks to however many were closed.
            where.append("closed_at IS NOT NULL")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        args.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM trades{clause} ORDER BY id DESC LIMIT ?", args
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def prune_signals(self, retention_days: int = 90) -> int:
        """Delete rejected-signal rows older than the retention window.

        The signals table is the only unbounded one (26 strategies × every bar).
        Executed/graded signals (real/shadow, or carrying a pnl) are kept — they
        are the learning record; stale *rejections* are just veto telemetry."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=retention_days)).isoformat()
        cur = self.conn.execute(
            "DELETE FROM signals WHERE ts < ? AND disposition LIKE 'rejected%' "
            "AND pnl IS NULL", (cutoff,))
        self.conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def trade_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT mode, COUNT(*) n FROM trades GROUP BY mode"
        ).fetchall()
        return {r["mode"] or "real": r["n"] for r in rows}

    @staticmethod
    def _parse_dt(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _row_to_trade(r: sqlite3.Row) -> Trade:
        keys = r.keys()
        # Load the persisted timestamps instead of letting them default to now() —
        # the equity curve, ordering and drawdown all depend on real trade times.
        opened = Database._parse_dt(r["opened_at"]) if "opened_at" in keys else None
        closed = Database._parse_dt(r["closed_at"]) if "closed_at" in keys else None
        fields = dict(
            symbol=r["symbol"],
            side=Side(r["side"]),
            volume=r["volume"],
            entry_price=r["entry_price"],
            exit_price=r["exit_price"],
            pnl=r["pnl"],
            strategy=r["strategy"] or "",
            mode=(r["mode"] if "mode" in keys else "real") or "real",
            context=json.loads(r["context"]) if r["context"] else {},
        )
        if opened is not None:
            fields["opened_at"] = opened
        if closed is not None:
            fields["closed_at"] = closed
        return Trade(**fields)

    # ── signals ──────────────────────────────────────────────────────────────

    def record_signal(self, signal: Signal, disposition: str, price: float | None,
                      lot: float | None = None, take_profit: float | None = None,
                      stop_loss: float | None = None, client_id: str | None = None,
                      sentiment: float | None = None, rl_p: float | None = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO signals
               (ts, strategy, symbol, side, conviction, price, disposition, rationale,
                lot, take_profit, stop_loss, client_id, sentiment, rl_p)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal.ts.isoformat(), signal.strategy, signal.symbol, signal.side.value,
                signal.conviction, price, disposition, signal.rationale,
                lot, take_profit, stop_loss, client_id, sentiment, rl_p,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_signal_outcome(self, client_id: str, pnl: float) -> None:
        """Grade the (executed) signal that produced a now-closed trade."""
        self.conn.execute(
            "UPDATE signals SET pnl=? WHERE client_id=? AND pnl IS NULL",
            (pnl, client_id),
        )
        self.conn.commit()

    def recent_signals(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── learning ─────────────────────────────────────────────────────────────

    def record_learning_event(
        self,
        strategy: str,
        hypothesis: str,
        param_updates: dict,
        accepted: bool,
        sharpe_before: float | None = None,
        sharpe_after: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO learning_events
               (ts, strategy, hypothesis, param_updates, accepted, sharpe_before, sharpe_after)
               VALUES (?,?,?,?,?,?,?)""",
            (_now(), strategy, hypothesis, json.dumps(param_updates),
             1 if accepted else 0, sharpe_before, sharpe_after),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_learning_events(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM learning_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["param_updates"] = json.loads(d["param_updates"]) if d["param_updates"] else {}
            d["accepted"] = bool(d["accepted"])
            out.append(d)
        return out

    def reset_all(self) -> dict[str, int]:
        """Wipe trade history: trades, signals, and learning events.

        Deliberately KEEPS the `candles` table — those are cached market price
        bars (the walk-forward validation backbone), not trade history, and
        re-accumulating them is slow and costs API calls. Returns the row count
        deleted per table so the caller can log/confirm what was cleared.
        """
        counts: dict[str, int] = {}
        for table in ("trades", "signals", "learning_events"):
            cur = self.conn.execute(f"DELETE FROM {table}")
            counts[table] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        self.conn.commit()
        # Reclaim the freed pages so the file actually shrinks on disk.
        try:
            self.conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass  # VACUUM can't run inside a transaction on some builds; harmless
        return counts

    def close(self) -> None:
        self.conn.close()
