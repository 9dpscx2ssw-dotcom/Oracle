"""The system scoreboard: is the WHOLE book improving?

Every learning layer optimizes locally (a strategy's params, a symbol's
inclusion, a signal's take/skip, a strategy's capital weight). None of them
answers the only question that defines a self-improving system: is the book's
risk-adjusted performance trending up across time?

This computes book-level metrics over the most recent 30 days of closed
trades and the 30 days before that, and states the verdict plainly. Published
in status.json every slow loop and logged — if this line doesn't say
"improving" across months, the system is not self-improving, whatever the
components claim.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

WINDOW_DAYS = 30


def _metrics(trades: list) -> dict:
    pnls = [t.pnl for t in trades if t.pnl is not None]
    if not pnls:
        return {"n": 0, "pnl": 0.0, "expectancy": 0.0, "sharpe": 0.0, "win_rate": 0.0}
    wins = sum(1 for p in pnls if p > 0)
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    sd = math.sqrt(var)
    if sd:
        sharpe = mean / sd
    else:
        # Zero variance: every trade identical. Direction still matters —
        # a window of pure profit must not score the same as pure loss.
        sharpe = math.copysign(9.99, mean) if mean else 0.0
    return {
        "n": len(pnls),
        "pnl": round(sum(pnls), 2),
        "expectancy": round(mean, 4),
        "sharpe": round(sharpe, 4),                     # per-trade, unannualized
        "win_rate": round(wins / len(pnls), 3),
    }


def compute(journal) -> dict:
    """Book-level current-vs-previous window comparison, real+shadow and real-only."""
    now = datetime.now(timezone.utc)
    cur_start = now - timedelta(days=WINDOW_DAYS)
    prev_start = now - timedelta(days=2 * WINDOW_DAYS)

    def _ts(t):
        ts = t.closed_at or t.opened_at
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    closed = [t for t in journal.recent(limit=5000)
              if t.pnl is not None and t.mode != "learning"]
    cur = [t for t in closed if _ts(t) >= cur_start]
    prev = [t for t in closed if prev_start <= _ts(t) < cur_start]

    cur_m, prev_m = _metrics(cur), _metrics(prev)
    if cur_m["n"] < 10 or prev_m["n"] < 10:
        verdict = "insufficient data"
    elif cur_m["sharpe"] > prev_m["sharpe"] and cur_m["expectancy"] >= prev_m["expectancy"]:
        verdict = "improving"
    elif cur_m["sharpe"] < prev_m["sharpe"] and cur_m["expectancy"] < prev_m["expectancy"]:
        verdict = "degrading"
    else:
        verdict = "flat"

    real_cur = _metrics([t for t in cur if t.mode == "real"])
    out = {
        "window_days": WINDOW_DAYS,
        "current": cur_m,
        "previous": prev_m,
        "real_current": real_cur,
        "verdict": verdict,
        "as_of": now.isoformat(),
    }
    log.info("Scoreboard: book %dd sharpe %.4f (prev %.4f), expectancy %.4f "
             "(prev %.4f), %d trades — %s",
             WINDOW_DAYS, cur_m["sharpe"], prev_m["sharpe"],
             cur_m["expectancy"], prev_m["expectancy"], cur_m["n"], verdict.upper())
    return out
