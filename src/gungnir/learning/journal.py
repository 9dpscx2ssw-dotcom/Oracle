"""Trade journal: the agent's memory. Thin wrapper over the DB that keeps a
context snapshot with each trade (so learning can correlate outcomes with the
conditions that produced them) and records every signal + learning event.
"""

from __future__ import annotations

from ..core import metrics
from ..data.models import Signal, Trade
from ..persistence.db import Database


class Journal:
    def __init__(self, db: Database):
        self.db = db

    # ── trades ──
    def record(self, trade: Trade, context: dict | None = None) -> int:
        if context:
            trade.context = {**trade.context, **context}
        metrics.trade_closed(trade.mode, trade.pnl)
        return self.db.record_trade(trade)

    def recent(
        self, strategy: str | None = None, mode: str | None = None, limit: int = 100
    ) -> list[Trade]:
        return self.db.recent_trades(strategy=strategy, mode=mode, limit=limit)

    def closed(self, strategy: str | None = None, limit: int = 100) -> list[Trade]:
        return self.db.recent_trades(strategy=strategy, limit=limit, closed_only=True)

    # ── signals ──
    def record_signal(self, signal: Signal, disposition: str, price: float | None,
                      **kw) -> int:
        metrics.inc_signal(disposition)
        return self.db.record_signal(signal, disposition, price, **kw)

    def update_signal_outcome(self, client_id: str, pnl: float) -> None:
        self.db.update_signal_outcome(client_id, pnl)

    def recent_signals(self, limit: int = 100) -> list[dict]:
        return self.db.recent_signals(limit=limit)

    # ── learning ──
    def record_learning_event(self, *args, **kwargs) -> int:
        return self.db.record_learning_event(*args, **kwargs)

    def recent_learning_events(self, limit: int = 100) -> list[dict]:
        return self.db.recent_learning_events(limit=limit)
