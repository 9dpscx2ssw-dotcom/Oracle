"""Pre-trade compliance gate: the last check before a LIVE order leaves.

Deliberately separate from PortfolioRisk: risk sizes and shapes orders,
compliance answers a different question — "is this order *permissible* at
all?" — with hard, simple rules an auditor can read:

  • absolute notional cap per order (fat-finger guard, independent of equity)
  • restricted symbol list (never trade these live)
  • daily live-order budget (a runaway loop can't machine-gun the account)

Every rejection names its rule. Applies to live orders only; shadow/learning
fills are simulation and stay unrestricted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..data.models import Order

log = logging.getLogger(__name__)


class PreTradeCompliance:
    def __init__(self, config: Config):
        c = config.get("compliance", default={}) or {}
        self.max_order_notional = c.get("max_order_notional")   # None = unlimited
        self.max_orders_per_day = int(c.get("max_orders_per_day", 200) or 0)
        self.restricted = {s.upper() for s in (c.get("restricted_symbols") or [])}
        self._day = None
        self._orders_today = 0
        # The daily budget exists to stop a runaway loop machine-gunning the
        # account — a crash-looping agent that resets its own counter on every
        # restart defeats it, so the count survives the process.
        self._state_path = Path(c.get("state_path", "data/compliance_state.json"))
        self._load_state()

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text())
            if data.get("day") == datetime.now(timezone.utc).date().isoformat():
                self._day = datetime.now(timezone.utc).date()
                self._orders_today = int(data.get("orders_today", 0))
                log.info("Restored compliance counter: %d live orders already "
                         "today", self._orders_today)
        except (OSError, ValueError, TypeError) as e:
            log.warning("Compliance state load failed (%s); counter starts at 0", e)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "day": (self._day or datetime.now(timezone.utc).date()).isoformat(),
                "orders_today": self._orders_today,
            }))
            tmp.replace(self._state_path)
        except OSError as e:
            log.warning("Compliance state save failed: %s", e)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if self._day != today:
            self._day = today
            self._orders_today = 0
            self._save_state()

    def check(self, order: Order, mark_price: float) -> tuple[bool, str | None]:
        """Return (allowed, reason). Call only for orders headed to the live broker."""
        self._roll_day()

        if order.symbol.upper() in self.restricted:
            return False, f"restricted symbol {order.symbol}"

        if self.max_orders_per_day and self._orders_today >= self.max_orders_per_day:
            return False, (f"daily live-order budget exhausted "
                           f"({self._orders_today}/{self.max_orders_per_day})")

        if self.max_order_notional:
            # Fail CLOSED on a missing mark: the fat-finger guard can't verify
            # an order it can't price, and live orders are exactly where an
            # unverifiable notional must not slip through.
            if not mark_price:
                return False, "no mark price available to verify order notional"
            notional = abs(order.volume * mark_price)
            if notional > float(self.max_order_notional):
                return False, (f"order notional {notional:,.0f} exceeds cap "
                               f"{float(self.max_order_notional):,.0f}")

        return True, None

    def count(self, order: Order) -> None:
        """Record a live order that was actually submitted."""
        self._roll_day()
        self._orders_today += 1
        self._save_state()
