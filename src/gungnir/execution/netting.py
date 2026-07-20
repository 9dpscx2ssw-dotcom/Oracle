"""Virtual per-strategy books with netted account execution.

Without netting, every strategy holds its own real position: five strategies
buying US500 in the same loop open five broker deals at the same price (5×
spread, 5× exposure to one idea), and a simultaneous buy+sell pair pays the
spread twice to hold net-zero risk. NettingBroker splits those two concerns:

  • **Virtual books** — every strategy order fills instantly on an internal
    per-(symbol, strategy) paper ledger, at the mark, with the configured cost
    model, brackets and MFE/MAE tracking. These round-trips are what the agent
    journals, the allocator weighs, and the RL policy learns from — attribution
    is exactly as if each strategy traded alone.
  • **Net account** — the wrapped account broker (PaperBroker in dry-run,
    Capital.com live) holds at most ONE position per symbol, tagged
    ``__net__``: the signed sum of the virtual books. ``reconcile(symbol)``
    (called by the agent once per symbol per fast loop) brings the account to
    that target with at most one close + one open.

Consequences worth knowing:
  • Net positions carry NO broker-side stop/take-profit — per-strategy brackets
    are meaningless on a netted book. Exits are agent-managed on the virtual
    books (`_manage_exits`), and the resulting net change flows to the account
    on the next reconcile. Set ``execution.netting: false`` to restore
    one-deal-per-strategy execution with broker-side brackets.
  • Account equity is the netted book's truth; the journal measures strategy
    skill from virtual fills. They can legitimately differ by the spread the
    netting saved.
  • Account positions NOT tagged ``__net__`` (adopted/manual deals) are left
    alone by reconciliation and still surfaced via ``open_positions`` so
    exposure counting and operator alerts see them.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..data.models import Order, Side, Trade
from .broker import Broker, PaperBroker

log = logging.getLogger(__name__)

NET_TAG = "__net__"

# Volumes at or below this are treated as flat (matches the 4-dp rounding used
# by the sizing path).
_EPS = 5e-5

# An order vet takes (order, mark_price) and answers whether the net order may
# be submitted (e.g. live pre-trade compliance).
OrderVet = Callable[[Order, float], bool]


class NettingBroker(Broker):
    """Wraps an account broker; strategies fill virtually, the account nets."""

    def __init__(self, account: Broker, cost=None, order_vet: OrderVet | None = None,
                 catastrophe_stop_mult: float = 1.5,
                 catastrophe_stop_pct: float = 0.05,
                 virtual_books_path: str | None = "data/virtual_books.json"):
        self.account = account
        self.virtual = PaperBroker(cost=cost)
        # Per-strategy virtual books carry the real (tight) brackets the agent
        # manages exits on. Persist them so a restart doesn't lose them: without
        # this, after a restart the virtual books are empty, _manage_exits has
        # nothing to act on, and the live net position rides on only its wide
        # catastrophe stop until strategies happen to re-emit.
        self._vbooks_path = Path(virtual_books_path) if virtual_books_path else None
        self._load_virtual()
        self._order_vet = order_vet
        self._dirty: set[str] = set()
        self._last_price: dict[str, float] = {}
        # Catastrophe stop: every net order carries a WIDE broker-side stop so
        # a dead agent (OOM, host reboot, network partition) cannot ride an
        # open position unbounded — agent-managed virtual exits remain the fine
        # control; this is the dead-man's brake. Distance = the widest virtual
        # stop on the net side × mult; pct-of-price fallback when no virtual
        # position carries a stop. mult<=0 disables (previous behavior).
        self.cat_mult = float(catastrophe_stop_mult)
        self.cat_pct = float(catastrophe_stop_pct)
        # Per-strategy trades force-closed because the account's net position
        # was closed externally (margin close-out, manual close on the
        # platform). Handed to the agent via drain_closed() for grading.
        self._pending_drained: list[Trade] = []
        self._blocked: dict[str, str] = {}

    # ── virtual-book persistence (restart survivability) ─────────────────────
    def _load_virtual(self) -> None:
        if self._vbooks_path is None:
            return
        try:
            if self._vbooks_path.exists():
                for rec in json.loads(self._vbooks_path.read_text()) or []:
                    kw = {}
                    if rec.get("opened_at"):
                        try:
                            kw["opened_at"] = datetime.fromisoformat(rec["opened_at"])
                        except ValueError:
                            pass
                    strat = rec.get("strategy", "")
                    self.virtual._positions[(rec["symbol"], strat)] = Trade(
                        symbol=rec["symbol"], side=Side(rec["side"]),
                        volume=rec["volume"], entry_price=rec.get("entry_price", 0.0),
                        strategy=strat, mode=rec.get("mode", "shadow"),
                        context=rec.get("context", {}), **kw)
                if self.virtual._positions:
                    log.warning("Rehydrated %d virtual strategy book(s) from %s "
                                "across a restart", len(self.virtual._positions),
                                self._vbooks_path)
        except (OSError, ValueError, KeyError) as e:
            log.warning("Virtual-book ledger unreadable (%s); starting empty", e)

    def _save_virtual(self) -> None:
        if self._vbooks_path is None:
            return
        try:
            recs = [
                {"symbol": sym, "strategy": strat, "side": t.side.value,
                 "volume": t.volume, "entry_price": t.entry_price, "mode": t.mode,
                 "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                 "context": t.context or {}}
                for (sym, strat), t in self.virtual._positions.items()
            ]
            self._vbooks_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._vbooks_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(recs))
            tmp.replace(self._vbooks_path)
        except OSError as e:
            log.warning("Virtual-book ledger save failed: %s", e)

    def block_symbol(self, symbol: str, reason: str) -> None:
        self._blocked[symbol] = reason

    def unblock_symbol(self, symbol: str) -> None:
        self._blocked.pop(symbol, None)

    # ── passthroughs / marks ─────────────────────────────────────────────────

    def mark(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = price
        self.virtual.mark(symbol, price)
        self.account.mark(symbol, price)

    async def account_equity(self) -> float:
        return await self.account.account_equity()

    async def balance(self) -> float:
        return await self.account.balance()

    def reset(self) -> None:
        """Dashboard reset: drop virtual books and the paper account state."""
        self.virtual.reset()
        self.account.reset()
        self._dirty.clear()
        self._pending_drained.clear()
        self._save_virtual()   # persist the now-empty books

    # ── the agent's view: per-strategy virtual positions ─────────────────────

    def position(self, symbol: str, strategy: str | None = None) -> Trade | None:
        return self.virtual.position(symbol, strategy)

    def positions_for(self, symbol: str) -> list[Trade]:
        return self.virtual.positions_for(symbol)

    def position_count(self) -> int:
        return self.virtual.position_count()

    async def open_positions(self) -> list[Trade]:
        # Drive the account's own reconciliation machinery (Capital.com adopts
        # unknown deals and detects broker-side closes here), then translate
        # external closes of OUR net position into per-strategy virtual closes
        # so attribution survives.
        account_positions = await self.account.open_positions()
        for closed in self.account.drain_closed():
            if closed.strategy == NET_TAG:
                log.warning(
                    "Net position on %s was closed account-side (%s); force-"
                    "closing its virtual books.", closed.symbol,
                    (closed.context or {}).get("close_reason", "unknown"))
                self._pending_drained.extend(
                    await self._flatten_virtual(closed.symbol, "net_closed_externally"))
                self._dirty.add(closed.symbol)
            else:
                # Adopted/manual deal closed broker-side — not ours to net;
                # pass through for the agent to grade as before.
                self._pending_drained.append(closed)
        # Virtual books are the strategy-facing truth; adopted/manual account
        # deals ride along so exposure counting and operator alerts see them.
        extra = [p for p in account_positions if p.strategy != NET_TAG]
        return list(await self.virtual.open_positions()) + extra

    def drain_closed(self) -> list[Trade]:
        out, self._pending_drained = self._pending_drained, []
        return out

    # ── strategy-facing execution: virtual fills ─────────────────────────────

    async def submit(self, order: Order) -> Trade | None:
        trade = await self.virtual.submit(order)
        if trade is not None:
            self._dirty.add(order.symbol)
            self._save_virtual()
        return trade

    async def close(self, symbol: str, strategy: str | None = None) -> Trade | None:
        closed = await self.virtual.close(symbol, strategy)
        if closed is not None:
            self._dirty.add(symbol)
            self._save_virtual()
            return closed
        # No virtual book matched — an adopted/manual account deal (surfaced by
        # open_positions) can still be closed directly. Never the net position:
        # reconcile owns that.
        for pos in self.account.positions_for(symbol):
            if pos.strategy == NET_TAG:
                continue
            if strategy is None or pos.strategy == strategy:
                return await self.account.close(symbol, pos.strategy)
        return None

    async def _flatten_virtual(self, symbol: str, reason: str) -> list[Trade]:
        out = []
        for pos in list(self.virtual.positions_for(symbol)):
            closed = await self.virtual.close(symbol, pos.strategy)
            if closed is not None:
                closed.context = {**(closed.context or {}), "close_reason": reason}
                out.append(closed)
        if out:
            self._save_virtual()
        return out

    # ── account reconciliation ───────────────────────────────────────────────

    def _net_target(self, symbol: str) -> float:
        """Signed sum of the virtual books on a symbol (BUY +, SELL −)."""
        total = 0.0
        for pos in self.virtual.positions_for(symbol):
            total += pos.volume if pos.side == Side.BUY else -pos.volume
        return total

    def _net_current(self, symbol: str) -> float:
        """Signed volume of the account's net-tagged position(s) on a symbol."""
        total = 0.0
        for pos in self.account.positions_for(symbol):
            if pos.strategy == NET_TAG:
                total += pos.volume if pos.side == Side.BUY else -pos.volume
        return total

    def _catastrophe_stop(self, symbol: str, side: Side) -> float | None:
        """Broker-side dead-man stop for a net order, or None when disabled.

        Wide on purpose: it must never fire before the agent's own virtual-book
        exits under normal operation, only bound the loss when the agent isn't
        running. When it does fire, the account-side close is detected by
        ``open_positions`` and the virtual books are flattened with attribution.
        """
        if self.cat_mult <= 0:
            return None
        mark = self._last_price.get(symbol, 0.0)
        if mark <= 0:
            return None
        # Widest stop distance among virtual positions on the net side.
        widest = 0.0
        want = side
        for pos in self.virtual.positions_for(symbol):
            if pos.side != want:
                continue
            stop = (pos.context or {}).get("stop")
            if stop:
                widest = max(widest, abs(pos.entry_price - float(stop)))
        dist = widest * self.cat_mult if widest > 0 else mark * self.cat_pct
        if dist <= 0:
            return None
        level = mark - dist if side == Side.BUY else mark + dist
        return round(level, 6) if level > 0 else None

    async def reconcile(self, symbol: str | None = None) -> None:
        # Dirty flags are only cleared on a successful reconcile, so the full
        # sweep (symbol=None) retries anything a failed loop left behind.
        symbols = {symbol} if symbol is not None else set(self._dirty)
        for sym in symbols:
            await self._reconcile_symbol(sym)

    async def _reconcile_symbol(self, symbol: str) -> None:
        if symbol in self._blocked:
            log.debug("Skipping net reconcile for %s: %s", symbol, self._blocked[symbol])
            return
        target = round(self._net_target(symbol), 4)
        current = self._net_current(symbol)
        if abs(target - current) <= _EPS:
            self._dirty.discard(symbol)
            return

        # INCREASES in the same direction submit only the DELTA — the common
        # case (strategies piling onto one idea) used to close the whole net
        # deal and reopen it larger, paying the full spread twice on the entire
        # position for a size bump. (The paper account merges same-side deals;
        # Capital.com just holds a second net-tagged deal, which _net_current
        # already sums.) Shrinks, flips and go-flat keep close-then-reopen —
        # partial closes need broker support the ABC doesn't promise.
        if current != 0 and target != 0 and (target > 0) == (current > 0) \
                and abs(target) > abs(current):
            side = Side.BUY if target > 0 else Side.SELL
            delta = round(abs(target) - abs(current), 4)
            if delta <= _EPS:
                self._dirty.discard(symbol)
                return
            order = Order(
                symbol=symbol, side=side, volume=delta,
                stop_loss=self._catastrophe_stop(symbol, side),
                client_id=f"{NET_TAG}:{symbol}:{int(time.time())}",
            )
            if self._order_vet is not None and not self._order_vet(
                    order, self._last_price.get(symbol, 0.0)):
                log.error("Net delta order for %s vetoed; account left at "
                          "current size.", symbol)
                return
            if await self.account.submit(order) is None:
                log.error("Net delta open failed for %s; will retry next loop.",
                          symbol)
                return
            self._dirty.discard(symbol)
            return

        # Close-then-reopen keeps exactly one net deal per symbol and needs
        # nothing beyond the Broker ABC (no partial-close support required).
        # With the default zero-cost paper model it is PnL-neutral; live it
        # trades a little churn for a much smaller order count overall.
        guard = 0
        while self.account.position(symbol, NET_TAG) is not None and guard < 16:
            guard += 1
            if await self.account.close(symbol, NET_TAG) is None:
                # Account refused/failed the close (e.g. transient API error):
                # leave the symbol dirty so the next loop retries.
                log.error("Net close failed for %s; will retry next loop.", symbol)
                return
            # Some account brokers expose every close through drain_closed(),
            # including closes explicitly requested by this reconciler. Drain
            # that event now: it is an internal reshape, not an external
            # broker-side close, and must never flatten the virtual books.
            # Preserve any unrelated event that happened to be queued.
            for closed in self.account.drain_closed():
                if closed.strategy == NET_TAG and closed.symbol == symbol:
                    log.debug("Consumed internal net reshape close for %s", symbol)
                else:
                    self._pending_drained.append(closed)

        if abs(target) > _EPS:
            side = Side.BUY if target > 0 else Side.SELL
            order = Order(
                symbol=symbol,
                side=side,
                volume=abs(target),
                stop_loss=self._catastrophe_stop(symbol, side),
                client_id=f"{NET_TAG}:{symbol}:{int(time.time())}",
            )
            if self._order_vet is not None and not self._order_vet(
                    order, self._last_price.get(symbol, 0.0)):
                # Vetoed reopen: the account stays flat(ter) — conservative —
                # and the symbol stays dirty so a later loop retries once
                # conditions change.
                log.error("Net order for %s vetoed; account left flat, "
                          "virtual books unchanged.", symbol)
                return
            if await self.account.submit(order) is None:
                log.error("Net open failed for %s; will retry next loop.", symbol)
                return
        self._dirty.discard(symbol)
