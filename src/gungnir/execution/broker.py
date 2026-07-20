"""Broker interface + an in-memory PaperBroker for dry-run/backtest.

The agent only ever talks to the Broker interface, so swapping cTrader for a
paper fill engine (or a different broker) changes nothing upstream.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..data.models import Order, Side, Trade

log = logging.getLogger(__name__)


class Broker(ABC):
    @abstractmethod
    async def account_equity(self) -> float: ...

    @abstractmethod
    async def balance(self) -> float:
        """Realized cash balance (equity minus unrealized PnL)."""

    @abstractmethod
    async def open_positions(self) -> list[Trade]: ...

    @abstractmethod
    async def submit(self, order: Order) -> Trade | None: ...

    @abstractmethod
    async def close(self, symbol: str, strategy: str | None = None) -> Trade | None: ...

    # ── shared optional interface (audit F-00a/F-18: the agent used to reach
    # into implementation internals via isinstance/getattr; these defaults make
    # every broker answer the same questions safely) ─────────────────────────

    def position(self, symbol: str, strategy: str | None = None) -> Trade | None:
        """The tracked open position for a symbol (optionally per strategy)."""
        return None

    def positions_for(self, symbol: str) -> list[Trade]:
        """Every tracked open position on a symbol, across strategies."""
        return []

    def position_count(self) -> int:
        """Number of tracked open positions."""
        return 0

    def mark(self, symbol: str, price: float) -> None:
        """Feed the latest price; used for paper fills and live PnL estimates."""

    def drain_closed(self) -> list[Trade]:
        """Positions that closed outside the agent's control (e.g. broker-side
        stop/TP) since the last call. The agent grades and journals these."""
        return []

    async def reconcile(self, symbol: str | None = None) -> None:
        """Flush any deferred execution state for a symbol (all symbols when
        None). A no-op everywhere except the NettingBroker, where it nets the
        per-strategy virtual books into one account position per symbol."""

    def reset(self) -> None:
        """Drop tracked paper state (dashboard reset); no-op for real brokers."""


class PaperBroker(Broker):
    """Fills instantly at the order's implied price. For `--dry-run` and tests.

    Positions are keyed by ``(symbol, strategy)`` so each strategy manages its own
    paper position independently. The shadow broker is shared by every strategy;
    keying on symbol alone made them fight over one slot per symbol — strategy B's
    opposite signal would close strategy A's position (at the same tick price, so
    ``pnl≈0``) and open its own, every tick, churning the journal with empty
    round-trips. Per-strategy keys keep each strategy's shadow track clean.

    The public API stays symbol-based; pass ``strategy`` to target a specific
    strategy's position, or omit it to act on the first match for a symbol.
    """

    def __init__(self, starting_equity: float = 10_000.0, cost=None):
        from ..backtest.costs import CostModel
        self.starting_equity = starting_equity
        self._equity = starting_equity
        self._positions: dict[tuple[str, str], Trade] = {}  # (symbol, strategy) -> Trade
        self._last_price: dict[str, float] = {}
        # Transaction costs so paper/shadow P&L matches what live would actually
        # net. Defaults to zero (mid-to-mid) to preserve existing behavior/tests.
        self.cost = cost or CostModel()

    def reset(self) -> None:
        """Drop all open positions and restore starting equity (dashboard reset)."""
        self._positions.clear()
        self._equity = self.starting_equity

    @staticmethod
    def _strategy_of(order: Order) -> str:
        return order.client_id.split(":")[0] if order.client_id else ""

    def mark(self, symbol: str, price: float) -> None:
        """Feed the latest price so PnL marks and exits are realistic."""
        self._last_price[symbol] = price
        # Track each open position's excursion extremes: MFE (max favorable,
        # ≥0) and MAE (max adverse, ≤0), in price units along the trade's
        # direction. Closed trades carry them in context, so bracket placement
        # can be judged against what the trade actually did while open.
        for (s, _), pos in self._positions.items():
            if s != symbol:
                continue
            direction = 1 if pos.side == Side.BUY else -1
            exc = direction * (price - pos.entry_price)
            ctx = pos.context or {}
            if exc > ctx.get("mfe", 0.0):
                ctx["mfe"] = round(exc, 6)
            if exc < ctx.get("mae", 0.0):
                ctx["mae"] = round(exc, 6)
            pos.context = ctx

    async def balance(self) -> float:
        return self._equity

    async def account_equity(self) -> float:
        unrealized = 0.0
        for (sym, _strat), pos in self._positions.items():
            px = self._last_price.get(sym, pos.entry_price)
            direction = 1 if pos.side == Side.BUY else -1
            unrealized += direction * (px - pos.entry_price) * pos.volume
        return self._equity + unrealized

    async def open_positions(self) -> list[Trade]:
        return list(self._positions.values())

    def position(self, symbol: str, strategy: str | None = None) -> Trade | None:
        if strategy is not None:
            return self._positions.get((symbol, strategy))
        # No strategy given: return the first position open on this symbol (compat).
        return next((p for (s, _), p in self._positions.items() if s == symbol), None)

    def positions_for(self, symbol: str) -> list[Trade]:
        """Every open position on this symbol, across strategies."""
        return [p for (s, _), p in self._positions.items() if s == symbol]

    def position_count(self) -> int:
        return len(self._positions)

    async def submit(self, order: Order) -> Trade | None:
        price = self._last_price.get(order.symbol)
        if price is None:
            log.warning("No mark price for %s; cannot paper-fill.", order.symbol)
            return None
        strategy = self._strategy_of(order)
        fill = self.cost.fill_price(order.side, price, opening=True)
        existing = self._positions.get((order.symbol, strategy))
        if existing is not None:
            # Same key twice used to silently OVERWRITE the old position —
            # leaking its PnL. Same side merges (volume-weighted entry, which
            # is what delta net orders need); opposite side closes the old
            # position first so nothing vanishes unaccounted.
            if existing.side == order.side:
                total = existing.volume + order.volume
                existing.entry_price = (
                    existing.entry_price * existing.volume + fill * order.volume
                ) / total
                existing.volume = total
                ctx = existing.context or {}
                if order.stop_loss is not None:
                    ctx["stop"] = order.stop_loss
                if order.take_profit is not None:
                    ctx["tp"] = order.take_profit
                existing.context = ctx
                log.info("[PAPER] merged %s %s +%.4f → vol=%.4f @ %.5f",
                         order.side.value, order.symbol, order.volume, total, fill)
                return existing
            log.warning("[PAPER] opposite-side submit on open %s/%s position; "
                        "closing it first.", order.symbol, strategy or "?")
            await self.close(order.symbol, strategy)
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            volume=order.volume,
            entry_price=fill,
            strategy=strategy,
            # Keep the bracket on the position so exits can be evaluated.
            context={"stop": order.stop_loss, "tp": order.take_profit},
        )
        self._positions[(order.symbol, strategy)] = trade
        log.info("[PAPER] opened %s %s vol=%.4f @ %.5f", order.side.value, order.symbol,
                 order.volume, price)
        return trade

    async def close(self, symbol: str, strategy: str | None = None) -> Trade | None:
        if strategy is not None:
            key = (symbol, strategy)
        else:
            key = next((k for k in self._positions if k[0] == symbol), None)
        if key is None:
            return None
        pos = self._positions.pop(key, None)
        if pos is None:
            return None
        mid = self._last_price.get(symbol, pos.entry_price)
        exit_fill = self.cost.fill_price(pos.side, mid, opening=False)   # adverse exit
        direction = 1 if pos.side == Side.BUY else -1
        commission = (self.cost.commission(pos.entry_price * pos.volume)
                      + self.cost.commission(exit_fill * pos.volume))
        pos.exit_price = exit_fill
        # PnL lands in the QUOTE currency (JPY for USDJPY) — convert to the
        # account currency before it touches equity, rewards, or the journal.
        from .fx import to_account_ccy
        raw_pnl = direction * (exit_fill - pos.entry_price) * pos.volume - commission
        pos.pnl = to_account_ccy(symbol, raw_pnl, self._last_price.get)
        self._equity += pos.pnl
        # Normalize the excursion extremes to R (one R = entry↔stop distance)
        # so MAE/MFE are comparable across symbols and position sizes.
        ctx = pos.context or {}
        stop = ctx.get("stop")
        risk_per_unit = abs(pos.entry_price - stop) if stop else 0.0
        if risk_per_unit > 0:
            if "mfe" in ctx:
                ctx["mfe_r"] = round(ctx["mfe"] / risk_per_unit, 2)
            if "mae" in ctx:
                ctx["mae_r"] = round(ctx["mae"] / risk_per_unit, 2)
            pos.context = ctx

        # Diagnostic: warn if trade opened and closed at same price (suggests tight TP/SL or manual close)
        if abs(pos.exit_price - pos.entry_price) < 1e-6:
            log.warning("[PAPER] %s closed with zero PnL: entry=%.5f, exit=%.5f, marked=%.5f, commission=%.2f",
                        symbol, pos.entry_price, pos.exit_price, mid, commission)
        else:
            log.info("[PAPER] closed %s: entry=%.5f, exit=%.5f, pnl=%.2f", symbol, pos.entry_price, pos.exit_price, pos.pnl)
        return pos
