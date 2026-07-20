"""Capital.com REST broker: account, positions, market orders.

Shares a :class:`CapitalComSession` with the market feed. Order placement is
two-step on Capital.com: ``POST /api/v1/positions`` returns a ``dealReference``,
then ``GET /api/v1/confirms/{dealReference}`` resolves it into a ``dealId`` and
the actual fill level. Closing is ``DELETE /api/v1/positions/{dealId}``.

Position tracking (audit F-02/F-03/F-05):
  • positions are keyed by **dealId**, with the owning strategy carried on the
    Trade — two strategies on one symbol no longer overwrite each other;
  • ``close()`` resolves the closing deal's fill level and computes realized
    PnL, so round-trips can be journaled and graded;
  • positions that vanish broker-side (stop/TP hit) are turned into completed
    Trades and queued for ``drain_closed()`` instead of being silently dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ..data.models import Order, Side, Trade
from .broker import Broker
from .capital_session import CapitalComSession

log = logging.getLogger(__name__)

# How long an unconfirmed order intent stays eligible for adoption-matching.
_INTENT_TTL = 300.0
# Volume tolerance when matching an adopted deal to a pending intent.
_INTENT_VOL_TOL = 0.01


class CapitalComBroker(Broker):
    def __init__(self, session: CapitalComSession,
                 positions_path: str = "data/live_positions.json"):
        self.session = session
        # dealId → our Trade. Durably persisted (below): a restart used to start
        # with an empty map, so pre-existing net deals were re-ADOPTED every
        # loop with a blank strategy tag — _net_current then read the live net
        # as flat and the agent stopped managing their exits (only the
        # catastrophe stop remained). The ledger rehydrates them with their real
        # strategy/__net__ tag so netting and agent-managed exits survive.
        self._positions_path = Path(positions_path)
        self._positions: dict[str, Trade] = self._load_positions()
        self._last_price: dict[str, float] = {}
        self._closed_externally: list[Trade] = []   # drained by the agent
        # Last successfully fetched account numbers. Served when a fetch fails:
        # returning 0.0 there told the risk layer the account was wiped out,
        # tripping the drawdown breakers on a transient API error.
        self._last_equity: float = 0.0
        self._last_balance: float = 0.0
        # Idempotency ledger: order intents recorded (and persisted) BEFORE the
        # POST leaves. If the POST times out after the broker accepted it, the
        # deal shows up as "unknown" on the next positions poll — matching it
        # back to a recent intent restores its strategy tag (critically,
        # __net__ tags stay nettable) instead of leaving duplicate unmanaged
        # exposure alongside the retried order.
        self._intents_path = Path("data/pending_orders.json")
        self._pending_intents: list[dict] = self._load_intents()

    async def connect(self) -> None:
        await self.session.connect()

    # ── account ────────────────────────────────────────────────────────────────
    async def _account(self) -> dict:
        res = await self.session.get("/api/v1/accounts")
        accounts = res.json().get("accounts", [])
        if not accounts:
            return {}
        # Prefer the account flagged 'preferred', else the first.
        return next((a for a in accounts if a.get("preferred")), accounts[0])

    async def account_equity(self) -> float:
        try:
            bal = (await self._account()).get("balance", {})
            self._last_equity = float(bal.get("balance", 0)) + float(bal.get("profitLoss", 0))
        except Exception as e:  # noqa: BLE001 — never break the loop on a data error
            log.error("Failed to fetch account equity: %s (serving last-known %.2f)",
                      e, self._last_equity)
        return self._last_equity

    async def balance(self) -> float:
        try:
            self._last_balance = float(
                (await self._account()).get("balance", {}).get("balance", 0))
        except Exception as e:  # noqa: BLE001
            log.error("Failed to fetch balance: %s (serving last-known %.2f)",
                      e, self._last_balance)
        return self._last_balance

    # ── position tracking ──────────────────────────────────────────────────────
    def position(self, symbol: str, strategy: str | None = None) -> Trade | None:
        for t in self._positions.values():
            if t.symbol != symbol:
                continue
            if strategy is None or t.strategy == strategy:
                return t
        return None

    def positions_for(self, symbol: str) -> list[Trade]:
        return [t for t in self._positions.values() if t.symbol == symbol]

    def position_count(self) -> int:
        return len(self._positions)

    # ── durable position ledger (restart survivability) ──────────────────────
    @staticmethod
    def _serialize_trade(t: Trade) -> dict:
        return {
            "symbol": t.symbol, "side": t.side.value, "volume": t.volume,
            "entry_price": t.entry_price, "strategy": t.strategy, "mode": t.mode,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "context": t.context or {},
        }

    def _load_positions(self) -> dict[str, Trade]:
        try:
            if self._positions_path.exists():
                data = json.loads(self._positions_path.read_text())
                out: dict[str, Trade] = {}
                for deal_id, rec in (data or {}).items():
                    kw = {}
                    if rec.get("opened_at"):
                        try:
                            kw["opened_at"] = datetime.fromisoformat(rec["opened_at"])
                        except ValueError:
                            pass
                    out[deal_id] = Trade(
                        symbol=rec["symbol"], side=Side(rec["side"]),
                        volume=rec["volume"], entry_price=rec.get("entry_price", 0.0),
                        strategy=rec.get("strategy", ""), mode=rec.get("mode", "real"),
                        context=rec.get("context", {}), **kw)
                if out:
                    log.warning("Rehydrated %d live position(s) from %s across a "
                                "restart", len(out), self._positions_path)
                return out
        except (OSError, ValueError, KeyError) as e:
            log.warning("Live-position ledger unreadable (%s); starting empty", e)
        return {}

    def _save_positions(self) -> None:
        try:
            self._positions_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._positions_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {d: self._serialize_trade(t) for d, t in self._positions.items()}))
            tmp.replace(self._positions_path)
        except OSError as e:
            log.warning("Live-position ledger save failed: %s", e)

    def drain_closed(self) -> list[Trade]:
        out, self._closed_externally = self._closed_externally, []
        return out

    def _finalize(self, trade: Trade, exit_price: float | None) -> Trade:
        """Stamp exit price / realized PnL onto a closing trade (best-effort)."""
        from .fx import to_account_ccy
        px = exit_price or self._last_price.get(trade.symbol)
        if px and trade.entry_price:
            direction = 1 if trade.side == Side.BUY else -1
            trade.exit_price = float(px)
            # Quote-currency PnL → account currency (JPY pnl on USDJPY was
            # flowing into USD equity/rewards ~100× over-scaled).
            raw = direction * (trade.exit_price - trade.entry_price) * trade.volume
            trade.pnl = to_account_ccy(trade.symbol, raw, self._last_price.get)
            if not exit_price:
                trade.context = {**(trade.context or {}), "exit_estimated": True}
        trade.closed_at = datetime.now(timezone.utc)
        return trade

    async def open_positions(self) -> list[Trade]:
        try:
            res = await self.session.get("/api/v1/positions")
            rows = res.json().get("positions", [])
            fetched: dict[str, dict] = {}
            for row in rows:
                pos, market = row.get("position", {}), row.get("market", {})
                deal_id = pos.get("dealId")
                if deal_id:
                    fetched[deal_id] = {"pos": pos, "symbol": market.get("epic", "")}

            # Update tracked positions from broker truth; adopt unknown deals
            # (opened outside the agent, or before a restart) so they can be
            # marked, counted against exposure, and manually closed. An unknown
            # deal is first matched against the pending-intent ledger: a POST
            # that timed out after acceptance is OUR order — restore its
            # strategy tag (a recovered __net__ deal stays nettable) instead of
            # adopting it as permanent unmanaged exposure.
            changed = False
            for deal_id, row in fetched.items():
                pos, symbol = row["pos"], row["symbol"]
                known = self._positions.get(deal_id)
                if known:
                    known.volume = float(pos.get("size", known.volume))
                    if pos.get("level"):
                        known.entry_price = float(pos["level"])
                    continue
                direction = pos.get("direction", "BUY")
                size = float(pos.get("size", 0))
                intent = self._match_intent(symbol, direction, size)
                if intent:
                    log.warning("Recovered orphan deal %s on %s from the pending-"
                                "order ledger (strategy=%s) — the submit response "
                                "was lost but the order filled.",
                                deal_id, symbol, intent["strategy"] or "?")
                ctx = {"deal_id": deal_id}
                if intent:
                    ctx["recovered"] = True
                    ctx["client_id"] = intent.get("client_id")
                else:
                    ctx["adopted"] = True
                self._positions[deal_id] = Trade(
                    symbol=symbol,
                    side=Side.BUY if direction == "BUY" else Side.SELL,
                    volume=size,
                    entry_price=float(pos.get("level", 0)),
                    strategy=(intent or {}).get("strategy", ""),
                    context=ctx,
                )
                changed = True

            # Tracked deals missing from the broker were closed broker-side
            # (stop/TP) — finalize them and queue for the agent to grade,
            # instead of silently dropping them (audit F-02).
            for deal_id in list(self._positions.keys()):
                if deal_id not in fetched:
                    trade = self._positions.pop(deal_id)
                    exit_px = await self._closing_level(deal_id)
                    trade.context = {**(trade.context or {}), "close_reason": "broker-side"}
                    self._closed_externally.append(self._finalize(trade, exit_px))
                    log.info("Position %s (deal %s) closed broker-side; queued for grading",
                             trade.symbol, deal_id)
                    changed = True

            if changed:
                self._save_positions()
            return list(self._positions.values())
        except Exception as e:  # noqa: BLE001
            log.error("Failed to fetch positions: %s", e)
            return list(self._positions.values())

    async def _closing_level(self, deal_id: str) -> float | None:
        """Best-effort lookup of the closing fill level from account history."""
        try:
            res = await self.session.get(
                "/api/v1/history/activity",
                params={"lastPeriod": 3600, "detailed": "true"},
            )
            for item in res.json().get("activities", []):
                detail = item.get("details") or {}
                if item.get("dealId") == deal_id or detail.get("dealId") == deal_id:
                    level = item.get("level") or detail.get("level")
                    if level:
                        return float(level)
        except Exception as e:  # noqa: BLE001 — estimate from mark instead
            log.debug("Closing-level lookup failed for %s: %s", deal_id, e)
        return None

    # ── idempotency ledger ─────────────────────────────────────────────────────

    def _load_intents(self) -> list[dict]:
        try:
            if self._intents_path.exists():
                data = json.loads(self._intents_path.read_text())
                if isinstance(data, list):
                    return data
        except (OSError, ValueError) as e:
            log.warning("Pending-order ledger unreadable (%s); starting empty", e)
        return []

    def _save_intents(self) -> None:
        try:
            self._intents_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._intents_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._pending_intents))
            tmp.replace(self._intents_path)
        except OSError as e:
            log.warning("Pending-order ledger save failed: %s", e)

    def _record_intent(self, order: Order) -> dict:
        intent = {
            "client_id": order.client_id,
            "symbol": order.symbol,
            "direction": "BUY" if order.side == Side.BUY else "SELL",
            "size": order.volume,
            "strategy": self._strategy_of(order),
            "ts": time.time(),
        }
        self._pending_intents.append(intent)
        self._save_intents()
        return intent

    def _clear_intent(self, intent: dict) -> None:
        self._pending_intents = [i for i in self._pending_intents if i is not intent
                                 and i.get("client_id") != intent.get("client_id")]
        self._save_intents()

    def _match_intent(self, symbol: str, direction: str, size: float) -> dict | None:
        """Find (and consume) a recent pending intent matching an adopted deal."""
        now = time.time()
        self._pending_intents = [
            i for i in self._pending_intents if now - i.get("ts", 0) < _INTENT_TTL
        ]
        for intent in self._pending_intents:
            if (intent["symbol"] == symbol and intent["direction"] == direction
                    and abs(intent["size"] - size) <= _INTENT_VOL_TOL * max(size, 1e-9)):
                self._clear_intent(intent)
                return intent
        self._save_intents()
        return None

    # ── orders ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _strategy_of(order: Order) -> str:
        return order.client_id.split(":")[0] if order.client_id else ""

    async def submit(self, order: Order) -> Trade | None:
        payload = {
            "epic": order.symbol,
            "direction": "BUY" if order.side == Side.BUY else "SELL",
            "size": order.volume,
        }
        if order.stop_loss:
            payload["stopLevel"] = order.stop_loss
        if order.take_profit:
            payload["profitLevel"] = order.take_profit
        # Ledger entry BEFORE the wire: if the POST times out after the broker
        # accepted it, the next positions poll matches the orphan deal back to
        # this intent instead of adopting it as unmanaged exposure.
        intent = self._record_intent(order)
        try:
            res = await self.session.post("/api/v1/positions", json=payload)
            deal_ref = res.json().get("dealReference")
            if not deal_ref:
                log.warning("No dealReference returned for %s", order.client_id)
                self._clear_intent(intent)
                return None

            # Resolve the reference into a confirmed deal (fill level + dealId).
            # Retry up to 3 times; the broker may need time to process.
            deal_id, fill, confirmed = deal_ref, None, False
            for attempt in range(3):
                try:
                    conf = (await self.session.get(f"/api/v1/confirms/{deal_ref}")).json()
                    if conf.get("dealStatus") == "REJECTED":
                        log.warning("Order rejected for %s: %s", order.symbol, conf.get("reason"))
                        self._clear_intent(intent)
                        return None
                    deal_id = conf.get("dealId", deal_ref)
                    if conf.get("level"):
                        fill = float(conf["level"])
                        confirmed = True
                        break
                except Exception as e:  # noqa: BLE001
                    if attempt < 2:
                        await asyncio.sleep(0.5)
                    else:
                        log.warning("Confirm lookup failed after 3 attempts for %s: %s "
                                    "(trade submitted but not confirmed)", order.symbol, e)

            # Never record entry_price=0 (audit F-05): fall back to the latest
            # mark and flag the estimate; downstream PnL math guards on it.
            estimated = False
            if fill is None:
                fill = self._last_price.get(order.symbol)
                estimated = fill is not None
            if fill is None:
                log.error("No fill or mark price for %s; tracking deal %s without entry "
                          "(PnL unavailable until reconciled)", order.symbol, deal_id)

            # Execution quality: fill vs the arrival mark, in bps, signed so
            # positive = adverse (paid more buying / received less selling).
            # Journaled on the trade and exported as a Prometheus histogram —
            # the number that says whether live fills match backtest costs.
            slippage_bps = None
            arrival = self._last_price.get(order.symbol)
            if fill and arrival and not estimated:
                direction = 1 if order.side == Side.BUY else -1
                slippage_bps = round(
                    direction * (float(fill) - arrival) / arrival * 10_000.0, 3)
                from ..core import metrics
                metrics.observe_slippage(slippage_bps)

            trade = Trade(
                symbol=order.symbol, side=order.side, volume=order.volume,
                entry_price=float(fill) if fill else 0.0,
                strategy=self._strategy_of(order),
                context={"stop": order.stop_loss, "tp": order.take_profit,
                         "deal_id": deal_id,
                         **({"slippage_bps": slippage_bps}
                            if slippage_bps is not None else {}),
                         **({"entry_estimated": True} if estimated or not fill else {})},
            )
            self._positions[deal_id] = trade
            self._save_positions()
            self._clear_intent(intent)
            status = "confirmed" if confirmed else "submitted (unconfirmed)"
            log.info("Order %s: %s %s vol=%.4f @ %.5f (deal %s)", status, order.side.value,
                     order.symbol, order.volume, trade.entry_price or 0, deal_id)
            return trade
        except Exception as e:  # noqa: BLE001
            # Intent stays in the ledger: the broker may have accepted the order
            # even though we never saw the response (timeout, dropped socket).
            # The next open_positions() poll adoption-matches it back.
            log.error("Failed to submit order for %s: %s (intent retained for "
                      "adoption-matching)", order.symbol, e)
            return None

    async def close(self, symbol: str, strategy: str | None = None) -> Trade | None:
        pos = self.position(symbol, strategy)
        deal_id = (pos.context or {}).get("deal_id") if pos else None
        if not pos or not deal_id:
            log.warning("No tracked deal to close for %s (strategy=%s)", symbol, strategy)
            return None
        try:
            res = await self.session.delete(f"/api/v1/positions/{deal_id}")
            self._positions.pop(deal_id, None)
            self._save_positions()
            # Resolve the closing fill so the round-trip carries realized PnL.
            exit_px = None
            close_ref = None
            try:
                close_ref = res.json().get("dealReference")
            except Exception:  # noqa: BLE001 — some responses have no body
                pass
            if close_ref:
                try:
                    conf = (await self.session.get(f"/api/v1/confirms/{close_ref}")).json()
                    if conf.get("level"):
                        exit_px = float(conf["level"])
                except Exception as e:  # noqa: BLE001
                    log.debug("Close confirm lookup failed for %s: %s", symbol, e)
            trade = self._finalize(pos, exit_px)
            log.info("Closed position %s (deal %s) exit=%.5f pnl=%s", symbol, deal_id,
                     trade.exit_price or 0.0,
                     f"{trade.pnl:.2f}" if trade.pnl is not None else "n/a")
            return trade
        except Exception as e:  # noqa: BLE001
            log.error("Failed to close position %s: %s", symbol, e)
            return None

    def mark(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = price

    async def aclose(self) -> None:
        await self.session.aclose()
