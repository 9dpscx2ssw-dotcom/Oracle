"""CapitalComBroker order/close/reconciliation tests (audit F-02/F-03/F-05/F-20).

The session is mocked; these pin the broker-side contracts:
  • positions tracked by dealId with the owning strategy;
  • position() answers per (symbol, strategy) so the agent's anti-stacking
    check works on the live broker;
  • close() resolves the closing fill and computes realized PnL;
  • positions that vanish broker-side are finalized and queued in
    drain_closed() instead of silently dropped.
"""

from __future__ import annotations

import asyncio

import pytest

from gungnir.data.models import Order, Side
from gungnir.execution.capital_com import CapitalComBroker


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    """Scriptable stand-in for CapitalComSession."""

    def __init__(self):
        self.posted: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        self.positions_payload = {"positions": []}
        self.confirm_payload = {"dealId": "D1", "dealStatus": "OPEN", "level": 100.0}
        self.deal_ref = "REF1"
        self.activity_payload = {"activities": []}

    async def post(self, path, json=None, **kw):
        self.posted.append((path, json or {}))
        return _Resp({"dealReference": self.deal_ref})

    async def get(self, path, **kw):
        if path.startswith("/api/v1/confirms/"):
            return _Resp(self.confirm_payload)
        if path.startswith("/api/v1/positions"):
            return _Resp(self.positions_payload)
        if path.startswith("/api/v1/history/activity"):
            return _Resp(self.activity_payload)
        return _Resp({})

    async def delete(self, path, **kw):
        self.deleted.append(path)
        return _Resp({"dealReference": "CLOSE-REF"})


def _order(symbol="US500", side=Side.BUY, strategy="stratA", volume=1.0):
    return Order(symbol=symbol, side=side, volume=volume,
                 client_id=f"{strategy}:{symbol}:123")


def test_submit_tracks_by_deal_id_and_strategy():
    s = _FakeSession()
    b = CapitalComBroker(s)
    trade = asyncio.run(b.submit(_order()))
    assert trade is not None
    assert trade.entry_price == 100.0
    assert trade.strategy == "stratA"
    assert b.position("US500", "stratA") is trade
    assert b.position("US500", "other") is None        # per-strategy lookup
    assert b.position_count() == 1


def test_two_strategies_same_symbol_do_not_overwrite():
    s = _FakeSession()
    b = CapitalComBroker(s)
    s.confirm_payload = {"dealId": "D1", "level": 100.0}
    t1 = asyncio.run(b.submit(_order(strategy="stratA")))
    s.deal_ref, s.confirm_payload = "REF2", {"dealId": "D2", "level": 101.0}
    t2 = asyncio.run(b.submit(_order(strategy="stratB", side=Side.SELL)))
    assert t1 is not None and t2 is not None
    assert b.position_count() == 2                     # F-05: no overwrite
    assert b.position("US500", "stratA").entry_price == 100.0
    assert b.position("US500", "stratB").entry_price == 101.0


def test_close_computes_realized_pnl():
    s = _FakeSession()
    b = CapitalComBroker(s)
    asyncio.run(b.submit(_order()))                    # entry 100.0, BUY, vol 1
    s.confirm_payload = {"level": 105.0}               # closing fill
    closed = asyncio.run(b.close("US500", "stratA"))
    assert closed is not None
    assert closed.exit_price == 105.0
    assert closed.pnl == pytest.approx(5.0)            # F-02: round-trip PnL
    assert b.position_count() == 0


def test_rejected_order_returns_none():
    s = _FakeSession()
    b = CapitalComBroker(s)
    s.confirm_payload = {"dealStatus": "REJECTED", "reason": "MARKET_CLOSED"}
    assert asyncio.run(b.submit(_order())) is None
    assert b.position_count() == 0


def test_no_fill_falls_back_to_mark_never_zero():
    s = _FakeSession()
    b = CapitalComBroker(s)
    b.mark("US500", 99.5)
    s.confirm_payload = {"dealId": "D1"}               # no level in confirm
    trade = asyncio.run(b.submit(_order()))
    assert trade is not None
    assert trade.entry_price == 99.5                   # F-05: marked, not 0
    assert trade.context.get("entry_estimated") is True


def test_vanished_position_is_queued_for_grading():
    s = _FakeSession()
    b = CapitalComBroker(s)
    b.mark("US500", 103.0)
    asyncio.run(b.submit(_order()))                    # entry 100.0
    # Broker reports no positions → the deal was closed broker-side (stop/TP).
    s.positions_payload = {"positions": []}
    open_now = asyncio.run(b.open_positions())
    assert open_now == []
    drained = b.drain_closed()
    assert len(drained) == 1                           # F-02: not dropped
    t = drained[0]
    assert t.pnl == pytest.approx(3.0)                 # estimated from mark
    assert t.context.get("close_reason") == "broker-side"
    assert b.drain_closed() == []                      # drained once
