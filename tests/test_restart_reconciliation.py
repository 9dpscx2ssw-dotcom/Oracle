"""Phase-1.1: durable position ledger + restart rehydration.

Before this fix, a process restart started CapitalComBroker with an empty
position map and NettingBroker with empty virtual books. Pre-existing live net
deals were then re-adopted every loop with a blank strategy tag (so the netting
layer read the live net as flat), and the agent stopped managing their exits.

These tests pin the restart contract:
  • a live deal tracked before a restart is rehydrated with its strategy tag
    (so a __net__ deal stays nettable) and is NOT re-adopted;
  • the netting virtual books survive a restart so agent-managed exits still
    have positions to act on.
"""

from __future__ import annotations

import asyncio

from gungnir.data.models import Order, Side
from gungnir.execution.broker import PaperBroker
from gungnir.execution.capital_com import CapitalComBroker
from gungnir.execution.netting import NET_TAG, NettingBroker


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.positions_payload = {"positions": []}
        self.confirm_payload = {"dealId": "D1", "level": 100.0}
        self.deal_ref = "REF1"

    async def post(self, path, json=None, **kw):
        return _Resp({"dealReference": self.deal_ref})

    async def get(self, path, **kw):
        if path.startswith("/api/v1/confirms/"):
            return _Resp(self.confirm_payload)
        if path.startswith("/api/v1/positions"):
            return _Resp(self.positions_payload)
        return _Resp({"activities": []})

    async def delete(self, path, **kw):
        return _Resp({"dealReference": "CLOSE-REF"})


def _order(symbol="US500", side=Side.BUY, strategy=NET_TAG, volume=1.0):
    return Order(symbol=symbol, side=side, volume=volume,
                 client_id=f"{strategy}:{symbol}:123")


def test_live_position_survives_restart_and_is_not_readopted(tmp_path):
    path = str(tmp_path / "live_positions.json")
    s = _FakeSession()
    b1 = CapitalComBroker(s, positions_path=path)
    trade = asyncio.run(b1.submit(_order(strategy=NET_TAG)))
    assert trade is not None
    deal_id = (trade.context or {}).get("deal_id")

    # Simulate a restart: brand-new broker instance, same ledger file. The
    # broker still reports the deal as open.
    b2 = CapitalComBroker(s, positions_path=path)
    assert b2.position_count() == 1                         # rehydrated
    recovered = next(iter(b2._positions.values()))
    assert recovered.strategy == NET_TAG                    # net tag preserved
    assert recovered.symbol == "US500"

    # The broker poll reports the same deal; it must be recognised as KNOWN,
    # not re-adopted (no spurious "adopted"/blank-strategy exposure).
    s.positions_payload = {"positions": [
        {"position": {"dealId": deal_id, "direction": "BUY", "size": 1.0,
                      "level": 100.0}, "market": {"epic": "US500"}}]}
    positions = asyncio.run(b2.open_positions())
    assert len(positions) == 1
    p = positions[0]
    assert p.strategy == NET_TAG
    assert not (p.context or {}).get("adopted")             # not re-adopted


def test_net_current_sees_rehydrated_deal_as_net(tmp_path):
    # The whole point: after restart, the netting layer must read its live net
    # from the rehydrated deal, not treat it as flat.
    path = str(tmp_path / "live_positions.json")
    s = _FakeSession()
    acct1 = CapitalComBroker(s, positions_path=path)
    asyncio.run(acct1.submit(_order(strategy=NET_TAG, side=Side.BUY, volume=2.0)))

    acct2 = CapitalComBroker(s, positions_path=path)   # "restart"
    nb = NettingBroker(acct2, catastrophe_stop_mult=0,
                       virtual_books_path=str(tmp_path / "vbooks.json"))
    assert nb._net_current("US500") == 2.0             # +2 long, not 0


def test_virtual_books_survive_restart(tmp_path):
    vpath = str(tmp_path / "vbooks.json")
    acct = PaperBroker()
    acct.mark("US500", 100.0)
    nb1 = NettingBroker(acct, catastrophe_stop_mult=0, virtual_books_path=vpath)
    nb1.mark("US500", 100.0)
    # Strategy opens a virtual position carrying its bracket.
    order = Order(symbol="US500", side=Side.BUY, volume=1.0, stop_loss=95.0,
                  take_profit=110.0, client_id="stratA:US500:1")
    asyncio.run(nb1.submit(order))
    assert nb1.position("US500", "stratA") is not None

    # Restart: fresh NettingBroker + fresh paper account, same virtual ledger.
    nb2 = NettingBroker(PaperBroker(), catastrophe_stop_mult=0,
                        virtual_books_path=vpath)
    pos = nb2.position("US500", "stratA")
    assert pos is not None                              # rehydrated
    assert pos.context.get("stop") == 95.0             # bracket preserved
    assert pos.context.get("tp") == 110.0
    assert nb2._net_target("US500") == 1.0
