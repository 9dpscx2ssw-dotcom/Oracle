"""NettingBroker: virtual per-strategy books + one net account position.

Guards the netted-execution layer: a same-loop burst of duplicate signals must
collapse into one account order, opposing signals must cancel instead of
holding two spread-paying positions, and per-strategy attribution (the food for
the journal/allocator/RL) must survive netting untouched.
"""

from gungnir.data.models import Order, Side
from gungnir.execution.broker import PaperBroker
from gungnir.execution.netting import NET_TAG, NettingBroker


def _order(strategy, symbol, side, vol, stop=None, tp=None):
    return Order(symbol=symbol, side=side, volume=vol, stop_loss=stop,
                 take_profit=tp, client_id=f"{strategy}:{symbol}:1")


def _netted(equity=10_000.0):
    account = PaperBroker(starting_equity=equity)
    return NettingBroker(account), account


def _net_positions(account, symbol):
    return [p for p in account.positions_for(symbol) if p.strategy == NET_TAG]


async def test_duplicate_burst_nets_to_one_account_order():
    nb, account = _netted()
    nb.mark("US500", 7565.9)
    for strat in ("a", "b", "c"):
        await nb.submit(_order(strat, "US500", Side.BUY, 2.0))
    await nb.reconcile("US500")

    # Three virtual books, ONE account position of the summed volume.
    assert len(nb.positions_for("US500")) == 3
    nets = _net_positions(account, "US500")
    assert len(nets) == 1
    assert nets[0].side == Side.BUY
    assert abs(nets[0].volume - 6.0) < 1e-9


async def test_opposing_signals_net_flat():
    nb, account = _netted()
    nb.mark("RTY", 2976.25)
    await nb.submit(_order("bull", "RTY", Side.BUY, 3.5))
    await nb.submit(_order("bear", "RTY", Side.SELL, 3.5))
    await nb.reconcile("RTY")

    assert len(nb.positions_for("RTY")) == 2      # both intents tracked
    assert _net_positions(account, "RTY") == []   # account holds nothing


async def test_partial_netting_carries_the_difference():
    nb, account = _netted()
    nb.mark("DE40", 24884.15)
    await nb.submit(_order("bull", "DE40", Side.BUY, 5.0))
    await nb.submit(_order("bear", "DE40", Side.SELL, 2.0))
    await nb.reconcile("DE40")

    nets = _net_positions(account, "DE40")
    assert len(nets) == 1
    assert nets[0].side == Side.BUY
    assert abs(nets[0].volume - 3.0) < 1e-9


async def test_strategy_close_shrinks_the_net():
    nb, account = _netted()
    nb.mark("GOLD", 4030.0)
    await nb.submit(_order("a", "GOLD", Side.BUY, 4.0))
    await nb.submit(_order("b", "GOLD", Side.BUY, 1.5))
    await nb.reconcile("GOLD")
    assert abs(_net_positions(account, "GOLD")[0].volume - 5.5) < 1e-9

    closed = await nb.close("GOLD", "a")
    assert closed is not None and closed.strategy == "a"
    await nb.reconcile("GOLD")
    nets = _net_positions(account, "GOLD")
    assert len(nets) == 1
    assert abs(nets[0].volume - 1.5) < 1e-9


async def test_direction_flip():
    nb, account = _netted()
    nb.mark("BTCUSD", 64000.0)
    await nb.submit(_order("a", "BTCUSD", Side.BUY, 1.0))
    await nb.reconcile("BTCUSD")
    assert _net_positions(account, "BTCUSD")[0].side == Side.BUY

    await nb.close("BTCUSD", "a")
    await nb.submit(_order("b", "BTCUSD", Side.SELL, 2.0))
    await nb.reconcile("BTCUSD")
    nets = _net_positions(account, "BTCUSD")
    assert len(nets) == 1
    assert nets[0].side == Side.SELL
    assert abs(nets[0].volume - 2.0) < 1e-9


async def test_per_strategy_pnl_attribution_survives_netting():
    nb, _ = _netted()
    nb.mark("EURUSD", 1.1000)
    await nb.submit(_order("winner", "EURUSD", Side.BUY, 10_000))
    await nb.submit(_order("loser", "EURUSD", Side.SELL, 10_000))
    await nb.reconcile("EURUSD")

    nb.mark("EURUSD", 1.1010)                      # +10 pips
    win = await nb.close("EURUSD", "winner")
    loss = await nb.close("EURUSD", "loser")
    assert win.pnl > 0 and loss.pnl < 0            # each graded on its own book
    assert abs(win.pnl - 10.0) < 1e-6
    assert abs(loss.pnl + 10.0) < 1e-6


async def test_reconcile_is_idempotent_when_clean():
    nb, account = _netted()
    nb.mark("US100", 29400.0)
    await nb.submit(_order("a", "US100", Side.BUY, 1.0))
    await nb.reconcile("US100")
    first = _net_positions(account, "US100")[0]

    ops = {"n": 0}
    orig = account.submit

    async def counting_submit(order):
        ops["n"] += 1
        return await orig(order)

    account.submit = counting_submit
    await nb.reconcile("US100")                    # nothing changed
    await nb.reconcile()                           # full sweep, still clean
    assert ops["n"] == 0
    assert _net_positions(account, "US100")[0] is first


class _QueueingPaperBroker(PaperBroker):
    """Paper account that exposes explicit closes through drain_closed()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = []

    async def close(self, symbol, strategy=None):
        trade = await super().close(symbol, strategy)
        if trade is not None:
            self._closed.append(trade)
        return trade

    def drain_closed(self):
        out, self._closed = self._closed, []
        return out


async def test_internal_net_reconcile_close_is_not_external_flatten():
    account = _QueueingPaperBroker()
    nb = NettingBroker(account)
    nb.mark("US100", 29400.0)
    await nb.submit(_order("consensus", "US100", Side.BUY, 1.0))
    await nb.reconcile("US100")

    # A target shrink causes NettingBroker to close the account deal internally.
    await nb.close("US100", "consensus")
    await nb.submit(_order("consensus", "US100", Side.BUY, 0.5))
    await nb.reconcile("US100")
    await nb.open_positions()

    # The explicit account close must not be interpreted as an external close
    # and must not flatten the replacement virtual position.
    assert nb.position("US100", "consensus") is not None
    assert abs(nb.position("US100", "consensus").volume - 0.5) < 1e-9
    assert nb.drain_closed() == []


async def test_external_net_close_flattens_virtual_books_with_attribution():
    nb, account = _netted()
    nb.mark("HK50", 25000.0)
    await nb.submit(_order("a", "HK50", Side.BUY, 1.0))
    await nb.submit(_order("b", "HK50", Side.BUY, 2.0))
    await nb.reconcile("HK50")

    # Simulate the account closing the net position outside our control
    # (margin close-out / manual close on the platform).
    net = await account.close("HK50", NET_TAG)
    assert net is not None
    account._closed_externally = [net]            # what drain_closed would return
    account.drain_closed = lambda: account.__dict__.pop("_closed_externally", [])

    await nb.open_positions()                     # detection point
    drained = nb.drain_closed()
    assert sorted(t.strategy for t in drained) == ["a", "b"]
    assert nb.positions_for("HK50") == []
    await nb.reconcile("HK50")                    # target 0, current 0 — stays flat
    assert _net_positions(account, "HK50") == []


async def test_order_vet_blocks_net_open_but_keeps_retrying():
    allow = {"ok": False}
    account = PaperBroker(starting_equity=10_000)
    nb = NettingBroker(account, order_vet=lambda order, price: allow["ok"])
    nb.mark("US30", 52600.0)
    await nb.submit(_order("a", "US30", Side.BUY, 1.0))
    await nb.reconcile("US30")
    assert _net_positions(account, "US30") == []  # vetoed → account stays flat

    allow["ok"] = True
    await nb.reconcile()                          # symbol stayed dirty → retried
    assert len(_net_positions(account, "US30")) == 1


async def test_account_equity_reflects_net_book():
    nb, account = _netted(equity=10_000.0)
    nb.mark("XRPUSD", 1.1000)
    await nb.submit(_order("bull", "XRPUSD", Side.BUY, 1_000))
    await nb.submit(_order("bear", "XRPUSD", Side.SELL, 1_000))
    await nb.reconcile("XRPUSD")

    nb.mark("XRPUSD", 1.2000)                     # a big move either way
    # Net-flat book: account equity is untouched by the offsetting intents.
    assert abs(await nb.account_equity() - 10_000.0) < 1e-6
