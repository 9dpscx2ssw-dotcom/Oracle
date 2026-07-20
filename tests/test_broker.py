"""PaperBroker position-keying tests.

Guards the fix for the shadow-broker churn: strategies sharing one broker used to
fight over a single position slot per symbol, opening and closing each other's
trades every tick (pnl≈0). Positions are now keyed per (symbol, strategy).
"""

import asyncio

from gungnir.data.models import Order, Side
from gungnir.execution.broker import PaperBroker


def _order(strategy, symbol, side, vol=0.01):
    return Order(symbol=symbol, side=side, volume=vol, client_id=f"{strategy}:{symbol}:1")


def test_strategies_do_not_flip_each_others_positions():
    async def run():
        b = PaperBroker(starting_equity=10_000)
        b.mark("US100", 29_400.0)
        await b.submit(_order("cci_macd", "US100", Side.SELL))
        await b.submit(_order("adx_momentum_ema", "US100", Side.BUY))
        # Opposite signals on the same symbol coexist — one per strategy, no churn.
        assert len(await b.open_positions()) == 2
        assert b.position("US100", "cci_macd").side == Side.SELL
        assert b.position("US100", "adx_momentum_ema").side == Side.BUY
        assert len(b.positions_for("US100")) == 2

    asyncio.run(run())


def test_same_strategy_replaces_its_own_position():
    async def run():
        b = PaperBroker(starting_equity=10_000)
        b.mark("US100", 29_400.0)
        await b.submit(_order("cci_macd", "US100", Side.SELL))
        await b.submit(_order("cci_macd", "US100", Side.BUY))  # same key → replaced
        assert len(await b.open_positions()) == 1
        assert b.position("US100", "cci_macd").side == Side.BUY

    asyncio.run(run())


def test_symbol_only_api_is_backward_compatible():
    async def run():
        b = PaperBroker(starting_equity=10_000)
        b.mark("EURUSD", 1.10)
        await b.submit(Order(symbol="EURUSD", side=Side.BUY, volume=1.0))  # no client_id
        assert b.position("EURUSD") is not None
        closed = await b.close("EURUSD")
        assert closed is not None and closed.symbol == "EURUSD"
        assert await b.open_positions() == []

    asyncio.run(run())


def test_close_targets_one_strategy():
    async def run():
        b = PaperBroker(starting_equity=10_000)
        b.mark("US100", 29_400.0)
        await b.submit(_order("a", "US100", Side.BUY))
        await b.submit(_order("b", "US100", Side.SELL))
        await b.close("US100", "a")
        remaining = await b.open_positions()
        assert len(remaining) == 1 and remaining[0].strategy == "b"

    asyncio.run(run())
