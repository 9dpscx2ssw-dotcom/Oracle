"""Transaction-cost model + cost-aware backtest tests."""

from __future__ import annotations

from gungnir.backtest import engine
from gungnir.backtest.costs import CostModel
from gungnir.data.models import Side
from gungnir.strategy.registry import _REGISTRY


def test_fill_price_crosses_the_spread_adversely():
    c = CostModel(spread_bps=10.0, slippage_bps=0.0)   # 10 bps full spread → 5 bps each side
    mid = 100.0
    # Opening a long lifts the ask (pay up); opening a short hits the bid.
    assert c.fill_price(Side.BUY, mid, opening=True) > mid
    assert c.fill_price(Side.SELL, mid, opening=True) < mid
    # Closing a long sells at the bid; closing a short buys at the ask.
    assert c.fill_price(Side.BUY, mid, opening=False) < mid
    assert c.fill_price(Side.SELL, mid, opening=False) > mid
    assert abs(c.fill_price(Side.BUY, mid, opening=True) - 100.05) < 1e-9


def test_commission_scales_with_notional():
    c = CostModel(commission_bps=2.0)
    assert abs(c.commission(10_000.0) - 2.0) < 1e-9
    assert c.commission(0.0) == 0.0


def test_zero_flag():
    assert CostModel().zero is True
    assert CostModel(spread_bps=1.0).zero is False


def test_costs_reduce_backtest_pnl():
    candles = engine.synthetic_candles("XBTUSD", n=400, seed=42)
    feats = engine.feature_store.build_kraken_series("XBTUSD", candles)
    strat = _REGISTRY["cci_macd"](mode="shadow", symbols=["XBTUSD"])

    free = engine.run(strat, candles, "XBTUSD", sl_pct=1.5, tp_pct=3.0, feats=feats)
    strat2 = _REGISTRY["cci_macd"](mode="shadow", symbols=["XBTUSD"])
    costed = engine.run(strat2, candles, "XBTUSD", sl_pct=1.5, tp_pct=3.0, feats=feats,
                        cost=CostModel(spread_bps=20.0, commission_bps=2.0, slippage_bps=2.0))

    assert free.metrics.n_trades == costed.metrics.n_trades > 0
    # Same trades, but every round-trip now pays spread+commission → strictly less.
    assert costed.metrics.total_pnl < free.metrics.total_pnl


def test_paper_broker_costs_make_flat_roundtrip_negative():
    import asyncio

    from gungnir.data.models import Order, Side
    from gungnir.execution.broker import PaperBroker

    async def run():
        b = PaperBroker(starting_equity=10_000, cost=CostModel(spread_bps=10.0, commission_bps=1.0))
        b.mark("EURUSD", 1.10)
        await b.submit(Order(symbol="EURUSD", side=Side.BUY, volume=1000.0, client_id="s:EURUSD:1"))
        closed = await b.close("EURUSD", "s")     # close at the same mark
        assert closed.pnl < 0                      # paid spread + commission both sides

    asyncio.run(run())
