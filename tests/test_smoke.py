"""Smoke tests: the pipeline wires together and produces sane outputs in dry-run."""

from __future__ import annotations

import asyncio

import numpy as np

from gungnir.config import Config
from gungnir.data.models import OrderBook, OrderBookLevel, Side, Signal
from gungnir.data.news_feed import CompositeNewsFeed, RSSNewsFeed, FinnhubNewsFeed
from gungnir.execution.broker import PaperBroker
from gungnir.features import indicators
from gungnir.features.orderbook import analyze
from gungnir.strategy.examples.mean_reversion import MeanReversion
from gungnir.strategy.examples.trend_following import TrendFollowing
from gungnir.features.feature_store import build
from gungnir.data.market_feed import StubMarketFeed, SyntheticMarketFeed


def test_indicators_basic():
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    assert indicators.ema(series, 3)[-1] > series[0]
    assert 0 <= indicators.rsi(series, 5) <= 100


def test_orderbook_imbalance():
    book = OrderBook(
        symbol="EURUSD",
        bids=[OrderBookLevel(price=1.0999, size=300)],
        asks=[OrderBookLevel(price=1.1001, size=100)],
    )
    feats = analyze(book)
    assert feats is not None
    assert feats.imbalance > 0  # more bid size => positive imbalance


def test_synthetic_feed_is_not_flat():
    """The paper feed must actually move, or strategies never fire (the 'no data'
    bug). Verify it yields a varied, validly-ordered candle series."""
    candles = asyncio.run(SyntheticMarketFeed().recent_candles("XBTUSD", "5m", 120))
    assert len(candles) == 120
    closes = [c.close for c in candles]
    assert len(set(round(c, 6) for c in closes)) > 10        # not a flat line
    for c in candles:                                        # OHLC sanity
        assert c.low <= c.open <= c.high and c.low <= c.close <= c.high


def test_synthetic_feed_advances_with_time():
    """A new candle should form as wall-clock time crosses period boundaries."""
    from unittest import mock
    feed = SyntheticMarketFeed()
    t0 = 1_900_000_000

    def last_close(t):
        with mock.patch("gungnir.data.market_feed.time.time", return_value=t):
            return asyncio.run(feed.recent_candles("XBTUSD", "1m", 30))[-1].close

    prices = [last_close(t0 + k * 60) for k in range(6)]      # six successive minutes
    assert len(set(round(p, 6) for p in prices)) > 1          # price moved over time


def test_strategies_emit_valid_signals():
    candles = asyncio.run(StubMarketFeed(mid=1.10).recent_candles("EURUSD", "M5", 100))
    fs = build("EURUSD", candles)
    for strat in (TrendFollowing(params={"min_conviction": 0.0}), MeanReversion()):
        for sig in strat.generate(fs):
            assert isinstance(sig, Signal)
            assert 0.0 <= sig.conviction <= 1.0
            assert sig.side in (Side.BUY, Side.SELL, Side.FLAT)


def test_paper_broker_roundtrip():
    async def _run():
        broker = PaperBroker(starting_equity=10_000)
        broker.mark("EURUSD", 1.10)
        from gungnir.data.models import Order

        trade = await broker.submit(Order(symbol="EURUSD", side=Side.BUY, volume=1.0))
        assert trade is not None
        broker.mark("EURUSD", 1.11)
        closed = await broker.close("EURUSD")
        assert closed.pnl is not None and closed.pnl > 0

    asyncio.run(_run())


def test_news_feeds():
    """Test RSS and composite feeds (Finnhub requires API key, so we just check it loads)."""
    from gungnir.config import Secrets

    async def _run():
        config = Config(
            {
                "data": {
                    "news": {"rss_feeds": ["https://feeds.bloomberg.com/markets/news.rss"]},
                    "symbols": ["EURUSD"],
                }
            },
            Secrets()
        )
        # Test RSSNewsFeed directly
        rss = RSSNewsFeed(config)
        items = await rss.fetch()
        # RSS may return items or empty list (depends on network/feed availability)
        assert isinstance(items, list)

        # Test CompositeNewsFeed (should include RSS, but Finnhub will be skipped without API key)
        composite = CompositeNewsFeed(config)
        items = await composite.fetch()
        assert isinstance(items, list)

        # Test FinnhubNewsFeed without API key (should return empty)
        finnhub = FinnhubNewsFeed(config)
        items = await finnhub.fetch()
        assert isinstance(items, list)
        # Without API key, should be empty
        assert len(items) == 0

    asyncio.run(_run())
