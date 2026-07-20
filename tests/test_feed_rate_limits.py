"""Rate-limit hardening: candle cache, stale-serving, snapshot sharing, and
session pacing/429 retry.

Guards the fix for the 429 storms: the agent re-fetched every candle series
every fast loop (~190 requests/30s over 21 symbols), Capital.com throttled it,
and each 429 returned [] — starving strategies AND exit management of data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from gungnir.config import Config, Secrets
from gungnir.data.capital_com_feed import CapitalComMarketFeed
from gungnir.execution.capital_session import CapitalComSession


def _config(**market) -> Config:
    return Config({"data": {"market": market}}, Secrets.from_env())


def _prices_payload(ts: datetime, bars: int = 3, step_min: int = 5) -> dict:
    prices = []
    for i in range(bars):
        t = ts + timedelta(minutes=step_min * i)
        prices.append({
            "snapshotTime": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "openPrice": {"bid": 100.0 + i}, "highPrice": {"bid": 101.0 + i},
            "lowPrice": {"bid": 99.0 + i}, "closePrice": {"bid": 100.5 + i},
            "lastTradedVolume": 10,
        })
    return {"prices": prices}


class _FakeSession:
    """Stands in for CapitalComSession: canned JSON per path prefix."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[str] = []
        self.fail = False

    async def get(self, path, **kwargs):
        self.calls.append(path)
        if self.fail:
            raise httpx.HTTPStatusError(
                "429", request=httpx.Request("GET", path),
                response=httpx.Response(429, request=httpx.Request("GET", path)))

        class R:
            def __init__(self, data):
                self._data = data

            def json(self):
                return self._data

        return R(self.payload)


async def test_candles_cached_until_next_bar():
    now = datetime.now(timezone.utc)
    session = _FakeSession(_prices_payload(now - timedelta(minutes=10)))
    feed = CapitalComMarketFeed(session, _config())

    first = await feed.recent_candles("US100", "5m", 3)
    again = await feed.recent_candles("US100", "5m", 3)
    assert len(first) == 3 and again == first
    assert len(session.calls) == 1          # second call served from cache


async def test_expired_cache_refetches_at_full_depth():
    # Newest bar opened two intervals ago → a new bar exists → refetch; and a
    # shallow n=2 request must not hollow out the 3-bar series.
    old = datetime.now(timezone.utc) - timedelta(minutes=20)
    session = _FakeSession(_prices_payload(old))
    feed = CapitalComMarketFeed(session, _config(candle_refetch_seconds=0))

    await feed.recent_candles("US100", "5m", 3)
    two = await feed.recent_candles("US100", "5m", 2)
    assert len(session.calls) == 2          # expired → refetched
    assert len(two) == 2                    # sliced to what was asked
    # The cache itself still holds the full series for deeper callers.
    assert len(feed._candles[("US100", "MINUTE_5")].candles) == 3


async def test_failed_refresh_serves_stale_candles():
    old = datetime.now(timezone.utc) - timedelta(minutes=20)
    session = _FakeSession(_prices_payload(old))
    feed = CapitalComMarketFeed(session, _config(candle_refetch_seconds=0))

    first = await feed.recent_candles("US100", "5m", 3)
    session.fail = True                      # next refresh 429s
    stale = await feed.recent_candles("US100", "5m", 3)
    assert stale == first                    # last-known data, not []


async def test_tick_and_orderbook_share_one_snapshot():
    session = _FakeSession({"snapshot": {"bid": 100.0, "offer": 100.2}})
    feed = CapitalComMarketFeed(session, _config(snapshot_ttl=60))

    book = await feed.orderbook("US100", 10)
    tick = await feed.latest_tick("US100")
    assert book is not None and tick is not None
    assert tick.bid == 100.0 and tick.ask == 100.2
    assert len(session.calls) == 1          # one /markets request served both


def _session_with_transport(handler, **kwargs) -> CapitalComSession:
    s = CapitalComSession("key", "id", "pw", demo=True, min_interval=0.0, **kwargs)
    s._client = httpx.AsyncClient(
        base_url=s.base_url, transport=httpx.MockTransport(handler))
    s._cst, s._security = "cst", "sec"       # skip login
    return s


async def test_session_retries_get_on_429_then_succeeds():
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    s = _session_with_transport(handler)
    r = await s.get("/api/v1/prices/US100")
    assert r.json() == {"ok": True}
    assert hits["n"] == 3                    # two 429s absorbed, third served


async def test_session_never_retries_post_on_429():
    hits = {"n": 0}

    def handler(request):
        hits["n"] += 1
        return httpx.Response(429)

    s = _session_with_transport(handler)
    try:
        await s.post("/api/v1/positions", json={})
        raise AssertionError("expected HTTPStatusError")
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 429
    assert hits["n"] == 1                    # orders are never auto-retried


async def test_market_status_rejects_closed_snapshot_and_weekend():
    session = _FakeSession({"snapshot": {"marketStatus": "CLOSED", "bid": 1.1, "offer": 1.2}})
    feed = CapitalComMarketFeed(session, _config())
    status = await feed.market_status("EURUSD", now=datetime(2026, 7, 19, 20, 25, tzinfo=timezone.utc))
    assert status.tradeable is False
    assert status.reason == "broker_closed"

async def test_market_status_allows_tradeable_crypto():
    session = _FakeSession({"snapshot": {"marketStatus": "TRADEABLE", "bid": 100.0, "offer": 100.1}})
    feed = CapitalComMarketFeed(session, _config())
    status = await feed.market_status("BTCUSD", now=datetime(2026, 7, 19, 20, 25, tzinfo=timezone.utc))
    assert status.tradeable is True
