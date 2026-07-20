"""Capital.com REST market-data feed (candles, ticks, top-of-book).

Uses a shared :class:`CapitalComSession` for authentication. The important API
details (which the previous version got wrong, hence the 400s):

  * resolution is an enum — MINUTE, MINUTE_5, …, HOUR_4, DAY — not a number;
  * ``from``/``to`` are ``YYYY-MM-DDTHH:MM:SS`` strings, not unix timestamps;
  * each price point nests bid/ask under openPrice/highPrice/lowPrice/closePrice,
    e.g. ``price["closePrice"]["bid"]``.

`symbol` is treated as the Capital.com *epic* (e.g. "BTCUSD"), so the configured
universe must use Capital.com epics when running live.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from ..config import Config
from ..execution.capital_session import CapitalComSession
from .feeds import MarketFeed, MarketStatus
from ..core.filters import calendar_allows
from .models import Candle, OrderBook, OrderBookLevel, Tick

log = logging.getLogger(__name__)

# Strategy/agent timeframe → Capital.com resolution enum. Accept both the
# lower-case ("5m") and cTrader-style ("M5") spellings used across the codebase.
RESOLUTION = {
    "1m": "MINUTE", "5m": "MINUTE_5", "15m": "MINUTE_15", "30m": "MINUTE_30",
    "1h": "HOUR", "4h": "HOUR_4", "1d": "DAY",
    "M1": "MINUTE", "M5": "MINUTE_5", "M15": "MINUTE_15", "M30": "MINUTE_30",
    "H1": "HOUR", "H4": "HOUR_4", "D1": "DAY",
}
_RES_MINUTES = {"MINUTE": 1, "MINUTE_5": 5, "MINUTE_15": 15, "MINUTE_30": 30,
                "HOUR": 60, "HOUR_4": 240, "DAY": 1440}


def _iso(dt: datetime) -> str:
    """Capital.com wants naive-looking ISO without timezone suffix."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_candles(symbol: str, timeframe: str, data: dict) -> list[Candle]:
    """Turn a /prices response into chronological Candles (oldest first)."""
    out: list[Candle] = []
    for p in data.get("prices", []):
        try:
            o, h = p["openPrice"], p["highPrice"]
            low, c = p["lowPrice"], p["closePrice"]

            # Mid price when both sides exist (bid-only candles put a systematic
            # short bias into every feature); fall back to whichever side is there.
            def pick(d: dict) -> float:
                bid, ask = d.get("bid"), d.get("ask")
                if bid is not None and ask is not None:
                    return (float(bid) + float(ask)) / 2.0
                return float(bid if bid is not None else (ask or 0)) or 0.0
            snap = p.get("snapshotTime", "")
            try:
                ts = datetime.fromisoformat(snap.replace("Z", "+00:00")) if snap else datetime.now(timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)
            out.append(Candle(
                symbol=symbol, timeframe=timeframe,
                open=pick(o), high=pick(h), low=pick(low), close=pick(c),
                volume=float(p.get("lastTradedVolume", 0) or 0), ts=ts,
            ))
        except (KeyError, TypeError, ValueError) as e:
            log.warning("Skipping malformed price point for %s: %s", symbol, e)
    return out


@dataclass
class _CandleCache:
    """One (symbol, timeframe) candle series with its refresh bookkeeping."""
    candles: list[Candle] = field(default_factory=list)
    n: int = 0                       # bars the cached fetch asked for
    fetched_mono: float = 0.0        # monotonic time of the last real fetch
    refresh_after: datetime | None = None   # when the next bar can exist


class CapitalComMarketFeed(MarketFeed):
    def __init__(self, session: CapitalComSession, config: Config):
        self.session = session
        self.config = config
        self.universe = [
            u["symbol"] for u in config.get("universe", default=[]) if u.get("enabled", True)
        ]
        self._min_size: dict[str, float | None] = {}   # cached dealing rules
        # Candle cache: a series only gains a new bar once per bar interval, so
        # re-fetching 250 bars every fast loop is almost pure waste — and the
        # cause of the 429 storms that starved strategies of data. Entries are
        # refreshed when the next bar can exist (never more often than
        # ``candle_refetch_seconds``), and served stale when a fetch fails —
        # the agent's own stale-data guard decides whether they're still
        # actionable.
        self._candles: dict[tuple[str, str], _CandleCache] = {}
        self._min_refetch = float(config.get(
            "data", "market", "candle_refetch_seconds", default=25.0) or 0.0)
        # Market snapshot (/markets/{epic}) cache: latest_tick and orderbook
        # both read it, usually within the same loop — share one request. The
        # TTL must exceed the fast-loop interval fraction spent per symbol or
        # every loop refetches every symbol (the 2s default did exactly that).
        self._snap: dict[str, tuple[float, dict]] = {}
        self._snap_ttl = float(config.get(
            "data", "market", "snapshot_ttl", default=20.0) or 0.0)
        # WebSocket quote stream (data.market.websocket, default on): fresh
        # bid/offer per symbol in milliseconds, with every REST path below as
        # the fallback whenever a symbol has no fresh streamed quote.
        self.stream = None
        self._ws_enabled = bool(config.get("data", "market", "websocket",
                                           default=True))

    async def connect(self) -> None:
        await self.session.connect()
        if self._ws_enabled:
            try:
                from .capital_ws import CapitalComQuoteStream
                self.stream = CapitalComQuoteStream(self.session, self.universe)
                await self.stream.start()
            except Exception as e:  # noqa: BLE001 — REST fallback covers this
                log.warning("Quote stream unavailable (%s); using REST "
                            "snapshots only", e)
                self.stream = None
        log.info("CapitalComMarketFeed ready for %d symbols (stream=%s)",
                 len(self.universe), "on" if self.stream else "off")

    def _stream_quote(self, symbol: str) -> tuple[float, float] | None:
        if self.stream is not None:
            return self.stream.quote(symbol)
        return None

    async def market_status(self, symbol: str, now: datetime | None = None) -> MarketStatus:
        """Return broker/calendar tradability and fail closed on ambiguity."""
        checked = now or datetime.now(timezone.utc)
        if not calendar_allows(symbol, checked):
            return MarketStatus(symbol, False, "weekend", checked, "calendar")
        try:
            snap = await self._snapshot(symbol)
            raw = (snap.get("marketStatus") or snap.get("status") or "")
            status = str(raw).strip().upper()
            if status in {"TRADEABLE", "OPEN", "OPENED"}:
                return MarketStatus(symbol, True, "broker_tradeable", checked, "capital_com")
            if status in {"CLOSED", "CLOSE", "SUSPENDED", "UNTRADEABLE", "OFFLINE"}:
                return MarketStatus(symbol, False, "broker_closed", checked, "capital_com")
            log.warning("Unknown market status for %s; blocking entries", symbol)
            return MarketStatus(symbol, False, "broker_status_unknown", checked, "capital_com")
        except Exception as e:  # noqa: BLE001 — market status is a hard gate
            log.error("Failed to determine market status for %s: %s", symbol, e)
            return MarketStatus(symbol, False, "broker_status_error", checked, "capital_com")

    async def _snapshot(self, symbol: str) -> dict:
        """The /markets/{epic} snapshot, shared between tick and orderbook."""
        cached = self._snap.get(symbol)
        if cached and time.monotonic() - cached[0] < self._snap_ttl:
            return cached[1]
        res = await self.session.get(f"/api/v1/markets/{symbol}")
        snap = res.json().get("snapshot", {}) or {}
        self._snap[symbol] = (time.monotonic(), snap)
        return snap

    async def prefetch_snapshots(self, symbols: list[str]) -> None:
        """Warm the snapshot cache for many symbols in a few batched requests.

        ``GET /api/v1/markets?epics=A,B,…`` returns up to ~50 markets per call —
        one loop over the whole universe costs 1–2 requests instead of one per
        symbol (the per-symbol path was the fast loop's dominant latency).
        Failures degrade to the per-symbol fallback in ``_snapshot``.
        """
        now = time.monotonic()
        stale = [s for s in symbols
                 if self._stream_quote(s) is None
                 and (s not in self._snap
                      or now - self._snap[s][0] >= self._snap_ttl)]
        for i in range(0, len(stale), 50):
            chunk = stale[i:i + 50]
            try:
                res = await self.session.get(
                    "/api/v1/markets", params={"epics": ",".join(chunk)})
                got = 0
                for detail in res.json().get("marketDetails", []) or []:
                    epic = (detail.get("instrument") or {}).get("epic")
                    snap = detail.get("snapshot") or {}
                    if epic and snap:
                        self._snap[epic] = (time.monotonic(), snap)
                        got += 1
                log.debug("Batched snapshot: %d/%d epics refreshed", got, len(chunk))
            except Exception as e:  # noqa: BLE001 — per-symbol fallback still works
                log.warning("Batched snapshot fetch failed (%s); falling back to "
                            "per-symbol requests", e)

    async def latest_tick(self, symbol: str) -> Tick | None:
        streamed = self._stream_quote(symbol)
        if streamed is not None:
            return Tick(symbol=symbol, bid=streamed[0], ask=streamed[1])
        try:
            snap = await self._snapshot(symbol)
            bid, ask = float(snap.get("bid", 0)), float(snap.get("offer", 0))
            if not (bid or ask):
                return None
            return Tick(symbol=symbol, bid=bid, ask=ask)
        except Exception as e:  # noqa: BLE001 — data fetch must never break the loop
            log.error("Failed to fetch tick for %s: %s", symbol, e)
            return None

    async def recent_candles(self, symbol: str, timeframe: str, n: int) -> list[Candle]:
        resolution = RESOLUTION.get(timeframe)
        if resolution is None:
            log.error("Unsupported timeframe for Capital.com: %s", timeframe)
            return []

        # Serve from cache while no new bar can exist yet (signals are decided
        # on closed bars only, so nothing actionable changes in between), and
        # never re-fetch the same series more often than the refetch floor.
        key = (symbol, resolution)
        cache = self._candles.get(key)
        now = datetime.now(timezone.utc)
        if (cache is not None and cache.candles and n <= cache.n
                and ((cache.refresh_after is not None and now < cache.refresh_after)
                     or time.monotonic() - cache.fetched_mono < self._min_refetch)):
            return cache.candles[-n:]

        # Refresh at the deepest depth any caller has asked for, so a shallow
        # request (e.g. the daily-change stat's n=2) can't hollow out a series
        # the strategies need 250 bars of.
        fetch_n = max(n, cache.n if cache else 0)
        span = timedelta(minutes=_RES_MINUTES[resolution] * (fetch_n + 2))
        try:
            res = await self.session.get(
                f"/api/v1/prices/{symbol}",
                params={"resolution": resolution, "max": fetch_n,
                        "from": _iso(now - span), "to": _iso(now)},
            )
            candles = _parse_candles(symbol, timeframe, res.json())
        except httpx.HTTPStatusError as e:
            # 404 = no price history for this resolution/window (e.g. market
            # closed, or the demo has no intraday data for this epic). Expected
            # and non-actionable, so don't log it as an error on every poll.
            if e.response.status_code == 404:
                log.debug("No candles for %s/%s in window (404)", symbol, timeframe)
            else:
                log.error("Failed to fetch candles for %s/%s: %s", symbol, timeframe, e)
            return self._stale_candles(key, symbol, timeframe, n)
        except Exception as e:  # noqa: BLE001
            log.error("Failed to fetch candles for %s/%s: %s", symbol, timeframe, e)
            return self._stale_candles(key, symbol, timeframe, n)

        if not candles:
            return self._stale_candles(key, symbol, timeframe, n)

        # The next bar opens one interval after the newest bar we got (which is
        # usually still forming). If the API lags publishing it, refresh_after
        # stays in the past and the refetch floor paces the retries.
        newest = candles[-1].ts
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        self._candles[key] = _CandleCache(
            candles=candles, n=fetch_n,
            fetched_mono=time.monotonic(),
            refresh_after=newest + timedelta(minutes=_RES_MINUTES[resolution]),
        )
        return candles[-n:]

    def _stale_candles(self, key: tuple[str, str], symbol: str, timeframe: str,
                       n: int) -> list[Candle]:
        """Last-known candles after a failed/empty fetch — better than starving
        the strategies and exit management; the agent's stale-data guard
        (audit F-10) still blocks decisions on genuinely decayed series."""
        cache = self._candles.get(key)
        if cache is None or not cache.candles:
            return []
        log.warning("Serving %d cached candles for %s/%s after failed refresh",
                    len(cache.candles), symbol, timeframe)
        return cache.candles[-n:]

    async def orderbook(self, symbol: str, depth: int) -> OrderBook | None:
        # Capital.com exposes only top-of-book (streamed quote or snapshot).
        streamed = self._stream_quote(symbol)
        if streamed is not None:
            return OrderBook(symbol=symbol,
                             bids=[OrderBookLevel(price=streamed[0], size=1.0)],
                             asks=[OrderBookLevel(price=streamed[1], size=1.0)])
        try:
            snap = await self._snapshot(symbol)
            bid, ask = float(snap.get("bid", 0)), float(snap.get("offer", 0))
            if not (bid or ask):
                return None
            return OrderBook(symbol=symbol,
                             bids=[OrderBookLevel(price=bid, size=1.0)],
                             asks=[OrderBookLevel(price=ask, size=1.0)])
        except Exception as e:  # noqa: BLE001
            log.error("Failed to fetch orderbook for %s: %s", symbol, e)
            return None

    async def min_deal_size(self, symbol: str) -> float | None:
        """Broker minimum deal size from /markets/{epic} dealingRules.minDealSize,
        cached per symbol (the agent was sizing positions below this, so the broker
        would reject them)."""
        if symbol in self._min_size:
            return self._min_size[symbol]
        size = None
        try:
            res = await self.session.get(f"/api/v1/markets/{symbol}")
            rules = (res.json().get("dealingRules") or {}).get("minDealSize") or {}
            if rules.get("value") is not None:
                size = float(rules["value"])
        except Exception as e:  # noqa: BLE001 — never break the loop over a rule lookup
            log.warning("Could not fetch dealing rules for %s: %s", symbol, e)
        self._min_size[symbol] = size
        return size
