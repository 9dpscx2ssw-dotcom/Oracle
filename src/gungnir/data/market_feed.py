"""Paper-mode market feeds: a flat stub and a moving synthetic random walk.

The live feed is Capital.com (`data/capital_com_feed.py`). These two exist so
`--dry-run` and tests can exercise the full pipeline without any broker.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque

from .models import Candle, OrderBook, OrderBookLevel, Tick
from .feeds import MarketFeed

class StubMarketFeed(MarketFeed):
    """Deterministic fake feed for dev/tests and `--dry-run` without a broker."""

    def __init__(self, mid: float = 1.10):
        self._mid = mid

    async def latest_tick(self, symbol: str) -> Tick | None:
        return Tick(symbol=symbol, bid=self._mid - 0.0001, ask=self._mid + 0.0001)

    async def recent_candles(self, symbol: str, timeframe: str, n: int) -> list[Candle]:
        return [
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                open=self._mid,
                high=self._mid + 0.001,
                low=self._mid - 0.001,
                close=self._mid,
            )
            for _ in range(n)
        ]

    async def orderbook(self, symbol: str, depth: int) -> OrderBook | None:
        bids = [OrderBookLevel(price=self._mid - 0.0001 * (i + 1), size=100) for i in range(depth)]
        asks = [OrderBookLevel(price=self._mid + 0.0001 * (i + 1), size=100) for i in range(depth)]
        return OrderBook(symbol=symbol, bids=bids, asks=asks)


class SyntheticMarketFeed(MarketFeed):
    """A *moving* paper feed for `--dry-run` so the whole pipeline is alive.

    `StubMarketFeed` returns a flat price, which means no strategy ever fires —
    so in paper mode the agent looks dead (no signals, trades, or RL). This feed
    instead walks each symbol's price with a deterministic, time-anchored random
    walk: increments are seeded by (symbol, timeframe, period) so the series is
    reproducible and *continuous across calls*, and it advances with wall-clock
    time so fresh candles keep arriving while the agent runs. That gives the
    strategies real structure to trade against without any external data source.
    """

    _PERIOD_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
                   "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

    # Approximate real price levels so paper mode looks credible (and stops/targets
    # are sensibly scaled per instrument). Unknown symbols fall back to a hashed base.
    _ANCHORS = {
        "US100": 28991.5, "US500": 7332.7, "US30": 51753.1, "RTY": 2999.5,
        "J225": 69475.0, "DE40": 24632.9, "UK100": 10460.5, "HK50": 22860.8,
        "GOLD": 4090.06, "XAUUSD": 4090.06,
        "BTCUSD": 59495.85, "ETHUSD": 1568.54, "XRPUSD": 1.04201,
        "SOLUSD": 70.8973, "DOGEUSD": 0.0728015,
        "EURUSD": 1.13835, "USDJPY": 161.729, "GBPUSD": 1.31933, "AUDUSD": 0.68921,
        "NZDUSD": 0.56369, "USDCHF": 0.80936, "USDCAD": 1.41910, "EURJPY": 184.092,
        "EURGBP": 0.86245, "EURAUD": 1.65023, "EURCHF": 0.92181, "EURCAD": 1.61544,
        "EURNZD": 2.01767, "USDMXN": 17.49913, "USDTRY": 46.47822, "USDPLN": 3.76651,
        "USDNOK": 9.92330, "USDCNH": 6.80315, "USDZAR": 16.46388,
    }

    def __init__(self, base: float = 100.0, vol: float = 0.004, history: int = 400):
        self.base = base          # nominal anchor price
        self.vol = vol            # per-period step size (~0.4% by default)
        self.history = history    # candles to seed so indicators have a warm-up
        self._series: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _seed(symbol: str) -> int:
        return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)

    def _now_period(self, timeframe: str) -> tuple[int, int]:
        m = self._PERIOD_MIN.get(timeframe, 5)
        return int(time.time() // (m * 60)), m

    def _step(self, symbol: str, timeframe: str, period: int) -> float:
        """Deterministic, ~zero-mean log-return for one period."""
        h = hashlib.sha256(f"{symbol}|{timeframe}|{period}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64          # uniform [0,1)
        return (u - 0.5) * 2.0 * self.vol

    def _closes(self, symbol: str, timeframe: str, n: int) -> list[float]:
        key = (symbol, timeframe)
        now, _ = self._now_period(timeframe)
        st = self._series.get(key)
        if st is None:
            maxlen = max(self.history, n + 5)
            closes: deque[float] = deque(maxlen=maxlen)
            # Anchor to a realistic level when known, else a per-symbol hashed base.
            price = self._ANCHORS.get(symbol, self.base * (0.5 + (self._seed(symbol) % 1000) / 1000.0))
            for per in range(now - maxlen + 1, now + 1):
                price *= math.exp(self._step(symbol, timeframe, per))
                closes.append(price)
            self._series[key] = {"last": now, "closes": closes}
        else:
            for per in range(st["last"] + 1, now + 1):
                st["closes"].append(st["closes"][-1] * math.exp(self._step(symbol, timeframe, per)))
            st["last"] = now
        return list(self._series[key]["closes"])[-n:]

    async def recent_candles(self, symbol: str, timeframe: str, n: int) -> list[Candle]:
        closes = self._closes(symbol, timeframe, n)
        out: list[Candle] = []
        for i, c in enumerate(closes):
            o = closes[i - 1] if i > 0 else c
            hi, lo = max(o, c) * 1.0008, min(o, c) * 0.9992
            out.append(Candle(symbol=symbol, timeframe=timeframe,
                              open=o, high=hi, low=lo, close=c, volume=100.0))
        return out

    async def latest_tick(self, symbol: str) -> Tick | None:
        c = await self.recent_candles(symbol, "1m", 1)
        px = c[-1].close
        return Tick(symbol=symbol, bid=px * 0.9999, ask=px * 1.0001)

    async def orderbook(self, symbol: str, depth: int) -> OrderBook | None:
        c = await self.recent_candles(symbol, "1m", 1)
        mid = c[-1].close
        bids = [OrderBookLevel(price=mid * (1 - 0.0001 * (i + 1)), size=100) for i in range(depth)]
        asks = [OrderBookLevel(price=mid * (1 + 0.0001 * (i + 1)), size=100) for i in range(depth)]
        return OrderBook(symbol=symbol, bids=bids, asks=asks)
