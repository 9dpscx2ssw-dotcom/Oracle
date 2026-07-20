"""Data feed interfaces. Concrete feeds live in market_feed/news_feed/macro_feed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import Candle, MacroIndicator, NewsItem, OrderBook, Tick


@dataclass(frozen=True)
class MarketStatus:
    """Fail-closed tradability result for one instrument."""
    symbol: str
    tradeable: bool
    reason: str
    checked_at: datetime
    source: str = "feed"


class MarketFeed(ABC):
    """Live price + order-book source (cTrader)."""

    @abstractmethod
    async def latest_tick(self, symbol: str) -> Tick | None: ...

    @abstractmethod
    async def recent_candles(self, symbol: str, timeframe: str, n: int) -> list[Candle]: ...

    @abstractmethod
    async def orderbook(self, symbol: str, depth: int) -> OrderBook | None: ...

    async def min_deal_size(self, symbol: str) -> float | None:
        """Broker minimum deal/lot size for an instrument, if known. Default None
        (unknown) so feeds without dealing rules don't impose one."""
        return None

    async def market_status(self, symbol: str, now: datetime | None = None) -> MarketStatus:
        """Apply the conservative calendar gate even for synthetic/paper feeds."""
        checked = now or datetime.now(timezone.utc)
        # Import lazily to avoid a module cycle during feed interface import.
        from ..core.filters import calendar_allows
        allowed = calendar_allows(symbol, checked)
        return MarketStatus(symbol, allowed,
                            "feed_default" if allowed else "weekend",
                            checked, "calendar")


class NewsFeed(ABC):
    @abstractmethod
    async def fetch(self) -> list[NewsItem]: ...


class MacroFeed(ABC):
    @abstractmethod
    async def fetch(self) -> list[MacroIndicator]: ...
