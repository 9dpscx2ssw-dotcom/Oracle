"""News ingestion via RSS (free), Finnhub, or NewsAPI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import feedparser

from ..config import Config
from .feeds import NewsFeed
from .models import NewsItem

log = logging.getLogger(__name__)


class RSSNewsFeed(NewsFeed):
    def __init__(self, config: Config):
        self.feeds: list[str] = config.get("data", "news", "rss_feeds", default=[])

    async def fetch(self) -> list[NewsItem]:
        import asyncio
        items: list[NewsItem] = []
        for url in self.feeds:
            if not url:      # null entry in the config list (audit F-19)
                continue
            try:
                # feedparser downloads synchronously — run it off the event
                # loop so the trading loop never blocks on a slow feed (F-06).
                parsed = await asyncio.to_thread(feedparser.parse, url)
            except Exception as exc:  # noqa: BLE001 — never let a bad feed kill the loop
                log.warning("RSS fetch failed for %s: %s", url, exc)
                continue
            for entry in parsed.entries[:25]:
                items.append(
                    NewsItem(
                        title=getattr(entry, "title", ""),
                        summary=getattr(entry, "summary", ""),
                        url=getattr(entry, "link", ""),
                        source=parsed.feed.get("title", url),
                        published=_parse_date(entry),
                    )
                )
        log.info("Fetched %d news items", len(items))
        return items


class FinnhubNewsFeed(NewsFeed):
    """Fetch company news from Finnhub API (free tier: 60 API calls/min)."""

    def __init__(self, config: Config):
        # Key lives in secrets (.env), not the YAML config — the old
        # config.get("credentials", ...) path never existed (audit F-19).
        self.api_key: str = getattr(config.secrets, "finnhub_api_key", "") or ""
        self.symbols: list[str] = [
            u["symbol"] for u in config.get("universe", default=[]) if u.get("enabled", True)
        ]
        if not self.api_key:
            log.warning("FINNHUB_API_KEY not set; FinnhubNewsFeed will return empty")

    async def fetch(self) -> list[NewsItem]:
        if not self.api_key or not self.symbols:
            return []

        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed; FinnhubNewsFeed disabled")
            return []

        items: list[NewsItem] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for symbol in self.symbols[:5]:  # Finnhub free tier: ~60 calls/min
                try:
                    resp = await client.get(
                        "https://finnhub.io/api/v1/company-news",
                        params={
                            "symbol": symbol,
                            "limit": 10,
                            "token": self.api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        for article in data:
                            items.append(
                                NewsItem(
                                    title=article.get("headline", ""),
                                    summary=article.get("summary", ""),
                                    url=article.get("url", ""),
                                    source=article.get("source", "Finnhub"),
                                    published=_parse_finnhub_date(
                                        article.get("datetime")
                                    ),
                                )
                            )
                except Exception as exc:
                    log.warning(
                        "Finnhub fetch failed for %s: %s", symbol, exc
                    )
                    continue
        log.info("Fetched %d Finnhub news items", len(items))
        return items


def _parse_date(entry) -> datetime:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class CompositeNewsFeed(NewsFeed):
    """Combine multiple news sources (RSS, Finnhub, etc.)."""

    def __init__(self, config: Config):
        self.feeds: list[NewsFeed] = []
        # Always add RSS if configured
        if config.get("data", "news", "rss_feeds", default=[]):
            self.feeds.append(RSSNewsFeed(config))
        # Add Finnhub if API key is present (from .env secrets, audit F-19)
        if getattr(config.secrets, "finnhub_api_key", ""):
            self.feeds.append(FinnhubNewsFeed(config))

    async def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for feed in self.feeds:
            try:
                fetched = await feed.fetch()
                items.extend(fetched)
            except Exception as exc:
                log.warning("Feed fetch failed: %s", exc)
        # Deduplicate by URL and sort by publish date descending
        seen = set()
        unique = []
        for item in items:
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)
        unique.sort(key=lambda x: x.published, reverse=True)
        return unique


def _parse_finnhub_date(unix_timestamp: Any) -> datetime:
    """Convert Finnhub's unix timestamp (seconds) to datetime."""
    if unix_timestamp:
        try:
            return datetime.fromtimestamp(int(unix_timestamp), tz=timezone.utc)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)
