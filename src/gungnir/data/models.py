"""Typed data objects passed between every layer of the agent.

Using pydantic models keeps the boundaries explicit: a feed produces these, a
strategy consumes them, the journal persists them. Nothing passes raw dicts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Market data ───────────────────────────────────────────────────────────────


class Tick(BaseModel):
    symbol: str
    bid: float
    ask: float
    ts: datetime = Field(default_factory=_utcnow)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class Candle(BaseModel):
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    ts: datetime = Field(default_factory=_utcnow)


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    """A depth snapshot. bids/asks are sorted best-first."""

    symbol: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_utcnow)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None


# ── External information ──────────────────────────────────────────────────────


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    url: str = ""
    source: str = ""
    symbols: list[str] = Field(default_factory=list)
    published: datetime = Field(default_factory=_utcnow)


class MacroIndicator(BaseModel):
    """A single macro datapoint, e.g. latest CPI or fed funds rate."""

    series_id: str          # e.g. "CPIAUCSL"
    name: str               # e.g. "CPI"
    value: float
    previous: float | None = None
    observation_date: datetime = Field(default_factory=_utcnow)


class Sentiment(BaseModel):
    symbol: str
    score: float            # -1 (bearish) .. +1 (bullish)
    confidence: float       # 0 .. 1
    rationale: str = ""
    ts: datetime = Field(default_factory=_utcnow)


class Prediction(BaseModel):
    """LLM/model fused view of where an asset is headed."""

    symbol: str
    direction: int          # -1 short, 0 neutral, +1 long
    confidence: float       # 0 .. 1
    horizon: str = "intraday"
    rationale: str = ""
    ts: datetime = Field(default_factory=_utcnow)


# ── Trading primitives ────────────────────────────────────────────────────────


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


class Signal(BaseModel):
    """A strategy's *intent*, before risk sizing. Conviction in [0, 1]."""

    strategy: str
    symbol: str
    side: Side
    conviction: float
    rationale: str = ""
    ts: datetime = Field(default_factory=_utcnow)


class Order(BaseModel):
    """A sized order ready for the broker."""

    symbol: str
    side: Side
    volume: float                 # in broker lots / units
    stop_loss: float | None = None
    take_profit: float | None = None
    client_id: str = ""           # idempotency key
    ts: datetime = Field(default_factory=_utcnow)


class Trade(BaseModel):
    """A round-trip (or open position) recorded in the journal."""

    symbol: str
    side: Side
    volume: float
    entry_price: float
    exit_price: float | None = None
    pnl: float | None = None
    strategy: str = ""
    mode: str = "real"          # real | shadow
    opened_at: datetime = Field(default_factory=_utcnow)
    closed_at: datetime | None = None
    # Snapshot of the context that produced this trade — fuel for learning.
    context: dict = Field(default_factory=dict)
