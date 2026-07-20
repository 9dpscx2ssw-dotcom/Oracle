"""Order-book analysis: imbalance, spread, microprice, depth slope.

These microstructure features are early signals of short-term pressure and are
fed into strategies and the LLM prediction context.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..data.models import OrderBook


class OrderBookFeatures(BaseModel):
    symbol: str
    spread: float
    mid: float
    microprice: float           # size-weighted fair price
    imbalance: float            # (bidvol - askvol) / (bidvol + askvol), -1..1
    bid_depth: float
    ask_depth: float
    depth_slope: float          # how fast size grows away from top of book


def analyze(book: OrderBook) -> OrderBookFeatures | None:
    if not book.bids or not book.asks:
        return None

    best_bid = book.bids[0]
    best_ask = book.asks[0]
    spread = best_ask.price - best_bid.price
    mid = (best_bid.price + best_ask.price) / 2.0

    bid_depth = sum(lvl.size for lvl in book.bids)
    ask_depth = sum(lvl.size for lvl in book.asks)
    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total else 0.0

    # Microprice: weight each side's best price by the *opposite* side's size.
    denom = best_bid.size + best_ask.size
    microprice = (
        (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / denom
        if denom
        else mid
    )

    depth_slope = _depth_slope(book)

    return OrderBookFeatures(
        symbol=book.symbol,
        spread=spread,
        mid=mid,
        microprice=microprice,
        imbalance=imbalance,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        depth_slope=depth_slope,
    )


def _depth_slope(book: OrderBook) -> float:
    """Average per-level cumulative-size growth, a proxy for liquidity thickness."""
    cum = 0.0
    growth = []
    prev = 0.0
    for lvl in list(book.bids) + list(book.asks):
        cum += lvl.size
        growth.append(cum - prev)
        prev = cum
    return float(sum(growth) / len(growth)) if growth else 0.0
