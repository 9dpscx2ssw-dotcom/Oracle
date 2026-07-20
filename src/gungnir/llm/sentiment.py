"""News/social sentiment scoring via the LLM."""

from __future__ import annotations

from ..data.models import NewsItem, Sentiment
from .client import LLMClient

_SYSTEM = (
    "You are a financial markets analyst. Score how the supplied headlines are "
    "likely to affect the given instrument over the next few hours. Respond ONLY "
    "with JSON: {\"score\": float in [-1,1], \"confidence\": float in [0,1], "
    "\"rationale\": short string}. Negative = bearish, positive = bullish."
)


_MARKET_SYSTEM = (
    "You are a financial markets analyst. Score how the supplied headlines are "
    "likely to affect broad risk appetite (equities, risk FX, crypto) over the "
    "next few hours. Respond ONLY with JSON: {\"score\": float in [-1,1], "
    "\"confidence\": float in [0,1], \"rationale\": short string}. "
    "Negative = risk-off/bearish, positive = risk-on/bullish."
)

MARKET = "MARKET"


def score(llm: LLMClient, symbol: str, news: list[NewsItem]) -> Sentiment:
    if not news:
        return Sentiment(symbol=symbol, score=0.0, confidence=0.0, rationale="no news")

    headlines = "\n".join(f"- {n.title}" for n in news[:30])
    prompt = f"Instrument: {symbol}\nHeadlines:\n{headlines}"
    data = llm.complete_json(prompt, system=_SYSTEM, max_tokens=300)

    return Sentiment(
        symbol=symbol,
        score=_clamp(data.get("score", 0.0), -1, 1),
        confidence=_clamp(data.get("confidence", 0.0), 0, 1),
        rationale=str(data.get("rationale", "")),
    )


def score_market(llm: LLMClient, news: list[NewsItem]) -> Sentiment:
    """One market-level sentiment for a news cycle.

    RSS items carry no per-symbol tags, so per-symbol scoring sent the *same*
    headline set to the LLM once per symbol — N× the tokens for N copies of one
    answer. One call per news cycle serves every symbol.
    """
    if not news:
        return Sentiment(symbol=MARKET, score=0.0, confidence=0.0, rationale="no news")
    headlines = "\n".join(f"- {n.title}" for n in news[:30])
    data = llm.complete_json(f"Headlines:\n{headlines}",
                             system=_MARKET_SYSTEM, max_tokens=300)
    return Sentiment(
        symbol=MARKET,
        score=_clamp(data.get("score", 0.0), -1, 1),
        confidence=_clamp(data.get("confidence", 0.0), 0, 1),
        rationale=str(data.get("rationale", "")),
    )


def _clamp(v, lo, hi) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0
