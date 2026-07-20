"""Fuse technical features + order book + sentiment + macro into a directional
prediction. The LLM reasons over the *summary* of features, not raw ticks.
"""

from __future__ import annotations

from ..data.models import MacroIndicator, Prediction, Sentiment
from .client import LLMClient

_SYSTEM = (
    "You are a discretionary macro trader. Given a structured snapshot of an "
    "instrument (technicals, order-book pressure, news sentiment, macro context), "
    "give a short-horizon directional view. Respond ONLY with JSON: "
    "{\"direction\": -1|0|1, \"confidence\": float in [0,1], \"horizon\": string, "
    "\"rationale\": short string}. Be conservative; prefer 0 when signals conflict."
)


def predict(
    llm: LLMClient,
    symbol: str,
    feature_summary: dict,
    sentiment: Sentiment | None,
    macro: list[MacroIndicator] | None,
) -> Prediction:
    macro_str = (
        ", ".join(f"{m.name}={m.value}" for m in macro) if macro else "none"
    )
    sent_str = (
        f"score={sentiment.score:.2f} conf={sentiment.confidence:.2f}"
        if sentiment
        else "none"
    )
    prompt = (
        f"Instrument: {symbol}\n"
        f"Technicals/orderbook: {feature_summary}\n"
        f"Sentiment: {sent_str}\n"
        f"Macro: {macro_str}\n"
        "Give your directional view."
    )
    data = llm.complete_json(prompt, system=_SYSTEM, max_tokens=300)

    direction = data.get("direction", 0)
    try:
        direction = int(direction)
    except (TypeError, ValueError):
        direction = 0
    direction = max(-1, min(1, direction))

    return Prediction(
        symbol=symbol,
        direction=direction,
        confidence=_clamp(data.get("confidence", 0.0)),
        horizon=str(data.get("horizon", "intraday")),
        rationale=str(data.get("rationale", "")),
    )


def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
