"""Post-trade reflection: the LLM reads recent trades and proposes parameter
nudges and hypotheses. This is the qualitative half of "learning over time";
the quantitative half lives in learning/optimizer.py.

Crucially, the LLM only *proposes*. learning/evaluator.py gates whether a
proposal is accepted, so the model can't degrade a profitable strategy on a whim.
"""

from __future__ import annotations

from ..data.models import Trade
from .client import LLMClient

_SYSTEM = (
    "You are a trading systems researcher reviewing a strategy's recent trades. "
    "Identify patterns in the losers and winners and propose small, testable "
    "parameter adjustments. Respond ONLY with JSON: "
    "{\"hypothesis\": short string, \"param_updates\": {param: new_value, ...}, "
    "\"confidence\": float in [0,1]}. Only suggest parameters that already exist. "
    "Keep changes incremental."
)


def reflect(
    llm: LLMClient,
    strategy: str,
    current_params: dict,
    recent_trades: list[Trade],
) -> dict:
    """Return {'hypothesis', 'param_updates', 'confidence'} (possibly empty)."""
    if not recent_trades:
        return {}

    summary = [
        {
            "symbol": t.symbol,
            "side": t.side.value,
            "pnl": t.pnl,
            "context": t.context,
        }
        for t in recent_trades[-50:]
    ]
    prompt = (
        f"Strategy: {strategy}\n"
        f"Current params: {current_params}\n"
        f"Recent trades: {summary}\n"
        "Propose adjustments."
    )
    data = llm.complete_json(prompt, system=_SYSTEM)
    if not isinstance(data.get("param_updates"), dict):
        return {}
    return data
