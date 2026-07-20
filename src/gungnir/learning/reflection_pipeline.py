"""Ties the learning pieces together for the slow loop:

    journal trades  ──▶  llm/reflection (propose param updates)
                     ──▶  optimizer/evaluator (gate the proposal)
                     ──▶  registry.update_params + save (commit if accepted)

This is where "learns from trading over time and tweaks the strategies" actually
happens. The LLM proposes; the evaluator decides; only credible improvements are
committed, so the agent can't talk itself into a worse strategy.
"""

from __future__ import annotations

import logging

from ..config import Config
from ..llm.client import LLMClient
from ..strategy.registry import StrategyRegistry
from .bayesian_reflection import optimize_strategy as bayesian_optimize
from .evaluator import evaluate
from .journal import Journal

log = logging.getLogger(__name__)


def run(config: Config, llm: LLMClient, registry: StrategyRegistry, journal: Journal, reflection_mode: str | None = None) -> None:
    min_trades = config.get("learning", "min_trades_before_tuning", default=30)
    lookback = config.get("learning", "reflection_lookback_trades", default=50)
    # Model-risk governance: when auto_apply is false, accepted proposals are
    # recorded as PENDING instead of self-applying — a human reviews the
    # Learning tab and applies them by editing data/strategies.yaml.
    auto_apply = bool(config.get("learning", "auto_apply", default=True))
    # reflection_mode can be overridden at runtime; fall back to config
    if reflection_mode is None:
        reflection_mode = config.get("learning", "reflection_mode", default="bayesian")  # llm or bayesian

    active = registry.active()
    if not active:
        return

    # Phase 1: Batch all strategies into one LLM call (if using LLM mode)
    if reflection_mode == "llm":
        proposals = _batch_llm_reflect(llm, active, journal, min_trades, lookback)
    else:
        # Phase 2: Bayesian optimization (deterministic, no LLM calls)
        proposals = _batch_bayesian_reflect(active, journal, min_trades, lookback)

    # Process proposals
    for strat in active:
        proposal = proposals.get(strat.name, {})
        updates = proposal.get("param_updates") if proposal else None
        if not updates:
            continue

        # Gate: walk-forward ONLY — replay current vs proposed params over the
        # locally stored price history (real bars the proposal was never fitted
        # on). The old "holdout" fallback re-scored journal trades through a
        # placeholder that returned them unchanged, so it could never accept
        # anything and silently pretended to validate. When history is too thin
        # to gate, the proposal is recorded as unvalidated and NOT applied.
        closed = journal.closed(strategy=strat.name, limit=lookback)
        if len(closed) < min_trades:
            continue
        before = evaluate(closed)
        wf = _walk_forward_accept(strat, updates, journal, closed)
        accepted = bool(wf)
        hypothesis = str(proposal.get("hypothesis", ""))
        if accepted and auto_apply:
            log.info("Strategy %s: applying %s (%s)", strat.name, updates, hypothesis)
            strat.update_params(updates)
        elif accepted:
            log.info("Strategy %s: proposal PASSED the gate but auto_apply=false — "
                     "recorded as pending: %s", strat.name, updates)
            hypothesis = f"PENDING APPROVAL (auto_apply=false): {hypothesis}"
        elif wf is None:
            log.info("Strategy %s: proposal %s NOT validated — insufficient "
                     "candle history for the walk-forward gate", strat.name, updates)
            hypothesis = f"UNVALIDATED (insufficient history): {hypothesis}"
        else:
            log.info("Strategy %s: rejected proposal %s (walk-forward showed "
                     "no credible edge)", strat.name, updates)

        # Record the reflection so the dashboard's Learning tab can show history.
        journal.record_learning_event(
            strategy=strat.name,
            hypothesis=hypothesis,
            param_updates=updates,
            accepted=accepted and auto_apply,
            sharpe_before=before.sharpe,
            sharpe_after=None,
        )


def _walk_forward_accept(strat, updates: dict, journal: Journal,
                         closed: list, min_bars: int = 300) -> bool | None:
    """Replay current vs proposed params over stored price history.

    Uses the candle store the agent accumulates while running. Backtests the
    strategy's 3 most-traded symbols with both parameter sets; the proposal is
    accepted only if it wins (higher Sharpe, no worse total PnL) on a majority
    of the tested symbols. Returns None when the store doesn't yet hold enough
    bars — the caller then records the proposal as unvalidated (never applies).
    """
    try:
        from ..backtest import engine
        from ..features import feature_store

        counts: dict[str, int] = {}
        for t in closed:
            if t.symbol:
                counts[t.symbol] = counts.get(t.symbol, 0) + 1
        symbols = [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
        if not symbols:
            return None

        wins, tested = 0, 0
        for sym in symbols:
            candles = journal.db.load_candles(sym, strat.timeframe, limit=2000)
            if len(candles) < min_bars:
                continue
            base = type(strat)(params=dict(strat.params), symbols=strat.symbols,
                               mode=strat.mode, timeframe=strat.timeframe)
            cand = type(strat)(params={**strat.params, **updates},
                               symbols=strat.symbols, mode=strat.mode,
                               timeframe=strat.timeframe)
            # Indicators are identical for both runs — compute once, share.
            feats = feature_store.build_kraken_series(sym, candles)
            r0 = engine.run(base, candles, sym, feats=feats)
            r1 = engine.run(cand, candles, sym, feats=feats)
            tested += 1
            if (r1.metrics.sharpe > r0.metrics.sharpe
                    and r1.metrics.total_pnl >= r0.metrics.total_pnl):
                wins += 1
        if tested == 0:
            return None
        verdict = wins * 2 > tested
        log.info("Walk-forward gate for %s: %d/%d symbols improved → %s",
                 strat.name, wins, tested, "accept" if verdict else "reject")
        return verdict
    except Exception as e:  # noqa: BLE001 — validation must not break the slow loop
        log.warning("Walk-forward gate failed for %s (%s); proposal left "
                    "unvalidated", strat.name, e)
        return None


def _batch_llm_reflect(llm: LLMClient, active: list, journal: Journal,
                       min_trades: int, lookback: int) -> dict[str, dict]:
    """Phase 1: Batch all strategies into a single LLM call (15× reduction)."""
    summaries = {}
    for strat in active:
        closed = journal.closed(strategy=strat.name, limit=lookback)
        if len(closed) < min_trades:
            continue
        summaries[strat.name] = {
            "params": strat.params,
            "recent_trades": [
                {"symbol": t.symbol, "side": t.side.value, "pnl": t.pnl}
                for t in closed[-20:]  # Last 20 trades
            ],
        }

    if not summaries:
        return {}

    # Single batched prompt
    _BATCH_SYSTEM = (
        "You are a trading systems researcher. Given multiple strategies and their "
        "recent trades, propose ONE focused parameter change per strategy to improve "
        "performance. Respond ONLY with JSON: "
        '{"strategy_name": {"param": new_value, "hypothesis": "...", "confidence": 0.5}, ...}'
    )

    prompt = (
        "Given these trading strategies and their recent trades, propose one "
        "focused parameter adjustment per strategy (or omit if no improvement seen).\n\n"
        f"Strategies: {summaries}\n\n"
        'Respond with JSON: {"strategy_name": {"param": value, "hypothesis": "...", '
        '"confidence": 0.7}, ...}'
    )

    data = llm.complete_json(prompt, system=_BATCH_SYSTEM)

    # Extract proposals (LLM might return structured dict or nested responses)
    proposals = {}
    for strat_name, proposal in data.items():
        if isinstance(proposal, dict) and "param" in proposal:
            proposals[strat_name] = {
                "param_updates": {k: v for k, v in proposal.items()
                                 if k not in ("hypothesis", "confidence")},
                "hypothesis": proposal.get("hypothesis", ""),
                "confidence": proposal.get("confidence", 0.5),
            }
    return proposals


def _batch_bayesian_reflect(active: list, journal: Journal,
                            min_trades: int, lookback: int) -> dict[str, dict]:
    """Phase 2: Optimize all strategies via Bayesian (scipy.optimize)."""
    proposals = {}
    for strat in active:
        closed = journal.closed(strategy=strat.name, limit=lookback)
        if len(closed) < min_trades:
            continue
        proposal = bayesian_optimize(strat, journal)
        if proposal:
            proposals[strat.name] = proposal
    return proposals
