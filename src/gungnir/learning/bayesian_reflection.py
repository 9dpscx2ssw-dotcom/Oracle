"""Bayesian optimization for strategy parameter tuning.

Replaces LLM reflection with deterministic, reproducible hyperparameter search
via scipy.optimize.differential_evolution. The fitness of a candidate parameter
set is a REAL backtest: replay the strategy over the locally accumulated candle
history (the same engine the walk-forward gate uses).

The previous version "rescored" journal trades through a placeholder that
returned them unchanged — every candidate had identical fitness, the optimizer
burned 30 DE generations per strategy per slow loop, and always returned {}.

Advantages over LLM reflection:
  • Deterministic: same history → same parameters every time
  • Grounded: fitness is measured on price data, not model opinion
  • Cost: 0 LLM tokens
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scipy.optimize import differential_evolution

from .journal import Journal

if TYPE_CHECKING:
    from ..strategy.registry import Strategy

log = logging.getLogger(__name__)

# Bars of stored history required before optimizing on it — below this the
# backtest is noise and any "improvement" is a fit to a handful of swings.
MIN_BARS = 300


def _top_symbol(trades) -> str | None:
    counts: dict[str, int] = {}
    for t in trades:
        if t.symbol:
            counts[t.symbol] = counts.get(t.symbol, 0) + 1
    return max(counts, key=counts.get) if counts else None


def optimize_strategy(strategy: "Strategy", journal: Journal) -> dict:
    """Search the strategy's parameter bounds for a higher-Sharpe variant.

    Returns {} unless a candidate credibly beats the current parameters on a
    backtest over stored candle history. The walk-forward gate downstream
    still has final say — this only *proposes*.
    """
    from ..backtest import engine
    from ..features import feature_store

    param_bounds = strategy.get_parameter_bounds()
    if not param_bounds:
        return {}

    trades = journal.closed(strategy=strategy.name, limit=50)
    if len(trades) < 20:
        return {}          # not enough live evidence that tuning matters yet

    symbol = _top_symbol(trades)
    if symbol is None:
        return {}
    candles = journal.db.load_candles(symbol, strategy.timeframe, limit=2000)
    if len(candles) < MIN_BARS:
        log.debug("Bayesian tuning skipped for %s: only %d stored bars on %s "
                  "(need %d)", strategy.name, len(candles), symbol, MIN_BARS)
        return {}

    # Indicators are identical for every candidate — compute once, share.
    feats = feature_store.build_kraken_series(symbol, candles)
    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]
    cls = type(strategy)

    def _run(params: dict) -> float:
        trial = cls(params=params, symbols=strategy.symbols,
                    mode=strategy.mode, timeframe=strategy.timeframe)
        result = engine.run(trial, candles, symbol, feats=feats)
        return result.metrics.sharpe

    baseline_sharpe = _run(dict(strategy.params))

    def fitness(x) -> float:
        return -_run(dict(zip(param_names, (float(v) for v in x))))

    try:
        # Small budget on purpose: each evaluation is a full backtest, and the
        # slow loop runs this for every active strategy.
        result = differential_evolution(
            fitness, bounds, maxiter=6, popsize=5, seed=42,
            tol=1e-3, polish=False,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Optimization failed for %s: %s", strategy.name, e)
        return {}

    optimized_sharpe = -float(result.fun)
    sharpe_gain = optimized_sharpe - baseline_sharpe
    # Require a real edge improvement (additive margin — multiplicative gates
    # invert around negative Sharpe).
    if sharpe_gain < max(0.05, 0.1 * abs(baseline_sharpe)):
        return {}

    updates = {name: round(float(v), 4) for name, v in zip(param_names, result.x)}
    return {
        "param_updates": updates,
        "hypothesis": (
            f"Backtest-optimized on {symbol}/{strategy.timeframe} "
            f"({len(candles)} bars): Sharpe {baseline_sharpe:.2f} → "
            f"{optimized_sharpe:.2f}"
        ),
        "confidence": min(0.9, 0.5 + sharpe_gain / 2.0),
    }
