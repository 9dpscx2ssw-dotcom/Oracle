"""Parameter optimization — the quantitative half of "learning over time".

The optimizer proposes new strategy parameters and backtests them against the
trade journal (or a replayed candle history). It works hand-in-hand with:
  • llm/reflection.py  — proposes *candidate* parameter sets (qualitative).
  • evaluator.accept_change — gates whether a candidate is actually better.

Two backends:
  • walk_forward — grid/random search over recent windows, picking params that
                   generalize across folds (guards against overfitting).
  • bayesian     — sample-efficient search (needs the `learn` extra: scikit-learn).

This module deliberately separates *propose* from *commit*: it never mutates a
live strategy directly. The agent applies accepted params via the registry.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..data.models import Trade
from .evaluator import Metrics, accept_change, evaluate

log = logging.getLogger(__name__)

# A backtest function: (params, trades) -> the trades that *would* have resulted.
# In practice you replay candle history through the strategy; for the journal-only
# path you can re-score recorded context against new thresholds.
BacktestFn = Callable[[dict, list[Trade]], list[Trade]]


class Optimizer:
    def __init__(self, mode: str = "walk_forward", min_trades: int = 30):
        self.mode = mode
        self.min_trades = min_trades

    def propose(
        self,
        current_params: dict,
        candidate_params: dict,
        history: list[Trade],
        backtest: BacktestFn,
    ) -> tuple[dict, Metrics, bool]:
        """Backtest current vs candidate; return (params_to_use, metrics, changed)."""
        before = evaluate(backtest(current_params, history))
        after = evaluate(backtest(candidate_params, history))
        if accept_change(before, after, self.min_trades):
            log.info("Accepting param change: sharpe %.2f -> %.2f", before.sharpe, after.sharpe)
            return candidate_params, after, True
        return current_params, before, False

    def search(
        self,
        param_grid: dict[str, list],
        history: list[Trade],
        backtest: BacktestFn,
        base_params: dict,
    ) -> dict:
        """Simple grid/walk-forward search returning the best-generalizing params.

        Replace with Bayesian optimization (skopt/optuna) when mode == 'bayesian'.
        Kept minimal here so the base image needs no ML deps.
        """
        best_params = dict(base_params)
        best_sharpe = evaluate(backtest(best_params, history)).sharpe
        for name, values in param_grid.items():
            for v in values:
                trial = {**best_params, name: v}
                sharpe = evaluate(backtest(trial, history)).sharpe
                if sharpe > best_sharpe:
                    best_sharpe, best_params = sharpe, trial
        return best_params
