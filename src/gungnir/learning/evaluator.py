"""Performance evaluation: turn a set of closed trades into metrics, and act as
the *gate* that decides whether a proposed parameter change is allowed through.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from ..data.models import Trade


class Metrics(BaseModel):
    n_trades: int
    win_rate: float
    expectancy: float       # average pnl per trade
    profit_factor: float
    sharpe: float
    max_drawdown: float
    total_pnl: float


def evaluate(trades: list[Trade]) -> Metrics:
    pnls = [t.pnl for t in trades if t.pnl is not None]
    if not pnls:
        return Metrics(n_trades=0, win_rate=0, expectancy=0, profit_factor=0,
                       sharpe=0, max_drawdown=0, total_pnl=0)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    mean = sum(pnls) / len(pnls)

    # Sharpe on per-trade *returns* (pnl / notional), not dollar PnL. The previous
    # formula annualized dollar PnL by sqrt(252) as if one trade == one day, which
    # inflated Sharpe by orders of magnitude for multi-trade-per-day strategies and
    # made the optimizer's accept-gate meaningless. This is an un-annualized,
    # per-trade, scale-free measure of risk-adjusted edge.
    rets = [
        t.pnl / abs(t.entry_price * t.volume)
        for t in trades
        if t.pnl is not None and t.entry_price and t.volume
    ]
    if len(rets) > 1:
        rmean = sum(rets) / len(rets)
        rstd = math.sqrt(sum((r - rmean) ** 2 for r in rets) / len(rets))
        sharpe = (rmean / rstd) if rstd else 0.0
    else:
        sharpe = 0.0

    return Metrics(
        n_trades=len(pnls),
        win_rate=len(wins) / len(pnls),
        expectancy=mean,
        profit_factor=(gross_win / gross_loss) if gross_loss else float("inf"),
        sharpe=sharpe,
        max_drawdown=_max_drawdown(pnls),
        total_pnl=sum(pnls),
    )


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return mdd


def failing_symbols(trades: list[Trade], min_trades: int = 30,
                    max_profit_factor: float = 0.5) -> list[str]:
    """Symbols on which this trade set shows a well-sampled negative edge.

    Groups closed trades by symbol; a symbol qualifies for pruning when it has
    at least ``min_trades`` closed trades AND a profit factor below
    ``max_profit_factor``. Small samples never qualify — a symbol must earn its
    exclusion with evidence, not a bad streak.
    """
    by_symbol: dict[str, list[Trade]] = {}
    for t in trades:
        if t.pnl is not None and t.symbol:
            by_symbol.setdefault(t.symbol, []).append(t)
    out = []
    for symbol, ts in by_symbol.items():
        m = evaluate(ts)
        if m.n_trades >= min_trades and m.profit_factor < max_profit_factor:
            out.append(symbol)
    return sorted(out)


def accept_change(before: Metrics, after: Metrics, min_trades: int = 30) -> bool:
    """Gate a proposed parameter change: only accept if it's a credible improvement.

    Used by the optimizer/reflection loop: backtest current params vs. proposed
    params over the journal, and only commit if the proposal beats the incumbent
    on a robust metric with enough sample size.
    """
    if after.n_trades < min_trades:
        return False
    # Require a real edge improvement, not noise.
    return after.sharpe > before.sharpe * 1.05 and after.expectancy >= before.expectancy
