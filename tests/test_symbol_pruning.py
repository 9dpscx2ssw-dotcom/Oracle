"""Per-symbol learned pruning: failing_symbols() decision rule, the
excluded_symbols blacklist on strategies, and registry persistence."""

from __future__ import annotations

from gungnir.data.models import Side, Trade
from gungnir.learning.evaluator import failing_symbols
from gungnir.strategy.kraken_strategies import KRAKEN_STRATEGIES
from gungnir.strategy.registry import StrategyRegistry


def _trade(symbol: str, pnl: float) -> Trade:
    return Trade(symbol=symbol, side=Side.BUY, volume=1.0, entry_price=100.0,
                 exit_price=100.0 + pnl, pnl=pnl, strategy="s")


def test_failing_symbols_requires_evidence_and_bad_pf():
    # GOLD: 30 trades, mostly losers (PF well under 0.5) → prune.
    gold = [_trade("GOLD", -1.0)] * 25 + [_trade("GOLD", 1.0)] * 5
    # EURUSD: same terrible PF but only 10 trades → not enough evidence.
    eur = [_trade("EURUSD", -1.0)] * 8 + [_trade("EURUSD", 1.0)] * 2
    # US500: 40 trades, profitable → never pruned.
    spx = [_trade("US500", 1.0)] * 25 + [_trade("US500", -1.0)] * 15
    out = failing_symbols(gold + eur + spx, min_trades=30, max_profit_factor=0.5)
    assert out == ["GOLD"]


def test_failing_symbols_empty_on_no_closed_trades():
    assert failing_symbols([], min_trades=30) == []


def test_excluded_symbols_veto_trades_symbol():
    strat = KRAKEN_STRATEGIES[0](symbols=[], excluded_symbols=["GOLD"])
    assert strat.trades_symbol("US500") is True     # open scope
    assert strat.trades_symbol("GOLD") is False     # blacklisted
    # Blacklist also beats an explicit opt-in scope.
    scoped = KRAKEN_STRATEGIES[0](symbols=["GOLD", "US500"], excluded_symbols=["GOLD"])
    assert scoped.trades_symbol("GOLD") is False
    assert scoped.trades_symbol("US500") is True


def test_registry_round_trips_excluded_symbols(tmp_path):
    path = tmp_path / "strategies.yaml"
    name = KRAKEN_STRATEGIES[0].name
    path.write_text(
        "strategies:\n"
        f"  - name: {name}\n"
        "    mode: shadow\n"
        "    excluded_symbols: [GOLD, EURUSD]\n"
    )
    reg = StrategyRegistry.from_yaml(path)
    strat = reg.get(name)
    assert strat is not None
    assert strat.excluded_symbols == ["GOLD", "EURUSD"]

    strat.excluded_symbols.append("BTCUSD")
    reg.save()
    reloaded = StrategyRegistry.from_yaml(path)
    assert reloaded.get(name).excluded_symbols == ["GOLD", "EURUSD", "BTCUSD"]
