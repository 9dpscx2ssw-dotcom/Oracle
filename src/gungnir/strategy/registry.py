"""Load strategies + their (tunable) params from YAML, and persist updates.

The registry is the bridge between config, the dashboard, and the learning layer:
  • the dashboard calls `set_mode` to turn strategies off/shadow/live,
  • the optimizer / reflection pass call `update_params`,
  • `save` writes mode + params back so changes survive restarts.

All strategies in the config are loaded (even `off` ones) so they can be toggled
on from the dashboard without an edit + restart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .base import Strategy
from .examples.mean_reversion import MeanReversion
from .examples.trend_following import TrendFollowing
from .kraken_strategies import KRAKEN_STRATEGIES

log = logging.getLogger(__name__)

# Map config `name` -> Strategy class. Register new strategies here.
_REGISTRY: dict[str, type[Strategy]] = {
    TrendFollowing.name: TrendFollowing,
    MeanReversion.name: MeanReversion,
}

# Register all 26 Kraken strategies
for strat_cls in KRAKEN_STRATEGIES:
    if strat_cls.name not in _REGISTRY:
        _REGISTRY[strat_cls.name] = strat_cls


def _mode_from_entry(entry: dict) -> str:
    if entry.get("mode") in Strategy.MODES:
        return entry["mode"]
    # Back-compat with the older `enabled` flag. New strategies start in shadow
    # so they are vetted before ever touching the real account.
    return "shadow" if entry.get("enabled", True) else "off"


class StrategyRegistry:
    def __init__(self, strategies: list[Strategy], path: Path | None = None):
        self.strategies = strategies
        self.path = path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategyRegistry":
        p = Path(path)
        raw: dict = {"strategies": []}
        if p.exists():
            try:
                raw = yaml.safe_load(p.read_text()) or {"strategies": []}
            except yaml.YAMLError as e:
                # A corrupt state file must not crash-loop the boot (audit F-13).
                log.error("Strategy state %s is corrupt (%s); starting with an "
                          "empty registry — restore from config/strategies.yaml "
                          "or the dashboard.", p, e)
        built: list[Strategy] = []
        for entry in raw.get("strategies", []):
            cls_ = _REGISTRY.get(entry["name"])
            if cls_ is None:
                log.warning("Unknown strategy '%s' in config; skipping.", entry["name"])
                continue
            built.append(
                cls_(
                    params=entry.get("params", {}),
                    symbols=entry.get("symbols", []),
                    mode=_mode_from_entry(entry),
                    timeframe=entry.get("timeframe", "5m"),
                    excluded_symbols=entry.get("excluded_symbols", []),
                )
            )
        log.info("Loaded %d strategies", len(built))
        return cls(built, p)

    def active(self) -> list[Strategy]:
        """Strategies that are not turned off."""
        return [s for s in self.strategies if s.enabled]

    def all(self) -> list[Strategy]:
        return self.strategies

    def get(self, name: str) -> Strategy | None:
        return next((s for s in self.strategies if s.name == name), None)

    def set_mode(self, name: str, mode: str) -> bool:
        strat = self.get(name)
        if strat is None or mode not in Strategy.MODES:
            return False
        strat.mode = mode
        log.info("Strategy %s mode -> %s", name, mode)
        self.save()
        return True

    def save(self) -> None:
        """Persist current mode + params back to YAML so changes survive restarts."""
        if self.path is None:
            return
        payload = {
            "strategies": [
                {
                    "name": s.name,
                    "mode": s.mode,
                    "enabled": s.enabled,
                    "symbols": s.symbols,
                    "excluded_symbols": s.excluded_symbols,
                    "timeframe": s.timeframe,
                    "params": s.params,
                }
                for s in self.strategies
            ]
        }
        # Atomic write (tmp + rename): a crash mid-write must never leave a
        # truncated file that blocks the next boot (audit F-13).
        import tempfile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=self.path.parent, delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(yaml.safe_dump(payload, sort_keys=False))
            tmp_path = tmp.name
        Path(tmp_path).replace(self.path)
        log.info("Saved strategy state to %s", self.path)
