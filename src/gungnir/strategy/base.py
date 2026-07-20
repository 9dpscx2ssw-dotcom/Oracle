"""Strategy interface.

A Strategy is a pure function of (FeatureSet) -> list[Signal], parameterized by a
mutable `params` dict that the learning layer may update over time. Keeping
strategies deterministic and side-effect-free makes them testable and lets the
optimizer backtest parameter changes against the journal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.models import Signal
from ..features.feature_store import FeatureSet


class Strategy(ABC):
    name: str = "base"

    # Correlation family for consensus voting (core/aggregator.py): strategies
    # sharing an indicator mechanism vote as one capped bloc, so clone count
    # can't manufacture consensus. Empty ⇒ the strategy is its own family.
    family: str = ""

    # Runtime mode, toggled from the dashboard:
    #   off    — strategy is dormant, emits no signals
    #   shadow — emits signals and "paper" trades them, but never touches the
    #            real account (use this to vet a strategy before going live)
    #   live   — trades the real account (only when not in global dry-run)
    MODES = ("off", "shadow", "live")

    # Tunable-parameter contract. DEFAULTS holds every parameter this strategy
    # reads (via ``p()``) with its out-of-the-box value; BOUNDS holds the
    # optimizer's (min, max) search range per parameter. A strategy with empty
    # BOUNDS is invisible to the Bayesian/walk-forward tuning loop.
    DEFAULTS: dict[str, float] = {}
    BOUNDS: dict[str, tuple[float, float]] = {}

    def __init__(
        self,
        params: dict | None = None,
        symbols: list[str] | None = None,
        mode: str = "shadow",
        timeframe: str = "5m",
        excluded_symbols: list[str] | None = None,
    ):
        self.params = params or {}
        self.symbols = symbols or []
        # Learned per-symbol blacklist: symbols this strategy has demonstrably
        # lost on (pruned by the slow loop). Overrides the opt-in scope above.
        self.excluded_symbols = excluded_symbols or []
        self.mode = mode if mode in self.MODES else "shadow"
        self.timeframe = timeframe

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @abstractmethod
    def generate(self, features: FeatureSet) -> list[Signal]:
        """Emit zero or more trade intents for this asset."""

    def p(self, key: str) -> float:
        """Read a tunable parameter, falling back to the class default."""
        try:
            return float(self.params.get(key, self.DEFAULTS.get(key, 0.0)))
        except (TypeError, ValueError):
            return float(self.DEFAULTS.get(key, 0.0))

    def update_params(self, updates: dict) -> None:
        """Apply (already-gated) parameter changes from the learning layer."""
        self.params.update(updates)

    def get_parameter_bounds(self) -> dict:
        """Optimization bounds {param: (min, max)} for tunable parameters."""
        return dict(self.BOUNDS)

    def trades_symbol(self, symbol: str) -> bool:
        if symbol in self.excluded_symbols:
            return False
        return not self.symbols or symbol in self.symbols
