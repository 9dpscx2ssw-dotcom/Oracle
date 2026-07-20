"""Reinforcement-learning decision layer.

The strategies generate hard signals from chart analysis; this package decides
which of those signals are actually worth taking, and learns from the outcome of
every decision — taken *or* skipped. See :mod:`policy` for the agent-facing API.
"""

from __future__ import annotations

from .policy import SKIP, TAKE, Decision, RLPolicy
from .state import FEATURE_NAMES, STATE_DIM

__all__ = ["RLPolicy", "Decision", "SKIP", "TAKE", "STATE_DIM", "FEATURE_NAMES"]
