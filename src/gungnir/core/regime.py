"""Market regime classification: the context every learning layer conditions on.

Four regimes from two orthogonal, deterministic axes:

    trend strength (ADX)  ×  volatility (ATR/price percentile vs own history)

    trend_low    trending, calm      — trend-followers' best conditions
    trend_high   trending, violent   — trends exist but stops get run
    range_low    ranging, calm       — mean-reversion's home turf
    range_high   ranging, violent    — chop; most strategies' worst regime

Deliberately simple: two thresholds a human can sanity-check, computed from
data the features already carry. Sophistication belongs in what the system
*does* with the regime (per-regime strategy stats, allocation weights), not
in the label itself. Every trade is stamped with its entry regime so the
journal accumulates regime-conditional evidence automatically.
"""

from __future__ import annotations

REGIMES = ("trend_low", "trend_high", "range_low", "range_high")

ADX_TREND = 25.0          # ADX at/above → trending
VOL_HIGH_PCTILE = 0.70    # ATR% above own 70th percentile → high vol


def classify(adx: float, vol_percentile: float) -> str:
    """Label the regime from trend strength and relative volatility.

    ``vol_percentile`` is where the current ATR/price sits within this
    symbol's own recent history (0..1) — volatility is always relative to
    the instrument, never an absolute threshold shared across FX and crypto.
    """
    trending = (adx or 0.0) >= ADX_TREND
    high_vol = (vol_percentile or 0.0) >= VOL_HIGH_PCTILE
    if trending:
        return "trend_high" if high_vol else "trend_low"
    return "range_high" if high_vol else "range_low"


def vol_percentile(history: list[float], current: float) -> float:
    """Fraction of recent ATR% readings at or below the current one."""
    if not history:
        return 0.5     # no history yet → neutral
    below = sum(1 for v in history if v <= current)
    return below / len(history)
