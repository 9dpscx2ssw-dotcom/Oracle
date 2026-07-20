"""Smart filtering: sentiment-based position sizing and market regime filtering (Phase 1)."""

from __future__ import annotations

from gungnir.core import filters
from gungnir.data.models import Sentiment, Side, Signal
from gungnir.risk.portfolio import PortfolioRisk
from gungnir.config import Config, Secrets


def _signal(side=Side.BUY) -> Signal:
    return Signal(strategy="test_strat", symbol="EURUSD", side=side,
                  conviction=0.7)


def _sentiment(score: float, confidence: float) -> Sentiment:
    return Sentiment(symbol="EURUSD", score=score, confidence=confidence,
                     rationale="test")


def test_market_regime_filter_allows_aligned_signals():
    """Signal + sentiment agree → allow."""
    signal = _signal(Side.BUY)
    sentiment = _sentiment(score=0.8, confidence=0.8)  # Bullish

    ok, why = filters.market_regime_filter(signal, sentiment)
    assert ok is True
    assert why is None


def test_market_regime_filter_blocks_conflicting_signals():
    """Signal conflicts with sentiment confidence → veto."""
    signal = _signal(Side.SELL)
    sentiment = _sentiment(score=0.8, confidence=0.8)  # Bullish, but we're selling

    ok, why = filters.market_regime_filter(signal, sentiment)
    assert ok is False
    assert why == "sentiment_euphoria"


def test_market_regime_filter_ignores_weak_sentiment():
    """Weak sentiment (confidence < 0.5) → no veto."""
    signal = _signal(Side.SELL)
    sentiment = _sentiment(score=0.8, confidence=0.3)  # Bullish but weak

    ok, why = filters.market_regime_filter(signal, sentiment)
    assert ok is True  # No veto on weak signal
    assert why is None


def test_market_regime_filter_allows_without_sentiment():
    """No sentiment → allow."""
    signal = _signal(Side.BUY)
    ok, why = filters.market_regime_filter(signal, None)
    assert ok is True
    assert why is None


def test_sentiment_scales_position_up_on_agreement():
    """When signal + sentiment agree, size up."""
    cfg = Config({"risk": {}}, Secrets())
    risk = PortfolioRisk(cfg)

    signal = _signal(Side.BUY)
    sentiment = _sentiment(score=0.9, confidence=0.8)  # Very bullish

    base_volume = 100.0
    scaled = risk.scale_with_sentiment(base_volume, signal, sentiment)

    # Strong agreement → scale up
    assert scaled > base_volume
    assert scaled <= base_volume * 1.3  # Max 30% boost


def test_sentiment_scales_position_down_on_disagreement():
    """When signal contradicts sentiment, size down."""
    cfg = Config({"risk": {}}, Secrets())
    risk = PortfolioRisk(cfg)

    signal = _signal(Side.BUY)
    sentiment = _sentiment(score=-0.9, confidence=0.8)  # Very bearish

    base_volume = 100.0
    scaled = risk.scale_with_sentiment(base_volume, signal, sentiment)

    # Strong disagreement → scale down
    assert scaled < base_volume
    assert scaled >= base_volume * 0.5  # Min 50% of base


def test_sentiment_scales_no_adjustment_on_weak_sentiment():
    """Weak sentiment → no scaling."""
    cfg = Config({"risk": {}}, Secrets())
    risk = PortfolioRisk(cfg)

    signal = _signal(Side.BUY)
    sentiment = _sentiment(score=0.9, confidence=0.3)  # Bullish but weak

    base_volume = 100.0
    scaled = risk.scale_with_sentiment(base_volume, signal, sentiment)

    # Weak sentiment → no adjustment
    assert scaled == base_volume


def test_sentiment_scales_no_adjustment_without_sentiment():
    """No sentiment → no scaling."""
    cfg = Config({"risk": {}}, Secrets())
    risk = PortfolioRisk(cfg)

    signal = _signal(Side.BUY)
    base_volume = 100.0
    scaled = risk.scale_with_sentiment(base_volume, signal, None)

    assert scaled == base_volume
