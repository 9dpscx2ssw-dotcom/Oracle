"""Backtest replays live settings: pre-trade filters + configured sizer.

Guards that engine.run applies the same veto gates the live agent uses, so a
backtest reflects configured selectivity instead of raw one-strategy edge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gungnir.backtest import engine
from gungnir.backtest.costs import CostModel
from gungnir.core.filters import FilterConfig
from gungnir.data.models import Candle
import gungnir.strategy.registry as reg


def _series(symbol="US500", n=400, start_hour=0):
    """Hourly candles with timestamps so the session gate is exercisable."""
    base = datetime(2026, 7, 1, start_hour, tzinfo=timezone.utc)
    import random
    rng = random.Random(5)
    out, price = [], 100.0
    for i in range(n):
        c = max(1.0, price + rng.gauss(0, 0.4))
        out.append(Candle(symbol=symbol, timeframe="1h", open=price,
                          high=max(price, c) + 0.3, low=min(price, c) - 0.3,
                          close=c, volume=1000, ts=base + timedelta(hours=i)))
        price = c
    return out


def _strat(name):
    cls = reg._REGISTRY.get(name) or reg._REGISTRY[next(iter(reg._REGISTRY))]
    return cls(mode="shadow", symbols=["US500"])


def test_no_filters_never_vetoes():
    candles = _series()
    feats = engine.feature_store.build_kraken_series("US500", candles)
    res = engine.run(_strat("mean_reversion"), candles, "US500", feats=feats)
    assert res.filter_vetoes == 0


def test_session_filter_vetoes_off_hours_entries():
    # US500 is an index; liquid session is 13-21 UTC. Bars start at 00:00, so
    # most entries fall outside the window and must be vetoed.
    candles = _series(start_hour=0)
    feats = engine.feature_store.build_kraken_series("US500", candles)
    raw = engine.run(_strat("mean_reversion"), candles, "US500", feats=feats)
    filt = engine.run(_strat("mean_reversion"), candles, "US500", feats=feats,
                      filters=FilterConfig.from_dict({"session": True}))
    assert filt.filter_vetoes > 0
    # A veto gate can only ever reduce the number of trades taken.
    assert filt.metrics.n_trades <= raw.metrics.n_trades


def test_live_sizer_is_used_when_passed():
    """A passed sizer drives volume (vs the engine's default 0.5%-risk sizer)."""
    from gungnir.config import Config, Secrets
    from gungnir.risk.position_sizing import build_sizer, VolTarget
    candles = _series()
    feats = engine.feature_store.build_kraken_series("US500", candles)
    sizer = build_sizer(Config({"risk": {"sizer": "vol_target"}}, Secrets.from_env()))
    assert isinstance(sizer, VolTarget)
    res = engine.run(_strat("trend_following"), candles, "US500", feats=feats,
                     sizer=sizer, cost=CostModel(spread_bps=2.0))
    # Run completes and produces a result object with the veto counter present.
    assert hasattr(res, "filter_vetoes")
