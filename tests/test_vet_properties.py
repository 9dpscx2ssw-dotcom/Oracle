"""Property-based invariants for PortfolioRisk.vet() — the last gate.

The audit history here is exactly the class of bug Hypothesis finds: the
minimum-lot floor re-inflating capped orders (F-01), leverage multiplying
position size (F-04). These properties must hold for ANY input, not just the
examples a human thought of.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from gungnir.config import Config, Secrets
from gungnir.data.models import Side, Signal
from gungnir.risk.portfolio import PortfolioRisk

_EPS = 1e-6


def _risk(equity: float, max_per_asset: float = 0.5, max_gross: float = 2.0,
          leverage: float = 30.0) -> PortfolioRisk:
    r = PortfolioRisk(Config({"risk": {
        "max_per_asset_exposure": max_per_asset,
        "max_portfolio_exposure": max_gross,
        "max_open_positions": 50,
        "max_daily_drawdown": 0.99,
        "max_intraday_drawdown": 0,
        "max_total_drawdown": 0,
        "min_confidence": 0.0,
        "leverage": leverage,
        "leverage_by_type": {},          # scalar leverage for every class
        "min_lot_by_type": {"forex": 0.0, "stocks": 0.0, "indices": 0.0,
                            "commodities": 0.0, "crypto": 0.0},
    }}, Secrets.from_env()))
    r.update_book("real", equity)
    r.update_book("shadow", equity)
    r.roll_day()
    return r


def _sig(conviction: float = 0.9, side: Side = Side.BUY) -> Signal:
    return Signal(strategy="s", symbol="EURUSD", side=side, conviction=conviction)


@settings(max_examples=200, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=1e7),
    raw_volume=st.floats(min_value=0.0, max_value=1e9),
    price=st.floats(min_value=0.0001, max_value=1e6),
    atr=st.floats(min_value=0.0, max_value=1e4),
)
def test_vet_never_exceeds_per_asset_cap(equity, raw_volume, price, atr):
    r = _risk(equity)
    order = r.vet(_sig(), raw_volume, price, atr)
    if order is not None:
        notional = order.volume * price
        assert notional <= r.max_per_asset * equity * (1 + _EPS)


@settings(max_examples=200, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=1e7),
    raw_volume=st.floats(min_value=0.0, max_value=1e9),
    price=st.floats(min_value=0.0001, max_value=1e6),
    existing=st.floats(min_value=0.0, max_value=1e7),
)
def test_vet_respects_gross_cap_with_existing_exposure(equity, raw_volume, price, existing):
    r = _risk(equity)
    r.open_exposure = {"US500": existing}
    order = r.vet(_sig(), raw_volume, price, 0.01)
    if order is not None:
        gross_after = existing + order.volume * price
        assert gross_after <= r.max_gross * equity * (1 + _EPS)


@settings(max_examples=200, deadline=None)
@given(
    equity=st.floats(min_value=100.0, max_value=1e7),
    raw_volume=st.floats(min_value=0.0, max_value=1e9),
    price=st.floats(min_value=0.0001, max_value=1e6),
    leverage=st.floats(min_value=1.0, max_value=500.0),
)
def test_vet_never_exceeds_margin_capacity(equity, raw_volume, price, leverage):
    r = _risk(equity, max_per_asset=100.0, max_gross=1000.0, leverage=leverage)
    order = r.vet(_sig(), raw_volume, price, 0.01)
    if order is not None:
        margin_cap = equity * (leverage / (1.0 + r.lev_safety))
        assert order.volume * price <= margin_cap * (1 + _EPS)


@settings(max_examples=200, deadline=None)
@given(
    raw_volume=st.floats(min_value=-1e6, max_value=1e9, allow_nan=False),
    price=st.floats(min_value=0.0001, max_value=1e6),
    conviction=st.floats(min_value=0.0, max_value=1.0),
)
def test_vet_output_is_positive_and_finite(raw_volume, price, conviction):
    import math
    r = _risk(10_000.0)
    order = r.vet(_sig(conviction=conviction), raw_volume, price, 0.01)
    if order is not None:
        assert order.volume > 0
        assert math.isfinite(order.volume)


@settings(max_examples=100, deadline=None)
@given(
    raw_volume=st.floats(min_value=0.001, max_value=1e6),
    price=st.floats(min_value=0.01, max_value=1e5),
    instrument_min=st.floats(min_value=0.0, max_value=100.0),
)
def test_min_lot_floor_never_breaches_caps(raw_volume, price, instrument_min):
    """The F-01 regression as a property: rounding a sub-minimum volume UP to
    the broker floor must still respect every cap — or reject."""
    r = _risk(10_000.0)
    order = r.vet(_sig(), raw_volume, price, 0.01, instrument_min=instrument_min)
    if order is not None:
        assert order.volume * price <= r.max_per_asset * 10_000.0 * (1 + _EPS)
        if instrument_min:
            assert order.volume >= instrument_min - _EPS


@settings(max_examples=100, deadline=None)
@given(equity_drop=st.floats(min_value=0.0, max_value=0.99))
def test_halted_book_never_emits_orders(equity_drop):
    """Once a breaker trips, vet() must return None regardless of the order."""
    r = _risk(10_000.0)
    r.max_daily_dd = 0.05
    r.update_book("real", 10_000.0 * (1.0 - equity_drop))
    order = r.vet(_sig(), 100.0, 1.10, 0.01, book="real")
    if equity_drop >= 0.05:
        assert order is None
    # (below the threshold, either outcome is legitimate — caps may still bind)
