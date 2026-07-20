"""Sizers must ignore leverage (audit F-04): leverage is a margin constraint
enforced as a max-notional cap in PortfolioRisk.vet(), not a size multiplier —
at 200x the old multiplier turned a 0.5%-risk trade into ~90% of equity."""

from __future__ import annotations

import pytest

from gungnir.config import Config, Secrets
from gungnir.data.models import Side, Signal
from gungnir.features.feature_store import FeatureSet
from gungnir.risk.position_sizing import build_sizer
from gungnir.risk.portfolio import PortfolioRisk


def _cfg(**risk) -> Config:
    return Config({"risk": risk}, Secrets.from_env())


def _feat(atr=1.0, price=100.0) -> FeatureSet:
    return FeatureSet(symbol="US500", last_price=price, atr=atr)


def _sig(conv=1.0) -> Signal:
    return Signal(strategy="s", symbol="US500", side=Side.BUY, conviction=conv)


@pytest.mark.parametrize("sizer_name", ["fixed_fractional", "vol_target", "kelly"])
def test_leverage_does_not_multiply_size(sizer_name):
    lo = build_sizer(_cfg(sizer=sizer_name, leverage=1.0))
    hi = build_sizer(_cfg(sizer=sizer_name, leverage=200.0))
    v_lo = lo.size(_sig(), _feat(), equity=10_000.0)
    v_hi = hi.size(_sig(), _feat(), equity=10_000.0)
    assert v_hi == pytest.approx(v_lo), (
        f"{sizer_name}: leverage changed the sized volume ({v_lo} → {v_hi})")


def test_fixed_fractional_risks_the_configured_fraction():
    sizer = build_sizer(_cfg(sizer="fixed_fractional", account_risk_per_trade=0.005,
                             stop_atr_mult=2.0, leverage=200.0))
    vol = sizer.size(_sig(conv=1.0), _feat(atr=1.0), equity=10_000.0)
    # risk_cash = 10_000 * 0.005 = $50; stop distance = 2*ATR = 2.0 → 25 units.
    assert vol == pytest.approx(25.0)


def test_vet_caps_notional_at_margin_capacity():
    cfg = _cfg(max_per_asset_exposure=100.0, max_portfolio_exposure=100.0,
               leverage=2.0, leverage_safety_margin=0.0)
    r = PortfolioRisk(cfg)
    r.equity = 10_000.0
    r.day_start_equity = 10_000.0
    # Sizer wants $1M notional; margin capacity at 2x is $20k → 200 units @ $100.
    order = r.vet(_sig(), raw_volume=10_000.0, price=100.0, atr=1.0)
    assert order is not None
    assert order.volume * 100.0 <= 20_000.0 + 1e-6


def test_margin_cap_uses_asset_class_leverage():
    """Capital.com grants 200:1 on indices but only 20:1 on crypto — the
    per-order margin cap must use the class the symbol belongs to."""
    from gungnir.data.models import Signal, Side

    cfg = _cfg(max_per_asset_exposure=1000.0, max_portfolio_exposure=1000.0,
               leverage=1.0, leverage_safety_margin=0.0,
               leverage_by_type={"indices": 200.0, "crypto": 20.0},
               min_lot_by_type={"indices": 0.001, "crypto": 0.001})
    r = PortfolioRisk(cfg)
    r.equity = 1_000.0
    r.day_start_equity = 1_000.0

    idx = Signal(strategy="s", symbol="US500", side=Side.BUY, conviction=0.8)
    cry = Signal(strategy="s", symbol="BTCUSD", side=Side.BUY, conviction=0.8)
    huge = 1e9
    o_idx = r.vet(idx, raw_volume=huge, price=100.0, atr=1.0)
    o_cry = r.vet(cry, raw_volume=huge, price=100.0, atr=1.0)
    assert o_idx is not None and o_cry is not None
    assert o_idx.volume * 100.0 <= 200_000.0 + 1e-6     # 200x on indices
    assert o_cry.volume * 100.0 <= 20_000.0 + 1e-6      # 20x on crypto
    assert o_cry.volume < o_idx.volume
