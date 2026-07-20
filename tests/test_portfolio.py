"""Tests for the portfolio risk gate (PortfolioRisk.vet).

These guard the capital-protection layer: the exposure/position caps only work if
`open_exposure` is populated (the agent refreshes it each loop), and the
daily-drawdown breaker must halt new entries. Regression coverage for the audit
finding that these limits were inert.
"""

from __future__ import annotations

from gungnir.config import Config, Secrets
from gungnir.data.models import Side, Signal
from gungnir.risk.portfolio import PortfolioRisk


def _risk(**raw) -> PortfolioRisk:
    cfg = Config({"risk": raw}, Secrets.from_env())
    r = PortfolioRisk(cfg)
    r.equity = 10_000.0
    r.day_start_equity = 10_000.0
    return r


def _sig(symbol="EURUSD", side=Side.BUY) -> Signal:
    return Signal(strategy="s", symbol=symbol, side=side, conviction=0.8)


def test_flat_or_nonpositive_volume_rejected():
    r = _risk()
    assert r.vet(_sig(side=Side.FLAT), 1.0, 1.10, 0.01) is None
    assert r.vet(_sig(), 0.0, 1.10, 0.01) is None


def test_daily_drawdown_halts_new_entries():
    r = _risk(max_daily_drawdown=0.03)
    r.equity = 9_600.0          # -4% on the day, past the 3% limit
    assert r.trading_halted() is True
    assert r.vet(_sig(), 1.0, 1.10, 0.01) is None


def test_max_positions_blocks_new_symbol_when_full():
    r = _risk(max_open_positions=2)
    r.open_exposure = {"AAA": 100.0, "BBB": 100.0}      # already at the cap
    assert r.vet(_sig(symbol="CCC"), 1.0, 1.10, 0.01) is None      # new symbol blocked
    # An existing symbol may still be (re)sized.
    assert r.vet(_sig(symbol="AAA"), 1.0, 1.10, 0.01) is not None


def test_per_asset_cap_shrinks_order():
    r = _risk(max_per_asset_exposure=0.5, max_portfolio_exposure=10.0)
    # cap = 0.5 * 10_000 = 5_000 notional; price 100 -> max 50 units. US500 is an
    # index (min_lot 0.01), so the shrunk 50 units stays well above the floor.
    order = r.vet(_sig(symbol="US500"), raw_volume=100.0, price=100.0, atr=1.0)
    assert order is not None
    assert order.volume <= 50.0 + 1e-9


def test_gross_exposure_cap_shrinks_order():
    r = _risk(max_portfolio_exposure=1.0, max_per_asset_exposure=10.0)
    r.open_exposure = {"AAA": 9_500.0}                  # gross cap = 10_000
    order = r.vet(_sig(symbol="US500"), raw_volume=100.0, price=100.0, atr=1.0)
    assert order is not None
    assert order.volume * 100.0 <= 500.0 + 1e-6         # only 500 notional headroom


def test_max_lot_caps_and_min_lot_floors():
    r = _risk(max_per_asset_exposure=10.0, max_portfolio_exposure=10.0)
    r.max_lot = 2.0
    r.min_lot = 0.01                                    # coherent floor below max_lot
    order = r.vet(_sig(), raw_volume=100.0, price=1.0, atr=0.01)
    assert order is not None and order.volume <= 2.0

    r2 = _risk(max_per_asset_exposure=10.0, max_portfolio_exposure=10.0)
    r2.min_lot = 5.0
    order2 = r2.vet(_sig(), raw_volume=1.0, price=1.0, atr=0.01)
    assert order2 is not None and order2.volume >= 5.0


def test_sub_minimum_order_rejected_not_sent(monkeypatch):
    """The fix: when the caps leave less headroom than the broker's minimum deal
    size, the order is REJECTED — not shrunk to a sub-minimum (or 0.0000) lot
    that the broker 400s on. This was the US100=0.001 / vol=0.0000 bug.
    """
    r = _risk(max_per_asset_exposure=0.5, max_portfolio_exposure=10.0)
    # Per-asset cap = 5_000; US100 already holds 4_999 → only $1 of headroom.
    r.open_exposure = {"US100": 4_999.0}
    # Broker minimum deal size for US100 is 0.1; $1 headroom at ~29k = ~0.00003
    # lots, far below it. Old code sent that (rounded to 0.0000); now → None.
    order = r.vet(_sig(symbol="US100"), raw_volume=1.0, price=29_000.0, atr=5.0,
                  instrument_min=0.1)
    assert order is None


def test_order_never_below_instrument_min_when_accepted():
    """Any order that IS returned respects the broker's minimum deal size."""
    r = _risk(max_per_asset_exposure=10.0, max_portfolio_exposure=10.0)
    order = r.vet(_sig(symbol="US100"), raw_volume=0.05, price=100.0, atr=1.0,
                  instrument_min=0.1)
    # 0.05 requested < 0.1 broker min, but the floored 0.1 fits the (ample) caps,
    # so it's rounded UP to exactly the minimum rather than rejected.
    assert order is not None and order.volume >= 0.1
