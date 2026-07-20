"""RL confidence-scaled sizing math + MAE/MFE excursion tracking."""

from __future__ import annotations

import asyncio

from gungnir.data.models import Order, Side
from gungnir.execution.broker import PaperBroker


# ── MAE/MFE tracking (PaperBroker) ────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def test_mark_tracks_excursions_and_close_normalizes_to_r():
    b = PaperBroker()
    b.mark("US500", 100.0)
    order = Order(symbol="US500", side=Side.BUY, volume=1.0,
                  stop_loss=98.0, take_profit=106.0, client_id="s:US500:1")
    _run(b.submit(order))

    b.mark("US500", 103.0)    # +3 favorable
    b.mark("US500", 99.0)     # -1 adverse
    b.mark("US500", 101.0)    # neither extreme moves

    pos = b.position("US500", "s")
    assert pos.context["mfe"] == 3.0
    assert pos.context["mae"] == -1.0

    closed = _run(b.close("US500", "s"))
    # Risk = |100 - 98| = 2 → MFE 3/2 = 1.5R, MAE -1/2 = -0.5R.
    assert closed.context["mfe_r"] == 1.5
    assert closed.context["mae_r"] == -0.5


def test_sell_side_excursions_are_direction_correct():
    b = PaperBroker()
    b.mark("EURUSD", 1.1000)
    order = Order(symbol="EURUSD", side=Side.SELL, volume=100.0,
                  stop_loss=1.1100, client_id="s:EURUSD:1")
    _run(b.submit(order))
    b.mark("EURUSD", 1.0900)   # price down = favorable for a short
    b.mark("EURUSD", 1.1050)   # price up = adverse
    pos = b.position("EURUSD", "s")
    assert pos.context["mfe"] > 0     # ~ +0.01
    assert pos.context["mae"] < 0     # ~ -0.005


# ── Confidence-scaled sizing math (mirrors the agent's formula) ──────────────

def _scale(p_take, thr=0.5, floor=0.5):
    edge = (p_take - thr) / max(1e-9, 1.0 - thr)
    return floor + (1.0 - floor) * min(max(edge, 0.0), 1.0)


def test_confidence_scale_bounds():
    assert _scale(0.5) == 0.5          # at threshold → floor
    assert _scale(1.0) == 1.0          # full conviction → full size
    assert _scale(0.75) == 0.75        # linear in between
    assert _scale(0.3) == 0.5          # below threshold clamps to floor
    assert _scale(2.0) == 1.0          # above 1 clamps to full


def test_confidence_scale_never_upsizes():
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        assert _scale(p) <= 1.0
