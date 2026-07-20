"""Crash protections: book-aware drawdown breakers, the intraday-from-peak
breaker, and the loss-streak cooldown.

Guards the fixes for the sustained-downtrend incident: 500+ losing shadow
trades over 16h with ZERO halts, because (a) the breakers only watched the
real account while every fill landed on the shadow book, (b) nothing measured
drawdown from the day's peak, and (c) stopped-out strategies re-entered
against the trend every bar with no cooldown.
"""

from __future__ import annotations

from gungnir.config import Config, Secrets
from gungnir.data.models import Side, Signal
from gungnir.risk.cooldown import LossStreakGuard
from gungnir.risk.portfolio import DrawdownTracker, PortfolioRisk


def _risk(**raw) -> PortfolioRisk:
    r = PortfolioRisk(Config({"risk": raw}, Secrets.from_env()))
    r.update_book("real", 10_000.0)
    r.update_book("shadow", 10_000.0)
    r.roll_day()
    return r


def _sig(symbol="EURUSD") -> Signal:
    return Signal(strategy="s", symbol=symbol, side=Side.BUY, conviction=0.8)


# ── book-aware breakers ──────────────────────────────────────────────────────

def test_shadow_book_bleed_halts_shadow_but_not_real():
    r = _risk(max_daily_drawdown=0.03)
    r.update_book("shadow", 9_500.0)          # -5% on the shadow book only
    assert r.trading_halted("shadow") is True
    assert r.trading_halted("real") is False
    assert r.vet(_sig(), 1.0, 1.10, 0.01, book="shadow") is None
    assert r.vet(_sig(), 1.0, 1.10, 0.01, book="real") is not None


def test_real_book_halt_does_not_bench_shadow():
    r = _risk(max_daily_drawdown=0.03)
    r.update_book("real", 9_500.0)
    assert r.trading_halted("real") is True
    assert r.vet(_sig(), 1.0, 1.10, 0.01, book="shadow") is not None


def test_day_roll_resets_both_books():
    r = _risk(max_daily_drawdown=0.03)
    r.update_book("shadow", 9_500.0)
    assert r.trading_halted("shadow") is True
    r.roll_day()                              # new UTC session: today's baseline
    assert r.trading_halted("shadow") is False


# ── intraday peak-to-trough breaker ─────────────────────────────────────────

def test_crash_after_runup_trips_intraday_breaker_not_daily():
    # Day starts at 10k, runs to 11k, crashes to 10.4k: only -~0% vs the day
    # start (daily breaker blind) but -5.5% from the day's peak.
    r = _risk(max_daily_drawdown=0.03, max_intraday_drawdown=0.05)
    r.update_book("real", 11_000.0)
    r.update_book("real", 10_400.0)
    assert r.trading_halted("real") is True
    assert r.vet(_sig(), 1.0, 1.10, 0.01, book="real") is None


def test_intraday_breaker_disabled_when_zero():
    r = _risk(max_daily_drawdown=0.5, max_intraday_drawdown=0)
    r.update_book("real", 11_000.0)
    r.update_book("real", 10_400.0)
    assert r.trading_halted("real") is False


def test_tracker_breach_reports_first_reason():
    t = DrawdownTracker()
    t.update(100.0)
    t.roll_day()
    t.update(80.0)
    assert "daily drawdown" in t.breach(0.03, 0.05, 0.25)
    assert t.breach(0.5, 0.5, 0.5) == ""


# ── loss-streak cooldown ─────────────────────────────────────────────────────

def test_three_straight_losses_bench_the_pairing():
    g = LossStreakGuard(max_streak=3, cooldown_minutes=60)
    for _ in range(3):
        g.record("mr", "US500", -50.0, now=1000.0)
    assert g.blocked_seconds("mr", "US500", now=1001.0) > 0
    # Other strategies / symbols are unaffected.
    assert g.blocked_seconds("mr", "GOLD", now=1001.0) == 0
    assert g.blocked_seconds("trend", "US500", now=1001.0) == 0


def test_win_resets_the_streak():
    g = LossStreakGuard(max_streak=3, cooldown_minutes=60)
    g.record("mr", "US500", -50.0, now=1000.0)
    g.record("mr", "US500", -50.0, now=1001.0)
    g.record("mr", "US500", +10.0, now=1002.0)     # winner: streak cleared
    g.record("mr", "US500", -50.0, now=1003.0)
    assert g.blocked_seconds("mr", "US500", now=1004.0) == 0


def test_cooldown_expires_then_one_probe_rebenches_on_loss():
    g = LossStreakGuard(max_streak=3, cooldown_minutes=1)
    for _ in range(3):
        g.record("mr", "US500", -50.0, now=1000.0)
    assert g.blocked_seconds("mr", "US500", now=1010.0) > 0       # benched
    assert g.blocked_seconds("mr", "US500", now=1000.0 + 61) == 0  # served
    # The probe re-entry loses again → benched immediately, not after 3 more.
    g.record("mr", "US500", -50.0, now=1000.0 + 62)
    assert g.blocked_seconds("mr", "US500", now=1000.0 + 63) > 0


def test_disabled_guard_never_blocks():
    g = LossStreakGuard(max_streak=0, cooldown_minutes=0)
    for _ in range(10):
        g.record("mr", "US500", -50.0, now=1000.0)
    assert g.blocked_seconds("mr", "US500", now=1001.0) == 0
