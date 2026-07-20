"""Phase-1 capital protections: catastrophe stops on net positions, breaker and
compliance state that survive restarts, and the order-idempotency ledger.

These guard the three worst live failure modes found in the engineering audit:
a dead agent riding unprotected exposure, breakers re-arming lower after every
restart, and a timed-out POST leaving a duplicate unmanaged position.
"""

from __future__ import annotations

import json

from gungnir.config import Config, Secrets
from gungnir.core.compliance import PreTradeCompliance
from gungnir.data.models import Order, Side
from gungnir.execution.broker import PaperBroker
from gungnir.execution.netting import NET_TAG, NettingBroker
from gungnir.risk.portfolio import PortfolioRisk


def _order(symbol="EURUSD", side=Side.BUY, vol=1.0, strategy="s1",
           stop=None, tp=None) -> Order:
    return Order(symbol=symbol, side=side, volume=vol, stop_loss=stop,
                 take_profit=tp, client_id=f"{strategy}:{symbol}:1")


# ── catastrophe stop on net positions ────────────────────────────────────────

async def test_net_order_carries_catastrophe_stop():
    account = PaperBroker()
    nb = NettingBroker(account, catastrophe_stop_mult=1.5)
    nb.mark("EURUSD", 1.1000)
    # Virtual stop 50 pips below entry → catastrophe stop 75 pips below mark.
    await nb.submit(_order(stop=1.0950))
    await nb.reconcile("EURUSD")
    net = account.position("EURUSD", NET_TAG)
    assert net is not None
    cat = (net.context or {}).get("stop")
    assert cat is not None
    assert abs(cat - (1.1000 - 1.5 * 0.0050)) < 1e-9


async def test_catastrophe_stop_pct_fallback_without_virtual_stops():
    account = PaperBroker()
    nb = NettingBroker(account, catastrophe_stop_mult=1.5, catastrophe_stop_pct=0.05)
    nb.mark("US500", 5000.0)
    await nb.submit(_order(symbol="US500", stop=None))
    await nb.reconcile("US500")
    net = account.position("US500", NET_TAG)
    assert (net.context or {}).get("stop") == 5000.0 * 0.95


async def test_catastrophe_stop_disabled_when_mult_zero():
    account = PaperBroker()
    nb = NettingBroker(account, catastrophe_stop_mult=0.0)
    nb.mark("EURUSD", 1.1000)
    await nb.submit(_order(stop=1.0950))
    await nb.reconcile("EURUSD")
    net = account.position("EURUSD", NET_TAG)
    assert (net.context or {}).get("stop") is None


async def test_catastrophe_stop_sell_side_sits_above_mark():
    account = PaperBroker()
    nb = NettingBroker(account, catastrophe_stop_mult=2.0)
    nb.mark("EURUSD", 1.1000)
    await nb.submit(_order(side=Side.SELL, stop=1.1050))
    await nb.reconcile("EURUSD")
    net = account.position("EURUSD", NET_TAG)
    assert net.side == Side.SELL
    assert (net.context or {}).get("stop") > 1.1000


# ── breaker-state persistence ────────────────────────────────────────────────

def _risk() -> PortfolioRisk:
    return PortfolioRisk(Config({"risk": {"max_total_drawdown": 0.25}},
                                Secrets.from_env()))


def test_breaker_peak_survives_restart(tmp_path):
    path = str(tmp_path / "risk_state.json")
    r1 = _risk()
    r1.update_book("real", 10_000.0)
    r1.roll_day()
    r1.update_book("real", 12_000.0)   # all-time peak
    r1.update_book("real", 9_800.0)    # -18% from peak: not yet tripped
    r1.save_state(path)

    r2 = _risk()                       # "restart"
    r2.load_state(path)
    assert r2.books["real"].peak == 12_000.0
    # A further slide past 25% from the RESTORED peak trips the total breaker.
    r2.update_book("real", 8_900.0)
    r2.books["real"].roll_day()        # fresh day baseline; total dd still binds
    assert r2.trading_halted("real") is True


def test_same_day_restart_restores_daily_baseline(tmp_path):
    path = str(tmp_path / "risk_state.json")
    r1 = _risk()
    r1.update_book("real", 10_000.0)
    r1.roll_day()
    r1.update_book("real", 11_000.0)   # intraday run-up
    r1.save_state(path)

    r2 = _risk()
    saved_day = r2.load_state(path)
    assert saved_day is not None       # same-day restart
    assert r2.books["real"].day_start == 10_000.0
    assert r2.books["real"].day_peak == 11_000.0


def test_missing_state_file_is_fine(tmp_path):
    r = _risk()
    assert r.load_state(str(tmp_path / "nope.json")) is None


# ── compliance-counter persistence ───────────────────────────────────────────

def _compliance(tmp_path, max_per_day=5) -> PreTradeCompliance:
    return PreTradeCompliance(Config({"compliance": {
        "max_orders_per_day": max_per_day,
        "state_path": str(tmp_path / "compliance_state.json"),
    }}, Secrets.from_env()))


def test_order_budget_survives_restart(tmp_path):
    c1 = _compliance(tmp_path, max_per_day=5)
    for _ in range(5):
        c1.count(_order())
    ok, why = c1.check(_order(), 1.10)
    assert not ok and "budget" in why

    c2 = _compliance(tmp_path, max_per_day=5)   # "restart"
    ok, why = c2.check(_order(), 1.10)
    assert not ok and "budget" in why           # counter restored, not reset


def test_notional_check_fails_closed_without_mark(tmp_path):
    c = PreTradeCompliance(Config({"compliance": {
        "max_order_notional": 1000,
        "state_path": str(tmp_path / "c.json"),
    }}, Secrets.from_env()))
    ok, why = c.check(_order(vol=1.0), 0.0)     # no mark price
    assert not ok and "mark price" in why


# ── idempotency ledger (unit level; the HTTP path is mocked out) ─────────────

def test_intent_ledger_matches_orphan_deal(tmp_path, monkeypatch):
    from gungnir.execution import capital_com
    monkeypatch.chdir(tmp_path)
    broker = capital_com.CapitalComBroker.__new__(capital_com.CapitalComBroker)
    broker._intents_path = tmp_path / "pending_orders.json"
    broker._pending_intents = []

    intent = broker._record_intent(_order(symbol="US500", vol=2.5, strategy=NET_TAG))
    assert json.loads(broker._intents_path.read_text())  # persisted

    # An adopted deal with matching symbol/side/size consumes the intent.
    match = broker._match_intent("US500", "BUY", 2.5)
    assert match is not None and match["strategy"] == NET_TAG
    assert broker._match_intent("US500", "BUY", 2.5) is None   # consumed
    assert intent["client_id"].startswith(NET_TAG)


def test_intent_ledger_ignores_stale_and_mismatched(tmp_path):
    from gungnir.execution import capital_com
    broker = capital_com.CapitalComBroker.__new__(capital_com.CapitalComBroker)
    broker._intents_path = tmp_path / "pending_orders.json"
    broker._pending_intents = []
    broker._record_intent(_order(symbol="US500", vol=2.5))

    assert broker._match_intent("US500", "SELL", 2.5) is None   # wrong side
    assert broker._match_intent("US500", "BUY", 5.0) is None    # wrong size
    # Stale intents expire.
    broker._pending_intents[0]["ts"] -= 10_000
    assert broker._match_intent("US500", "BUY", 2.5) is None
