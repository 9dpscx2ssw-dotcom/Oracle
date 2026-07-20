"""Phase-3 money math: the vol-target sizer, FX PnL conversion, delta-order
reconciliation, paper merge-on-submit, and the control race fix."""

from __future__ import annotations

from gungnir.config import Config, Secrets
from gungnir.data.models import Candle, Order, Side, Signal
from gungnir.execution.broker import PaperBroker
from gungnir.execution.fx import quote_currency, to_account_ccy
from gungnir.execution.netting import NET_TAG, NettingBroker
from gungnir.features.feature_store import KrakenFeatureSet
from gungnir.risk.position_sizing import VolTarget


# ── VolTarget sizing ─────────────────────────────────────────────────────────

def _features(symbol="EURUSD", price=1.10, atr=0.0005, tf="5m") -> KrakenFeatureSet:
    return KrakenFeatureSet(
        symbol=symbol, last_price=price, atr=atr,
        candles=[Candle(symbol=symbol, timeframe=tf, open=price, high=price,
                        low=price, close=price)])


def _sizer(target=0.10) -> VolTarget:
    return VolTarget(Config({"risk": {"vol_target_annual": target}},
                            Secrets.from_env()))


def test_vol_target_produces_sane_notional():
    """The old formula (equity×target/ATR, no time scaling) asked for ~$1M of
    EURUSD on a $10k account — every order cap-saturated and the sizing chain
    upstream was erased. Annualized, the same inputs stay inside the caps."""
    sig = Signal(strategy="s", symbol="EURUSD", side=Side.BUY, conviction=0.5)
    vol = _sizer().size(sig, _features(), 10_000.0)
    notional = vol * 1.10
    assert 500 < notional < 10_000            # sane, cap-free size
    # Old formula for comparison: 10_000*0.10*0.5/0.0005 = 1_000_000 units.
    assert vol < 100_000


def test_vol_target_scales_with_timeframe():
    """Same ATR on a slower bar = lower per-year vol = larger position."""
    sig = Signal(strategy="s", symbol="EURUSD", side=Side.BUY, conviction=1.0)
    v_5m = _sizer().size(sig, _features(tf="5m"), 10_000.0)
    v_1h = _sizer().size(sig, _features(tf="1h"), 10_000.0)
    assert v_1h > v_5m
    # sqrt(12) ratio between 5m and 1h bars-per-year
    assert abs(v_1h / v_5m - 12 ** 0.5) < 0.01


def test_vol_target_zero_atr_rejects():
    sig = Signal(strategy="s", symbol="EURUSD", side=Side.BUY, conviction=1.0)
    assert _sizer().size(sig, _features(atr=0.0), 10_000.0) == 0.0


# ── FX PnL conversion ────────────────────────────────────────────────────────

def test_quote_currency_detection():
    assert quote_currency("USDJPY") == "JPY"
    assert quote_currency("EURGBP") == "GBP"
    assert quote_currency("EURUSD") == "USD"
    assert quote_currency("US500") == "USD"       # non-FX epics quote USD
    assert quote_currency("AAPL") == "USD"
    assert quote_currency("BTCUSD") == "USD"


def test_pnl_converts_via_inverse_pair():
    # ¥15,000 of USDJPY PnL at USDJPY=150 → $100.
    marks = {"USDJPY": 150.0}
    assert abs(to_account_ccy("USDJPY", 15_000.0, marks.get) - 100.0) < 1e-9


def test_pnl_converts_via_direct_pair():
    # £80 of EURGBP PnL at GBPUSD=1.25 → $100.
    marks = {"GBPUSD": 1.25}
    assert abs(to_account_ccy("EURGBP", 80.0, marks.get) - 100.0) < 1e-9


def test_pnl_passthrough_when_no_rate():
    assert to_account_ccy("EURNOK", 500.0, {}.get) == 500.0   # warned, unchanged


async def test_paper_close_converts_jpy_pnl_to_usd():
    b = PaperBroker(starting_equity=10_000.0)
    b.mark("USDJPY", 150.0)
    await b.submit(Order(symbol="USDJPY", side=Side.BUY, volume=1_000.0,
                         client_id="s:USDJPY:1"))
    b.mark("USDJPY", 151.5)                      # +1.5 JPY × 1000 = ¥1,500
    closed = await b.close("USDJPY", "s")
    assert closed is not None
    assert abs(closed.pnl - 1_500.0 / 151.5) < 1e-6     # ≈ $9.90, not $1,500
    assert abs(await b.account_equity() - (10_000.0 + closed.pnl)) < 1e-6


# ── paper merge-on-submit (no more silent overwrite) ─────────────────────────

async def test_same_side_resubmit_merges_not_overwrites():
    b = PaperBroker()
    b.mark("US500", 5000.0)
    await b.submit(Order(symbol="US500", side=Side.BUY, volume=1.0, client_id="s:US500:1"))
    b.mark("US500", 5100.0)
    t = await b.submit(Order(symbol="US500", side=Side.BUY, volume=1.0, client_id="s:US500:2"))
    assert t.volume == 2.0
    assert abs(t.entry_price - 5050.0) < 1e-9    # volume-weighted entry


async def test_opposite_side_resubmit_closes_first():
    b = PaperBroker(starting_equity=10_000.0)
    b.mark("US500", 5000.0)
    await b.submit(Order(symbol="US500", side=Side.BUY, volume=1.0, client_id="s:US500:1"))
    b.mark("US500", 5100.0)
    t = await b.submit(Order(symbol="US500", side=Side.SELL, volume=1.0, client_id="s:US500:2"))
    assert t.side == Side.SELL and t.volume == 1.0
    # The long's +100 PnL was realized (not silently discarded).
    assert abs(await b.balance() - 10_100.0) < 1e-9


# ── delta-order reconciliation ───────────────────────────────────────────────

class _CountingAccount(PaperBroker):
    def __init__(self):
        super().__init__()
        self.submits = 0
        self.closes = 0

    async def submit(self, order):
        self.submits += 1
        return await super().submit(order)

    async def close(self, symbol, strategy=None):
        self.closes += 1
        return await super().close(symbol, strategy)


def _order(strategy: str, vol: float, side=Side.BUY) -> Order:
    return Order(symbol="US500", side=side, volume=vol,
                 client_id=f"{strategy}:US500:1")


async def test_net_increase_submits_delta_only():
    account = _CountingAccount()
    nb = NettingBroker(account, catastrophe_stop_mult=0)
    nb.mark("US500", 5000.0)
    await nb.submit(_order("s1", 1.0))
    await nb.reconcile("US500")
    assert (account.submits, account.closes) == (1, 0)
    # Second strategy piles on: the account should see ONE delta order, no close.
    await nb.submit(_order("s2", 0.5))
    await nb.reconcile("US500")
    assert (account.submits, account.closes) == (2, 0)
    net = account.position("US500", NET_TAG)
    assert net.volume == 1.5


async def test_net_decrease_still_closes_and_reopens():
    account = _CountingAccount()
    nb = NettingBroker(account, catastrophe_stop_mult=0)
    nb.mark("US500", 5000.0)
    await nb.submit(_order("s1", 1.0))
    await nb.submit(_order("s2", 0.5))
    await nb.reconcile("US500")
    await nb.close("US500", "s2")               # net shrinks 1.5 → 1.0
    await nb.reconcile("US500")
    net = account.position("US500", NET_TAG)
    assert net is not None and abs(net.volume - 1.0) < 1e-9
    assert account.closes >= 1                  # shrink used close-then-reopen


async def test_net_flip_closes_then_opens():
    account = _CountingAccount()
    nb = NettingBroker(account, catastrophe_stop_mult=0)
    nb.mark("US500", 5000.0)
    await nb.submit(_order("s1", 1.0, Side.BUY))
    await nb.reconcile("US500")
    await nb.close("US500", "s1")
    await nb.submit(_order("s1", 2.0, Side.SELL))
    await nb.reconcile("US500")
    net = account.position("US500", NET_TAG)
    assert net.side == Side.SELL and abs(net.volume - 2.0) < 1e-9


# ── control one-shot consumption (lost-update fix) ───────────────────────────

def test_clear_keys_preserves_concurrent_writes(tmp_path):
    from gungnir.core.control import Control
    c = Control(tmp_path / "control.json")
    c.write({"reset_shadow": True, "paused": False})
    # A dashboard write lands AFTER the agent read its stale copy…
    c.write({**c.read(), "paused": True})
    # …and the agent consumes only its one-shot key.
    c.clear_keys("reset_shadow")
    data = c.read()
    assert "reset_shadow" not in data
    assert data["paused"] is True               # the concurrent write survived
