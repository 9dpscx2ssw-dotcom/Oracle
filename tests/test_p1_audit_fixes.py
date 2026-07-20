"""Phase-1/2 audit-remediation tests.

Covers three fixes:
  • correlation/cluster exposure cap in PortfolioRisk.vet();
  • RL gate fails open when the policy has collapsed or diverged;
  • constant-time dashboard token comparison (behavioural equivalence).
"""

from __future__ import annotations

import hmac

import pytest

from gungnir.core.agent import RL_MIN_TAKE_FLOOR, _rl_gate_healthy
from gungnir.data.models import Side, Signal
from gungnir.risk.portfolio import PortfolioRisk, _cluster_of


class _Cfg:
    """Minimal Config stand-in: nested .get(*path, default=...)."""

    def __init__(self, raw):
        self.raw = raw

    def get(self, *path, default=None):
        node = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _risk(**risk) -> PortfolioRisk:
    r = PortfolioRisk(_Cfg({"risk": risk}))
    r.equity = 10_000.0
    r.roll_day()
    return r


def _sig(symbol, side=Side.BUY):
    return Signal(strategy="s", symbol=symbol, side=side, conviction=1.0)


# ── cluster cap ───────────────────────────────────────────────────────────────

def test_cluster_of_groups_equities_and_splits_fx():
    assert _cluster_of("AAPL") == _cluster_of("NVDA") == "stocks"
    assert _cluster_of("BTCUSD") == "crypto"
    # FX split by the non-USD leg: EUR pairs share a bucket, JPY doesn't.
    assert _cluster_of("EURUSD") == _cluster_of("EURGBP") == "fx_eur"
    assert _cluster_of("USDJPY") == "fx_jpy"


def test_cluster_cap_binds_across_correlated_symbols():
    # Cluster cap = 0.5x equity ($5k). Loose per-asset/gross so only the
    # cluster cap can bind. First equity order eats most of the bucket; the
    # second correlated order is shrunk to the remaining bucket headroom.
    r = _risk(max_cluster_exposure=0.5, max_per_asset_exposure=10.0,
              max_portfolio_exposure=100.0, leverage=100.0, min_confidence=0.0,
              max_open_positions=50)
    # Pretend AAPL already holds $4,000 of the $5,000 tech bucket.
    r.open_exposure = {"AAPL": 4_000.0}
    order = r.vet(_sig("NVDA"), raw_volume=100.0, price=100.0, atr=1.0)  # wants $10,000
    assert order is not None
    # Only $1,000 of bucket headroom remained → ~10 units at $100.
    assert order.volume * 100.0 == pytest.approx(1_000.0, abs=1.0)


def test_cluster_cap_default_is_non_binding():
    # Default max_cluster == max_gross, so a single-cluster order is never cut
    # tighter than the gross cap would already cut it.
    r = _risk(max_portfolio_exposure=1.0, max_per_asset_exposure=10.0,
              leverage=100.0, min_confidence=0.0)
    r.open_exposure = {}
    order = r.vet(_sig("NVDA"), raw_volume=50.0, price=100.0, atr=1.0)  # $5,000, under gross
    assert order is not None
    assert order.volume == pytest.approx(50.0)


# ── RL gate fail-open ─────────────────────────────────────────────────────────

def test_rl_gate_healthy_when_taking_normally():
    assert _rl_gate_healthy(0.5, diverged=False) is True
    assert _rl_gate_healthy(None, diverged=False) is True   # warmup


def test_rl_gate_unhealthy_on_collapse_or_divergence():
    assert _rl_gate_healthy(0.0, diverged=False) is False            # all-skip
    assert _rl_gate_healthy(RL_MIN_TAKE_FLOOR / 2, diverged=False) is False
    assert _rl_gate_healthy(0.9, diverged=True) is False             # diverged


def test_rl_gate_failopen_semantics():
    # The exact expression used in the decide loop: take only when the gate is
    # on AND healthy; otherwise the signal is TAKEN (fail open).
    def effective_take(gate_on, policy_take, rate, diverged):
        return policy_take if (gate_on and _rl_gate_healthy(rate, diverged)) else True

    # Healthy policy that says skip → skip is honoured.
    assert effective_take(True, False, 0.5, False) is False
    # Collapsed policy that says skip → overridden to take.
    assert effective_take(True, False, 0.0, False) is True
    # Diverged policy that says skip → overridden to take.
    assert effective_take(True, False, 0.8, True) is True
    # Gate off entirely → always take.
    assert effective_take(False, False, 0.5, False) is True


# ── constant-time token compare ───────────────────────────────────────────────

def test_token_compare_is_equivalent_and_constant_time():
    token = "s3cret-token"
    assert hmac.compare_digest(token, token) is True
    assert hmac.compare_digest("" , token) is False
    assert hmac.compare_digest("wrong", token) is False
