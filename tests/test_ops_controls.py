"""Operational controls: audit chain, pre-trade compliance, alert throttling."""

from __future__ import annotations

import json

from gungnir.config import Config, Secrets
from gungnir.core.alerts import Alerter
from gungnir.core.compliance import PreTradeCompliance
from gungnir.data.models import Order, Side
from gungnir.persistence.audit import AuditLog


def _order(symbol="US500", volume=1.0):
    return Order(symbol=symbol, side=Side.BUY, volume=volume, client_id="s:x:1")


# ── audit trail ────────────────────────────────────────────────────────────────

def test_audit_chain_verifies_and_survives_restart(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(p)
    a.record("live_order", symbol="US500", volume=1.0)
    a.record("live_close", symbol="US500", pnl=12.5)
    # A new instance chains from the tail on disk (restart continuity).
    b = AuditLog(p)
    b.record("kill_engaged", source="file")
    intact, n, bad = b.verify()
    assert intact and n == 3 and bad is None


def test_audit_tampering_is_detected(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(p)
    a.record("live_order", symbol="US500", volume=1.0)
    a.record("live_close", symbol="US500", pnl=12.5)
    # Retroactively edit the first entry's payload.
    lines = p.read_text().splitlines()
    doctored = json.loads(lines[0])
    doctored["volume"] = 100.0
    lines[0] = json.dumps(doctored)
    p.write_text("\n".join(lines) + "\n")
    intact, _, _ = AuditLog(p).verify()
    assert intact is False


# ── pre-trade compliance ───────────────────────────────────────────────────────

def _compliance(**c) -> PreTradeCompliance:
    # Isolated state file: the day-counter persists across restarts by design,
    # so tests must not share the default data/compliance_state.json.
    import tempfile
    c.setdefault("state_path",
                 tempfile.mktemp(prefix="compliance_", suffix=".json"))
    return PreTradeCompliance(Config({"compliance": c}, Secrets.from_env()))


def test_restricted_symbol_blocked():
    c = _compliance(restricted_symbols=["USDTRY"])
    ok, why = c.check(_order(symbol="USDTRY"), mark_price=32.0)
    assert not ok and "restricted" in why
    assert c.check(_order(symbol="US500"), mark_price=5000.0)[0]


def test_notional_cap_blocks_fat_finger():
    c = _compliance(max_order_notional=10_000)
    assert c.check(_order(volume=1.0), mark_price=5_000.0)[0]          # $5k ok
    ok, why = c.check(_order(volume=10.0), mark_price=5_000.0)         # $50k no
    assert not ok and "notional" in why


def test_daily_order_budget():
    c = _compliance(max_orders_per_day=2)
    o = _order()
    assert c.check(o, 100.0)[0]
    c.count(o)
    c.count(o)
    ok, why = c.check(o, 100.0)
    assert not ok and "budget" in why


# ── alert throttling ───────────────────────────────────────────────────────────

def test_alerter_throttles_per_key(monkeypatch):
    sent = []
    a = Alerter("https://example.invalid/hook", min_interval=900)
    monkeypatch.setattr(a, "_post", lambda k, m: sent.append(k))
    # send() spawns a thread only when not throttled; patching _post before
    # means the thread target records synchronously enough for this test via
    # direct call — call the internal path deterministically instead:
    a.send("halt", "first")            # passes throttle, spawns thread
    a.send("halt", "second")           # throttled, no thread
    a.send("kill", "other key")        # different key, passes
    import time
    time.sleep(0.2)                    # let daemon threads run
    assert sent.count("halt") == 1
    assert sent.count("kill") == 1


def test_alerter_disabled_without_url():
    a = Alerter("")
    assert not a.enabled
    a.send("halt", "should only log")   # must not raise


def test_critical_alerts_bypass_throttle(monkeypatch):
    sent = []
    a = Alerter("https://example.invalid/hook", min_interval=900)
    monkeypatch.setattr(a, "_post", lambda k, m: sent.append(m))
    a.send("kill", "engaged", critical=True)
    a.send("kill", "disengaged", critical=True)   # would be throttled if not critical
    import time
    time.sleep(0.2)
    assert sent == ["engaged", "disengaged"]


def test_alerter_telegram_only_counts_as_enabled():
    assert Alerter("", telegram_token="123:abc", telegram_chat_id="42").enabled
    assert not Alerter("", telegram_token="123:abc").enabled   # chat id required


def test_alerter_delivery_routing(monkeypatch):
    """{message} URLs go out as GET with the text encoded; plain webhook URLs
    as JSON POST; Telegram to the Bot API — all fanned out per alert."""
    calls = []

    class _FakeHttpx:
        @staticmethod
        def get(url, timeout=None):
            calls.append(("GET", url))

        @staticmethod
        def post(url, json=None, timeout=None):
            calls.append(("POST", url, json))

    import sys
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    a = Alerter("https://api.callmebot.com/whatsapp.php?phone=+491&apikey=k&text={message}",
                telegram_token="123:abc", telegram_chat_id="42")
    a._post("halt", "drawdown breaker engaged")

    gets = [c for c in calls if c[0] == "GET"]
    posts = [c for c in calls if c[0] == "POST"]
    assert len(gets) == 1 and "text=" in gets[0][1] and "{message}" not in gets[0][1]
    assert "drawdown" in gets[0][1] or "drawdown" in __import__("urllib.parse", fromlist=["unquote"]).unquote(gets[0][1])
    assert len(posts) == 1 and "api.telegram.org/bot123:abc" in posts[0][1]
    assert posts[0][2]["chat_id"] == "42" and "drawdown" in posts[0][2]["text"]

    # Discord/Slack shape: plain URL → JSON POST with content+text keys.
    calls.clear()
    b = Alerter("https://discord.com/api/webhooks/x")
    b._post("kill", "engaged")
    assert calls and calls[0][0] == "POST"
    assert set(calls[0][2]) == {"content", "text"}
