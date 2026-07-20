"""Prometheus branch: metrics export, WebSocket quote streaming, Codex CLI
provider, and slippage tracking."""

from __future__ import annotations

import asyncio
import json
import stat

from gungnir.config import Config, Secrets
from gungnir.core import metrics
from gungnir.data.capital_ws import CapitalComQuoteStream
from gungnir.llm.client import CodexClient, build_llm


# ── metrics module ───────────────────────────────────────────────────────────

def test_metrics_helpers_never_raise():
    """Every helper is safe whether or not prometheus_client is installed."""
    metrics.observe_loop("fast", 1.2)
    metrics.set_equity("real", 10_000.0)
    metrics.set_open_positions(3)
    metrics.set_halted("shadow", True)
    metrics.inc_signal("rejected_risk")
    metrics.trade_closed("shadow", -12.5)      # negative PnL must not raise
    metrics.trade_closed("real", None)
    metrics.inc_api_429()
    metrics.ws_connected(True)
    metrics.inc_ws_quote()
    metrics.cap_saturation(True)
    metrics.observe_slippage(-1.5)


def test_metrics_setup_respects_disabled():
    cfg = Config({"metrics": {"enabled": False}}, Secrets.from_env())
    assert metrics.setup(cfg) is False


def test_metrics_values_are_scrapeable():
    import prometheus_client
    metrics.set_equity("real", 12_345.0)
    metrics.trade_closed("real", -100.0)
    output = prometheus_client.generate_latest().decode()
    assert 'gungnir_equity{book="real"} 12345.0' in output
    assert "gungnir_realized_pnl" in output


# ── WebSocket quote stream (against a local in-process server) ───────────────

class _FakeSession:
    cst = "test-cst"
    security_token = "test-token"

    async def connect(self):
        pass


async def _serve_quotes(port: int, got_subscribe: asyncio.Event):
    """Minimal Capital.com-shaped streaming server on localhost."""
    import websockets

    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("destination") == "marketData.subscribe":
                got_subscribe.set()
                assert msg.get("cst") == "test-cst"
                await ws.send(json.dumps(
                    {"destination": "marketData.subscribe", "status": "OK"}))
                for bid, ofr in ((1.1000, 1.1002), (1.1005, 1.1007)):
                    await ws.send(json.dumps({
                        "destination": "quote",
                        "payload": {"epic": "EURUSD", "bid": bid, "ofr": ofr},
                    }))

    return await websockets.serve(handler, "127.0.0.1", port)


async def test_stream_receives_and_serves_quotes(unused_tcp_port=None):
    port = 8765
    got_subscribe = asyncio.Event()
    server = await _serve_quotes(port, got_subscribe)
    try:
        stream = CapitalComQuoteStream(
            _FakeSession(), ["EURUSD"], url=f"ws://127.0.0.1:{port}")
        await stream.start()
        await asyncio.wait_for(got_subscribe.wait(), timeout=5)
        for _ in range(50):                      # wait for the quote to land
            if stream.quote("EURUSD") is not None:
                break
            await asyncio.sleep(0.05)
        q = stream.quote("EURUSD")
        assert q is not None
        assert q == (1.1005, 1.1007)             # newest quote wins
        assert stream.quote("US500") is None     # unknown symbol → REST path
        await stream.stop()
        assert stream.connected is False
    finally:
        server.close()
        await server.wait_closed()


async def test_stream_quote_expires():
    stream = CapitalComQuoteStream(_FakeSession(), ["EURUSD"], max_quote_age=0.0)
    stream._quotes["EURUSD"] = (1.1, 1.2, 0.0)   # ancient monotonic stamp
    assert stream.quote("EURUSD") is None


async def test_stream_reconnects_after_drop():
    """A server that dies mid-stream must trigger reconnect, not a dead task."""
    import websockets
    port = 8766
    connections = []

    async def handler(ws):
        connections.append(1)
        await ws.close()                          # drop immediately

    server = await websockets.serve(handler, "127.0.0.1", port)
    try:
        stream = CapitalComQuoteStream(
            _FakeSession(), ["EURUSD"], url=f"ws://127.0.0.1:{port}")
        await stream.start()
        for _ in range(100):
            if len(connections) >= 2:
                break
            await asyncio.sleep(0.1)
        assert len(connections) >= 2              # it came back
        await stream.stop()
    finally:
        server.close()
        await server.wait_closed()


# ── Codex CLI provider ───────────────────────────────────────────────────────

def _fake_codex(tmp_path, script_body: str) -> str:
    path = tmp_path / "codex"
    path.write_text("#!/bin/sh\n" + script_body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _codex_config(cmd: str, **extra) -> Config:
    return Config({"llm": {"provider": "codex", "codex_cmd": cmd,
                           "codex_timeout_seconds": 10, **extra}},
                  Secrets.from_env())


def test_build_llm_selects_codex(tmp_path):
    cmd = _fake_codex(tmp_path, "exit 0\n")
    assert isinstance(build_llm(_codex_config(cmd)), CodexClient)


def test_codex_parses_output_last_message(tmp_path):
    # The stub mimics `codex exec --output-last-message <file> <prompt>`:
    # it writes JSON (wrapped in prose, as models do) to the -o file.
    body = """
out=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--output-last-message" ]; then out="$a"; fi
  prev="$a"
done
echo 'Here is my analysis: {"score": 0.4, "confidence": 0.8, "rationale": "ok"}' > "$out"
"""
    client = CodexClient(_codex_config(_fake_codex(tmp_path, body)))
    data = client.complete_json("Instrument: EURUSD", system="score it")
    assert data == {"score": 0.4, "confidence": 0.8, "rationale": "ok"}
    # Second call is served from the LRU cache (stub not re-run — same answer).
    assert client.complete_json("Instrument: EURUSD", system="score it") == data


def test_codex_missing_binary_cools_down(tmp_path):
    client = CodexClient(_codex_config(str(tmp_path / "nope-codex")))
    assert client.complete_json("x") == {}
    assert client._cooldown_until > 0            # not retried every loop


def test_codex_repeated_failures_trip_cooldown(tmp_path):
    cmd = _fake_codex(tmp_path, "echo 'usage limit reached' >&2\nexit 1\n")
    client = CodexClient(_codex_config(cmd))
    for _ in range(3):
        assert client.complete_json(f"p{_}") == {}
    assert client._cooldown_until > 0


# ── slippage bookkeeping (sign convention) ───────────────────────────────────

def test_slippage_sign_convention():
    """BUY filled above arrival = adverse = positive; SELL below = positive."""
    # Mirrors the formula in CapitalComBroker.submit.
    def slip(side_dir: int, fill: float, arrival: float) -> float:
        return side_dir * (fill - arrival) / arrival * 10_000.0

    assert slip(+1, 1.1002, 1.1000) > 0          # buy, paid up → adverse
    assert slip(+1, 1.0998, 1.1000) < 0          # buy, price improved
    assert slip(-1, 1.0998, 1.1000) > 0          # sell, hit lower → adverse
    assert slip(-1, 1.1002, 1.1000) < 0          # sell, price improved


def test_metrics_dont_break_paper_flow(tmp_path):
    """Journal wiring: recording signals/trades with metrics attached works."""
    from gungnir.data.models import Side, Signal, Trade
    from gungnir.learning.journal import Journal
    from gungnir.persistence.db import Database
    db = Database(tmp_path / "m.db")
    j = Journal(db)
    j.record_signal(Signal(strategy="s", symbol="EURUSD", side=Side.BUY,
                           conviction=0.5), "shadow", 1.1)
    j.record(Trade(symbol="EURUSD", side=Side.BUY, volume=1.0,
                   entry_price=1.1, exit_price=1.101, pnl=-5.0, mode="shadow"))
    db.close()

