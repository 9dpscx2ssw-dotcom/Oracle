"""/api/telemetry: Prometheus text parsing and card-shaped JSON."""

from __future__ import annotations

from gungnir.dashboard.server import (
    _parse_prom_text, _slippage_buckets, _telemetry_from,
)

SAMPLE = """\
# HELP gungnir_loop_seconds Wall time of the last completed loop iteration
# TYPE gungnir_loop_seconds gauge
gungnir_loop_seconds{loop="fast"} 3.7
gungnir_loop_seconds{loop="slow"} 0.39
gungnir_equity{book="real"} 10142.55
gungnir_equity{book="shadow"} 9987.1
gungnir_open_positions 3.0
gungnir_halted{book="real"} 0.0
gungnir_halted{book="shadow"} 1.0
gungnir_signals_total{disposition="real"} 128.0
gungnir_signals_total{disposition="shadow"} 1204.0
gungnir_signals_total{disposition="rejected_risk"} 2240.0
gungnir_signals_total{disposition="rejected_cooldown"} 671.0
gungnir_trades_closed_total{mode="real"} 62.0
gungnir_trades_closed_total{mode="shadow"} 741.0
gungnir_realized_pnl{mode="real"} 214.3
gungnir_realized_pnl{mode="shadow"} -88.45
gungnir_api_429_total 2.0
gungnir_ws_connected 1.0
gungnir_ws_quotes_total 48211.0
gungnir_cap_saturation_total{result="capped"} 9.0
gungnir_cap_saturation_total{result="full"} 132.0
gungnir_fill_slippage_bps_bucket{le="-5.0"} 1.0
gungnir_fill_slippage_bps_bucket{le="0.0"} 12.0
gungnir_fill_slippage_bps_bucket{le="2.0"} 43.0
gungnir_fill_slippage_bps_bucket{le="+Inf"} 62.0
gungnir_fill_slippage_bps_count 62.0
gungnir_fill_slippage_bps_sum 49.6
"""


def test_parse_prom_text_labels_and_scalars():
    m = _parse_prom_text(SAMPLE)
    assert ({"loop": "fast"}, 3.7) in m["gungnir_loop_seconds"]
    assert m["gungnir_open_positions"] == [({}, 3.0)]
    # comments and HELP/TYPE lines are ignored
    assert "# HELP" not in str(m.keys())


def test_slippage_buckets_decumulate():
    m = _parse_prom_text(SAMPLE)
    buckets = _slippage_buckets(m)
    # cumulative 1, 12, 43, 62 → per-bucket 1, 11, 31, 19
    assert [b["n"] for b in buckets] == [1, 11, 31, 19]
    assert buckets[0]["from"] is None and buckets[0]["to"] == -5.0
    assert buckets[-1]["to"] is None            # +Inf tail


def test_telemetry_shape():
    t = _telemetry_from(_parse_prom_text(SAMPLE),
                        {"rl": {"brier": 0.21, "recent_take_rate": 0.62}})
    assert t["loop"] == {"fast": 3.7, "slow": 0.39}
    assert t["equity"]["real"] == 10142.55
    assert t["open_positions"] == 3
    assert t["halted"] == {"real": False, "shadow": True}
    assert t["signals"]["rejected_risk"] == 2240
    assert t["trades"]["real"] == {"n": 62, "pnl": 214.3}
    assert t["trades"]["shadow"]["pnl"] == -88.45
    assert t["api_429"] == 2
    assert t["ws"] == {"connected": True, "quotes": 48211}
    assert t["cap"] == {"capped": 9, "total": 141}
    assert t["slippage"]["n"] == 62 and t["slippage"]["mean"] == 0.8
    assert t["rl"]["brier"] == 0.21


def test_endpoint_degrades_when_agent_unreachable(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from gungnir.dashboard.server import create_app
    monkeypatch.setenv("GUNGNIR_METRICS_URL", "http://127.0.0.1:9/metrics")
    monkeypatch.setenv("GUNGNIR_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("GUNGNIR_STATUS_PATH", str(tmp_path / "none.json"))
    client = TestClient(create_app())
    r = client.get("/api/telemetry")
    assert r.status_code == 200                  # degrade, never 500
    body = r.json()
    assert body["available"] is False
    assert "scrape_url" in body and "reason" in body
