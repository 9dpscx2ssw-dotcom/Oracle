"""Capital.com feed: resolution mapping, ISO date format, and price parsing.

These lock in the request/response contract that previously caused the 400s
(numeric resolutions, unix timestamps, and bid/ask read off the wrong nesting).
The live HTTP path needs real credentials and is exercised on deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gungnir.data.capital_com_feed import RESOLUTION, _iso, _parse_candles


def test_resolution_codes_are_enums_not_numbers():
    assert RESOLUTION["5m"] == "MINUTE_5"
    assert RESOLUTION["1h"] == "HOUR"
    assert RESOLUTION["4h"] == "HOUR_4"
    assert RESOLUTION["1d"] == "DAY"
    # cTrader-style spellings used elsewhere in the codebase also resolve.
    assert RESOLUTION["M15"] == "MINUTE_15"
    assert RESOLUTION["H4"] == "HOUR_4"


def test_iso_has_no_timezone_suffix():
    s = _iso(datetime(2026, 6, 28, 19, 5, 0, tzinfo=timezone.utc))
    assert s == "2026-06-28T19:05:00"


def test_parse_candles_uses_mid_price_from_nested_prices():
    # Mid = (bid+ask)/2 — bid-only candles biased every feature short; a missing
    # side falls back to whichever exists (see test below).
    data = {"prices": [{
        "snapshotTime": "2026-06-28T18:00:00",
        "openPrice": {"bid": 100.0, "ask": 100.2},
        "highPrice": {"bid": 101.0, "ask": 101.2},
        "lowPrice": {"bid": 99.5, "ask": 99.7},
        "closePrice": {"bid": 100.5, "ask": 100.7},
        "lastTradedVolume": 1234,
    }]}
    candles = _parse_candles("BTCUSD", "5m", data)
    assert len(candles) == 1
    c = candles[0]
    assert (c.open, c.high, c.low, c.close, c.volume) == (100.1, 101.1, 99.6, 100.6, 1234.0)
    assert c.symbol == "BTCUSD" and c.timeframe == "5m"


def test_parse_candles_falls_back_to_single_side():
    data = {"prices": [{
        "snapshotTime": "2026-06-28T18:00:00",
        "openPrice": {"bid": 100.0},
        "highPrice": {"ask": 101.2},
        "lowPrice": {"bid": 99.5},
        "closePrice": {"bid": 100.5},
    }]}
    c = _parse_candles("BTCUSD", "5m", data)[0]
    assert (c.open, c.high, c.low, c.close) == (100.0, 101.2, 99.5, 100.5)


def test_parse_candles_skips_malformed_points():
    data = {"prices": [
        {"snapshotTime": "x", "openPrice": {"bid": 1.0}},          # missing high/low/close
        {"snapshotTime": "2026-06-28T18:05:00", "openPrice": {"bid": 1.0},
         "highPrice": {"bid": 1.1}, "lowPrice": {"bid": 0.9}, "closePrice": {"bid": 1.05}},
    ]}
    candles = _parse_candles("X", "5m", data)
    assert len(candles) == 1          # the malformed point is skipped, not fatal
    assert candles[0].close == 1.05


def test_parse_candles_empty_response():
    assert _parse_candles("X", "5m", {}) == []
