"""Prometheus metrics: the agent's operational surface for Grafana/alerting.

Everything here is OPTIONAL and fail-safe: if ``prometheus_client`` isn't
installed, every function is a no-op and the agent runs exactly as before —
status.json remains the dashboard contract; this is the machine-readable twin
for real monitoring (scrape ``http://agent:9109/metrics``).

Wire points (one per concern, chosen to be single choke points):
  • Scheduler        → loop duration per label
  • Agent status     → equity per book, open positions, halted flags,
                       cap-saturation counters
  • Journal          → signals by disposition, closed trades by mode, PnL sum
  • CapitalComSession→ API 429s
  • Quote stream     → connection state, quotes received
  • CapitalComBroker → live fill slippage (bps, adverse-positive)
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via the no-op test
    _AVAILABLE = False

_started = False

if _AVAILABLE:
    LOOP_SECONDS = Gauge("gungnir_loop_seconds",
                         "Wall time of the last completed loop iteration",
                         ["loop"])
    EQUITY = Gauge("gungnir_equity", "Account equity per book", ["book"])
    OPEN_POSITIONS = Gauge("gungnir_open_positions",
                           "Open positions on the account broker")
    HALTED = Gauge("gungnir_halted",
                   "1 when a drawdown breaker has halted the book", ["book"])
    SIGNALS = Counter("gungnir_signals_total",
                      "Signals recorded, by disposition", ["disposition"])
    TRADES_CLOSED = Counter("gungnir_trades_closed_total",
                            "Closed round-trips, by mode", ["mode"])
    # Gauge, not Counter: losses decrement, and Counter.inc() rejects negatives.
    REALIZED_PNL = Gauge("gungnir_realized_pnl",
                         "Cumulative realized PnL (account ccy), by mode",
                         ["mode"])
    API_429 = Counter("gungnir_api_429_total",
                      "HTTP 429 responses from the broker API")
    WS_CONNECTED = Gauge("gungnir_ws_connected",
                         "1 while the quote stream is connected")
    WS_QUOTES = Counter("gungnir_ws_quotes_total",
                        "Quote messages received over the stream")
    CAP_SATURATION = Counter("gungnir_cap_saturation_total",
                             "Vetted orders, by whether the caps cut them",
                             ["result"])
    SLIPPAGE_BPS = Histogram(
        "gungnir_fill_slippage_bps",
        "Live fill slippage vs arrival mark, bps (positive = adverse)",
        buckets=(-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0))


def setup(config) -> bool:
    """Start the scrape endpoint if enabled and the client library exists."""
    global _started
    if _started:
        return True
    enabled = bool(config.get("metrics", "enabled", default=True))
    if not enabled:
        return False
    if not _AVAILABLE:
        log.info("prometheus_client not installed; metrics endpoint disabled "
                 "(pip install prometheus-client)")
        return False
    port = int(config.get("metrics", "port", default=9109) or 9109)
    try:
        start_http_server(port)
        _started = True
        log.info("Prometheus metrics on :%d/metrics", port)
        return True
    except OSError as e:
        log.warning("Metrics endpoint failed to start on :%d: %s", port, e)
        return False


# ── recording helpers (no-ops without the client library) ────────────────────

def observe_loop(label: str, seconds: float) -> None:
    if _AVAILABLE:
        LOOP_SECONDS.labels(loop=label).set(seconds)


def set_equity(book: str, value: float) -> None:
    if _AVAILABLE:
        EQUITY.labels(book=book).set(value)


def set_open_positions(n: int) -> None:
    if _AVAILABLE:
        OPEN_POSITIONS.set(n)


def set_halted(book: str, halted: bool) -> None:
    if _AVAILABLE:
        HALTED.labels(book=book).set(1.0 if halted else 0.0)


def inc_signal(disposition: str) -> None:
    if _AVAILABLE:
        SIGNALS.labels(disposition=disposition).inc()


def trade_closed(mode: str, pnl: float | None) -> None:
    if _AVAILABLE:
        TRADES_CLOSED.labels(mode=mode).inc()
        if pnl:
            REALIZED_PNL.labels(mode=mode).inc(pnl)   # Gauge.inc accepts negatives


def inc_api_429() -> None:
    if _AVAILABLE:
        API_429.inc()


def ws_connected(connected: bool) -> None:
    if _AVAILABLE:
        WS_CONNECTED.set(1.0 if connected else 0.0)


def inc_ws_quote() -> None:
    if _AVAILABLE:
        WS_QUOTES.inc()


def cap_saturation(capped: bool) -> None:
    if _AVAILABLE:
        CAP_SATURATION.labels(result="capped" if capped else "full").inc()


def observe_slippage(bps: float) -> None:
    if _AVAILABLE:
        SLIPPAGE_BPS.observe(bps)
