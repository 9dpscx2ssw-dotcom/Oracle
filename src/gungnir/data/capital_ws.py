"""Capital.com WebSocket quote stream.

Streams live bid/offer for the universe over one socket instead of polling
``/markets`` snapshots — quotes arrive in milliseconds, so position marks and
agent-managed exits track the market tick-by-tick rather than at the REST
snapshot cadence. The REST path remains the fallback everywhere: if this
socket is down, disabled, or the ``websockets`` package is missing, the feed
behaves exactly as before.

Protocol (https://open-api.capital.com/ → WebSocket API):
  • connect to ``wss://api-streaming-capital.backend-capital.com/connect``
  • subscribe:  {"destination": "marketData.subscribe", "correlationId": …,
                 "cst": …, "securityToken": …, "payload": {"epics": [...]}}
    (max ~40 epics per subscription message — chunked here)
  • quotes:     {"destination": "quote",
                 "payload": {"epic": …, "bid": …, "ofr": …, "timestamp": …}}
  • the server drops idle connections after ~10 minutes — a ping is sent on a
    timer, and any drop triggers reconnect + resubscribe with fresh session
    tokens (login is re-run through the shared REST session if needed).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from ..core import metrics

log = logging.getLogger(__name__)

STREAM_URL = "wss://api-streaming-capital.backend-capital.com/connect"
_MAX_EPICS_PER_SUB = 40
_PING_EVERY = 300.0          # seconds; server times out idle sockets at ~10 min
_RECV_TIMEOUT = 30.0         # wake regularly to ping / notice the stop flag


class CapitalComQuoteStream:
    """Background task holding the freshest bid/offer per epic."""

    def __init__(self, session, symbols: list[str], *, url: str = STREAM_URL,
                 max_quote_age: float = 30.0):
        self.session = session          # CapitalComSession (tokens + login)
        self.symbols = [s for s in symbols if s]
        self.url = url
        self.max_quote_age = float(max_quote_age)
        self._quotes: dict[str, tuple[float, float, float]] = {}  # bid, ofr, mono
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.connected = False

    # ── consumer side ────────────────────────────────────────────────────────

    def quote(self, symbol: str) -> tuple[float, float] | None:
        """(bid, offer) if we hold one fresher than ``max_quote_age``."""
        q = self._quotes.get(symbol)
        if q is not None and time.monotonic() - q[2] <= self.max_quote_age:
            return q[0], q[1]
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        self.connected = False
        metrics.ws_connected(False)

    # ── the stream loop ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            log.warning("websockets not installed; quote streaming disabled "
                        "(REST snapshots remain the price source)")
            return
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self.session.connect()      # fresh tokens if needed
                async with websockets.connect(self.url, open_timeout=15,
                                              close_timeout=5) as ws:
                    await self._subscribe(ws)
                    self.connected = True
                    metrics.ws_connected(True)
                    log.info("Quote stream connected (%d symbols)",
                             len(self.symbols))
                    backoff = 1.0
                    await self._pump(ws)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — reconnect on anything
                log.warning("Quote stream dropped (%s); reconnecting in %.0fs",
                            str(e)[:200], backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                self.connected = False
                metrics.ws_connected(False)

    def _auth_fields(self) -> dict:
        return {"cst": self.session.cst,
                "securityToken": self.session.security_token}

    async def _subscribe(self, ws) -> None:
        for i in range(0, len(self.symbols), _MAX_EPICS_PER_SUB):
            chunk = self.symbols[i:i + _MAX_EPICS_PER_SUB]
            await ws.send(json.dumps({
                "destination": "marketData.subscribe",
                "correlationId": str(i),
                **self._auth_fields(),
                "payload": {"epics": chunk},
            }))

    async def _pump(self, ws) -> None:
        last_ping = time.monotonic()
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                self._handle(raw)
            if time.monotonic() - last_ping >= _PING_EVERY:
                await ws.send(json.dumps({
                    "destination": "ping", "correlationId": "ping",
                    **self._auth_fields(),
                }))
                last_ping = time.monotonic()

    def _handle(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        dest = msg.get("destination")
        if dest == "quote":
            p = msg.get("payload") or {}
            epic = p.get("epic")
            bid, ofr = p.get("bid"), p.get("ofr")
            if epic and bid is not None and ofr is not None:
                self._quotes[epic] = (float(bid), float(ofr), time.monotonic())
                metrics.inc_ws_quote()
        elif dest == "marketData.subscribe":
            status = msg.get("status")
            if status and status != "OK":
                log.warning("Quote subscription rejected: %s",
                            json.dumps(msg)[:300])
