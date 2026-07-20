"""Shared Capital.com REST session: login, token refresh, authenticated requests.

Capital.com is *not* a plain API-key REST API. You authenticate once with
``POST /api/v1/session`` (API key + account identifier + password) and receive
two tokens in the response headers — ``CST`` and ``X-SECURITY-TOKEN`` — that must
accompany every subsequent request. The tokens expire 10 minutes after the last
use, so this session re-authenticates transparently on a 401.

The market feed and the broker share one instance so a single login covers both.

Docs: https://open-api.capital.com/  ·  base URLs:
  live  https://api-capital.backend-capital.com
  demo  https://demo-api-capital.backend-capital.com
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx

log = logging.getLogger(__name__)

LIVE_URL = "https://api-capital.backend-capital.com"
DEMO_URL = "https://demo-api-capital.backend-capital.com"


class CapitalComSession:
    def __init__(
        self,
        api_key: str,
        identifier: str,
        password: str,
        *,
        demo: bool = False,
        base_url: str | None = None,
        timeout: float = 20.0,
        min_interval: float = 0.12,
        max_get_attempts: int = 3,
    ):
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        self.base_url = base_url or (DEMO_URL if demo else LIVE_URL)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"X-CAP-API-KEY": api_key, "Content-Type": "application/json"},
        )
        self._cst: str | None = None
        self._security: str | None = None
        self._lock = asyncio.Lock()
        # Rate-limit guard: Capital.com throttles bursts hard (the demo API
        # especially), and the agent naturally fires many concurrent fetches.
        # Every request reserves a slot ``min_interval`` after the previous
        # one, so concurrency never turns into a burst.
        self.min_interval = max(0.0, float(min_interval))
        self.max_get_attempts = max(1, int(max_get_attempts))
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()

    async def _pace(self) -> None:
        """Reserve the next request slot; sleeps outside the lock so waiting
        requests queue in order without serializing the actual HTTP calls."""
        if self.min_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    @property
    def authenticated(self) -> bool:
        return bool(self._cst and self._security)

    @property
    def cst(self) -> str:
        """Session token for the streaming API (empty until login)."""
        return self._cst or ""

    @property
    def security_token(self) -> str:
        """X-SECURITY-TOKEN for the streaming API (empty until login)."""
        return self._security or ""

    async def login(self) -> None:
        """Create a session and capture the CST / X-SECURITY-TOKEN headers."""
        async with self._lock:
            r = await self._client.post(
                "/api/v1/session",
                json={"identifier": self.identifier, "password": self.password},
            )
            r.raise_for_status()
            self._cst = r.headers.get("CST")
            self._security = r.headers.get("X-SECURITY-TOKEN")
            if not (self._cst and self._security):
                raise RuntimeError(
                    "Capital.com login succeeded but CST/X-SECURITY-TOKEN headers were missing"
                )
            env = "demo" if self.base_url == DEMO_URL else "live"
            log.info("Capital.com session established (%s)", env)

    async def connect(self) -> None:
        if not self.authenticated:
            await self.login()

    def _auth_headers(self) -> dict[str, str]:
        return {"CST": self._cst or "", "X-SECURITY-TOKEN": self._security or ""}

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Authenticated request; re-logins once and retries on a 401 (expired
        token). GETs are additionally retried on a 429 with backoff (honoring
        Retry-After) — reads are idempotent. POST/DELETE are never auto-retried
        on 429: an order that was throttled must not risk a double submit."""
        extra = kwargs.pop("headers", {})
        if not self.authenticated:
            await self.login()
        attempts = self.max_get_attempts if method.upper() == "GET" else 1
        for attempt in range(attempts):
            await self._pace()
            r = await self._client.request(
                method, path, headers={**self._auth_headers(), **extra}, **kwargs)
            if r.status_code == 401:
                log.info("Capital.com token expired; re-authenticating")
                await self.login()
                await self._pace()
                r = await self._client.request(
                    method, path, headers={**self._auth_headers(), **extra}, **kwargs
                )
            if r.status_code == 429:
                from ..core import metrics
                metrics.inc_api_429()
            if r.status_code == 429 and attempt < attempts - 1:
                retry_after = r.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 0.8 * (2 ** attempt)
                delay += random.uniform(0.0, 0.3)
                log.debug("429 from %s; retrying in %.1fs (attempt %d/%d)",
                          path, delay, attempt + 1, attempts)
                await asyncio.sleep(delay)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")  # loop always returns or raises

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", path, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()
