"""Macro data from FRED (free): CPI, interest rates, unemployment, etc."""

from __future__ import annotations

import logging

import httpx

from ..config import Config
from .feeds import MacroFeed
from .models import MacroIndicator

log = logging.getLogger(__name__)

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredMacroFeed(MacroFeed):
    def __init__(self, config: Config):
        self.api_key = config.secrets.fred_api_key
        # {logical_name: fred_series_id}
        self.series: dict[str, str] = config.get("data", "macro", "fred_series", default={})

    async def fetch(self) -> list[MacroIndicator]:
        if not self.api_key:
            log.warning("FRED_API_KEY not set; skipping macro fetch.")
            return []

        out: list[MacroIndicator] = []
        async with httpx.AsyncClient(timeout=20) as client:
            for name, series_id in self.series.items():
                try:
                    obs = await self._latest_observations(client, series_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("FRED fetch failed for %s: %s", series_id, exc)
                    continue
                if not obs:
                    continue
                latest = obs[-1]
                prev = obs[-2]["value"] if len(obs) >= 2 else None
                out.append(
                    MacroIndicator(
                        series_id=series_id,
                        name=name,
                        value=latest["value"],
                        previous=prev,
                    )
                )
        log.info("Fetched %d macro indicators", len(out))
        return out

    async def _latest_observations(self, client: httpx.AsyncClient, series_id: str):
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 2,
        }
        resp = await client.get(_FRED_URL, params=params)
        resp.raise_for_status()
        rows = resp.json().get("observations", [])
        # Re-sort ascending and coerce numeric, dropping FRED's "." missing marker.
        cleaned = [
            {"date": r["date"], "value": float(r["value"])}
            for r in reversed(rows)
            if r.get("value") not in (".", "", None)
        ]
        return cleaned
