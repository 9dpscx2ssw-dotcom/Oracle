"""Shadow tracker for the offline RL advisory.

The offline policy emits an advisory target position per symbol each loop but never
trades. To build evidence for (or against) promoting it, we grade each advisory
against what actually happened next: when a new advisory arrives for a symbol, the
*previous* one is scored by the realized price move since it was issued.

Accumulates, across loops and restarts (persisted to JSON):
  • hit rate   — fraction of directional advisories whose sign matched the move,
  • shadow return — Σ sign(advice) · realized_return (what blindly following it
    would have earned, gross of costs — promotion needs this clearly positive),
  • per-action counts and a rolling window of recent graded calls.

It is purely observational; it never places or influences a trade.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

_SIGN = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}


class AdvisoryTracker:
    def __init__(self, path: str = "data/advisory_track.json", recent: int = 500):
        self.path = path
        self._recent_max = recent
        self._last: dict[str, dict] = {}      # symbol -> last advisory issued
        self.n_dir = 0                        # graded, directional (non-flat) calls
        self.hits = 0
        self.cum_return = 0.0                 # cumulative shadow return
        self.by_action: dict[str, int] = {"LONG": 0, "SHORT": 0, "FLAT": 0}
        self.recent: deque[dict] = deque(maxlen=recent)

    def record(self, symbol: str, action: str, price: float, ts: str) -> dict | None:
        """Log a new advisory for `symbol`, grading the previous one against the
        move since it was issued. Returns the graded record (or None on the first
        observation / a zero price)."""
        graded = None
        prev = self._last.get(symbol)
        if prev and prev.get("price"):
            ret = (price - prev["price"]) / prev["price"]
            sign = _SIGN.get(prev["action"], 0.0)
            self.by_action[prev["action"]] = self.by_action.get(prev["action"], 0) + 1
            if sign != 0.0:
                self.n_dir += 1
                correct = (ret > 0 and sign > 0) or (ret < 0 and sign < 0)
                self.hits += 1 if correct else 0
                self.cum_return += sign * ret
                graded = {
                    "ts": ts, "symbol": symbol, "action": prev["action"],
                    "realized_return": round(ret, 6), "correct": bool(correct),
                    "shadow_return": round(sign * ret, 6),
                }
                self.recent.append(graded)
        self._last[symbol] = {"action": action, "price": price, "ts": ts}
        return graded

    def snapshot(self) -> dict:
        return {
            "n_graded": self.n_dir,
            "hit_rate": round(self.hits / self.n_dir, 3) if self.n_dir else 0.0,
            "cum_shadow_return": round(self.cum_return, 4),
            "by_action": dict(self.by_action),
            "recent": list(self.recent)[-25:],
        }

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self) -> None:
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "last": self._last, "n_dir": self.n_dir, "hits": self.hits,
                "cum_return": self.cum_return, "by_action": self.by_action,
                "recent": list(self.recent),
            }))
        except OSError as e:
            log.warning("Advisory tracker save failed: %s", e)

    def load(self) -> bool:
        p = Path(self.path)
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text())
            self._last = d.get("last", {})
            self.n_dir = int(d.get("n_dir", 0))
            self.hits = int(d.get("hits", 0))
            self.cum_return = float(d.get("cum_return", 0.0))
            self.by_action = d.get("by_action", {"LONG": 0, "SHORT": 0, "FLAT": 0})
            self.recent = deque(d.get("recent", []), maxlen=self._recent_max)
            return True
        except (OSError, ValueError, TypeError) as e:
            log.warning("Advisory tracker load failed: %s", e)
            return False
