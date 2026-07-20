"""Append-only, hash-chained audit trail.

Every consequential action — orders, closes, mode changes, kill-switch events,
resets, applied parameter changes — is appended as one JSON line. Each entry
carries the SHA-256 of the previous entry, so any retroactive edit or deletion
breaks the chain and is detectable with `verify()`. This is the institutional
"who did what, when" record: separate from the trading DB, human-readable,
and cheap enough to write on every event.

The log is advisory evidence, not a control: writing it must never block or
break trading, so all failures degrade to a logged warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str | Path = "data/audit.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._last_hash = self._tail_hash()

    def _tail_hash(self) -> str:
        """Hash of the last entry on disk (chain continues across restarts)."""
        try:
            if not self.path.exists():
                return _GENESIS
            last = None
            with self.path.open("rb") as f:
                for line in f:
                    if line.strip():
                        last = line
            if not last:
                return _GENESIS
            return json.loads(last).get("hash", _GENESIS)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Audit log tail unreadable (%s); chaining from genesis", e)
            return _GENESIS

    @staticmethod
    def _entry_hash(prev: str, payload: str) -> str:
        return hashlib.sha256((prev + payload).encode()).hexdigest()

    def record(self, event: str, **fields) -> None:
        """Append one event. Never raises into the caller."""
        try:
            with self._lock:
                body = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    **fields,
                    "prev": self._last_hash,
                }
                payload = json.dumps(body, sort_keys=True, default=str)
                body["hash"] = self._entry_hash(self._last_hash, payload)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(body, default=str) + "\n")
                self._last_hash = body["hash"]
        except Exception as e:  # noqa: BLE001 — audit must never break trading
            log.warning("Audit write failed for %s: %s", event, e)

    def verify(self) -> tuple[bool, int, str | None]:
        """Walk the chain. Returns (intact, entries_checked, first_bad_ts)."""
        prev, n = _GENESIS, 0
        try:
            if not self.path.exists():
                return True, 0, None
            with self.path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    claimed = entry.pop("hash", "")
                    payload = json.dumps(entry, sort_keys=True, default=str)
                    if entry.get("prev") != prev or self._entry_hash(prev, payload) != claimed:
                        return False, n, entry.get("ts")
                    prev = claimed
                    n += 1
            return True, n, None
        except (OSError, json.JSONDecodeError):
            return False, n, None

    def tail(self, limit: int = 100) -> list[dict]:
        """Most recent entries, newest last (for the dashboard)."""
        try:
            if not self.path.exists():
                return []
            lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
            return [json.loads(ln) for ln in lines[-limit:]]
        except (OSError, json.JSONDecodeError):
            return []
