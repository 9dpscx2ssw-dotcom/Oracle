"""Control channel between the dashboard and the agent.

The dashboard and agent are separate processes that share the ./data volume. To
keep them decoupled, the dashboard never calls the agent directly — it writes a
small `control.json` of *desired* state, and the agent reads + applies it at the
top of every fast loop. Last write wins; the file is written atomically.

Contract:
    {
      "strategies": { "<name>": "off" | "shadow" | "live", ... },
      "paused": bool,            # global pause: agent opens no new positions
      "updated_at": iso8601
    }
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class Control:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> dict:
        if not self.path.exists():
            return {"strategies": {}, "instruments": {}, "paused": False,
                    "kill": False, "risk_settings": {}, "runtime": {}}
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"strategies": {}, "instruments": {}, "paused": False,
                    "kill": False, "risk_settings": {}, "runtime": {}}
        data.setdefault("strategies", {})
        data.setdefault("instruments", {})
        data.setdefault("paused", False)
        data.setdefault("kill", False)
        data.setdefault("risk_settings", {})
        data.setdefault("runtime", {})
        return data

    def write(self, data: dict) -> None:
        import logging
        log = logging.getLogger(__name__)
        data = {**data, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=self.path.parent, delete=False, suffix=".tmp"
            ) as tmp:
                json.dump(data, tmp)
                tmp_path = tmp.name
            Path(tmp_path).replace(self.path)
            log.debug("Wrote control state to %s", self.path)
        except Exception as e:
            log.error("Failed to write control.json at %s: %s", self.path, e)
            raise

    def clear_keys(self, *keys: str) -> None:
        """Consume one-shot command keys without clobbering concurrent writes.

        The agent used to read the whole doc at loop start, act, then write its
        (by now stale) copy back — losing any dashboard write that landed in
        between. Re-reading immediately before deleting only the consumed keys
        shrinks the race window from a whole loop iteration to microseconds and
        never overwrites unrelated fields with stale values.
        """
        data = self.read()
        changed = False
        for k in keys:
            if k in data:
                del data[k]
                changed = True
        if changed:
            self.write(data)

    def set_strategy_mode(self, name: str, mode: str) -> dict:
        data = self.read()
        data["strategies"][name] = mode
        self.write(data)
        return data

    def set_instrument_enabled(self, symbol: str, enabled: bool) -> dict:
        data = self.read()
        data["instruments"][symbol] = bool(enabled)
        self.write(data)
        return data

    def set_paused(self, paused: bool) -> dict:
        data = self.read()
        data["paused"] = bool(paused)
        self.write(data)
        return data
