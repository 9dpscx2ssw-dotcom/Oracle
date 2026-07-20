#!/usr/bin/env python3
"""View RL convergence status and trends.

Usage:
    python scripts/convergence_report.py               # Show report
    python scripts/convergence_report.py --raw         # Show raw entries
    python scripts/convergence_report.py --tail N      # Show last N entries
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Load ConvergenceMonitor straight from its file, NOT via the gungnir package:
# importing gungnir.learning.rl would pull in the RL policy → pandas/numpy,
# which aren't installed on the host (they live in the Docker image). The
# monitor itself is stdlib-only, so this script runs anywhere.
_mod_path = Path(__file__).parent.parent / "src/gungnir/learning/rl/convergence_monitor.py"
_spec = importlib.util.spec_from_file_location("convergence_monitor", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ConvergenceMonitor = _mod.ConvergenceMonitor


def main():
    args = sys.argv[1:]
    log_path = Path("data/rl_convergence.jsonl")

    if not log_path.exists():
        print(f"No convergence log found at {log_path}")
        print("Logs are created automatically when RL is enabled and running.")
        sys.exit(1)

    monitor = ConvergenceMonitor(log_path)

    if "--raw" in args:
        # Show raw JSON entries
        with open(log_path) as f:
            for i, line in enumerate(f, 1):
                print(f"[{i}] {line.rstrip()}")

    elif "--tail" in args:
        # Show last N entries
        try:
            idx = args.index("--tail")
            n = int(args[idx + 1])
        except (ValueError, IndexError):
            print("Usage: --tail N")
            sys.exit(1)

        with open(log_path) as f:
            lines = f.readlines()
            for line in lines[-n:]:
                data = json.loads(line)
                ts = data.get("timestamp", "?")
                cum_reward = data.get("cumulative_reward", 0)
                avg_reward = data.get("avg_reward", 0)
                epsilon = data.get("epsilon", 0)
                updates = data.get("updates", 0)
                takes = data.get("action_counts", {}).get("take", 0)
                skips = data.get("action_counts", {}).get("skip", 0)
                print(
                    f"{ts:19} | "
                    f"Reward={cum_reward:7.2f} | "
                    f"AvgR={avg_reward:8.4f} | "
                    f"ε={epsilon:5.3f} | "
                    f"Updates={updates:4d} | "
                    f"Take/Skip={takes:3d}/{skips:3d}"
                )

    else:
        # Show full report (default)
        print(monitor.report())
        print()

        # Check for divergence
        is_div, reason = monitor.check_divergence()
        if is_div:
            print(f"⚠️  WARNING: {reason}\n")
            sys.exit(1)  # Exit with error code if diverging
        else:
            print("✓ No divergence detected.\n")


if __name__ == "__main__":
    main()
