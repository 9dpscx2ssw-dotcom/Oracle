"""RL convergence tracker: log metrics, detect divergence, surface trends."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class ConvergenceMonitor:
    """Track RL convergence by logging snapshots and detecting divergence."""

    def __init__(self, log_path: str | Path = "data/rl_convergence.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_snapshot = None

    def record(self, snapshot: dict) -> None:
        """Log a policy snapshot with timestamp and analysis."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **snapshot,
        }
        # Append to JSONL file (one JSON object per line)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._last_snapshot = entry

    def check_divergence(self) -> tuple[bool, str]:
        """Detect divergence patterns; return (is_diverging, reason)."""
        if not self.log_path.exists() or self._last_snapshot is None:
            return False, ""

        # Read last 10 entries to check trends
        entries = []
        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    entries.append(json.loads(line))
        except Exception as e:
            log.warning("Failed to read convergence log: %s", e)
            return False, ""

        if len(entries) < 2:
            return False, ""

        latest = entries[-1]
        prev = entries[-2]

        # Check 1: Reward collapse (>20% drop in last entry)
        prev_cumulative = prev.get("cumulative_reward", 0)
        curr_cumulative = latest.get("cumulative_reward", 0)
        if prev_cumulative > 0 and curr_cumulative < prev_cumulative * 0.8:
            return True, (
                f"Reward collapsed: {prev_cumulative:.2f} → {curr_cumulative:.2f} "
                f"({(1 - curr_cumulative/prev_cumulative)*100:.0f}% drop)"
            )

        # Check 2: Avg reward went negative
        avg_reward = latest.get("avg_reward", 0)
        if avg_reward < -0.05:
            return True, f"Avg reward negative: {avg_reward:.4f}"

        # Check 3: Epsilon not decaying (stuck)
        if len(entries) >= 5:
            epsilons = [e.get("epsilon", 0.1) for e in entries[-5:]]
            if epsilons[-1] == epsilons[-4]:  # No change over 4 entries
                return True, f"Exploration stuck at ε={epsilons[-1]:.3f}"

        # Check 4: Action mix inverted (was taking, now all-skipping or vice versa)
        actions = latest.get("action_counts", {})
        prev_actions = prev.get("action_counts", {})
        if prev_actions.get("take", 0) > 0 and actions.get("take", 0) == 0:
            return True, "Policy stopped taking signals (action mix collapsed)"
        if prev_actions.get("skip", 0) > 0 and actions.get("skip", 0) == 0:
            return True, "Policy stopped skipping signals (action mix inverted)"

        return False, ""

    def summary(self, hours: int = 24) -> dict:
        """Return convergence summary over the last N hours."""
        if not self.log_path.exists():
            return {"entries": 0, "status": "no_data"}

        entries = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.warning("Failed to read convergence log: %s", e)
            return {"entries": 0, "status": "error", "error": str(e)}

        if not entries:
            return {"entries": 0, "status": "no_data"}

        # Filter by time if needed (simplified: just use last N entries)
        recent = entries[-20:]  # Last ~10 hours at 30-min intervals

        first = recent[0]
        last = recent[-1]

        # Compute trends
        reward_start = first.get("cumulative_reward", 0)
        reward_end = last.get("cumulative_reward", 0)
        reward_change = reward_end - reward_start
        reward_pct = (reward_change / reward_start * 100) if reward_start > 0 else 0

        epsilon_start = first.get("epsilon", 0.3)
        epsilon_end = last.get("epsilon", 0.3)

        avg_reward_start = first.get("avg_reward", 0)
        avg_reward_end = last.get("avg_reward", 0)

        # Recent action mix
        last_actions = last.get("action_counts", {})
        total = last_actions.get("take", 0) + last_actions.get("skip", 0)
        take_rate = last_actions.get("take", 0) / total if total > 0 else 0

        return {
            "entries": len(entries),
            "hours_tracked": len(recent) * 0.5,  # 30-min intervals
            "status": "converging" if reward_pct > 0 else "flat" if reward_pct > -5 else "diverging",
            "cumulative_reward": {
                "start": round(reward_start, 2),
                "end": round(reward_end, 2),
                "change": round(reward_change, 2),
                "percent_change": round(reward_pct, 1),
            },
            "avg_reward": {
                "start": round(avg_reward_start, 4),
                "end": round(avg_reward_end, 4),
                "trend": "improving" if avg_reward_end > avg_reward_start else "declining",
            },
            "exploration": {
                "epsilon_start": round(epsilon_start, 3),
                "epsilon_end": round(epsilon_end, 3),
                "decaying": epsilon_end < epsilon_start,
            },
            "action_mix": {
                "take_rate": round(take_rate, 3),
                "take_count": last_actions.get("take", 0),
                "skip_count": last_actions.get("skip", 0),
            },
            "updates": last.get("updates", 0),
            "states_learned": last.get("states_learned", 0),
        }

    def report(self) -> str:
        """Human-readable convergence report."""
        summary = self.summary()
        if summary.get("entries", 0) == 0:
            return "No convergence data yet."

        s = summary
        lines = [
            f"📊 RL Convergence Report ({s['hours_tracked']:.1f}h tracked, {s['entries']} samples)",
            f"Status: {s['status'].upper()}",
            "",
            f"Cumulative Reward: {s['cumulative_reward']['start']:.2f} → {s['cumulative_reward']['end']:.2f} "
            f"({s['cumulative_reward']['percent_change']:+.1f}%)",
            f"Avg Reward: {s['avg_reward']['start']:.4f} → {s['avg_reward']['end']:.4f} "
            f"({s['avg_reward']['trend']})",
            f"Exploration: ε {s['exploration']['epsilon_start']:.3f} → {s['exploration']['epsilon_end']:.3f} "
            f"({'✓ decaying' if s['exploration']['decaying'] else '✗ not decaying'})",
            f"Action Mix: {s['action_mix']['take_rate']*100:.1f}% take, "
            f"{(1-s['action_mix']['take_rate'])*100:.1f}% skip "
            f"({s['action_mix']['take_count']} / {s['action_mix']['skip_count']})",
            f"Updates: {s['updates']} gradient steps | States: {s['states_learned']} in buffer",
        ]

        is_diverging, reason = self.check_divergence()
        if is_diverging:
            lines.append("")
            lines.append(f"⚠️  DIVERGENCE DETECTED: {reason}")

        return "\n".join(lines)
