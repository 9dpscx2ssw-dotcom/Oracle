"""RL convergence tracking and divergence detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gungnir.learning.rl.convergence_monitor import ConvergenceMonitor


def _snapshot(**kwargs) -> dict:
    """Stub RL policy snapshot."""
    return {
        "enabled": True,
        "mode": "active",
        "updates": kwargs.get("updates", 100),
        "cumulative_reward": kwargs.get("cumulative_reward", 50.0),
        "avg_reward": kwargs.get("avg_reward", 0.02),
        "epsilon": kwargs.get("epsilon", 0.2),
        "states_learned": kwargs.get("states_learned", 434),
        "action_counts": kwargs.get("action_counts", {"take": 48, "skip": 52}),
        "reward_history": kwargs.get("reward_history", [0.01, 0.02, 0.03]),
    }


def test_monitor_logs_snapshot():
    """Monitor records snapshots to JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(updates=10))
        monitor.record(_snapshot(updates=20))
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2


def test_monitor_detects_reward_collapse():
    """Divergence detection: reward drops 20%+."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(cumulative_reward=100.0))
        monitor.record(_snapshot(cumulative_reward=79.0))  # 21% drop
        is_div, reason = monitor.check_divergence()
        assert is_div
        assert "collapsed" in reason.lower()


def test_monitor_detects_negative_avg_reward():
    """Divergence detection: avg reward goes negative."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(avg_reward=0.01))
        monitor.record(_snapshot(avg_reward=-0.1))  # Negative
        is_div, reason = monitor.check_divergence()
        assert is_div
        assert "negative" in reason.lower()


def test_monitor_detects_stuck_exploration():
    """Divergence detection: epsilon not decaying."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        # Add 5 entries with same epsilon
        for _ in range(5):
            monitor.record(_snapshot(epsilon=0.3))
        is_div, reason = monitor.check_divergence()
        assert is_div
        assert "stuck" in reason.lower()


def test_monitor_detects_action_collapse():
    """Divergence detection: action mix goes to all-skip or all-take."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(action_counts={"take": 50, "skip": 50}))
        monitor.record(_snapshot(action_counts={"take": 0, "skip": 100}))  # All skip
        is_div, reason = monitor.check_divergence()
        assert is_div
        assert "stopped" in reason.lower()


def test_monitor_summary():
    """Summary computes trends over time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(cumulative_reward=40.0, avg_reward=0.01, epsilon=0.30))
        monitor.record(_snapshot(cumulative_reward=50.0, avg_reward=0.02, epsilon=0.25))
        monitor.record(_snapshot(cumulative_reward=60.0, avg_reward=0.025, epsilon=0.20))

        summary = monitor.summary()
        assert summary["status"] in ("converging", "flat", "diverging")
        assert summary["cumulative_reward"]["change"] == 20.0  # 60 - 40
        assert summary["cumulative_reward"]["percent_change"] > 0
        assert summary["exploration"]["decaying"]  # 0.20 < 0.30


def test_monitor_report_readable():
    """Report generates human-readable output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        monitor.record(_snapshot(cumulative_reward=50.0))
        monitor.record(_snapshot(cumulative_reward=60.0))

        report = monitor.report()
        assert "Convergence Report" in report
        assert "50.00" in report or "60.00" in report
        assert "Status:" in report


def test_monitor_no_divergence_when_healthy():
    """Healthy convergence doesn't trigger divergence alerts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "convergence.jsonl"
        monitor = ConvergenceMonitor(path)
        # Steady improvement
        for i in range(5):
            monitor.record(_snapshot(
                cumulative_reward=40.0 + i * 10,
                avg_reward=0.01 + i * 0.005,
                epsilon=0.30 - i * 0.02,
                updates=100 + i * 10,
            ))
        is_div, reason = monitor.check_divergence()
        assert not is_div
