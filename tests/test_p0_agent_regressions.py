"""P0 regression tests for deterministic agent safety defects."""

from __future__ import annotations

import ast
from pathlib import Path


AGENT_SOURCE = Path(__file__).parents[1] / "src/gungnir/core/agent.py"


def _method(name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(AGENT_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing async method {name}")


def test_fast_step_loads_positions_only_after_assignment():
    """Risk refresh must not load positions before the local is initialized."""
    method = _method("fast_step")
    assignments = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "positions"
                for target in node.targets)
    ]
    loads = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and node.id == "positions"
        and isinstance(node.ctx, ast.Load)
    ]
    assert assignments, "fast_step must initialize positions"
    assert loads, "fast_step must consume positions"
    assert min(assignments) < min(loads), "positions is consumed before initialization"


def test_consensus_step_initializes_go_live_before_use():
    """Consensus mode must derive execution mode from Agent state."""
    method = _method("_consensus_step")
    assignments = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "go_live"
                for target in node.targets)
    ]
    loads = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and node.id == "go_live"
        and isinstance(node.ctx, ast.Load)
    ]
    assert assignments, "consensus_step must initialize go_live"
    assert loads, "consensus_step must use its execution mode"
    assert min(assignments) < min(loads), "go_live is consumed before initialization"
    paper_mode_loads = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and node.attr == "_paper_mode"
    ]
    assert paper_mode_loads, "consensus execution must use the Agent runtime mode"
