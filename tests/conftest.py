"""Shared test fixtures.

Isolate every test's working directory into a throwaway tmp dir. Several
components persist small JSON ledgers under a relative ``data/`` path
(pending-order intents, compliance counter, risk-breaker state, and now the
live-position and virtual-book ledgers). Without isolation those writes would
land in the repo's real ``data/`` directory — polluting the running system's
state and leaking between tests (e.g. a position submitted in one test would
be rehydrated by the next broker constructed). chdir-ing into ``tmp_path`` per
test keeps every relative ``data/...`` write hermetic. No test depends on the
repo-root cwd (all source/config reads are ``__file__``-relative).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield
