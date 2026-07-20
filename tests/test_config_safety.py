from __future__ import annotations

import pytest

from gungnir.config import Config, Secrets


def test_dry_run_rejects_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("GUNGNIR_DRY_RUN", "definitely-not-a-bool")
    config = Config({"agent": {"dry_run": True}}, Secrets())
    with pytest.raises(ValueError, match="GUNGNIR_DRY_RUN"):
        _ = config.dry_run


def test_dry_run_accepts_explicit_false(monkeypatch):
    monkeypatch.setenv("GUNGNIR_DRY_RUN", "false")
    config = Config({"agent": {"dry_run": True}}, Secrets())
    assert config.dry_run is False


def test_dry_run_defaults_safe_when_environment_missing(monkeypatch):
    monkeypatch.delenv("GUNGNIR_DRY_RUN", raising=False)
    config = Config({}, Secrets())
    assert config.dry_run is True


def test_capital_com_demo_rejects_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("CAPITAL_COM_DEMO", "maybe")
    with pytest.raises(ValueError, match="CAPITAL_COM_DEMO"):
        Secrets.from_env()
