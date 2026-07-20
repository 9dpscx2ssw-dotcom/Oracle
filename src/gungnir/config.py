"""Config loading: YAML for tunables + .env for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be one of true/false/1/0/yes/no/on/off")


class Secrets(BaseModel):
    """Pulled from environment / .env. Never logged, never serialized to disk."""

    capital_com_api_key: str = ""
    capital_com_identifier: str = ""        # account login (usually your email)
    capital_com_password: str = ""          # the API key's custom password
    # SAFE DEFAULT: demo unless explicitly disabled (audit F-00b — a missing or
    # typo'd CAPITAL_COM_DEMO used to silently select the live-money endpoint).
    capital_com_demo: bool = True
    capital_com_api_url: str = ""           # optional explicit base-url override
    anthropic_api_key: str = ""
    # Haiku by default: sentiment/prediction are 3-field JSON tasks over headlines —
    # a frontier model is ~10× the cost for no measurable edge on them.
    anthropic_model: str = "claude-haiku-4-5-20251001"
    alert_webhook_url: str = ""             # operator alerts (Discord/Slack/CallMeBot URL)
    alert_telegram_bot_token: str = ""      # Telegram bot token for alerts
    alert_telegram_chat_id: str = ""        # Telegram chat id to send alerts to
    fred_api_key: str = ""
    newsapi_key: str = ""
    finnhub_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        load_dotenv()
        return cls(
            capital_com_api_key=os.getenv("CAPITAL_COM_API_KEY", ""),
            capital_com_identifier=os.getenv("CAPITAL_COM_IDENTIFIER", ""),
            capital_com_password=os.getenv("CAPITAL_COM_PASSWORD", ""),
            capital_com_demo=_env_bool("CAPITAL_COM_DEMO", True),
            capital_com_api_url=os.getenv("CAPITAL_COM_API_URL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
            alert_telegram_bot_token=os.getenv("ALERT_TELEGRAM_BOT_TOKEN", ""),
            alert_telegram_chat_id=os.getenv("ALERT_TELEGRAM_CHAT_ID", ""),
            fred_api_key=os.getenv("FRED_API_KEY", ""),
            newsapi_key=os.getenv("NEWSAPI_KEY", ""),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        )


class Config:
    """Thin wrapper around the parsed YAML config + secrets.

    Kept as a dict-backed object so adding config keys doesn't require schema
    churn during early development. Tighten into pydantic models as it settles.
    """

    def __init__(self, raw: dict[str, Any], secrets: Secrets):
        self.raw = raw
        self.secrets = secrets

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def dry_run(self) -> bool:
        if "GUNGNIR_DRY_RUN" in os.environ:
            return _env_bool("GUNGNIR_DRY_RUN", True)
        return bool(self.get("agent", "dry_run", default=True))

    @classmethod
    def load(cls, config_path: str | Path) -> "Config":
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        return cls(raw or {}, Secrets.from_env())
