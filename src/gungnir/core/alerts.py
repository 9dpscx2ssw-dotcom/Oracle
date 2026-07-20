"""Operator alerting: webhook, Telegram, WhatsApp/Signal (via URL template).

A solo-run trading system's biggest operational gap is that nobody is watching
the logs. When something state-changing happens — trading halted, kill switch
engaged, loop crash-looping, RL collapse, reconciliation divergence — this
pushes a message to every configured channel:

  • ALERT_WEBHOOK_URL
      - POST JSON {"content", "text"} → Discord / Slack / Mattermost webhooks
      - if the URL contains "{message}", the alert is URL-encoded into it and
        sent as GET → CallMeBot (WhatsApp & Signal), ntfy, and similar
        query-parameter receivers
  • ALERT_TELEGRAM_BOT_TOKEN + ALERT_TELEGRAM_CHAT_ID
      - native Telegram Bot API sendMessage

Alerts are throttled per key (default 15 min) so a persistent condition pages
once, not once per loop, and are sent from a daemon thread so a slow or dead
receiver can never stall the trading loop.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse

log = logging.getLogger(__name__)


class Alerter:
    def __init__(self, webhook_url: str = "", telegram_token: str = "",
                 telegram_chat_id: str = "", min_interval: float = 900.0):
        self.url = (webhook_url or "").strip()
        self.tg_token = (telegram_token or "").strip()
        self.tg_chat = (telegram_chat_id or "").strip()
        self.min_interval = min_interval
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        if not self.enabled:
            log.info("No alert channel configured (ALERT_WEBHOOK_URL / "
                     "ALERT_TELEGRAM_*) — operator alerts are log-only.")

    @property
    def enabled(self) -> bool:
        return bool(self.url or (self.tg_token and self.tg_chat))

    def send(self, key: str, message: str, critical: bool = False) -> None:
        """Fire an alert. Throttled per key unless ``critical`` — state changes
        a human must see (kill engage AND disengage) always deliver."""
        log.warning("ALERT [%s]: %s", key, message)
        if not self.enabled:
            return
        if not critical:
            now = time.monotonic()
            with self._lock:
                last = self._last_sent.get(key)
                # None-sentinel, not 0.0: monotonic() starts near zero in a
                # fresh process, so "now - 0.0 < interval" would silently
                # swallow every alert for the first min_interval after boot.
                if last is not None and now - last < self.min_interval:
                    return
                self._last_sent[key] = now
        threading.Thread(target=self._post, args=(key, message), daemon=True).start()

    # ── delivery (daemon thread) ───────────────────────────────────────────────

    def _post(self, key: str, message: str) -> None:
        body = f"🛡 Gungnir [{key}] {message}"
        for deliver in (self._deliver_webhook, self._deliver_telegram):
            try:
                deliver(body)
            except Exception as e:  # noqa: BLE001 — alerting must never break trading
                log.warning("Alert delivery FAILED (%s): %s", deliver.__name__, e)

    def _deliver_webhook(self, body: str) -> None:
        if not self.url:
            return
        import httpx
        if "{message}" in self.url:
            # Query-parameter receivers: CallMeBot (WhatsApp/Signal), ntfy, …
            r = httpx.get(self.url.replace("{message}", urllib.parse.quote(body)),
                          timeout=10.0)
        else:
            # JSON-body receivers: Discord ("content") / Slack ("text").
            r = httpx.post(self.url, json={"content": body, "text": body}, timeout=10.0)
        # 4xx/5xx don't raise on their own — without this, a bad webhook URL
        # fails forever in silence.
        r.raise_for_status()
        log.info("Alert delivered via webhook (%d)", r.status_code)

    def _deliver_telegram(self, body: str) -> None:
        if not (self.tg_token and self.tg_chat):
            return
        import httpx
        r = httpx.post(
            f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
            json={"chat_id": self.tg_chat, "text": body},
            timeout=10.0,
        )
        # Telegram reports config errors (bad token → 401, bad/never-messaged
        # chat_id → 400) in the response body; surface the description.
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            data = {}
        if r.status_code != 200 or not data.get("ok", False):
            raise RuntimeError(
                f"Telegram sendMessage failed ({r.status_code}): "
                f"{data.get('description', r.text[:200])} — check "
                "ALERT_TELEGRAM_BOT_TOKEN / ALERT_TELEGRAM_CHAT_ID, and make "
                "sure you have messaged the bot at least once.")
        log.info("Alert delivered via Telegram")
