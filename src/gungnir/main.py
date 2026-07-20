"""Entry point: build the agent from config and run the two-clock scheduler.

    python -m gungnir.main --config config/config.yaml [--dry-run]

In `--dry-run` (default) the agent uses a PaperBroker and a StubMarketFeed unless
real Capital.com credentials are present, so you can run the full pipeline end-to-end
without touching a live account.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from .config import Config
from .core.agent import Agent
from .core.scheduler import Scheduler
from .data.capital_com_feed import CapitalComMarketFeed
from .data.macro_feed import FredMacroFeed
from .data.market_feed import SyntheticMarketFeed
from .data.news_feed import CompositeNewsFeed
from .execution.broker import PaperBroker
from .execution.capital_com import CapitalComBroker
from .execution.capital_session import CapitalComSession
from .learning.journal import Journal
from .llm.client import build_llm
from .persistence.db import Database
from .risk.portfolio import PortfolioRisk
from .risk.position_sizing import build_sizer
from .strategy.registry import StrategyRegistry

log = logging.getLogger("gungnir")


def build_agent(config: Config) -> Agent:
    sec = config.secrets
    # Live needs the full credential set: API key + account login + password.
    live = not config.dry_run and bool(
        sec.capital_com_api_key and sec.capital_com_identifier and sec.capital_com_password
    )

    if live:
        from .execution.capital_session import DEMO_URL, LIVE_URL
        resolved_url = sec.capital_com_api_url or (DEMO_URL if sec.capital_com_demo else LIVE_URL)
        endpoint = "demo" if resolved_url == DEMO_URL else "live"
        # The URL override silently beats the demo flag — refuse ambiguity.
        if sec.capital_com_api_url and sec.capital_com_demo and endpoint == "live":
            raise RuntimeError(
                "CAPITAL_COM_DEMO=true but CAPITAL_COM_API_URL points at the LIVE "
                "endpoint. Remove the URL override or set CAPITAL_COM_DEMO=false.")
        # Real money requires an explicit second switch — a missing env var or a
        # typo must never be enough to trade a live account (audit F-00b).
        if endpoint == "live" and os.getenv("CAPITAL_COM_ALLOW_LIVE", "").strip().lower() not in ("1", "true", "yes"):
            raise RuntimeError(
                "Refusing to connect to the LIVE Capital.com endpoint: set "
                "CAPITAL_COM_ALLOW_LIVE=true to confirm real-money trading, or set "
                "CAPITAL_COM_DEMO=true for the demo account.")
        log.warning("Capital.com endpoint: %s (%s)", endpoint.upper(), resolved_url)
        session = CapitalComSession(
            sec.capital_com_api_key, sec.capital_com_identifier, sec.capital_com_password,
            demo=sec.capital_com_demo, base_url=sec.capital_com_api_url or None,
            min_interval=float(config.get(
                "data", "market", "min_request_interval", default=0.12) or 0.0),
        )
        market = CapitalComMarketFeed(session, config)
        broker = CapitalComBroker(session)
    else:
        # A moving synthetic feed (not a flat stub) so paper mode actually
        # produces signals → trades → RL, and the dashboard fills with data.
        from .backtest.costs import CostModel
        market = SyntheticMarketFeed()
        broker = PaperBroker(cost=CostModel.from_config(config))

    news = CompositeNewsFeed(config)
    macro = FredMacroFeed(config)
    llm = build_llm(config)

    strategies = StrategyRegistry.from_yaml(_strategies_path(config))

    sizer = build_sizer(config)
    risk = PortfolioRisk(config)
    db = Database(config.get("persistence", "db_path", default="data/gungnir.db"))
    journal = Journal(db)

    # "connected" ≠ "real money": the broker connection may target the demo
    # endpoint. Keep the two concepts separate in the banner (audit F-00b).
    if live:
        mode = f"Capital.com-connected ({'DEMO' if sec.capital_com_demo and not sec.capital_com_api_url else 'see endpoint log above'})"
    else:
        mode = "DRY-RUN (paper)"
    log.info("Gungnir starting in %s mode over %d strategies",
             mode, len(strategies.active()))

    return Agent(config, market, news, macro, llm, strategies, sizer, risk, broker, journal)


def _strategies_path(config: Config) -> str:
    """Resolve the (mutable) strategies state file.

    The agent rewrites this on every learning cycle and on dashboard toggles, so
    it must be writable. We keep it in the data volume (not the read-only config
    mount), seeding it on first run from config/strategies.yaml or the example.
    """
    configured = config.get("strategies_path", default="data/strategies.yaml")
    path = Path(configured)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for seed in (Path("config/strategies.yaml"), Path("config/strategies.example.yaml")):
            if seed.exists():
                try:
                    path.write_text(seed.read_text())
                    log.info("Seeded strategies state at %s from %s", path, seed)
                    # A fresh seed means fresh intent: drop stale per-strategy
                    # mode overrides left in control.json by earlier sessions —
                    # _apply_control re-applies them every loop and would
                    # silently override the seed's modes (and make dashboard
                    # toggles appear broken by fighting user changes).
                    try:
                        from .core.control import Control
                        ctrl = Control(config.get("dashboard", "control_path",
                                                  default="data/control.json"))
                        data = ctrl.read()
                        if data.get("strategies"):
                            data["strategies"] = {}
                            ctrl.write(data)
                            log.info("Cleared stale strategy modes from control.json "
                                     "(fresh strategies seed)")
                    except Exception as e:  # noqa: BLE001 — best-effort cleanup
                        log.warning("Could not clear stale control modes: %s", e)
                    break
                except (OSError, PermissionError) as e:
                    log.warning("Could not write to %s: %s. Using seed file directly.", path, e)
                    # If we can't write, just use the seed file directly
                    return str(seed)
    return str(path)


async def run(config: Config) -> None:
    # If live, connect the real feeds/broker before looping.
    agent = build_agent(config)
    for component in (agent.market, agent.broker):
        connect = getattr(component, "connect", None)
        if callable(connect):
            await connect()

    scheduler = Scheduler(
        fast_seconds=config.get("agent", "fast_loop_seconds", default=30),
        slow_seconds=config.get("agent", "slow_loop_seconds", default=3600),
        on_critical=lambda label, n: agent.alerter.send(
            "loop-failure", f"{label} loop has failed {n} consecutive iterations — "
            "the agent is not trading correctly."),
    )
    # Share the loop-duration dict so status.json shows iteration wall times.
    agent.loop_durations = scheduler.last_duration
    # Prometheus scrape endpoint (metrics.enabled, default on; no-op without
    # prometheus_client installed).
    from .core import metrics
    metrics.setup(config)

    # Stop cleanly on SIGINT/SIGTERM (e.g. `docker compose down`/restart) so the
    # finally block can flush learned state before the process exits.
    import signal
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, scheduler.stop)
        except (NotImplementedError, RuntimeError):
            pass  # not supported on this platform (e.g. Windows)

    try:
        await scheduler.run(agent.fast_step, agent.slow_step)
    finally:
        log.info("Flushing learned state before exit…")
        agent.persist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gungnir trading agent")
    parser.add_argument("--config", default="config/config.example.yaml")
    parser.add_argument("--dry-run", action="store_true", help="force paper trading")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["GUNGNIR_DRY_RUN"] = "true"

    logging.basicConfig(
        level=os.getenv("GUNGNIR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs every request URL at INFO — which leaks API keys passed as query
    # params (e.g. the FRED key). Quiet it to WARNING so secrets stay out of logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Work out of the box: if the chosen config is missing, fall back to the
    # bundled example so a fresh container still starts (in dry-run by default).
    config_path = args.config
    if not Path(config_path).exists():
        fallback = "config/config.example.yaml"
        log.warning("Config %s not found; falling back to %s", config_path, fallback)
        config_path = fallback

    config = Config.load(config_path)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
