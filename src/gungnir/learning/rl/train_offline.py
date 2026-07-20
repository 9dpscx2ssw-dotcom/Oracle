"""CLI: train the offline Double-DQN on a replay of real (or synthetic) candles.

    python -m gungnir.learning.rl.train_offline --symbol EURUSD --timeframe 1h \
        --bars 1000 --epochs 40 --spread-bps 8 --commission-bps 1

Prefers real Capital.com candles when credentials are present, falling back to a
synthetic series otherwise. Trains offline (no live exploration), saves the policy
to data/offline_dqn.npz, and prints diagnostics plus a greedy roll-out. This is an
advisory artifact — it is NOT wired into live execution.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ...backtest import engine
from ...backtest.costs import CostModel
from ...config import Config
from ...features.feature_store import build_kraken_series
from .env import TradingEnv
from .iql import IQL
from .offline import collect_transitions, evaluate_policy, train_offline
from .validation import run_walk_forward

log = logging.getLogger(__name__)


def _load_candles(cfg: Config, symbol: str, timeframe: str, bars: int):
    sec = cfg.secrets
    if sec.capital_com_api_key and sec.capital_com_identifier and sec.capital_com_password:
        try:
            from ...data.capital_com_feed import CapitalComMarketFeed
            from ...execution.capital_session import CapitalComSession

            async def _fetch():
                session = CapitalComSession(
                    sec.capital_com_api_key, sec.capital_com_identifier,
                    sec.capital_com_password, demo=sec.capital_com_demo,
                    base_url=sec.capital_com_api_url or None)
                feed = CapitalComMarketFeed(session, cfg)
                await feed.connect()
                return await feed.recent_candles(symbol, timeframe, n=bars)

            candles = asyncio.run(_fetch())
            if candles and len(candles) >= 100:
                return candles, "capital.com"
            log.warning("Capital.com returned %d candles; using synthetic", len(candles or []))
        except Exception as e:  # noqa: BLE001
            log.warning("Capital.com fetch failed (%s); using synthetic", e)
    return engine.synthetic_candles(symbol, n=bars, seed=42), "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train offline trading DQN")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--bars", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--passes", type=int, default=4, help="random-exploration passes")
    ap.add_argument("--algo", choices=("dqn", "iql"), default="dqn")
    ap.add_argument("--walk-forward", action="store_true",
                    help="purged walk-forward out-of-sample validation instead of a single fit")
    ap.add_argument("--spread-bps", type=float, default=0.0)
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--out", default="data/offline_dqn.npz")
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)
    cost = CostModel(args.spread_bps, args.commission_bps, args.slippage_bps)

    candles, source = _load_candles(cfg, args.symbol, args.timeframe, args.bars)
    log.info("Loaded %d %s candles (%s) for %s | algo=%s",
             len(candles), args.timeframe, source, args.symbol, args.algo)

    if args.walk_forward:
        out = run_walk_forward(candles, cost=cost, algo=args.algo,
                               epochs=args.epochs, passes=args.passes)
        log.info("Walk-forward (%d folds): mean OOS return=%.4f",
                 out["n_folds"], out["mean_oos_return"])
        for f in out["folds"]:
            log.info("  train%s test%s -> return=%.4f actions=%s",
                     f["train"], f["test"], f["total_return"], f["action_counts"])
        return

    feats = build_kraken_series(args.symbol, candles)
    if args.algo == "iql":
        env = TradingEnv(candles, feats, cost=cost)
        data = []
        for p in range(args.passes):
            data += collect_transitions(env, epsilon=1.0, seed=p)
        agent = IQL(seed=0)
        diag = agent.train(data, epochs=args.epochs)
    else:
        agent, diag = train_offline(candles, feats, cost=cost,
                                    epochs=args.epochs, exploration_passes=args.passes)
    agent.save(args.out)
    res = evaluate_policy(TradingEnv(candles, feats, cost=cost), agent)
    log.info("Done. transitions=%d", diag.get("transitions", 0))
    log.info("Greedy roll-out: total_return=%.4f action_counts=%s (saved to %s)",
             res["total_return"], res["action_counts"], args.out)


if __name__ == "__main__":
    main()
