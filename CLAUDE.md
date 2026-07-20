# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Gungnir is an autonomous, self-improving multi-strategy trading agent. It ingests
market data (ticks, candles, order-book depth), news, and macro series; runs ~26
technical strategies across timeframes; consults an LLM for sentiment/prediction/
reflection; sizes and risk-checks orders; executes them (live or paper); and learns
from closed trades via parameter optimization and a reinforcement-learning gate. A
separate FastAPI dashboard monitors and controls it.

## Commands

```bash
pip install -e ".[dev]"              # base + pytest/ruff/mypy
pip install -e ".[dev,dashboard]"    # add FastAPI/uvicorn for the console
pip install -e ".[learn]"            # add scikit-learn (heavier optimizers)

# Run the agent (defaults to --dry-run: PaperBroker + SyntheticMarketFeed)
python -m gungnir.main --config config/config.example.yaml --dry-run

# Dashboard (separate process; reads status.json + gungnir.db, writes control.json)
python -m gungnir.dashboard --port 8080

# Tests
python -m pytest                     # whole suite
python -m pytest tests/test_rl.py    # one file
python -m pytest tests/test_rl.py::test_name   # one test
python -m pytest -k "sizing"         # by keyword

ruff check src tests                 # lint (line-length 100, target py311)
mypy src                             # type check

# Offline RL training (advisory artifact — NOT wired into live execution)
python -m gungnir.learning.rl.train_offline --symbol EURUSD --timeframe 1h --bars 1000

python scripts/convergence_report.py # RL convergence status from data/rl_convergence.jsonl
```

`pytest.ini_options` sets `asyncio_mode = "auto"`, so async tests need no marker.
There is no CI workflow in-repo; run ruff/mypy/pytest locally before pushing.

## The stale-docs trap (read this first)

`README.md` and `ARCHITECTURE.md` describe an older design and are wrong in two
load-bearing ways. Trust the code, not those docs:

- **LLM provider is Anthropic Claude (or local Ollama, or `none`) — not Gemini.**
  Config key `llm.provider` selects `anthropic` | `ollama` | `none`. Secrets are
  `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (default `claude-opus-4-8`). See
  `llm/client.py::build_llm`.
- **The live broker/feed is Capital.com REST — not cTrader.** `main.py` wires
  `CapitalComSession` / `CapitalComBroker` / `CapitalComMarketFeed`. The entire
  `execution/ctrader*.py` + `data/market_feed.py` cTrader stack is **legacy**, kept
  only for its adapter tests; it is not on the live path. Symbols in `config` are
  Capital.com epics (e.g. `US100`, `EURUSD`, `AAPL`).

## Architecture

Loosely-coupled layers passing typed objects (`data/models.py`: `Tick`, `Candle`,
`OrderBook`, `NewsItem`, `MacroIndicator`, `Signal`, `Order`, `Trade`) through a
central orchestrating **Agent** (`core/agent.py`). Every boundary is an ABC
(`Broker`, `LLMClient`, `Strategy`, `DataFeed`, `PositionSizer`) so the live impl is
swappable.

**Two clocks** (`core/scheduler.py`): a fast loop (~30s: sense→decide→act) and a
slow loop (~1h: learn/reflect/optimize). One fast tick, `Agent.step()`:

1. Ingest market/news/macro → normalize to typed objects.
2. Build features: `features/indicators.py` (+ `kraken_indicators.py`) and
   `features/orderbook.py` (imbalance, spread, microprice) → `FeatureSet`.
3. Sense (LLM, advisory only): `llm/sentiment.py`, `llm/prediction.py`.
4. Decide: each active `Strategy` emits `Signal`s from the `FeatureSet` + LLM context.
5. **RL gate** (`learning/rl/`): an actor-critic policy scores each signal's P(take)
   and can block it (`rl.gate_signals`) and confidence-scale its size. Skipped
   signals are shadow-filled to learn the counterfactual. This sits between strategy
   output and sizing.
   **Consensus aggregation** (`core/aggregator.py`, `aggregation.mode: consensus`,
   default off): collapses all stances into ONE account decision per symbol —
   conviction × allocator × RL-P(take) weighted vote, family-capped (each
   `Strategy.family` bloc ≤ 40% of the vote), 35%-opposing veto on entries,
   EMA + enter/exit hysteresis. Strategies then shadow-trade for attribution
   only; the account trades the `consensus` book.
6. Size & risk-check: `risk/position_sizing.py` (vol_target | kelly | fixed_fractional)
   then `risk/portfolio.py` (exposure/correlation/drawdown caps — the last gate).
7. Execute → netted (`execution/netting.py`, `execution.netting: true` default):
   each strategy fills on its own **virtual book** (attribution for journal/
   allocator/RL), while the account broker — Capital.com (live) or PaperBroker
   (dry-run / shadow) — holds one net position per symbol, reconciled once per
   symbol per fast loop. Net positions carry no broker-side brackets; exits are
   agent-managed on the virtual books.
8. Record round-trips to the journal (`learning/journal.py`) on close.
9. Slow loop: `learning/evaluator.py` scores trades; `learning/optimizer.py` +
   `bayesian_reflection.py` / `llm/reflection.py` propose param tweaks (gated by
   `learning.auto_apply`); `learning/allocator.py` sets per-strategy sizing
   multipliers from recent regime-conditional edge.

**Core design invariants:**
- **LLM is advisor, never trigger.** All calls are rate-limited, cached, and return
  a structured fallback on any error so a slow/down LLM can never stall or crash the
  trading loop.
- **Risk is the boss.** The portfolio manager can shrink or kill any order and is the
  final gate before execution.
- **Backtest reuses live code.** `backtest/engine.py` replays candles bar-by-bar
  through the *same* Strategy + FeatureSet + sizing path, so backtests can't drift
  from live behavior. Driven by `POST /api/backtest`.

## Strategies

`strategy/kraken_strategies.py` defines the 26 real strategies (exported as
`KRAKEN_STRATEGIES`); `strategy/examples/` holds two reference strategies. The
`StrategyRegistry` (`strategy/registry.py`) loads them and their tunable params.
Each strategy runs in one of three **modes**: `off` / `shadow` (paper-trades on a
separate shadow broker, never touches the real account — new strategies start here)
/ `live`. Live falls back to shadow unless not in dry-run and the broker is connected.

## Agent ↔ dashboard contract (file-based, decoupled)

The dashboard is a separate read-only process that **never imports agent memory**.
Three JSON/DB files on the shared `data/` volume are the entire contract:
- `data/gungnir.db` — trade journal (history, metrics, equity curve).
- `data/status.json` — live snapshot the agent writes atomically each fast loop.
- `data/control.json` — dashboard→agent control (strategy modes, global pause),
  applied by the agent at the top of its next fast loop.

All mutable state lives under `data/` (journal, learned params in
`data/strategies.yaml`, RL policy in `data/rl_policy.npz`, tokens, status/control).
`config/` is read-only at runtime.

## Config & secrets

`config.py`: YAML tunables via `Config.load()` (dict-backed, accessed with
`config.get("a", "b", default=...)`), secrets from `.env` via `Secrets.from_env()`.
Copy `config/config.example.yaml` → `config/config.yaml` and `.env.example` → `.env`.

**Live-trading safety switches (do not weaken these):**
- `dry_run` defaults **true** (config or `GUNGNIR_DRY_RUN`); dry-run uses
  PaperBroker + SyntheticMarketFeed.
- `CAPITAL_COM_DEMO` defaults **true** — anything but an explicit `false`/`0`/`no`
  stays on the demo endpoint (audit F-00b).
- Connecting to the LIVE endpoint additionally requires `CAPITAL_COM_ALLOW_LIVE=true`;
  `main.py` refuses live otherwise and errors on demo/URL ambiguity.
- `compliance:` block applies hard rules (notional cap, daily order budget,
  restricted symbols) to **live** orders only.

## Deployment

Docker-first for a TrueNAS homelab: `docker compose up -d --build` brings up two
containers (`gungnir` agent + `dashboard`). `docker-entrypoint.sh` runs as root to
`chown` the `data` volume to `PUID:PGID` (default `568:568`) then `gosu`-drops to
that unprivileged user. See `docs/TRUENAS.md`.
