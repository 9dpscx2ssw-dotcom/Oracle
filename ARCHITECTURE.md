# Architecture

Gungnir is built as a set of loosely-coupled layers that pass typed data objects
through a central orchestrating **Agent**. Every layer is swappable behind an
interface so you can change broker, LLM, or data source without touching the rest.

```
                         ┌─────────────────────────────────────────────┐
                         │                  Agent (brain)               │
                         │   core/agent.py — the orchestration loop     │
                         └─────────────────────────────────────────────┘
        ingest                 reason / decide                 act / learn
          │                          │                             │
┌─────────▼─────────┐   ┌────────────▼─────────────┐   ┌───────────▼───────────┐
│   DATA SOURCES    │   │     FEATURES + LLM       │   │  EXECUTION + RISK     │
│  data/            │   │  features/  llm/         │   │  execution/  risk/    │
│                   │   │                          │   │                       │
│ • market_feed     │   │ • indicators (TA)        │   │ • netting broker      │
│   (Capital.com:   │──▶│ • orderbook analysis     │──▶│   (virtual books)     │
│   ticks, candles) │   │ • feature_store          │   │ • position sizing     │
│ • news_feed       │   │ • sentiment   (Claude)   │   │ • portfolio risk      │
│ • macro_feed      │   │ • prediction  (Claude)   │   │   (per-book breakers) │
│   (CPI, rates)    │   │ • reflection  (Claude)   │   └───────────┬───────────┘
│                   │   │ • RL gate                │               │
└───────────────────┘   └──────────────────────────┘               │
          ▲                                                         │
          │                  ┌──────────────────────┐              │
          └──────────────────│  LEARNING + MEMORY    │◀─────────────┘
                             │  learning/            │   trade outcomes
                             │ • journal (memory)    │
                             │ • evaluator (metrics) │
                             │ • optimizer (tuning)  │
                             └──────────────────────┘
                                        │
                             ┌──────────▼───────────┐
                             │   PERSISTENCE        │
                             │   persistence/db.py  │  SQLite (default) / TS DB
                             └──────────────────────┘
```

## The loop (one "tick" of the agent)

`core/agent.py::Agent.step()` runs on a schedule (`core/scheduler.py`):

1. **Ingest** — pull the latest market data (with rate-limit hardening: request
   pacing, candle caching per symbol/timeframe, 2s snapshot TTL), order book
   snapshots, fresh news, and any updated macro releases. Normalize into typed
   objects (`data/models.py`).
2. **Feature build** — compute technical indicators and order-book metrics
   (imbalance, spread, microprice, depth slope) into a `FeatureSet`.
3. **Sense (LLM)** — `llm/sentiment.py` scores news/social tone;
   `llm/prediction.py` fuses features + news + macro into a directional bias
   with a confidence and a short rationale. The LLM is used as a *reasoning and
   summarization* layer, **not** as the sole trade trigger.
4. **Decide** — each active `Strategy` consumes the `FeatureSet` + LLM context
   and emits zero or more `Signal`s (intent: long/short/flat, target asset,
   conviction).
5. **Gate (RL policy)** — an actor-critic policy (`learning/rl/`) scores each
   signal's P(take) and can block it or confidence-scale its size. Skipped signals
   are shadow-filled to learn the counterfactual.
6. **Size & risk-check** — `risk/position_sizing.py` converts conviction +
   volatility into a position size; `risk/portfolio.py` vetoes/sizes against
   account-level limits (max exposure, per-asset caps, per-book drawdown breakers,
   min confidence). Loss-streak cooldown (`risk/cooldown.py`) benches a
   strategy-symbol pair after N consecutive losses.
7. **Execute (netted)** — surviving orders go to `execution/netting.py`, which
   fills each on its strategy's virtual book (preserving RL/journal/allocator
   attribution) and reconciles one net position per symbol on the account broker
   (Capital.com live, PaperBroker in `--dry-run`). Same-second duplicate signals
   net to one account order; opposing signals cancel.
8. **Record** — fills and context snapshot are written to the trade journal
   (`learning/journal.py`).
9. **Learn (periodic)** — on a slower cadence, `learning/evaluator.py` scores
   closed trades; `learning/optimizer.py` + `llm/reflection.py` propose strategy
   parameter tweaks; `learning/allocator.py` sets per-strategy sizing multipliers
   from recent regime-conditional edge.

## Key design choices

- **LLM as advisor, not trigger.** LLMs are rate-limited and non-deterministic.
  Claude (Anthropic), Ollama (local), or none (disabled) provides *sentiment,
  news digestion, macro interpretation, and post-trade reflection*. Hard trade
  logic stays in deterministic strategy + risk code so a bad/slow LLM call can
  never blow up the account. All LLM calls are cached, rate-limited, and have
  safe fallbacks.
- **Everything is an interface.** `Broker`, `LLMClient`, `Strategy`, `DataFeed`,
  `PositionSizer` are abstract base classes. Capital.com / Claude / FRED are just
  one implementation each.
- **Two clocks.** A fast loop (seconds–minutes) for sensing/trading; a slow loop
  (hours–days) for learning, retraining, and parameter optimization.
- **Netted execution.** Strategies fill on per-strategy virtual books (preserving
  attribution for RL / allocator / journal), while the account holds one net
  position per symbol. Same-second duplicate signals on one symbol net to one
  account order; opposing signals cancel; the net position is reconciled once per
  symbol per fast loop.
- **Rate-limit hardening.** Request pacing (slot-based, 0.12s between requests),
  429 retry (exponential backoff on GETs only; POST/DELETE never auto-retry for
  order safety), candle caching (per-symbol per-timeframe, refreshed only when the
  next bar can exist, ~25s floor), and snapshot sharing (2s TTL on ticks +
  order-book snapshots) prevent data starvation during Capital.com bursts.
- **Crash protections.** Drawdown tracking is per-book (real account + shadow book
  separately), with independent daily / intraday-from-peak / total breakers. A
  loss-streak cooldown benches a strategy-symbol pairing after N consecutive
  losses, serving probes that can re-trigger the cooldown if they lose again.
- **Adaptation via reflection + optimization.** "Learning over time" = (a)
  numerical parameter optimization over the trade journal (walk-forward / Bayesian),
  plus (b) an LLM reflection pass that reads recent losing/winning trades and
  proposes hypotheses and parameter nudges, gated by the evaluator.
- **Risk is the boss.** The portfolio risk manager can shrink or kill any order.
  It owns global limits and is the last gate before execution.

## Component map

| Module | Responsibility |
|---|---|
| `core/agent.py` | Orchestrates one full sense→decide→act→learn cycle |
| `core/scheduler.py` | Fast loop / slow loop timing |
| `core/events.py` | Event & message types passed between layers |
| `execution/capital_session.py` | Capital.com REST API + request pacing + 429 retry logic |
| `execution/capital_com_feed.py` | Capital.com market feed with candle caching + snapshot sharing |
| `execution/capital_com_broker.py` | Capital.com live execution |
| `execution/netting.py` | Netted broker: per-strategy virtual books + symbol-scoped net reconciliation |
| `data/news_feed.py` | RSS / NewsAPI ingestion |
| `data/macro_feed.py` | FRED CPI, interest rates, macro series |
| `data/models.py` | `Tick`, `Candle`, `OrderBook`, `NewsItem`, `MacroIndicator`, `Signal`, `Order`, `Trade` |
| `features/indicators.py` | EMA/RSI/ATR/etc. technical features |
| `features/orderbook.py` | Imbalance, spread, microprice, depth analysis |
| `features/feature_store.py` | Assemble per-asset `FeatureSet` |
| `llm/client.py` | Provider-agnostic LLM interface: Anthropic / Ollama / none |
| `llm/sentiment.py` | News/social sentiment scoring |
| `llm/prediction.py` | Fuse signals → directional bias + rationale |
| `llm/reflection.py` | Post-trade reflection → strategy hypotheses |
| `strategy/base.py` | `Strategy` interface |
| `strategy/registry.py` | Load/store strategies + their (tunable) params |
| `strategy/kraken_strategies.py` | 26 production strategies (trend, mean-reversion, momentum, etc.) |
| `risk/position_sizing.py` | Kelly / vol-target / fixed-fractional sizers |
| `risk/portfolio.py` | Account-level exposure, correlation, per-book drawdown limits + breakers |
| `risk/cooldown.py` | Loss-streak cooldown: bench strategy-symbol pairs after N losses |
| `execution/broker.py` | `Broker` interface + paper broker |
| `learning/journal.py` | Trade + context memory store |
| `learning/evaluator.py` | Sharpe, win rate, expectancy, drawdown metrics |
| `learning/optimizer.py` | Parameter optimization over the journal |
| `learning/rl/` | Actor-critic RL policy for signal gating + confidence scaling |
| `learning/allocator.py` | Continuous capital allocator from regime-conditional edge |
| `persistence/db.py` | SQLite access: trades, signals, learning events |
| `backtest/engine.py` | Replay a strategy over candles; synthetic data generator |
| `core/control.py` | Dashboard → agent control file (strategy modes, pause) |
| `dashboard/server.py` | FastAPI console API (read + control) + tabbed UI |

## Monitoring dashboard

The dashboard is a separate, read-only process (its own container). The contract
between agent and dashboard is just two files on the shared `./data` volume:

- **`data/gungnir.db`** — the trade journal (history, metrics, equity curve).
- **`data/status.json`** — a live snapshot the agent writes atomically at the end
  of every fast loop (equity, drawdown, halt state, open positions, and the
  per-symbol "view": price, RSI, order-book imbalance, sentiment, LLM prediction).

Because the dashboard never imports or touches agent memory, it cannot affect
trading — worst case it shows slightly stale numbers. The `/api/*` JSON endpoints
also make it trivial to point Grafana or a phone widget at the same data.

### Control (dashboard → agent)

Control flows the other way through a third file, **`data/control.json`**, written
by the dashboard and read by the agent at the top of every fast loop:

```
{ "strategies": { "trend_following": "live" }, "paused": false }
```

This keeps the two processes decoupled while still allowing the dashboard to
turn strategies **off / shadow / live** and globally pause new entries. Nothing
in the dashboard ever calls into the agent directly.

### Modes & shadow trading

Each strategy runs in one of three modes:

- **off** — dormant, emits no signals.
- **shadow** — emits signals and paper-trades them on a separate shadow broker,
  but never touches the real account. New strategies start here so they can be
  evaluated risk-free.
- **live** — trades the real account (only when not in global dry-run and the
  cTrader broker is connected; otherwise it falls back to shadow).

Every signal is recorded with its *disposition* (`real`, `shadow`,
`rejected_risk`, `rejected_paused`). Trades are written to the journal only when
they close, as complete round-trips with realized P/L, so metrics stay clean.

### Netted execution model

`execution/netting.py` implements netted execution to reduce slippage and account
volatility:

- Each strategy fills on its own **virtual broker** (PaperBroker), preserving
  per-strategy attribution for RL, journal, and allocator scoring.
- The **account broker** (Capital.com or PaperBroker) holds one **net position per
  symbol**, reconciled once per symbol per fast loop.
- Same-loop duplicate signals on one symbol net to one account order (avoiding
  redundant fills); opposing signals net flat (avoiding spread-paying pairs).
- External closes (e.g., manual broker closes, stop-loss triggers) are adopted
  per-strategy via flatten logic, preserving round-trip tracking.

The reconciliation flow:
1. Drain closed positions from the account and flatten corresponding virtual books.
2. Compute net exposure per symbol from all virtual books.
3. Open or adjust one account order per symbol to hold the net; close if net is flat.
4. Record fills and reconciliation events for RL attribution.

### Crash protections

Three independent safeguards prevent the "sustained-downtrend incident" (500+ losing
trades over 16h with zero halts):

**Per-book drawdown tracking** (`risk/portfolio.py`):
- Separate `DrawdownTracker` for the real account and shadow/paper book.
- Each tracks daily drawdown (vs. day start), intraday drawdown (vs. today's peak,
  catches crashes after run-ups), and total drawdown (vs. all-time peak).
- `trading_halted(book)` gates new entries separately per book; shadow book halts
  don't prevent real orders (and vice versa), isolating shadow strategies.

**Loss-streak cooldown** (`risk/cooldown.py`):
- `LossStreakGuard` tracks consecutive losses per (strategy, symbol) round-trip.
- After `loss_streak_trades` (default 3) losses, the pairing is benched for
  `loss_streak_cooldown_minutes` (default 60).
- A win clears the streak early; probes after cooldown expires can re-trigger
  the cooldown immediately if they lose again, preventing counter-trend loops.

**Reinforcement learning gate** (`learning/rl/`):
- An actor-critic policy scores each signal's `P(take)` and can block it.
- Blocked signals are shadow-filled to learn the counterfactual, so the policy
  learns from skipped high-loss trades.

### Backtesting

`backtest/engine.py` replays a candle series bar-by-bar through the **same**
`Strategy` + `FeatureSet` + sizing code the live agent uses, so a backtest can't
silently drift from live behaviour. It ships a reproducible synthetic-data
generator, and can equally take real Capital.com candles from the market feed.
The console's Strategies tab drives it via `POST /api/backtest`.

### Capital.com REST API integration

`execution/capital_session.py` owns a connection to the Capital.com REST API
(`{demo,live}.capitalapi.com`). The session provides rate-limit hardening:

- **Request pacing** — all requests wait `min_request_interval` (default 0.12s) after
  the previous request via a monotonic-time slot (no blocking); prevents burst storms
  that saturate Capital.com's rate limiters.
- **429 retry logic** — GET requests retry on 429 (too many requests) with exponential
  backoff + jitter, honoring `Retry-After` headers. POST/DELETE never auto-retry (order
  safety: a retry on order submission could duplicate the fill).
- **Candle caching** — `capital_com_feed.py` holds a per-(symbol, timeframe) cache,
  refreshed only when the next bar can exist (never more often than ~25s floor). On
  fetch failure, the cache serves the last-known series (stale-serve) to prevent
  strategy starvation.
- **Snapshot sharing** — ticks + order-book snapshots share a 2s TTL, reducing
  redundant calls on same-loop burst requests.

The feed maps Capital.com REST responses → `Candle`, `Tick`, `OrderBook`. The broker
manages position reconciliation (net position per symbol) and emits orders. Live and
demo modes coexist: dry-run uses `PaperBroker` (in-memory), live uses
`CapitalComBroker` (REST).

**Token persistence.** OAuth access tokens (if used) are persisted and refreshed
before expiry via `data/ctrader_token.json` (gitignored). Safety switches:
- `dry_run` defaults **true** (config or `GUNGNIR_DRY_RUN`).
- `CAPITAL_COM_DEMO` defaults **true** (anything but explicit `false`/`0`/`no` stays on demo).
- Live trading requires both **not** in dry-run and `CAPITAL_COM_ALLOW_LIVE=true`.

## Scaling later

- Swap SQLite → TimescaleDB/Postgres for tick history.
- Add a message bus (Redis/NATS) if you split feeds and trading into separate
  containers.
- Add a small dashboard (FastAPI + a chart) reading the journal DB.
