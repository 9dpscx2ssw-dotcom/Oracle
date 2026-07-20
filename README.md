# Gungnir

An autonomous, self-improving trading agent with **26 Kraken strategies**, powered by a
**free LLM** (Google Gemini), designed to run 24/7 in a **TrueNAS homelab** via Docker.
Integrates with **Capital.com REST API** for live trading, plus **shadow mode** for
risk-free strategy vetting.

> *Gungnir — Odin's spear, said to never miss its mark.*

## What it does

| Capability | Where it lives |
|---|---|
| Digests live market data (ticks, candles, **order book depth**) from Capital.com | `data/capital_com_feed.py`, `features/feature_store.py` |
| Runs **26 Kraken trading strategies** on multiple timeframes (1m–1d) | `strategy/kraken_strategies.py` |
| Executes trades via Capital.com REST API or paper broker | `execution/capital_com.py`, `execution/broker.py` |
| **Shadow trades** (paper-trade any strategy risk-free) | `core/agent.py`, `execution/broker.py` |
| Learns from past trades & tweaks strategy params over time | `learning/` |
| Predicts the market from **news + macro (CPI, rates)** | `data/news_feed.py`, `data/macro_feed.py`, `llm/prediction.py` |
| Watches **sentiment** from market news | `llm/sentiment.py` |
| Manages **risk / position sizing** across multiple assets | `risk/` |
| Orchestrates everything on a loop | `core/agent.py`, `main.py` |
| **Backtesting** strategies over historical/synthetic data | `backtest/` |
| **Web console** (8 tabs, strategy on/off, backtests, sortable lists) | `dashboard/` |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design and data flow.

## Quick start (dev)

```bash
cp .env.example .env          # fill in Capital.com + Gemini + FRED keys
pip install -e ".[dev]"
python -m gungnir.main --config config/config.example.yaml --dry-run
```

The `--dry-run` flag uses paper trading with a stub market feed. To trade with
real Capital.com data, set `GUNGNIR_DRY_RUN=false` and provide `CAPITAL_COM_API_KEY`.

## Run on TrueNAS SCALE (Docker)

```bash
cp .env.example .env          # fill in secrets + TZ / PUID / PGID / DATA_DIR
docker compose up -d --build
docker compose logs -f gungnir
```

This brings up **two** containers: the `gungnir` agent and the `dashboard`
console. Open **`http://<truenas-ip>:8080`**.

The image runs as a **non-root** user (default `568:568`, the TrueNAS `apps`
user) so it plays nicely with a dataset-backed volume. All mutable state (journal
DB, learned params, status/control files, refreshed tokens) lives in the `data`
volume; `config` is mounted read-only. Point `DATA_DIR`/`CONFIG_DIR` at a dataset
and chown it to `PUID:PGID`.

📖 **Full step-by-step: [`docs/TRUENAS.md`](./docs/TRUENAS.md)** — datasets,
permissions, Install-via-YAML vs Dockge/Portainer, updating, and troubleshooting.

## Web console

A tabbed web console for monitoring **and** control:

```bash
pip install -e ".[dashboard]"
python -m gungnir.dashboard --port 8080      # then open http://localhost:8080
```

**Tabs**
- **Overview** — account balance, equity, running & closed P/L, executed real vs
  shadow trade counts, latest signal with a confidence wheel, equity curve, and
  per-strategy performance bars.
- **Instruments** — every symbol with price, RSI, ATR, order-book imbalance,
  spread, sentiment, and the LLM prediction.
- **Intelligence & Sentiment** — sentiment + predictions per symbol, macro
  indicators (CPI / rates), and the latest market news.
- **Strategies & Metrics** — turn each of the **26 strategies off / shadow / live**, see
  overall and per-strategy metrics (Sharpe, win rate, expectancy), and **run backtests**
  on synthetic candles with custom parameters.
- **Learning** — history of the LLM's reflection proposals and which parameter
  changes were accepted vs rejected by the evaluator.
- **Signals** — every generated signal and its disposition (real / shadow /
  rejected). **Sortable.**
- **Trades** — closed-trade history with mode and P/L. **Sortable.**
- **Settings** — the live config and a global pause switch.

**How control stays safe.** Reads come from `data/status.json` + the journal DB.
Writes (strategy mode, pause) go through `data/control.json`, which the agent
applies at the top of its next loop — the dashboard never touches agent memory.
Strategies default to **shadow** so they are vetted on paper before you ever flip
one to **live**. JSON API under `/api/*` (`/api/docs` for the schema).

## Required free accounts / keys

- **Capital.com REST API** — Get a free API key from https://capital.com/trading/api.
  Works with demo and live accounts. Set `CAPITAL_COM_API_KEY` in `.env`. 
  **Start with `GUNGNIR_DRY_RUN=true` and validate in `shadow` mode first.**
- **Google Gemini** — free API key from https://aistudio.google.com/ (generous
  free tier on `gemini-2.0-flash`). Used for sentiment, news reasoning, and
  strategy reflection.
- **FRED** — free API key from https://fred.stlouisfed.org/ for CPI, interest
  rates, and other macro series.
- **News** — start with free RSS feeds; optionally Finnhub or NewsAPI keys for
  additional market news coverage.

## ⚠️ Safety

This is a framework, not financial advice or a guaranteed money printer.
**Always run against a demo account first.** The agent ships in `--dry-run`
(paper) mode by default; live trading requires an explicit config flag plus
hard risk limits. Markets can and will take your money.
