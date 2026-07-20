"""Read-only monitoring dashboard (FastAPI) for the Gungnir agent.

Runs as its own process/container. Reads two things the agent publishes:
  • the trade journal SQLite DB  (history, metrics, equity curve)
  • data/status.json             (live equity, open positions, per-symbol views)

It never imports or mutates agent state, so it can't affect trading.
"""
