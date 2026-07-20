# Running Gungnir on TrueNAS SCALE

This guide targets **TrueNAS SCALE 24.10 "Electric Eel" or newer**, which runs
Docker / Docker Compose natively. Two containers are deployed from one image:
the **agent** and the read-only **dashboard** (port 8080).

> The agent ships with **all 26 Kraken trading strategies** loaded and runs in
> **shadow mode** by default (paper-trading) so you can vet strategies before
> live trading. Real market data requires a Capital.com API key.

---

## 1. Create a dataset for persistent state

In the TrueNAS UI: **Datasets → Add Dataset**, e.g. `tank/apps/gungnir`.
Then open a shell (**System → Shell**) and create the sub-dirs + a config:

```sh
mkdir -p /mnt/tank/apps/gungnir/{data,config}
# Copy the example config in (from a clone of the repo, or paste it):
cp config/config.example.yaml      /mnt/tank/apps/gungnir/config/config.yaml
cp config/strategies.example.yaml  /mnt/tank/apps/gungnir/config/strategies.yaml   # optional
```

### Permissions

The containers run as a non-root user (default **568:568**, the TrueNAS `apps`
user). The **data** dir must be writable by that uid; **config** can stay
read-only:

```sh
chown -R 568:568 /mnt/tank/apps/gungnir/data
chmod -R 0775     /mnt/tank/apps/gungnir/data
```

If you prefer a different user, set `PUID`/`PGID` (below) and chown to match.

---

## 2. Provide secrets (`.env`)

Create `/mnt/tank/apps/gungnir/.env` from [`.env.example`](../.env.example) and
fill in your keys. The essentials:

```ini
# Capital.com REST API (for real market data) — session auth needs ALL THREE.
# Create an API key under Settings -> API integrations; you set a custom password there.
CAPITAL_COM_API_KEY=...           # the API key
CAPITAL_COM_IDENTIFIER=...        # your account login (usually your email)
CAPITAL_COM_PASSWORD=...          # the custom password set for the API key
CAPITAL_COM_DEMO=true             # start on the demo endpoint; set false for live
# CAPITAL_COM_API_URL=            # optional; leave blank to pick live/demo automatically

# LLM for sentiment analysis (Google Gemini — free tier)
GEMINI_API_KEY=...               # get from https://aistudio.google.com/

# Macro economic data from FRED
FRED_API_KEY=...                 # get from https://fred.stlouisfed.org/ (free)

# News sources (optional)
FINNHUB_API_KEY=...              # optional, for additional news

# Runtime
GUNGNIR_DRY_RUN=false            # set to false to use real Capital.com market data

TZ=America/New_York
PUID=568
PGID=568
DATA_DIR=/mnt/tank/apps/gungnir/data
CONFIG_DIR=/mnt/tank/apps/gungnir/config
DASHBOARD_PORT=8080
```

---

## 3. Deploy the stack

### Option A — Custom App (Install via YAML)

**Apps → Discover Apps → ⋮ (top-right) → Install via YAML**, paste the contents
of [`docker-compose.yml`](../docker-compose.yml), and set the env vars above.
Because the compose uses `build: .`, either:

- point the app at a checkout of this repo on the NAS, **or**
- pre-build the image once from the Shell and switch to it (Option B).

### Option B — Shell / Dockge / Portainer (recommended)

Clone the repo onto the NAS and bring it up with Compose. This builds the image
and starts both services:

```sh
cd /mnt/tank/apps/gungnir
git clone https://github.com/9dpscx2ssw-dotcom/gungnir.git src-repo
cd src-repo
ln -sf /mnt/tank/apps/gungnir/.env .env       # reuse the .env created above
docker compose up -d --build
docker compose logs -f gungnir
```

Or import `docker-compose.yml` into **Dockge** / **Portainer** and set the same
environment variables there.

---

## 4. Open the console

Browse to `http://<truenas-ip>:8080`. Use the **Strategies** tab to flip a
strategy from **shadow** to **live** once you trust it, and the **Pause** button
or **Settings** tab to halt new entries.

---

## Updating

```sh
cd /mnt/tank/apps/gungnir/src-repo
git pull
docker compose up -d --build
```

State (journal DB, learned params, tokens) lives in the `data` dataset and
survives rebuilds.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PermissionError` writing `/app/data/...` | `chown -R $PUID:$PGID` the data dir; confirm `user:` matches |
| Dashboard shows no market data | Set `GUNGNIR_DRY_RUN=false` **and** all three Capital.com vars (`CAPITAL_COM_API_KEY`, `CAPITAL_COM_IDENTIFIER`, `CAPITAL_COM_PASSWORD`); restart `docker compose up -d` |
| Agent shows only synthetic data | In dry-run (or without the full Capital.com credential set) the agent uses a moving `SyntheticMarketFeed`; provide all three credentials and `GUNGNIR_DRY_RUN=false` to switch to live Capital.com data |
| Capital.com `401`/`error.invalid.details` | Wrong identifier/password, or key not enabled for that environment — confirm the custom password and that `CAPITAL_COM_DEMO` matches where the key was created |
| Capital.com `error.invalid.api.key` | The API key value is wrong or revoked; regenerate under Settings → API integrations |
| Dashboard shows "agent has not published status yet" | The agent writes `status.json` after its first fast loop; check `docker compose logs brynnhildr` |
| Healthcheck unhealthy at first | Expected for ~90s on cold start (it waits for the first heartbeat) |
| Universe symbols not found live | Capital.com uses *epics* (e.g. `BTCUSD`); set the universe symbols to Capital.com epics when running live |
| No new trades appearing | Check that at least one strategy is in `live` mode (switch from `shadow` in the dashboard Strategies tab) and that portfolio risk limits haven't halted trading |
| Container can't reach the internet | TrueNAS app network policy — ensure outbound HTTPS is allowed |
