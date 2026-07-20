# Gungnir Audit — 2026-07-03

Audited at branch `claude/backtest-ui-improvements-nu9cr4`, commit `a85939d`. Read-only audit; no code was modified. The two pre-known issues from the audit brief were re-verified first (§4, F-00a/F-00b).

---

## 1. Executive summary

Gungnir's architecture is sound in outline — clean layering (data → features → strategy → risk → RL gate → broker), a real ABC broker interface, atomic writes for the dashboard contract files, and a genuinely thoughtful RL design with counterfactual grading. The paper/shadow trading pipeline works and is well-guarded. **The live trading pipeline is not production-ready and should not run against real money in its current state.**

The three most dangerous issues: **(1)** the by-type minimum-lot floor is applied *after* the exposure caps and rounds capped orders back **up**, silently defeating the per-asset cap, the gross-exposure cap, and the dashboard max-lot control — three of the failing tests in the repo prove the caps do not bind; **(2)** live positions are fire-and-forget: nothing manages their exits, no realized P&L is ever computed or journaled for them, and the RL policy never learns from a single real trade, so internal state and broker state diverge by design; **(3)** the "don't stack positions" check is `isinstance(broker, PaperBroker)`-only, so a persistent signal condition (which is how most of the 28 strategies emit — level-based, not edge-triggered) submits a **new live order every 30 seconds** until the (defeated, see #1) exposure caps stop it. Combined with the `leverage: 200` size multiplier and the current relaxed config (`max_per_asset_exposure: 20`, `max_portfolio_exposure: 100`), this is an account-destroying combination.

Would I let this run live tomorrow? **No.** Demo: yes, it is useful exactly as it is being used — to shake these bugs out. The fix plan in §6 sequences the work; the P0 items are days, not weeks.

---

## 2. System map

**Modules** (`src/gungnir/`): `core` (agent orchestration, scheduler, filters, control channel), `execution` (Broker ABC, PaperBroker, CapitalComBroker + session; legacy cTrader), `data` (market/news/macro feeds, pydantic models), `features` (indicator computation, FeatureSet/KrakenFeatureSet), `strategy` (base, 26 kraken + 2 example strategies, registry), `risk` (portfolio caps/brackets, 3 sizers), `learning` (journal, evaluator, reflection pipeline, RL policy/network/state, offline RL), `llm` (Claude client with rate-limit/cooldown, sentiment/prediction/reflection), `persistence` (SQLite), `dashboard` (FastAPI server + static SPA, separate process, communicates via `data/status.json` / `data/control.json` / the DB).

**Lifecycle**: `main.py:build_agent` picks live vs paper (`live = not dry_run and full Capital.com creds`, main.py:41-43). `Scheduler` (scheduler.py:33-42) runs `fast_step` (30s) and `slow_step` (1h) as two coroutines on one event loop; each iteration's exception is caught and logged, and the loop continues. `fast_step`: apply control → manual closes → refresh equity/exposure from broker → fetch news → for each of 72 symbols sequentially: fetch candles (all TFs) + book → build features → publish view → mark brokers with tick mid → manage paper exits → for each strategy: generate signals → gates (pause, offline-veto, context filters, sentiment regime veto) → conviction blend → size → `PortfolioRisk.vet` → RL gate → submit to real or shadow broker → write status + heartbeat.

**Order path to the real broker** (the only one): agent.py:797-801 `go_live = strat.mode=="live" and not self._paper_mode and not isinstance(self.broker, PaperBroker)` → `broker.submit(order)` → `CapitalComBroker.submit` POSTs `/api/v1/positions` with broker-side `stopLevel`/`profitLevel`. Guards on the path: strategy mode (default `shadow`, registry.py:41-43), runtime paper/live mode, global pause, `min_confidence`, context filters, sentiment veto, exposure caps (defeated — F-01), RL gate (during warmup: takes everything), daily-drawdown halt.

**Paper vs live**: `PaperBroker` keys positions by `(symbol, strategy)`; `CapitalComBroker` by symbol only. Both implement the `Broker` ABC (`account_equity/balance/open_positions/submit/close`) — but the agent needs `position()`, `positions_for()`, `mark()`, per-strategy `close()`, which only PaperBroker has, so the agent branches on `isinstance(broker, PaperBroker)` in five places. That duck-typed edge is where the original `positions.clear()` bug lived and where F-03/F-05 live now.

**State**: `data/strategies.yaml` (modes+params, rewritten hourly and on toggle), `data/rl_policy.npz` + `.state.npz` (weights + buffer/telemetry, saved hourly and on shutdown), `data/gungnir.db` (trades, signals, learning events), `data/status.json`/`control.json` (dashboard contract, atomic).

---

## 3. Findings table

| ID | Sev | Area | Location | Summary | Status |
|----|-----|------|----------|---------|--------|
| F-00a | — | B | agent.py:283-285 (was ~262) | `positions.clear()` crash — **already fixed** this session (`7b147c6`); root cause + siblings verified | Confirmed fixed |
| F-00b | Critical | A | config.py:20,41-42; capital_session.py:43; main.py:70 | Demo is opt-in: missing/typo'd `CAPITAL_COM_DEMO` → **live endpoint**; URL override silently beats demo flag; "LIVE mode" log ≠ live money | Confirmed |
| F-01 | Critical | A | portfolio.py:145-147 | Min-lot floor applied after caps rounds orders back **up** — per-asset cap, gross cap, and max-lot all defeated (3 failing tests prove it) | Confirmed |
| F-02 | Critical | A | capital_com.py:149-161, 77-80; agent.py:318-339, 363-366, 809-810 | Live positions fire-and-forget: no exit management, no realized P&L, never journaled, RL never graded; broker/internal state diverge silently | Confirmed |
| F-03 | Critical | A | agent.py:809-810; kraken_strategies.py (level-based signals, e.g. :27-30) | Anti-stacking check is PaperBroker-only → live orders re-submitted every 30s while a signal condition persists | Confirmed |
| F-04 | Critical | A | position_sizing.py:41-45, 63-66 | `leverage` config multiplies every sizer's output (200x → ×181.8) with no margin/notional sanity check | Confirmed |
| F-05 | High | B | capital_com.py:140, 150-158; 130-137 | Live position tracking keyed by symbol: 2 live strategies on one symbol overwrite each other; fallback `entry_price=0` poisons P&L math | Confirmed |
| F-06 | High | B | news_feed.py:26; agent.py:203; agent.py:863 + llm/client.py:39 | Blocking calls on the event loop: sync `feedparser.parse` HTTP every fast loop (`poll_seconds` ignored); sync reflection with rate-limiter sleeps freezes both loops | Confirmed |
| F-07 | High | A/D | docker-compose.yml healthcheck; scheduler.py:36-38 | Healthcheck only tests heartbeat **existence** — a crash-looping or frozen agent stays "healthy" forever; errors visible only in logs | Confirmed |
| F-08 | High | A | dashboard/__main__.py:10; server.py (no auth middleware) | Dashboard binds 0.0.0.0 with **no auth** — anyone on the network can flip PAPER→LIVE, promote strategies, close positions, reset the RL policy | Confirmed |
| F-09 | High | B | server.py:271 vs agent.py:938 | Dashboard reads `st["latest_signal"]`; agent publishes `"signal"` → server overwrites the real field with an empty block at :307. Root cause of the persistent "No active signal" card | Confirmed |
| F-10 | Medium | A | agent.py:185-188; portfolio.py:89-96 | Drawdown breaker baseline resets to current equity daily (forgives losses at UTC midnight); halt blocks entries but never flattens or alerts; no stale-data or error-rate breaker exists | Confirmed |
| F-11 | Medium | B | agent.py:257-262 vs portfolio.py:67-80 | Dashboard `min_lot`/`max_lot` knobs write `risk.min_lot`/`max_lot` — attributes `vet()` no longer reads since the by-type refactor (dead controls; 3rd failing test encodes the old contract) | Confirmed |
| F-12 | Medium | C | agent.py:714-718; base.py:60-61 | `strat.trades_symbol()` never called by the agent — per-strategy symbol scoping ignored; all 28 strategies run on all 72 symbols | Confirmed |
| F-13 | Medium | B | registry.py:54, 108; network.py:151-157 | `strategies.yaml` written non-atomically and `from_yaml` crashes on corrupt YAML (boot failure); `rl_policy.npz` non-atomic (load is tolerant → silent policy loss instead) | Confirmed |
| F-14 | Medium | C | capital_com_feed.py:100-107; kraken strategies read `closes[-1]` | Candle window extends to `now` → last bar is the forming bar; level-based strategies evaluate and can flip on intra-bar noise (repaint churn); no closed-bar discipline | Suspected |
| F-15 | Medium | C | policy.py:116-119, config `entropy_coef: 0.01`; agent.py (C2: no live grading) | RL trains only on shadow/learning fills (train/serve skew vs live fills); all-skip is a positive-reward equilibrium when the signal stream loses — observed `P(take)=0.000` in production | Suspected |
| F-16 | Medium | C | kraken_strategies.py:59-71 (`bb_macd_sma`) et al. | Several strategies look logically inverted/degenerate (S3 buys above mid on *negative* MACD; live win rates 1.8-11% over 100+ trades) yet stay enabled by default | Confirmed (empirically) |
| F-17 | Medium | C/D | reflection accept gate evaluator.py:76-86 | Reflection re-scores proposals on the same lookback window it tuned on (in-sample); walk-forward optimizer exists but isn't in the reflection path | Suspected |
| F-18 | Low | B | agent.py:423-425 | `_portfolio_heat` reads private `_positions` via `getattr(..., {})` — silently degrades to 0 heat on rename (same drift family as F-00a) | Confirmed |
| F-19 | Low | B | news_feed.py:24-29 ("RSS fetch failed for None"); :48 `config.get("credentials", ...)` never populated | Feed list can contain `None`; Finnhub feed reads a config path that doesn't exist (dead) | Confirmed |
| F-20 | Low | D | tests/ | 77 pass / 3 fail; the 3 failures are **real regressions** (F-01, F-11) sitting unfixed in the suite; no test covers `_apply_control`, `CapitalComBroker.submit/close`, or the scheduler | Confirmed |

Git history secret scan: clean — no real `.env` ever committed; only example placeholders (one old `.env.example` revision shipped the **live** URL as the `CAPITAL_COM_API_URL` example, which interacts badly with F-00b's precedence).

---

## 4. Finding details

### F-00a — The known `positions.clear()` bug (fixed; siblings checked)
Root cause confirmed as described in the brief: `_apply_control` reached into `PaperBroker` internals that don't exist (`positions`, `equity`, `starting_equity`), and because `_apply_control()` is the *first* statement of `fast_step`, the exception killed every iteration before any trading, while scheduler.py:36-38 logged and continued — a permanent, invisible crash-loop (see F-07 for why the healthcheck never noticed). Fixed this session in `7b147c6` by adding `PaperBroker.reset()` and using it; the same commit fixed the sibling in the same handler: `reset_rl` rebuilt the policy with `RLPolicy(self.config)` — a `TypeError` at call time since the constructor is keyword-only (caught, logged, reset silently no-op'd). Sibling scan across the repo found one remaining member of the family, downgraded to F-18.

### F-00b — Live is the default endpoint; three switches pretend to be one
```python
# config.py:20
capital_com_demo: bool = False          # True → use the demo endpoint
# config.py:41-42
capital_com_demo=os.getenv("CAPITAL_COM_DEMO", "").strip().lower() in ("1", "true", "yes"),
# capital_session.py:43
self.base_url = base_url or (DEMO_URL if demo else LIVE_URL)
```
Failure scenario: `.env` on the new box omits `CAPITAL_COM_DEMO`, or someone types `CAPITAL_COM_DEMO=True ` with a stray character the parser doesn't recognize, or sets `CAPITAL_COM_API_URL` (an old `.env.example` revision showed the **live** URL as its example) — the session silently authenticates against the live endpoint while everyone believes they're on demo. Meanwhile `main.py:70` logs `"LIVE Capital.com mode"` whenever real credentials are present, which trains operators to ignore the word "LIVE" in logs. The only truthful line is `capital_session.py:71-72` ("session established (demo)").
**Fix**: default `capital_com_demo=True`; refuse to start (or require an explicit `I_UNDERSTAND_LIVE=true` env) when resolving to the live URL; log `endpoint=demo|live account=<id>` at WARNING on every startup; warn loudly if `base_url` contradicts `demo`.

### F-01 — Min-lot floor defeats every exposure cap
```python
# portfolio.py:124-147 (abridged)
if existing + notional > per_asset_cap:
    notional = max(0.0, per_asset_cap - existing)     # cap shrinks…
...
final_volume = notional / price
floor = max(min_lot or 0.0, instrument_min or 0.0)
if floor and 0 < final_volume < floor:
    final_volume = floor                              # …floor un-shrinks
```
Failure scenario: equity $925, per-asset cap leaves $300 headroom on EURUSD → capped volume 0.0003 lots → forex floor (100 units) rounds it to **100 units**, an order ~300× the risk budget, and the same logic applies to the gross cap. This is not theoretical: `tests/test_portfolio.py::test_per_asset_cap_shrinks_order`, `::test_gross_exposure_cap_shrinks_order`, and `::test_max_lot_caps_and_min_lot_floors` all fail against current code — the caps demonstrably do not bind. The floor was added to stop broker rejections of dust orders; it now overrides the risk layer it lives in.
**Fix**: if capped volume < floor → **reject** (`return None`), never round up. Rounding up is only legitimate when the *uncapped* size was below the floor **and** the floored notional still fits every cap. Repair the three tests as the regression harness.

### F-02 — Live trades are fire-and-forget
Evidence chain:
- `CapitalComBroker.close()` (capital_com.py:149-161) returns the *stored entry-side* Trade — no exit price, no PnL.
- `_manage_exits` (agent.py:318-324) iterates `{shadow_broker} ∪ {broker if PaperBroker} ∪ {learn_broker}` — the live broker is **never** exit-managed by the agent; exits happen only via broker-side stop/TP.
- When a broker-side bracket fires, the position vanishes from `/positions`; `open_positions()` (capital_com.py:77-80) just deletes it from local tracking with a warning. No round-trip is journaled, no signal outcome updated, no RL grading (`learn_from_trade` needs `trade.pnl`, policy.py:155).
- The opposite-direction close path can't fire for live positions either, because `existing` is hardcoded `None` for non-PaperBrokers (agent.py:809-810), and `_execute_manual_closes` (agent.py:213-223) calls `broker.close()` directly, bypassing `_close()`'s journaling.

Failure scenario: every live trade ever placed disappears from the system's memory the moment it exits. "Executed (Real)" stays `0 graded / $0` forever (visible in the user's dashboard screenshots today), the RL policy trains exclusively on paper outcomes, and reflection tunes strategies on a journal that contains no real fills. If a stop is rejected or a position is partially closed broker-side, nothing reconciles.
**Fix**: after a position disappears (or on close), fetch the closing deal from Capital.com's transaction/activity history (`/api/v1/history/transactions` or the confirms endpoint), build the completed Trade with real exit price/PnL, and route it through `_close()`'s grading path. Add periodic reconciliation: broker positions vs `_positions` vs journal opens; alert on divergence instead of silently deleting.

### F-03 — Persistent signals stack live orders every 30 seconds
```python
# agent.py:809-810
existing = (broker.position(symbol, strat.name)
            if isinstance(broker, PaperBroker) else None)
```
The 26 kraken strategies are level-based, not edge-triggered — e.g. `cci_macd` (kraken_strategies.py:27-28) emits BUY on every loop while `cci > 100 and macd > 0` holds, which can persist for hours. On the shadow broker, the same-direction guard silently skips; on the live broker the guard is bypassed, so each loop submits a fresh order (each with its own brackets). `CapitalComBroker._positions[symbol]` (capital_com.py:140) then tracks only the newest deal, orphaning the rest (compounds F-02/F-05). The stack grows until the per-asset cap binds — which F-01 shows may be never for small accounts, and the current live config (`max_per_asset_exposure: 20`, i.e. 2000% of equity per symbol) makes the cap meaningless anyway.
**Fix**: track live positions per `(symbol, strategy)` using the dealId map, apply the same same-direction skip / opposite-direction close logic to the live broker, and add an idempotency guard (client_id based) so one signal condition maps to at most one open live position per strategy.

### F-04 — `leverage` is a raw size multiplier
```python
# position_sizing.py:41-44 (same pattern in all three sizers)
if self.leverage > 1.0:
    usable_leverage = self.leverage / (1.0 + self.safety_margin)
    base_size = base_size * usable_leverage
```
`FixedFractional` already sizes so the *stop* loses `risk_per_trade` of equity; multiplying by 181.8 means a stop-out loses **~90% of equity** at the current `leverage: 200`, `account_risk_per_trade: 0.005`. Leverage determines *margin required*, not *how much to risk* — conflating them turns every conservative sizer into a margin-call machine. There is no check anywhere that `notional ≤ equity × available_leverage`, nor any per-order absolute notional ceiling (`max_lot_by_type` is all `null` in the live config).
**Fix**: remove the multiplier from the sizers. Use leverage only as a *constraint*: `max_notional = equity × usable_leverage`, enforced in `vet()` alongside the exposure caps. If "size up with leverage" is genuinely desired, make it an explicit, capped multiplier with a distinct name.

### F-05 — Symbol-keyed live tracking and `entry_price=0` fallback
capital_com.py:140 `self._positions[order.symbol] = trade` — a second live strategy on the same symbol overwrites the first's dealId; `close(symbol)` (:150-158) then closes whichever deal was stored last, and the other position leaks (F-02 ensures nobody notices). Separately, :130-137 falls back to `entry_price=0` when no fill/mark is known; `_unrealized` (agent.py:352-358) would then report `mark × volume` as profit — dashboard shows an absurd gain and, worse, the journal stores a poisoned entry if it ever closes through `_close`.
**Fix**: key by dealId with a `(symbol, strategy) → dealId` index; treat unknown-fill as an open incident (re-query confirms/positions until resolved), never as price 0.

### F-06 — Blocking the event loop (news every 30s, reflection for minutes)
- agent.py:203 `self._news = await self.news.fetch()` runs **every fast loop**; `RSSNewsFeed.fetch` (news_feed.py:26) calls `feedparser.parse(url)` — a synchronous HTTP download on the event loop, twice per loop. `data.news.poll_seconds: 600` exists in config and is read by nothing.
- agent.py:863 `reflection_pipeline.run(...)` is synchronous inside `slow_step`; in `llm` mode it makes LLM calls through `_RateLimiter.wait()` → `time.sleep` (client.py:36-39) on the event loop. Because both loops share one loop (scheduler.py:28-31), a slow reflection freezes trading, exit management, and the status file for its whole duration. (The fast-path LLM calls were made non-blocking this session, `c1d9bb7`; these two remain.)
**Fix**: cache news with `poll_seconds` TTL and fetch via `asyncio.to_thread`; run `reflection_pipeline.run` in a thread (`await asyncio.to_thread(...)`).

### F-07 — Healthcheck can't detect a dead agent
docker-compose.yml healthcheck: `exists('/app/data/heartbeat')` — the file is written once and never removed, so after the first loop the check passes forever, through crash-loops (F-00a ran undetected for a while, proving it), event-loop freezes (F-06), and feed outages. The scheduler's catch-all (scheduler.py:36-38) is the right call for resilience, but only if something *counts* consecutive failures.
**Fix**: healthcheck on heartbeat **freshness** (`mtime < now - 3×fast_loop`); have `_heartbeat()` write a timestamp; add a consecutive-failure counter in the scheduler that pauses trading and screams (status flag the dashboard renders red) after N straight failed iterations.

### F-08 — Unauthenticated dashboard controls a real-money account
`python -m gungnir.dashboard` binds `0.0.0.0:8080` (dashboard/__main__.py:10) with no authentication on any endpoint. Since this session wired PAPER/LIVE (`/api/risk` `PAPER_TRADE`) and strategy promotion (`/api/strategies/{name}/mode` → `live`), anyone with network reach — the screenshots show it exposed on a Tailscale IP — can flip the bot live, promote strategies, close positions, reset the RL policy, or change risk limits. The existing `test_server_security.py` covers input hygiene, not authn.
**Fix**: at minimum a bearer token / basic-auth middleware gated on an env secret, plus confirm-header for state-changing endpoints; document that the port must never be forwarded beyond a trusted network.

### F-09 — Dashboard "Latest Signal" reads a field the agent never writes
server.py:271 `sig = st.get("latest_signal")` — the agent publishes `"signal"` (agent.py `_write_status`, key `"signal": self._latest_signal`). `signal_block` is therefore always the empty skeleton, and server.py:307 `"signal": signal_block` **overwrites** the agent's real field in the merged response. Every agent-side attempt to fix the "No active signal" card this week was fighting this server-side shadowing.
**Fix**: read `st.get("signal")` (and map its `best_trade` shape), or stop overwriting the raw field. One-line root cause; verify by curling `/api/status` and checking `signal.best_trade.symbol` is non-empty while the agent is signaling.

### F-10 — Circuit breakers: one exists, and it forgives too easily
`trading_halted()` (portfolio.py:89-96) is the only breaker. Baseline resets to *current* equity at each UTC date change (agent.py:185-188) — lose 2.9% at 23:50 UTC and the counter re-arms at 00:00 from the lower base; a multi-day bleed never halts. The halt blocks new entries only: open live positions stay open (with F-02, unwatched), and nothing notifies a human. There is no stale-data breaker (feed errors return `[]`/`None` and the loop happily re-marks on old prices — capital_com_feed.py:117-120), and no error-rate breaker (F-07).
**Fix**: track peak-equity drawdown in addition to daily; on halt, optionally flatten and definitely alert (status flag + log at ERROR); skip a symbol's trading when its last successful candle/tick is older than N× timeframe.

### F-11 — Dead dashboard risk knobs
agent.py:257-262 writes `self.risk.min_lot` / `self.risk.max_lot` from the Settings tab; `vet()` reads only `min_lot_by_type`/`max_lot_by_type` (portfolio.py:140-147) since the by-type refactor. The knobs silently do nothing (and `test_max_lot_caps_and_min_lot_floors` fails, encoding the old contract). **Fix**: map the knobs onto the by-type dicts (apply to all types, or per-type UI), and delete the dead attributes.

### F-12 — Per-strategy symbol scoping is ignored
`Strategy.trades_symbol()` (base.py:60-61) is called by the two example strategies only; the agent's loop (agent.py:714-718) runs every active strategy on every universe symbol. `strategies.yaml` `symbols:` lists are decorative. Cost: 28×72 signal evaluations/loop and strategies trading instruments they were never designed for. **Fix**: `if not strat.trades_symbol(symbol): continue` at agent.py:715.

### F-13 — Non-atomic state writes; asymmetric load tolerance
`registry.save()` uses bare `write_text` (registry.py:108) — a crash mid-write leaves truncated YAML and `from_yaml` (registry.py:54) then raises at boot: the bot won't start until someone hand-fixes the file. `rl_policy.npz`/`.state.npz` saves are also non-atomic (network.py:151-157, policy.py save), but their loads swallow corruption and silently start a **fresh policy** — arguably worse, because months of learning evaporate without an error. `status.json`/`control.json` already do tmp+rename correctly (control.py:46-51); copy that pattern. **Fix**: tmp+`os.replace` for both; on RL load failure, log at ERROR and keep a `.bak` of the previous good save.

### F-14 — Forming-bar evaluation (Suspected)
`recent_candles` requests a window ending at `now` (capital_com_feed.py:100-107); Capital.com includes the current, still-forming bar, and every strategy reads `closes[-1]`. Level-based rules then trigger on intra-bar wiggles and "un-trigger" a minute later — churn that inflates trade counts and degrades every downstream statistic (the shadow journal's thousands of round-trips are partly this). Not confirmed end-to-end because I can't verify Capital.com's partial-bar semantics from here. **Fix if confirmed**: drop the last bar (or require `ts` older than one full period) for signal evaluation; keep the live bar only for marking/exits.

### F-15 — RL: train/serve skew and the all-skip equilibrium (Suspected)
The policy is graded exclusively on paper fills (F-02): shadow fills use `CostModel` mid-price assumptions, so if live fills differ systematically, the policy optimizes the wrong distribution the day live grading is added. Separately, `reward = -pnl_norm` for skips (policy.py:132) makes "skip everything" a positive-expectation strategy whenever the signal stream is net-losing per trade — and with `entropy_coef: 0.01` and ε→0.02, nothing structurally prevents collapse; production already showed `P(take)=0.000`. **Fix**: monitor take-rate with an alarm band; consider a small per-skip opportunity penalty or higher entropy floor; grade live trades (F-02) before trusting the policy near real money.

### F-16 — Degenerate strategies stay enabled
`bb_macd_sma` (kraken_strategies.py:59-71) buys when price > BB-mid **and MACD < 0** (and vice versa) — as written it fades momentum while above the mean, and its live record (1.8% win rate over 228 trades in the user's dashboard) matches a broken rule rather than bad luck. Several others sit near-zero (`fvg_m15` 5.9%/170, `adx_momentum_ema` 9.6%/187). Nothing auto-demotes a strategy on sustained negative expectancy; reflection only *tunes params*, it never turns a strategy off. **Fix**: review S3's intended logic against its source; add an auto-demotion rule (e.g. off/shadow after N trades with PF < 0.5).

### F-17 — In-sample acceptance gate (Suspected)
`accept_change` (evaluator.py:76-86) compares before/after on the **same** lookback window the proposal was fit on (reflection_pipeline.py:54-57 re-scores the identical `closed` set). The `walk_forward` optimizer exists (learning/optimizer.py) but isn't part of this gate. With 28 strategies × several params each tuned on ≤50-trade windows, accepted "improvements" will frequently be noise. **Fix**: hold out the most recent K trades from the fit window for the accept comparison, or route acceptance through the walk-forward optimizer.

### F-18/F-19/F-20 — Low severity
- F-18: agent.py:423-425 reads `getattr(self.shadow_broker, "_positions", {})` — a rename silently zeroes portfolio heat (RL input #14). Add `PaperBroker.position_count()`.
- F-19: `"RSS fetch failed for None"` in production logs means a `None` slipped into the feed list (and feedparser is handed `None`); FinnhubNewsFeed reads `config.get("credentials", ...)` — a path that is never populated (secrets live on `config.secrets`), so it's permanently dead.
- F-20: the suite (77 pass) is decent on pure logic (RL math, filters, costs, features) but has **zero** coverage of `_apply_control` (would have caught F-00a), `CapitalComBroker` order/close/confirm flows (test_capital.py covers the session/feed), or scheduler failure behavior; and the 3 failing portfolio tests are real regressions left red — a red suite normalizes ignoring the harness.

---

## 5. Fix plan

**P0 — before the bot runs again at all (demo included):**
1. **F-01** floor-vs-caps: reject instead of round-up (portfolio.py `vet()`); un-break the 3 failing tests — they are the verification.
2. **F-11** rewire or remove the dead min/max-lot knobs (same file/function as #1 — do together).
3. **F-06** stop blocking the loop: news TTL + `to_thread`; reflection in `to_thread`. Verify: fast-loop iteration time logged < 5s with LLM enabled.
4. **F-07** heartbeat freshness healthcheck + consecutive-failure trading pause. Verify: kill the feed, watch the container go unhealthy.
5. **F-09** dashboard `latest_signal` → `signal` (one line). Verify: Latest Signal card populates.

**P1 — before any strategy goes `live` with real (non-demo) money:**
6. **F-00b** demo-by-default + startup endpoint banner + URL/flag contradiction guard. Verify: unset env → connects to demo; explicit live requires opt-in.
7. **F-03 + F-05** live position tracking by `(symbol, strategy)`/dealId, same-direction skip and opposite-close for the live broker, no `entry_price=0` trades. Verify: new test — persistent signal over 3 fast loops yields exactly 1 live submit.
8. **F-02** live exit capture + journaling + RL grading via transaction history; periodic broker↔internal↔journal reconciliation with divergence alert. Verify on demo: bracket exit produces a journaled round-trip with real PnL and an RL update.
9. **F-04** leverage as constraint, not multiplier (touches all three sizers + `vet()` — do with #7's risk work). Verify: with leverage 200, a 0.5% risk trade still risks 0.5% at the stop.
10. **F-08** dashboard auth. Verify: unauthenticated `/api/risk` POST → 401.
11. **F-10** breaker hardening: peak-drawdown baseline, halt alerting, stale-data skip.

**P2 — can wait, schedule soon:**
12. **F-13** atomic YAML/NPZ writes (+ `.bak` for the policy).
13. **F-12** enforce `trades_symbol` (one line + config cleanup).
14. **F-14** closed-bar evaluation (verify partial-bar semantics on demo first).
15. **F-16** fix/disable degenerate strategies; auto-demotion rule.
16. **F-15/F-17** RL take-rate alarm; out-of-sample acceptance gate.
17. **F-19/F-20** feed cleanups; tests for `_apply_control`, `CapitalComBroker` (mock the session), scheduler failure counting.

---

## 6. What I couldn't assess

- **Capital.com account semantics**: whether the demo account is in netting or hedging mode (determines what "two live strategies, one symbol" does server-side), actual per-instrument margin factors, and whether `DELETE /positions/{dealId}` supports partial closes. Needs broker docs / demo experimentation.
- **Partial-bar behavior of `/prices`** (F-14): needs a live capture comparing consecutive polls inside one bar.
- **Whether demo fills approximate live fills** (spread/slippage) — determines how much F-15's train/serve skew matters. Needs paired demo/live data or broker documentation.
- **Strategy statistical significance**: win rates quoted are from the user's running instance (shadow fills, forming-bar churn included). No backtest data exists in the repo to separate "bad strategy" from "bad measurement" beyond F-16's logic reading.
- **The offline RL / IQL / advisory subsystem** (learning/rl/offline.py, iql.py, train_offline.py): disabled by default and not on the trading path; skimmed for interface only, not audited for correctness.
- **Docker entrypoint privilege handling** (chown-then-gosu): reviewed the compose comments only, not the entrypoint script's failure modes.
