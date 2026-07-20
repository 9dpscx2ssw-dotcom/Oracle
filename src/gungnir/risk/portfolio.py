"""Portfolio-level risk: the last gate before execution.

Owns account-wide limits and can veto or shrink any order:
  • gross exposure / leverage cap
  • per-asset exposure cap
  • max open positions
  • drawdown circuit-breakers (halt new entries when breached), tracked
    **per book**: the real account and the shadow/paper book each get their
    own daily, intraday-from-peak, and total-drawdown breakers. Before this,
    the breakers only watched the real account — in demo-connected mode with
    shadow-mode strategies, the shadow book could bleed indefinitely while
    the breakers stared at an untouched account (the 16h incident).

These bind on the live `open_exposure` map, which the agent refreshes from broker
positions each loop. (A correlation/cluster cap is intentionally not implemented
yet — do not assume one exists.)

This is intentionally conservative: when in doubt, size down or skip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Config
from ..data.models import Order, Side, Signal

log = logging.getLogger(__name__)


@dataclass
class DrawdownTracker:
    """Drawdown state for one book (real account or shadow/paper)."""
    equity: float = 0.0
    day_start: float = 0.0
    day_peak: float = 0.0
    peak: float = 0.0

    def roll_day(self) -> None:
        """New UTC session: today's baseline and peak restart from here."""
        self.day_start = self.equity
        self.day_peak = self.equity

    def update(self, equity: float) -> None:
        self.equity = equity
        self.peak = max(self.peak, equity)
        self.day_peak = max(self.day_peak, equity)

    def breach(self, max_daily: float, max_intraday: float, max_total: float) -> str:
        """The first tripped breaker's description, or '' when none tripped."""
        if self.day_start > 0:
            dd = (self.day_start - self.equity) / self.day_start
            if dd >= max_daily:
                return f"daily drawdown {dd:.2%} >= {max_daily:.2%}"
        # From today's PEAK, not just the day's start — a crash that follows a
        # morning run-up would otherwise hide inside the daily allowance.
        if max_intraday and self.day_peak > 0:
            idd = (self.day_peak - self.equity) / self.day_peak
            if idd >= max_intraday:
                return (f"intraday drawdown {idd:.2%} from today's peak "
                        f">= {max_intraday:.2%}")
        if max_total and self.peak > 0:
            tdd = (self.peak - self.equity) / self.peak
            if tdd >= max_total:
                return f"total drawdown {tdd:.2%} >= {max_total:.2%} from peak"
        return ""

# Market type categorization by symbol prefix/pattern
_MARKET_TYPES = {
    # Indices
    "US100": "indices", "US500": "indices", "US30": "indices", "RTY": "indices",
    "J225": "indices", "DE40": "indices", "UK100": "indices", "HK50": "indices",
    # Commodities
    "GOLD": "commodities",
    # Crypto
    "BTCUSD": "crypto", "ETHUSD": "crypto", "XRPUSD": "crypto",
    "SOLUSD": "crypto", "DOGEUSD": "crypto",
    # US Stocks & ETFs
    "AAPL": "stocks", "MSFT": "stocks", "NVDA": "stocks", "TSLA": "stocks",
    "GOOGL": "stocks", "AMZN": "stocks", "META": "stocks", "INTC": "stocks",
    "AMD": "stocks", "NFLX": "stocks", "ADBE": "stocks", "QCOM": "stocks",
    "AVGO": "stocks", "MU": "stocks", "SNDK": "stocks", "MRVL": "stocks",
    "SMCI": "stocks", "ORCL": "stocks", "COIN": "stocks", "PLTR": "stocks",
    "UBER": "stocks", "ARM": "stocks", "BABA": "stocks", "TSM": "stocks",
    "LLY": "stocks", "SOFI": "stocks", "IONQ": "stocks", "QBTS": "stocks",
    "RGTI": "stocks", "CORZ": "stocks", "DELL": "stocks", "NOW": "stocks",
    "SPCE": "stocks", "GME": "stocks", "MARA": "stocks", "HIMS": "stocks",
    "AMC": "stocks", "SPY": "stocks", "QQQ": "stocks", "IVV": "stocks",
}

# Currency codes for detecting FX pairs (EURUSD, USDJPY, EURGBP, …).
_CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "MXN", "TRY",
        "PLN", "NOK", "SEK", "DKK", "ZAR", "CNH", "SGD", "HKD"}


def _get_market_type(symbol: str) -> str:
    """Market type from the symbol name.

    Unknown symbols used to default to *forex* — 200:1 broker leverage — so a
    newly listed stock epic absent from the table got a 10× too-generous margin
    cap. FX pairs are now detected structurally (two ISO currency codes), and
    everything unrecognized falls back to the MOST conservative class (stocks,
    20:1): an unknown instrument should be under-margined, never over.
    """
    known = _MARKET_TYPES.get(symbol)
    if known:
        return known
    s = symbol.upper()
    if len(s) == 6 and s[:3] in _CCY and s[3:] in _CCY:
        return "forex"
    return "stocks"


class PortfolioRisk:
    def __init__(self, config: Config):
        self.max_gross = config.get("risk", "max_portfolio_exposure", default=2.0)
        self.max_per_asset = config.get("risk", "max_per_asset_exposure", default=0.5)
        self.max_positions = config.get("risk", "max_open_positions", default=5)
        self.max_daily_dd = config.get("risk", "max_daily_drawdown", default=0.03)
        self.min_confidence = config.get("risk", "min_confidence", default=0.3)
        self.stop_atr_mult = config.get("risk", "stop_atr_mult", default=2.0)
        self.tp_atr_mult = config.get("risk", "tp_atr_mult", default=3.0)

        # Market-type-specific lot size limits (can be overridden per type from dashboard).
        min_by_type = config.get("risk", "min_lot_by_type", default={})
        max_by_type = config.get("risk", "max_lot_by_type", default={})
        self.min_lot_by_type = {
            "indices": float(min_by_type.get("indices", 0.01)),
            "commodities": float(min_by_type.get("commodities", 0.01)),
            "forex": float(min_by_type.get("forex", 100.0)),
            "crypto": float(min_by_type.get("crypto", 0.001)),
            "stocks": float(min_by_type.get("stocks", 1.0)),
        }
        self.max_lot_by_type = {
            "indices": float(max_by_type["indices"]) if max_by_type.get("indices") else None,
            "commodities": float(max_by_type["commodities"]) if max_by_type.get("commodities") else None,
            "forex": float(max_by_type["forex"]) if max_by_type.get("forex") else None,
            "crypto": float(max_by_type["crypto"]) if max_by_type.get("crypto") else None,
            "stocks": float(max_by_type["stocks"]) if max_by_type.get("stocks") else None,
        }
        # Dashboard overrides: when set they take precedence over the by-type
        # tables for every market type (audit F-11 — these knobs were dead).
        self.min_lot: float | None = None
        self.max_lot: float | None = None

        # Leverage bounds *margin capacity* (max notional per order), never
        # position size (audit F-04 — it used to multiply the sizers' output).
        # Brokers grant different leverage per asset class (e.g. Capital.com:
        # 200:1 FX/indices/commodities but only 20:1 shares/crypto), so a
        # per-type table overrides the scalar; unlisted types use the scalar.
        self.leverage = float(config.get("risk", "leverage", default=1.0) or 1.0)
        lev_by_type = config.get("risk", "leverage_by_type", default={}) or {}
        self.leverage_by_type = {
            k: float(v) for k, v in lev_by_type.items() if v
        }
        self.lev_safety = float(config.get("risk", "leverage_safety_margin", default=0.10) or 0.0)
        # Peak-to-trough breaker on top of the daily one (audit F-10): the daily
        # baseline resets every UTC midnight, so a multi-day bleed never tripped.
        self.max_total_dd = float(config.get("risk", "max_total_drawdown", default=0.25) or 0.0)
        # Intraday peak-to-trough breaker: halts when equity falls this far
        # from TODAY'S peak (0 disables). Catches a crash that follows a
        # run-up, which the daily (from-open) breaker can't see.
        self.max_intraday_dd = float(
            config.get("risk", "max_intraday_drawdown", default=0.05) or 0.0)

        # Live state updated by the agent from broker account info. One
        # drawdown tracker per book — orders are vetted against the book they
        # actually fill on.
        self.books: dict[str, DrawdownTracker] = {
            "real": DrawdownTracker(), "shadow": DrawdownTracker(),
        }
        self.open_exposure: dict[str, float] = {}  # symbol -> notional
        self._halted: dict[str, bool] = {"real": False, "shadow": False}
        # Cap-saturation telemetry: when most orders come out cut by the caps,
        # the sizers upstream (conviction, allocator, sentiment scaling) are
        # being erased — the exact failure the audit found with the old
        # VolTarget formula. Published to status; warned on in the slow loop.
        self.cap_stats = {"total": 0, "capped": 0}

    # ── back-compat accessors (the real account's tracker) ───────────────────

    @property
    def equity(self) -> float:
        return self.books["real"].equity

    @equity.setter
    def equity(self, value: float) -> None:
        self.books["real"].update(value)

    @property
    def day_start_equity(self) -> float:
        return self.books["real"].day_start

    @day_start_equity.setter
    def day_start_equity(self, value: float) -> None:
        self.books["real"].day_start = value

    @property
    def peak_equity(self) -> float:
        return self.books["real"].peak

    @peak_equity.setter
    def peak_equity(self, value: float) -> None:
        self.books["real"].peak = value

    def update_book(self, book: str, equity: float) -> None:
        tracker = self.books.get(book)
        if tracker is not None:
            tracker.update(equity)

    # ── breaker-state persistence ────────────────────────────────────────────
    # The total-drawdown breaker measures from the all-time peak; without
    # persistence every restart re-armed it from the (lower) current equity —
    # a restart after a 20% loss quietly granted a fresh 25% allowance.

    def save_state(self, path: str) -> None:
        import json
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "day": datetime.now(timezone.utc).date().isoformat(),
                "books": {
                    name: {"equity": t.equity, "day_start": t.day_start,
                           "day_peak": t.day_peak, "peak": t.peak}
                    for name, t in self.books.items()
                },
            }
            with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False,
                                             suffix=".tmp") as tmp:
                json.dump(payload, tmp)
                tmp_path = tmp.name
            Path(tmp_path).replace(p)
        except OSError as e:
            log.warning("Risk state save failed: %s", e)

    def load_state(self, path: str) -> str | None:
        """Restore breaker state; returns the saved UTC day (ISO) or None.

        Same-day restart restores everything (the daily breaker keeps binding
        on today's true baseline). A restart on a later day restores only the
        all-time ``peak`` — the daily fields re-baseline at the next roll_day.
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            saved_day = data.get("day")
            same_day = saved_day == datetime.now(timezone.utc).date().isoformat()
            for name, vals in (data.get("books") or {}).items():
                t = self.books.get(name)
                if t is None:
                    continue
                t.peak = float(vals.get("peak", 0.0))
                if same_day:
                    t.equity = float(vals.get("equity", 0.0))
                    t.day_start = float(vals.get("day_start", 0.0))
                    t.day_peak = float(vals.get("day_peak", 0.0))
            log.info("Restored breaker state from %s (day=%s, same_day=%s)",
                     p, saved_day, same_day)
            return saved_day if same_day else None
        except (OSError, ValueError, TypeError) as e:
            log.warning("Risk state load failed (%s); starting fresh", e)
            return None

    def roll_day(self) -> None:
        """UTC session rollover: restart every book's daily baseline and peak."""
        for tracker in self.books.values():
            tracker.roll_day()

    # ── circuit breaker ──────────────────────────────────────────────────────

    def trading_halted(self, book: str = "real") -> bool:
        tracker = self.books.get(book)
        if tracker is None:
            return False
        reason = tracker.breach(self.max_daily_dd, self.max_intraday_dd,
                                self.max_total_dd)
        halted = bool(reason)
        # Log loudly on the transition only, not once per loop.
        if halted and not self._halted[book]:
            log.error("TRADING HALTED (%s book): %s. New entries blocked until "
                      "recovery.", book, reason)
        elif self._halted[book] and not halted:
            log.warning("Trading halt cleared (%s book); new entries allowed again.", book)
        self._halted[book] = halted
        return halted

    # ── the gate ─────────────────────────────────────────────────────────────

    def vet(self, signal: Signal, raw_volume: float, price: float, atr: float,
            instrument_min: float = 0.0, book: str = "real") -> Order | None:
        """Apply account-level limits; return a final Order or None to reject.

        ``instrument_min`` is the broker's minimum deal size for this symbol; a
        sized volume below it is rounded up so the order isn't rejected as too
        small (the agent was previously sending sub-minimum lots). ``book`` is
        where this order will fill — its own drawdown breakers gate it."""
        if signal.side == Side.FLAT or raw_volume <= 0:
            return None
        # Confidence gate: reject signals below minimum conviction threshold.
        if signal.conviction < self.min_confidence:
            log.debug("Signal %s rejected: conviction %.2f < %.2f", signal.symbol, signal.conviction, self.min_confidence)
            return None
        if self.trading_halted(book):
            return None
        if len(self.open_exposure) >= self.max_positions and signal.symbol not in self.open_exposure:
            log.info("Max open positions reached; skipping %s.", signal.symbol)
            return None

        notional = raw_volume * price
        requested_notional = notional
        equity = self.equity or 1.0

        # Margin capacity: notional per order can never exceed what the broker
        # actually extends for this asset class (with safety buffer). E.g. on
        # Capital.com, FX/indices/commodities get 200:1 but shares/crypto only
        # 20:1 — sizing crypto against 200:1 would just get orders rejected.
        symbol_type = _get_market_type(signal.symbol)
        leverage = self.leverage_by_type.get(symbol_type, self.leverage)
        margin_cap = equity * (leverage / (1.0 + self.lev_safety))
        if notional > margin_cap:
            notional = margin_cap

        # Per-asset cap.
        per_asset_cap = self.max_per_asset * equity
        existing = self.open_exposure.get(signal.symbol, 0.0)
        if existing + notional > per_asset_cap:
            notional = max(0.0, per_asset_cap - existing)

        # Gross exposure cap.
        gross = sum(abs(v) for v in self.open_exposure.values())
        gross_cap = self.max_gross * equity
        if gross + notional > gross_cap:
            notional = max(0.0, gross_cap - gross)

        if notional <= 0:
            return None

        import math
        final_volume = notional / price
        # Lot-size limits. Dashboard overrides beat the by-type tables.
        market_type = _get_market_type(signal.symbol)
        min_lot = self.min_lot if self.min_lot is not None else self.min_lot_by_type.get(market_type, 0.01)
        max_lot = self.max_lot if self.max_lot is not None else self.max_lot_by_type.get(market_type)

        if max_lot:
            final_volume = min(final_volume, max_lot)
        # Quantize DOWN to the 4-dp volume step. Plain round() half-up could
        # push a cap-saturated volume back over the cap it was just cut to —
        # found by the property tests (notional 50.0004 vs a 50.00 cap).
        final_volume = math.floor(final_volume * 10_000) / 10_000.0
        # Minimum-lot floor: only round UP when the floored order still fits
        # every cap above. The floor exists to avoid broker dust-rejections; it
        # must never override the risk caps (audit F-01 — it used to re-inflate
        # capped orders back to full size). The floor itself is quantized UP so
        # it never lands below the broker minimum.
        floor = max(min_lot or 0.0, instrument_min or 0.0)
        floor_q = math.ceil(floor * 10_000) / 10_000.0 if floor > 0 else 0.0
        if floor_q and 0 < final_volume < floor_q:
            floored_notional = floor_q * price
            fits_caps = (
                existing + floored_notional <= per_asset_cap + 1e-9
                and gross + floored_notional <= gross_cap + 1e-9
                and floored_notional <= margin_cap + 1e-9
                and not (max_lot and floor_q > max_lot)
            )
            if fits_caps:
                final_volume = floor_q
            else:
                # Can't reach the broker's minimum without breaching a risk cap
                # (typical once earlier strategies on this symbol have eaten the
                # per-asset/gross headroom). Sending a sub-minimum order just
                # earns a broker rejection — the 400s and vol=0.0000 paper junk
                # in the logs — so reject cleanly here instead.
                return None
        # Final guard: a volume that collapses to zero (tiny sliver of leftover
        # cap, no floor to catch it) is not a tradeable order.
        if final_volume <= 0:
            return None
        self.cap_stats["total"] += 1
        was_capped = final_volume * price < requested_notional * 0.999
        if was_capped:
            self.cap_stats["capped"] += 1
        from ..core import metrics
        metrics.cap_saturation(was_capped)
        stop, tp = self._brackets(signal.side, price, atr)
        return Order(
            symbol=signal.symbol,
            side=signal.side,
            volume=final_volume,
            stop_loss=stop,
            take_profit=tp,
            client_id=f"{signal.strategy}:{signal.symbol}:{int(signal.ts.timestamp())}",
        )

    def _brackets(self, side: Side, price: float, atr: float) -> tuple[float | None, float | None]:
        if atr <= 0:
            return None, None
        # Ensure minimum distance from entry to prevent immediate exits
        # Use at least 0.1% of price as minimum bracket distance
        min_distance = max(atr * self.stop_atr_mult, price * 0.001)
        if side == Side.BUY:
            stop = price - min_distance
            tp = price + max(atr * self.tp_atr_mult, price * 0.0015)
            return stop, tp
        else:
            stop = price + min_distance
            tp = price - max(atr * self.tp_atr_mult, price * 0.0015)
            return stop, tp

    def scale_with_sentiment(self, volume: float, signal: Signal, sentiment) -> float:
        """Scale position size based on signal-sentiment alignment.

        When signal direction agrees with sentiment, size up (confidence boost).
        When they conflict, size down (disagreement penalty).
        If sentiment is weak/neutral, no adjustment.
        """
        if sentiment is None or sentiment.confidence < 0.4:
            return volume  # Weak sentiment: no adjustment

        is_bullish = signal.side == Side.BUY
        sentiment_agrees = (is_bullish and sentiment.score > 0.2) or \
                          (not is_bullish and sentiment.score < -0.2)

        if sentiment_agrees:
            # Signal + sentiment aligned → size up
            abs_score = abs(sentiment.score)
            multiplier = 1.0 + (abs_score * 0.3)  # +30% max if strong agreement
            return volume * multiplier
        else:
            # Signal contradicts sentiment → size down
            abs_score = abs(sentiment.score)
            multiplier = max(0.5, 1.0 - (abs_score * 0.5))  # -50% max if strong disagreement
            return volume * multiplier
