"""Pre-trade context filters — the selectivity layer the agent lacked.

The strategies generate plenty of signals; what separates a disciplined system
from signal-spam is *not taking* trades when the context is wrong. Each filter is
independently toggleable (dashboard) and, when on, can only **veto** a signal —
never create or enlarge one. They also feed regime context the offline RL can use.

Filters:
  • trend       — signal side must agree with the fast/slow EMA trend
  • volatility  — skip dead chop and extreme (news-spike) volatility
  • volume      — skip thin volume (poor liquidity / wide spreads)
  • session     — skip outside the instrument's liquid hours (UTC)
  • spread      — skip when the quoted spread (bps) is too wide
  • regime      — route by regime: no mean-reversion in trends, no trend-following
                  in ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..data.models import Side

# Liquid UTC hour windows per instrument category (start <= hour < end).
_SESSIONS: dict[str, list[tuple[int, int]]] = {
    "Indices": [(13, 21)],       # US cash session (approx)
    "FX": [(7, 21)],             # London + New York
    "Commodities": [(7, 21)],
    "Crypto": [(0, 24)],         # 24/7
}

_CATEGORIES = {
    "Indices": {"US100", "US500", "US30", "RTY", "J225", "DE40", "UK100", "HK50",
                "NAS100", "SPX500", "DJI30"},
    "Commodities": {"GOLD", "XAUUSD", "SILVER", "XAGUSD", "OIL", "WTI", "BRENT", "NATGAS"},
    "Crypto": {"BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD", "DOGEUSD", "ADAUSD", "LTCUSD", "XBTUSD"},
}

# Strategies that fade extremes (mean-reversion) vs. ride trends.
REVERSION = {"mean_reversion", "cci_reversal", "bb_rsi", "multi_bb",
             "bb_macd_sma", "bb_rsi_cutting"}


def category(symbol: str) -> str:
    for cat, members in _CATEGORIES.items():
        if symbol in members:
            return cat
    return "FX"


def calendar_allows(symbol: str, now: datetime | None = None) -> bool:
    """Conservative calendar gate before broker-specific status checks.

    Crypto is 24/7; all other configured instruments are blocked on weekends.
    Broker status remains authoritative for weekday closes and holidays.
    """
    now = now or datetime.now(timezone.utc)
    return category(symbol) == "Crypto" or now.weekday() < 5


def in_session(symbol: str, hour: int, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return calendar_allows(symbol, now) and any(
        s <= now.hour < e for s, e in _SESSIONS.get(category(symbol), [(0, 24)])
    )


def classify_regime(f) -> str:
    """trend_up / trend_down / range from ADX + EMA stack."""
    adx = float(getattr(f, "adx", 0.0) or 0.0)
    if adx >= 25.0:
        return "trend_up" if (f.ema_fast or 0) >= (f.ema_slow or 0) else "trend_down"
    return "range"


def market_regime_filter(signal, sentiment) -> tuple[bool, str | None]:
    """Sentiment-based market regime filter: veto trades that conflict with sentiment.

    Returns (allowed, veto_reason). Use this as a secondary confirmation layer when
    sentiment is available and confident.
    """
    if sentiment is None or sentiment.confidence < 0.5:
        return True, None  # Weak sentiment: don't filter

    sentiment_score = sentiment.score
    is_bullish_signal = signal.side == Side.BUY

    # Market in panic (bearish sentiment): reject BUY signals
    if sentiment_score < -0.7 and sentiment.confidence > 0.7:
        if is_bullish_signal:
            return False, "sentiment_panic"

    # Market in euphoria (bullish sentiment): reject SELL signals
    if sentiment_score > 0.7 and sentiment.confidence > 0.7:
        if not is_bullish_signal:
            return False, "sentiment_euphoria"

    return True, None


@dataclass
class FilterConfig:
    trend: bool = False
    volatility: bool = False
    volume: bool = False
    session: bool = False
    spread: bool = False
    regime: bool = False
    # Parameters.
    vol_min: float = 0.0001       # atr/price floor (dead market below this)
    vol_max: float = 0.03         # atr/price ceiling (too wild above this)
    min_volume_ratio: float = 0.5  # last bar volume vs recent average
    max_spread_bps: float = 15.0
    adx_trend: float = 25.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "FilterConfig":
        d = d or {}
        f = cls()
        for k in vars(f):
            if k in d and d[k] is not None:
                setattr(f, k, type(getattr(f, k))(d[k]))
        return f


def _spread_bps(features) -> float | None:
    ob = getattr(features, "orderbook", None)
    last = features.last_price or 0.0
    if ob is None or not getattr(ob, "spread", None) or not last:
        return None
    return ob.spread / last * 10_000.0


def apply(signal, features, strategy: str, cfg: FilterConfig, symbol: str,
          now: datetime | None = None) -> tuple[bool, str | None]:
    """Return (allowed, veto_reason). Only enabled filters can veto."""
    if signal.side == Side.FLAT:
        return True, None
    now = now or datetime.now(timezone.utc)

    if cfg.session:
        windows = _SESSIONS.get(category(symbol), [(0, 24)])
        if not calendar_allows(symbol, now) or not any(s <= now.hour < e for s, e in windows):
            return False, "session"

    if cfg.spread:
        sb = _spread_bps(features)
        if sb is not None and sb > cfg.max_spread_bps:
            return False, "spread"

    if cfg.volatility:
        last = features.last_price or 0.0
        vol = (features.atr / last) if last else 0.0
        if vol < cfg.vol_min or vol > cfg.vol_max:
            return False, "volatility"

    if cfg.volume:
        candles = getattr(features, "candles", None) or []
        vols = [getattr(c, "volume", 0.0) or 0.0 for c in candles[-20:]]
        if len(vols) >= 5 and sum(vols) > 0:
            avg = sum(vols) / len(vols)
            if avg > 0 and vols[-1] < cfg.min_volume_ratio * avg:
                return False, "volume"

    if cfg.trend:
        up = (features.ema_fast or 0) >= (features.ema_slow or 0)
        if (signal.side == Side.BUY and not up) or (signal.side == Side.SELL and up):
            return False, "trend"

    if cfg.regime:
        reg = classify_regime(features)
        is_rev = strategy in REVERSION
        if reg.startswith("trend") and is_rev:
            return False, "regime"
        if reg == "range" and not is_rev:
            return False, "regime"

    return True, None
