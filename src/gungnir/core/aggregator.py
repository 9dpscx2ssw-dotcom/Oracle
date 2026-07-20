"""Consensus signal aggregation: many strategy stances → one decision per symbol.

Why this exists: 26 level-based strategies emitting independently makes the net
position flip-flop as individual strategies cross their thresholds — decision
churn. This module collapses their stances into a single smoothed consensus per
symbol and applies three churn-killers, in order of impact:

  1. **Conflict veto** — if the opposing side carries ≥ ``veto_opposing`` of the
     total vote weight (default 35%), no NEW position is opened. Disagreement is
     exactly where whipsaw is worst; refusing to trade there is cheaper than
     being on the wrong side of half your own book.
  2. **EMA smoothing** — the signed consensus is exponentially smoothed, so a
     single strategy joining/leaving a small stance book cannot swing the
     decision in one bar. Secondary to hysteresis: stances already carry
     persistence (level semantics), so the default alpha is mild (0.6).
  3. **Hysteresis** — enter only when the smoothed consensus clears
     ``enter_threshold``; exit only when it falls below the *lower*
     ``exit_threshold`` or flips sign. Between the two bands: hold, emit
     nothing. ``min_hold_bars`` additionally blocks consensus-driven exits right
     after entry (bracket stops still protect downside).
  4. **Family capping** — the 26 strategies are not independent voters: six
     HMA-Donchian clones agreeing is closer to ONE signal than six. Each stance
     carries a family tag, and no family may contribute more than
     ``family_cap`` of the total vote weight; excess is scaled down before the
     tally. Agreement across *different* signal types is what moves the
     consensus — and a structural dissenter (mean-reverters in a trend) can
     neither be drowned out below the veto threshold by sheer clone count, nor
     dominate it.

Votes are **stances, not edges**: a strategy that emitted BUY keeps voting BUY
until it releases the condition (or flips). Each stance's weight folds in
everything the system knows about signal quality — LLM-blended conviction ×
allocator regime-edge × RL P(take) — so the learners shape the vote *softly*
instead of hard-gating individual signals.

Pure and deterministic: no I/O, no clocks. The Agent owns when ``decide`` is
called (once per fresh-bar pass) and what it does with the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data.models import Side


@dataclass
class AggDecision:
    """One consensus verdict for one symbol on one bar."""

    action: str                 # "enter" | "exit" | "hold" | "veto" | "none"
    side: Side | None = None    # trade direction for "enter" (or held direction)
    strength: float = 0.0       # |smoothed consensus| in [0, 1] — sizes the trade
    consensus: float = 0.0      # signed smoothed consensus in [-1, 1]
    opposing: float = 0.0       # opposing fraction of total vote weight [0, 1]
    n_stances: int = 0          # strategies currently voting
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SymbolState:
    ema: float = 0.0
    short_ema: float = 0.0      # independent scalp lane (1M–15M)
    bars_held: int = 0          # decide() passes since the last consensus entry
    # strategy -> (side, effective weight, family, horizon)
    stances: dict[str, tuple[Side, float, str, str]] = field(default_factory=dict)


class SignalAggregator:
    """Conviction-weighted consensus with a conflict veto and hysteresis."""

    def __init__(self, *, veto_opposing: float = 0.35, ema_alpha: float = 0.6,
                 enter_threshold: float = 0.25, exit_threshold: float = 0.10,
                 min_hold_bars: int = 2, family_cap: float = 0.4,
                 horizon_weights: dict[str, float] | None = None) -> None:
        self.veto_opposing = veto_opposing
        self.ema_alpha = ema_alpha
        self.enter_threshold = enter_threshold
        # Exit band must sit below the entry band or hysteresis degenerates
        # into a single threshold (the churn this class exists to remove).
        self.exit_threshold = min(exit_threshold, enter_threshold)
        self.min_hold_bars = max(0, min_hold_bars)
        self.family_cap = family_cap      # outside (0, 1) disables capping
        self.horizon_weights = {
            # Support both strategy/config spellings (1H and H1).
            "M1": 0.80, "1M": 0.80, "M3": 0.85, "3M": 0.85,
            "M5": 0.90, "5M": 0.90, "M15": 1.00, "15M": 1.00,
            "M30": 1.05, "30M": 1.05, "H1": 1.15, "1H": 1.15,
            "H4": 1.20, "4H": 1.20, "D1": 1.25, "1D": 1.25,
        }
        if horizon_weights:
            self.horizon_weights.update({str(k).upper(): max(0.0, float(v))
                                         for k, v in horizon_weights.items()})
        self._state: dict[str, _SymbolState] = {}

    # ── stance book (level semantics over edge-triggered plumbing) ─────────

    def set_stance(self, symbol: str, strategy: str, side: Side,
                   weight: float, family: str | None = None,
                   horizon: str | None = None) -> None:
        """Record/refresh a strategy's current directional stance on a symbol.

        ``family`` groups correlated strategies for the vote cap; it defaults
        to the strategy's own name (a family of one — no capping effect).
        """
        st = self._state.setdefault(symbol, _SymbolState())
        hz = str(horizon or "M5").upper()
        factor = self.horizon_weights.get(hz, 1.0)
        st.stances[strategy] = (side, max(0.0, weight) * factor,
                                family or strategy, hz)

    def clear_stance(self, symbol: str, strategy: str) -> None:
        """The strategy released its condition — it no longer votes."""
        st = self._state.get(symbol)
        if st is not None:
            st.stances.pop(strategy, None)

    # ── consensus math ─────────────────────────────────────────────────────

    @staticmethod
    def _horizon_minutes(horizon: str) -> int | None:
        hz = str(horizon).upper().strip()
        for suffix, mult in (("M", 1), ("H", 60), ("D", 1440)):
            if hz.endswith(suffix):
                try: return int(hz[:-1]) * mult
                except ValueError: return None
        return None

    def _tally(self, stances: dict[str, tuple[Side, float, str, str]]
               ) -> tuple[float, float, dict[str, Any]]:
        """(signed consensus S in [-1,1], opposing fraction vs sign(S)).

        Family cap first: any family holding more than ``family_cap`` of the
        pre-cap total has its stances scaled down to exactly that share (one
        deterministic pass against the original total), so clone-count can't
        manufacture consensus strength.
        """
        total = sum(w for _, w, _, _ in stances.values())
        if total <= 0:
            return 0.0, 0.0, {"raw_score": 0.0, "buy_weight": 0.0,
                              "sell_weight": 0.0, "effective_total": 0.0,
                              "families": {}, "horizons": {}}
        cap_on = 0.0 < self.family_cap < 1.0
        fam_tot: dict[str, float] = {}
        for _, w, fam, _ in stances.values():
            fam_tot[fam] = fam_tot.get(fam, 0.0) + w
        scale = {fam: (min(1.0, self.family_cap * total / fw)
                       if cap_on and fw > 0 else 1.0)
                 for fam, fw in fam_tot.items()}
        eff = [(side, w * scale[fam], fam, hz)
               for side, w, fam, hz in stances.values()]
        eff_total = sum(w for _, w, _, _ in eff)
        if eff_total <= 0:
            return 0.0, 0.0, {"raw_score": 0.0, "buy_weight": 0.0,
                              "sell_weight": 0.0, "effective_total": 0.0,
                              "families": {}, "horizons": {}}
        buy = sum(w for side, w, _, _ in eff if side == Side.BUY)
        sell = sum(w for side, w, _, _ in eff if side == Side.SELL)
        signed = buy - sell
        s = signed / eff_total
        families = {}
        horizons = {}
        for side, w, fam, hz in eff:
            entry = families.setdefault(fam, {"buy": 0.0, "sell": 0.0, "weight": 0.0})
            entry["buy" if side == Side.BUY else "sell"] += w
            entry["weight"] += w
            horizons[hz] = horizons.get(hz, 0.0) + w
        diag = {"raw_score": round(s, 4), "buy_weight": round(buy, 4),
                "sell_weight": round(sell, 4), "effective_total": round(eff_total, 4),
                "families": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                             for k, v in families.items()},
                "horizons": {k: round(v, 4) for k, v in horizons.items()}}
        if s == 0.0:
            return 0.0, 0.5, diag             # perfectly split book
        losing = Side.SELL if s > 0 else Side.BUY
        opp = sum(w for side, w, _, _ in eff if side == losing) / eff_total
        return s, opp, diag

    def decide(self, symbol: str, position_side: Side | None) -> AggDecision:
        """Advance the consensus one bar and produce a verdict.

        ``position_side`` is the CURRENT consensus position on the account book
        (None when flat) — queried live by the caller, so bracket exits that
        closed the position out-of-band are self-correcting here.
        """
        st = self._state.setdefault(symbol, _SymbolState())
        s, opp, diag = self._tally(st.stances)
        st.ema = self.ema_alpha * s + (1.0 - self.ema_alpha) * st.ema
        ema = st.ema
        side = Side.BUY if ema > 0 else Side.SELL if ema < 0 else None
        short_stances = {k: v for k, v in st.stances.items()
                         if (self._horizon_minutes(v[3]) or 10**9) <= 15}
        short_s, short_opp, short_diag = self._tally(short_stances)
        st.short_ema = self.ema_alpha * short_s + (1.0 - self.ema_alpha) * st.short_ema
        short_ema = st.short_ema
        short_side = (Side.BUY if short_ema > 0 else Side.SELL if short_ema < 0 else None)
        n = len(st.stances)
        diag = {**diag, "ema_score": round(ema, 4),
                "short_lane": {**short_diag, "ema_score": round(short_ema, 4),
                               "opposing": round(short_opp, 4),
                               "n_stances": len(short_stances)},
                "opposing": round(opp, 4),
                "entry_threshold": self.enter_threshold,
                "exit_threshold": self.exit_threshold,
                "veto_threshold": self.veto_opposing}
        base: dict[str, Any] = dict(consensus=round(ema, 4),
                                    opposing=round(opp, 4), n_stances=n,
                                    diagnostics=diag)

        if position_side is None:
            st.bars_held = 0
            short_qualifies = (short_side is not None and len(short_stances) > 0
                               and abs(short_ema) >= self.enter_threshold
                               and short_opp < self.veto_opposing)
            if short_qualifies and (side is None or abs(ema) < self.enter_threshold
                                    or opp >= self.veto_opposing or short_side != side):
                return AggDecision(action="enter", side=short_side,
                                   strength=min(1.0, abs(short_ema)), **base)
            if side is None or abs(ema) < self.enter_threshold:
                return AggDecision(action="none", **base)
            if opp >= self.veto_opposing:
                return AggDecision(action="veto", side=side,
                                   strength=abs(ema), **base)
            return AggDecision(action="enter", side=side,
                               strength=min(1.0, abs(ema)), **base)

        # In a position: hold inside the hysteresis band; exit on sign flip or
        # when conviction drains below the (lower) exit band.
        st.bars_held += 1
        flipped = side is not None and side != position_side
        drained = abs(ema) < self.exit_threshold
        if (flipped or drained) and st.bars_held > self.min_hold_bars:
            st.bars_held = 0
            return AggDecision(action="exit", side=position_side,
                               strength=abs(ema), **base)
        return AggDecision(action="hold", side=position_side,
                           strength=abs(ema), **base)
