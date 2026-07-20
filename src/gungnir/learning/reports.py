"""Performance reporting: daily / weekly breakdowns and a plain-language summary.

One source of truth for both surfaces that report performance — the Telegram
digest (``format_telegram``) and the dashboard's Reports tab (the JSON shape
straight off ``build``). Keeping them on the same functions means the number
you read on your phone is the number on the console, always.

Everything here is a pure function of a list of closed ``Trade`` objects, so it
is deterministic and unit-tested. The narrative summary is composed from the
computed figures (never an LLM guess), so it can't misstate P&L.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ..data.models import Side, Trade
from .evaluator import evaluate


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _closed_at(t: Trade) -> datetime | None:
    return _aware(t.closed_at or t.opened_at)


def _avg_confidence(trades: list[Trade]) -> float | None:
    vals = [
        float((t.context or {}).get("confidence"))
        for t in trades
        if (t.context or {}).get("confidence") is not None
    ]
    return sum(vals) / len(vals) if vals else None


def _avg_slippage(trades: list[Trade]) -> float | None:
    vals = [
        float((t.context or {}).get("slippage_bps"))
        for t in trades
        if (t.context or {}).get("slippage_bps") is not None
    ]
    return sum(vals) / len(vals) if vals else None


def _row(name: str, trades: list[Trade]) -> dict:
    """Metrics for one group (strategy or instrument), JSON-safe."""
    m = evaluate(trades)
    modes: dict[str, int] = defaultdict(int)
    for t in trades:
        modes[t.mode or "real"] += 1
    pf = m.profit_factor
    return {
        "name": name,
        "trades": m.n_trades,
        "win_rate": round(m.win_rate, 3),
        "pnl": round(m.total_pnl, 2),
        "expectancy": round(m.expectancy, 2),
        "profit_factor": (None if pf == float("inf") else round(pf, 2)),
        "avg_confidence": (round(c, 3) if (c := _avg_confidence(trades)) is not None
                           else None),
        "real": modes.get("real", 0),
        "shadow": modes.get("shadow", 0),
    }


def _grouped(trades: list[Trade], key) -> list[dict]:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[key(t) or "?"].append(t)
    rows = [_row(name, ts) for name, ts in groups.items()]
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


def _extreme_trade(trades: list[Trade], *, best: bool) -> dict | None:
    graded = [t for t in trades if t.pnl is not None]
    if not graded:
        return None
    t = (max if best else min)(graded, key=lambda x: x.pnl)
    return {
        "symbol": t.symbol,
        "strategy": t.strategy or "?",
        "side": t.side.value,
        "pnl": round(t.pnl, 2),
        "mode": t.mode or "real",
    }


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def _returns(trades: list[Trade]) -> list[float]:
    """Per-trade scale-free returns (pnl / notional), matching evaluator.sharpe."""
    return [
        t.pnl / abs(t.entry_price * t.volume)
        for t in trades
        if t.pnl is not None and t.entry_price and t.volume
    ]


def _sortino(rets: list[float]) -> float:
    """Un-annualized Sortino on per-trade returns: mean / downside deviation."""
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    downside = [r for r in rets if r < 0]
    if not downside:
        return 0.0
    dd = math.sqrt(sum(r * r for r in downside) / len(rets))
    return mean / dd if dd else 0.0


def _streaks(pnls: list[float]) -> tuple[int, int, int]:
    """(longest win streak, longest loss streak, current streak).

    ``pnls`` must be chronological. Current streak is signed: positive for a
    winning run, negative for a losing run, 0 if the last trade was flat.
    """
    max_w = max_l = run = sign = 0
    for p in pnls:
        s = 1 if p > 0 else (-1 if p < 0 else 0)
        if s == 0:
            run = sign = 0
            continue
        run = run + 1 if s == sign else 1
        sign = s
        if s > 0:
            max_w = max(max_w, run)
        else:
            max_l = max(max_l, run)
    return max_w, max_l, run * sign


def _hold_minutes(trades: list[Trade]) -> float | None:
    spans = []
    for t in trades:
        o, c = _aware(t.opened_at), _aware(t.closed_at)
        if o and c and c >= o:
            spans.append((c - o).total_seconds() / 60.0)
    return sum(spans) / len(spans) if spans else None


def _distribution(pnls: list[float], bins: int = 9) -> dict:
    """Symmetric-around-zero P&L histogram, so the profit/loss balance reads visually."""
    if not pnls:
        return {"bins": [], "max_abs": 0.0}
    mx = max(abs(min(pnls)), abs(max(pnls))) or 1.0
    step = 2 * mx / bins
    out = []
    for i in range(bins):
        lo = -mx + i * step
        hi = lo + step
        count = sum(1 for p in pnls
                    if (lo <= p < hi) or (i == bins - 1 and p == hi))
        out.append({"lo": round(lo, 2), "hi": round(hi, 2),
                    "mid": round((lo + hi) / 2, 2), "count": count})
    return {"bins": out, "max_abs": round(mx, 2)}


def _equity_curve(trades: list[Trade]) -> list[dict]:
    """Cumulative P&L over closed trades in chronological order."""
    ordered = sorted(
        (t for t in trades if t.pnl is not None),
        key=lambda t: _closed_at(t) or datetime.min.replace(tzinfo=timezone.utc),
    )
    pts, cum = [], 0.0
    for t in ordered:
        if t.pnl is None:
            continue
        cum += t.pnl
        c = _closed_at(t)
        pts.append({"t": c.isoformat() if c else None,
                    "pnl": round(t.pnl, 2), "cum": round(cum, 2)})
    return pts


def _sparks(chrono: list[Trade]) -> dict:
    """Running metric series over chronological trades — one honest sparkline per KPI."""
    cum = gross_win = gross_loss = peak = mdd = conf_sum = 0.0
    n_win = conf_n = 0
    pnl: list[float] = []
    wr: list[float] = []
    pf: list[float | None] = []
    trd: list[int] = []
    conf: list[float | None] = []
    dd: list[float] = []
    for idx, t in enumerate(chrono, 1):
        p = t.pnl
        if p is None:
            continue
        cum += p
        if p > 0:
            gross_win += p
            n_win += 1
        elif p < 0:
            gross_loss += -p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        c = (t.context or {}).get("confidence")
        if c is not None:
            conf_sum += float(c)
            conf_n += 1
        pnl.append(round(cum, 2))
        wr.append(round(n_win / idx, 4))
        pf.append(round(gross_win / gross_loss, 3) if gross_loss else None)
        trd.append(idx)
        conf.append(round(conf_sum / conf_n, 4) if conf_n else None)
        dd.append(round(mdd, 2))
    return {"pnl": pnl, "win_rate": wr, "profit_factor": pf,
            "trades": trd, "confidence": conf, "drawdown": dd}


def _matrix(trades: list[Trade]) -> dict:
    """Strategy x instrument P&L cross-tab, rows/cols sorted by total edge."""
    cells: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    inst_tot: dict[str, float] = defaultdict(float)
    strat_tot: dict[str, float] = defaultdict(float)
    for t in trades:
        p = t.pnl
        if p is None:
            continue
        i, s = t.symbol or "?", t.strategy or "?"
        cells[i][s] += p
        inst_tot[i] += p
        strat_tot[s] += p
    insts = sorted(inst_tot, key=lambda k: inst_tot[k], reverse=True)
    strats = sorted(strat_tot, key=lambda k: strat_tot[k], reverse=True)
    grid = [[round(cells[i][s], 2) if s in cells[i] else None for s in strats]
            for i in insts]
    return {
        "instruments": insts,
        "strategies": strats,
        "cells": grid,
        "inst_totals": [round(inst_tot[i], 2) for i in insts],
        "strat_totals": [round(strat_tot[s], 2) for s in strats],
    }


def _combos(trades: list[Trade]) -> list[dict]:
    """Every instrument x strategy pairing, sorted by P&L descending."""
    groups: dict[tuple[str, str], list[Trade]] = defaultdict(list)
    for t in trades:
        if t.pnl is None:
            continue
        groups[(t.symbol or "?", t.strategy or "?")].append(t)
    rows: list[dict] = []
    for (i, s), ts in groups.items():
        m = evaluate(ts)
        rows.append({"name": f"{i} × {s}", "instrument": i, "strategy": s,
                     "trades": m.n_trades, "pnl": round(m.total_pnl, 2),
                     "win_rate": round(m.win_rate, 3)})
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


def window(trades: list[Trade], label: str, start: datetime, end: datetime) -> dict:
    """Full performance breakdown for one time window."""
    graded = [t for t in trades if t.pnl is not None]
    chrono = sorted(
        graded,
        key=lambda t: _closed_at(t) or datetime.min.replace(tzinfo=timezone.utc),
    )
    m = evaluate(graded)
    wins = [t.pnl for t in graded if t.pnl > 0]
    losses = [t.pnl for t in graded if t.pnl < 0]
    real = [t for t in graded if (t.mode or "real") == "real"]
    shadow = [t for t in graded if t.mode == "shadow"]
    real_pnl = round(sum(t.pnl for t in real if t.pnl is not None), 2)
    shadow_pnl = round(sum(t.pnl for t in shadow if t.pnl is not None), 2)
    pf = m.profit_factor

    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss else None
    kelly = (m.win_rate - (1 - m.win_rate) / payoff) if payoff else None
    recovery = (m.total_pnl / m.max_drawdown) if m.max_drawdown else None
    pnls = [t.pnl for t in graded if t.pnl is not None]
    max_w, max_l, cur_streak = _streaks([t.pnl for t in chrono if t.pnl is not None])

    return {
        "label": label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "trades": m.n_trades,
        "win_rate": round(m.win_rate, 3),
        "wins": len(wins),
        "losses": len(losses),
        "pnl": round(m.total_pnl, 2),
        "real_pnl": real_pnl,
        "shadow_pnl": shadow_pnl,
        "expectancy": round(m.expectancy, 2),
        "profit_factor": (None if pf == float("inf") else round(pf, 2)),
        "sharpe": round(m.sharpe, 3),
        "max_drawdown": round(m.max_drawdown, 2),
        "gross_win": round(sum(wins), 2),
        "gross_loss": round(sum(losses), 2),
        "avg_confidence": (round(c, 3) if (c := _avg_confidence(graded)) is not None
                           else None),
        "avg_slippage_bps": (round(s, 2) if (s := _avg_slippage(real)) is not None
                             else None),
        # extended, institution-standard risk/return metrics
        "metrics": {
            "sortino": round(_sortino(_returns(graded)), 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "payoff_ratio": (round(payoff, 2) if payoff is not None else None),
            "kelly": (round(kelly, 3) if kelly is not None else None),
            "recovery_factor": (round(recovery, 2) if recovery is not None else None),
            "std_dev": round(_std(pnls), 2),
            "max_win_streak": max_w,
            "max_loss_streak": max_l,
            "current_streak": cur_streak,
            "avg_hold_min": (round(h, 1) if (h := _hold_minutes(graded)) is not None
                             else None),
        },
        "signal_dist": {
            "real": {"trades": len(real), "pnl": real_pnl},
            "shadow": {"trades": len(shadow), "pnl": shadow_pnl},
        },
        "equity_curve": _equity_curve(graded),
        "sparks": _sparks(chrono),
        "distribution": _distribution(pnls),
        "matrix": _matrix(graded),
        "combos": _combos(graded),
        "best_trade": _extreme_trade(graded, best=True),
        "worst_trade": _extreme_trade(graded, best=False),
        "by_strategy": _grouped(graded, lambda t: t.strategy),
        "by_instrument": _grouped(graded, lambda t: t.symbol),
        "by_side": {
            "buy": _row("buy", [t for t in graded if t.side == Side.BUY]),
            "sell": _row("sell", [t for t in graded if t.side == Side.SELL]),
        },
    }


def _delta_block(cur: list[Trade], prev: list[Trade]) -> dict:
    """Headline metric deltas: current window minus the immediately prior window."""
    c, p = evaluate(cur), evaluate(prev)

    def pf_delta() -> float | None:
        if p.profit_factor in (0.0, float("inf")) or c.profit_factor == float("inf"):
            return None
        return round(c.profit_factor - p.profit_factor, 2)

    cc, pc = _avg_confidence(cur), _avg_confidence(prev)
    return {
        "prev_trades": p.n_trades,
        "pnl": round(c.total_pnl - p.total_pnl, 2),
        "win_rate": round(c.win_rate - p.win_rate, 4),
        "profit_factor": pf_delta(),
        "trades": c.n_trades - p.n_trades,
        "avg_confidence": (round(cc - pc, 4) if cc is not None and pc is not None
                           else None),
        "max_drawdown": round(c.max_drawdown - p.max_drawdown, 2),
    }


def _summary_text(daily: dict, weekly: dict) -> str:
    """Deterministic narrative built from the computed figures."""
    if daily["trades"] == 0:
        base = "No trades closed today."
    else:
        verdict = ("a profitable day" if daily["pnl"] > 0
                   else "a losing day" if daily["pnl"] < 0 else "a flat day")
        base = (f"{verdict.capitalize()}: {daily['trades']} trades closed, "
                f"{daily['win_rate'] * 100:.0f}% won, "
                f"net {daily['pnl']:+,.2f}.")
        strat = daily["by_strategy"]
        if strat:
            top = strat[0]
            base += f" Best strategy {top['name']} ({top['pnl']:+,.2f})"
            if len(strat) > 1 and strat[-1]["pnl"] < 0:
                w = strat[-1]
                base += f", worst {w['name']} ({w['pnl']:+,.2f})"
            base += "."
        inst = daily["by_instrument"]
        if inst and inst[0]["pnl"] > 0:
            base += f" Top instrument {inst[0]['name']} ({inst[0]['pnl']:+,.2f})."
    if weekly["trades"]:
        base += (f" Over 7 days: {weekly['trades']} trades, "
                 f"{weekly['win_rate'] * 100:.0f}% won, net {weekly['pnl']:+,.2f}, "
                 f"expectancy {weekly['expectancy']:+.2f}/trade.")
    return base


def build(journal, now: datetime | None = None, *, limit: int = 5000) -> dict:
    """Assemble the full daily + weekly report from the journal."""
    now = now or datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_day_start = day_start - timedelta(days=1)
    week_start = now - timedelta(days=7)
    prev_week_start = week_start - timedelta(days=7)

    # Real performance uses completed journal round-trips. Shadow execution is
    # virtual and its ledger can retain provisional P/L rows without a
    # ``closed_at`` marker; include those rows in the report so shadow activity
    # remains visible, while the Trades page's Closed endpoint stays strict.
    closed = journal.closed(limit=limit)
    shadow_observed = [
        t for t in journal.recent(limit=limit)
        if t.mode == "shadow" and t.pnl is not None and t.closed_at is None
    ]
    graded = [t for t in closed if t.pnl is not None and t.mode != "learning"] + shadow_observed

    def between(lo: datetime, hi: datetime) -> list[Trade]:
        return [t for t in graded if (c := _closed_at(t)) and lo <= c < hi]

    day_trades = [t for t in graded if (c := _closed_at(t)) and c >= day_start]
    week_trades = [t for t in graded if (c := _closed_at(t)) and c >= week_start]

    daily = window(day_trades, "Today (UTC)", day_start, now)
    weekly = window(week_trades, "Last 7 days", week_start, now)
    return {
        "generated_at": now.isoformat(),
        "daily": daily,
        "weekly": weekly,
        "deltas": {
            "daily": _delta_block(day_trades, between(prev_day_start, day_start)),
            "weekly": _delta_block(week_trades, between(prev_week_start, week_start)),
        },
        "summary": _summary_text(daily, weekly),
    }


def format_telegram(report: dict, *, equity: float | None = None,
                    verdict: str | None = None) -> str:
    """Render the report as a compact Telegram/alert message."""
    d, w = report["daily"], report["weekly"]

    def pf(v):
        return "∞" if v is None else f"{v:.2f}"

    lines = ["📊 Daily report"]
    if equity is not None:
        lines.append(f"Equity {equity:,.2f}")
    lines.append(report["summary"])
    lines.append("")
    lines.append(
        f"Today: {d['trades']} trades · win {d['win_rate'] * 100:.0f}% · "
        f"P&L {d['pnl']:+,.2f} · PF {pf(d['profit_factor'])} · "
        f"exp {d['expectancy']:+.2f}")
    if d.get("avg_confidence") is not None:
        lines.append(f"  avg confidence {d['avg_confidence'] * 100:.0f}%"
                     + (f" · avg slippage {d['avg_slippage_bps']:+.1f}bps"
                        if d.get("avg_slippage_bps") is not None else ""))
    if d["best_trade"]:
        b = d["best_trade"]
        lines.append(f"  best {b['symbol']}/{b['strategy']} {b['pnl']:+,.2f}")
    if d["worst_trade"] and d["worst_trade"]["pnl"] < 0:
        x = d["worst_trade"]
        lines.append(f"  worst {x['symbol']}/{x['strategy']} {x['pnl']:+,.2f}")

    top_s = ", ".join(f"{r['name']} {r['pnl']:+.0f}" for r in d["by_strategy"][:3])
    if top_s:
        lines.append(f"Strategies: {top_s}")
    top_i = ", ".join(f"{r['name']} {r['pnl']:+.0f}" for r in d["by_instrument"][:3])
    if top_i:
        lines.append(f"Instruments: {top_i}")

    lines.append("")
    lines.append(
        f"7d: {w['trades']} trades · win {w['win_rate'] * 100:.0f}% · "
        f"P&L {w['pnl']:+,.2f} · PF {pf(w['profit_factor'])} · "
        f"sharpe {w['sharpe']:+.2f}")
    if verdict:
        lines.append(f"Book verdict: {verdict}")
    return "\n".join(lines)
