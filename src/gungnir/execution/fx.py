"""Quote-currency → account-currency PnL conversion.

``(exit − entry) × volume`` yields PnL in the instrument's QUOTE currency:
USD for EURUSD/US500/AAPL, but JPY for USDJPY, GBP for EURGBP. Before this,
those raw numbers flowed straight into equity, RL rewards, the cooldown and
the allocator — a ¥5,000 USDJPY win counted ~100× a $50 EURUSD win.

Conversion uses the live marks the brokers already hold: for quote currency Q
and account currency A, prefer the direct pair QA (multiply), else AQ
(divide). When neither pair is marked (Q's pair isn't in the universe), the
amount passes through unchanged with a once-per-symbol warning — wrong but
visible, exactly as before, rather than silently zeroed.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

# ISO currency codes used to recognize 6-letter FX pairs.
_CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "MXN", "TRY",
        "PLN", "NOK", "SEK", "DKK", "ZAR", "CNH", "SGD", "HKD"}

_warned: set[str] = set()


def quote_currency(symbol: str) -> str:
    """The currency a symbol's PnL is denominated in (USD for non-FX epics)."""
    s = symbol.upper()
    if len(s) == 6 and s[:3] in _CCY and s[3:] in _CCY:
        return s[3:]
    return "USD"


def to_account_ccy(symbol: str, amount: float,
                   price_lookup: Callable[[str], float | None],
                   account_ccy: str = "USD") -> float:
    """Convert a quote-currency amount into the account currency.

    ``price_lookup`` maps an FX pair name to its latest mark (or None) — the
    brokers pass their own ``_last_price.get``.
    """
    quote = quote_currency(symbol)
    if quote == account_ccy or amount == 0.0:
        return amount
    direct = price_lookup(f"{quote}{account_ccy}")     # e.g. GBPUSD
    if direct:
        return amount * direct
    inverse = price_lookup(f"{account_ccy}{quote}")    # e.g. USDJPY
    if inverse:
        return amount / inverse
    if symbol not in _warned:
        _warned.add(symbol)
        log.warning(
            "No %s→%s rate marked for %s PnL conversion (add %s%s or %s%s to "
            "the universe); recording unconverted quote-currency PnL.",
            quote, account_ccy, symbol, quote, account_ccy, account_ccy, quote)
    return amount
