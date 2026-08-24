"""Market proxy built from a companion KRX price database.

**Limitation you should know before trusting anything downstream**: this
project has no market-cap or shares-outstanding data of its own.
prices.db's schema (prices(date, ticker, close), tickers(ticker, name))
only has close prices, so the originally-wanted "시총 상위 200종목"
(top-200-by-market-cap) filter cannot actually be computed here - market
cap needs shares outstanding, which isn't in this database and isn't
derivable from price alone. What's implemented instead is an
equal-weighted return across tickers with sufficiently complete price
history (>= MIN_COVERAGE of trading days in range), which excludes
thinly-traded/delisted names that would otherwise swing the average on a
handful of stale prints. That is a coverage/liquidity filter, not a
market-cap filter - small/illiquid names are almost certainly
over-represented relative to their real index weight. Treat
`market_return` as a rough regime proxy, not KOSPI. If a real market-cap
series becomes available later, swap it in here; validate.py doesn't care
how market_proxy() computed its numbers.

Also derived here:
  - cross-sectional volatility: the daily cross-sectional stdev of
    individual ticker returns (how dispersed stocks were that day, not
    how much the average moved)
  - a trading-day calendar (the actual dates on which at least one ticker
    printed a price - not a generic business-day calendar, which would
    wrongly include KRX holidays), so a weekend/holiday post's date can be
    rolled forward to the next real trading day instead of being dropped
    or matched to a closed market

Usage
-----
    python market.py                                   # summary stats to stdout
    python market.py --prices-db ../pairs-trading-krx/prices.db
    python market.py --start 2026-07-01 --min-coverage 0.9
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_PRICES_DB = Path(__file__).parent / ".." / "krx-pairs" / "prices.db"
MIN_COVERAGE = 0.95  # a ticker needs a price on >=95% of trading days in range to count


def load_prices(db_path: Path | str, start: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = "SELECT date, ticker, close FROM prices"
    params: tuple = ()
    if start:
        query += " WHERE date >= ?"
        params = (start,)
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    conn.close()
    return df


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Distinct dates on which at least one ticker actually traded."""
    return pd.DatetimeIndex(sorted(prices["date"].unique()))


def next_trading_day(calendar: pd.DatetimeIndex, date) -> pd.Timestamp | None:
    """Roll a (possibly non-trading) date forward to the next date the
    market was open - for mapping a weekend/holiday post to the first
    session the market could actually react in. Returns None if `date` is
    after the calendar's last session."""
    date = pd.Timestamp(date)
    idx = calendar.searchsorted(date)
    if idx >= len(calendar):
        return None
    return calendar[idx]


def market_proxy(prices: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Equal-weighted return proxy + cross-sectional volatility, one row per
    trading day. See the module docstring for why this isn't a real index."""
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    coverage = wide.notna().mean()
    eligible = coverage[coverage >= min_coverage].index
    wide = wide[eligible]

    returns = wide.pct_change(fill_method=None)
    out = pd.DataFrame({
        "market_return": returns.mean(axis=1),
        "cross_sectional_vol": returns.std(axis=1),
        "n_tickers": returns.notna().sum(axis=1),
    })
    return out.dropna(subset=["market_return"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-db", default=str(DEFAULT_PRICES_DB), help="Path to prices.db")
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = parser.parse_args()

    prices = load_prices(args.prices_db, args.start)
    if prices.empty:
        raise SystemExit(f"No rows in {args.prices_db}" + (f" on/after {args.start}" if args.start else ""))

    proxy = market_proxy(prices, args.min_coverage)
    cal = trading_calendar(prices)

    print(f"{len(proxy)} trading day(s) of proxy data; "
          f"{proxy['n_tickers'].iloc[-1]} eligible ticker(s) as of last day (coverage>={args.min_coverage:.0%})")
    print(proxy.tail(10).to_string(index=False))
    print(f"\nTrading calendar: {cal.min().date()} .. {cal.max().date()} ({len(cal)} session(s))")


if __name__ == "__main__":
    main()
