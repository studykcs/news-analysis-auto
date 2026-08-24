"""Validate the sentiment index against realized KRX returns.

This is the layer CLAUDE.md calls out as completely missing: a sentiment
index is a list of numbers until it's checked against what the market
actually did. Three independent checks:

  contemporaneous - corr(index_t, market_return_t), recap included and
    excluded side by side. If excluding recap posts makes the correlation
    collapse, that means the index's apparent signal was mostly same-day
    market recaps restating that day's return (CLAUDE.md #7) - report that
    plainly, don't quietly keep the recap-included number.

  leadlag - corr(index_t, market_return_{t+k}) for k=-5..+5, plus an OLS
    regression return_{t+1} ~ a + b*index_t + c*return_t with Newey-West
    (HAC, maxlags=5) standard errors, so the coefficient on index_t is
    read after controlling for today's own return (does the index add
    anything return_t doesn't already tell you?).

  eventstudy - for scores with an explicit ticker, cumulative abnormal
    return (ticker return minus the market proxy, summed) from event day
    -5 to +10 trading days, split by direction=+1 vs -1, with sample
    sizes and a one-sample t-test per horizon per group.

IMPORTANT, and non-negotiable: results are reported exactly as computed,
including a null result ("no relationship found"). This script does not
search over --shrink-k, --level, confidence cutoffs, or lag windows for
something that looks significant - if you rerun with different parameters
after seeing a result you don't like, you are doing the thing CLAUDE.md
explicitly says not to do. Small groups (<30) are flagged, and leadlag
tests 11 lags at once - that multiple-comparisons exposure is stated in
the output, not left for the reader to notice on their own.

Usage
-----
    python validate.py contemporaneous
    python validate.py leadlag
    python validate.py eventstudy
    python validate.py all
    python validate.py all --prices-db ../pairs-trading-krx/prices.db --level market
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import market
from index import METHODS, daily_index
from market import load_prices, market_proxy, next_trading_day, trading_calendar
from store import get_connection

MIN_GROUP_N = 30
LEADLAG_RANGE = range(-5, 6)
EVENT_WINDOW = range(-5, 11)


def _index_series(conn, level, method, shrink_k, include_recap) -> pd.Series:
    df = daily_index(conn, level=level, method=method, shrink_k=shrink_k, include_recap=include_recap)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["index_value"]
    s.index = pd.to_datetime(s.index)
    return s


def _market_series(prices_db, start) -> tuple[pd.Series, pd.DataFrame]:
    prices = load_prices(prices_db, start)
    if prices.empty:
        raise SystemExit(f"No rows in {prices_db}")
    proxy = market_proxy(prices)
    s = proxy.set_index("date")["market_return"]
    return s, prices


def cmd_contemporaneous(conn, args) -> None:
    mkt, _ = _market_series(args.prices_db, args.start)
    print("=== Contemporaneous correlation: corr(index_t, market_return_t) ===\n")

    corrs = {}
    for include_recap in (False, True):
        idx = _index_series(conn, args.level, args.method, args.shrink_k, include_recap)
        merged = pd.DataFrame({"index": idx, "market": mkt}).dropna()
        label = "recap 포함" if include_recap else "recap 제외"
        if merged.empty:
            print(f"[{label}] n=0 - no overlapping dates")
            corrs[include_recap] = None
            continue
        corr = merged["index"].corr(merged["market"])
        corrs[include_recap] = corr
        n = len(merged)
        warn = "  (n<30, 해석 주의)" if n < MIN_GROUP_N else ""
        print(f"[{label}] n={n} corr={corr:+.3f}{warn}")

    if corrs.get(False) is not None and corrs.get(True) is not None:
        drop = abs(corrs[True]) - abs(corrs[False])
        print()
        if drop > 0.15:
            print(
                f"NOTE: recap 제외 시 상관이 {corrs[True]:+.3f} -> {corrs[False]:+.3f}로 크게 줄었습니다. "
                "이는 recap 포함 지수의 상관관계 대부분이 '그날 있었던 일을 그대로 재서술'하는 "
                "게시물에서 왔다는 뜻입니다 - 즉 그 상관은 지수의 선행성이 아니라 순환참조입니다."
            )
        else:
            print("NOTE: recap 포함/제외 상관이 비슷합니다 - recap이 상관관계를 주도하고 있지는 않아 보입니다.")


def cmd_leadlag(conn, args) -> None:
    mkt, _ = _market_series(args.prices_db, args.start)
    idx = _index_series(conn, args.level, args.method, args.shrink_k, args.include_recap)

    print("=== Lead-lag correlation: corr(index_t, market_return_t+k), k=-5..+5 ===")
    print(f"(recap {'포함' if args.include_recap else '제외'}; 11개 lag를 동시에 검정하는 다중비교임에 유의)\n")

    rows = []
    for k in LEADLAG_RANGE:
        shifted = mkt.shift(-k)
        merged = pd.DataFrame({"index": idx, "market": shifted}).dropna()
        n = len(merged)
        corr = merged["index"].corr(merged["market"]) if n >= 3 else float("nan")
        rows.append({"k": k, "corr": corr, "n": n})
    lag_df = pd.DataFrame(rows)
    print(lag_df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n--- Regression: return_t+1 ~ a + b*index_t + c*return_t (Newey-West HAC, maxlags=5) ---")
    df = pd.DataFrame({
        "return_t1": mkt.shift(-1),
        "index_t": idx,
        "return_t": mkt,
    }).dropna()
    n = len(df)
    if n < MIN_GROUP_N:
        print(f"n={n} < {MIN_GROUP_N} - 표본이 작아 회귀 결과를 신뢰하기 어렵습니다. 참고용으로만 출력합니다.")
    if n < 5:
        print("표본이 너무 적어 회귀를 실행하지 않습니다.")
        return

    X = sm.add_constant(df[["index_t", "return_t"]])
    y = df["return_t1"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print(f"n={n}")
    for name in ("const", "index_t", "return_t"):
        b = model.params[name]
        se = model.bse[name]
        t = model.tvalues[name]
        p = model.pvalues[name]
        print(f"  {name:<10} coef={b:+.5f}  NW_se={se:.5f}  t={t:+.2f}  p={p:.3f}")


def cmd_eventstudy(conn, args) -> None:
    mkt, prices = _market_series(args.prices_db, args.start)
    cal = trading_calendar(prices)
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = wide.pct_change(fill_method=None)

    scored = pd.read_sql_query(
        """
        SELECT s.chat_id, s.message_id, s.ticker, s.direction, i.date AS event_date
        FROM scores s
        JOIN items i ON i.chat_id = s.chat_id AND i.message_id = s.message_id
        WHERE s.ticker IS NOT NULL AND s.direction IS NOT NULL AND s.direction != 0
        """,
        conn,
    )
    scored["event_date"] = pd.to_datetime(scored["event_date"].str.slice(0, 10))

    print(f"=== Event study: ticker-tagged items, CAR (ticker - market) over t-5..t+10 ===")
    print(f"{len(scored)} ticker-tagged, direction != 0 item(s) found in scores\n")

    records = []
    skipped_no_price = 0
    for _, row in scored.iterrows():
        ticker = row["ticker"]
        if ticker not in returns.columns:
            skipped_no_price += 1
            continue
        t0 = next_trading_day(cal, row["event_date"])
        if t0 is None:
            continue
        t0_pos = cal.get_indexer([t0])[0]

        car = 0.0
        car_by_k = {}
        for k in EVENT_WINDOW:
            pos = t0_pos + k
            if 0 <= pos < len(cal):
                d = cal[pos]
                r = returns.loc[d, ticker] if d in returns.index else np.nan
                m = mkt.loc[d] if d in mkt.index else np.nan
                if pd.notna(r) and pd.notna(m):
                    car += (r - m)
            car_by_k[k] = car
        records.append({"ticker": ticker, "direction": row["direction"], **{f"k{k}": car_by_k[k] for k in EVENT_WINDOW}})

    if skipped_no_price:
        print(f"(참고: {skipped_no_price}건은 해당 ticker의 가격 데이터가 prices.db에 없어 제외됨)\n")

    ev = pd.DataFrame(records)
    if ev.empty:
        print("이벤트가 하나도 매칭되지 않았습니다 (ticker가 prices.db에 없거나 거래일 매핑 실패).")
        return

    horizons = [0, 1, 3, 5, 10]
    for direction, label in ((1, "direction=+1"), (-1, "direction=-1")):
        group = ev[ev["direction"] == direction]
        n = len(group)
        warn = f"  *** n={n} < {MIN_GROUP_N}, 결과 해석에 주의 ***" if n < MIN_GROUP_N else ""
        print(f"[{label}] n={n}{warn}")
        for k in horizons:
            col = f"k{k}"
            vals = group[col].dropna()
            if len(vals) < 2:
                print(f"  t+{k:>2}: n={len(vals)} (표본 부족)")
                continue
            mean_car = vals.mean()
            t_stat, p_val = stats.ttest_1samp(vals, 0.0)
            print(f"  t+{k:>2}: mean CAR={mean_car:+.4f}  n={len(vals)}  t={t_stat:+.2f}  p={p_val:.3f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["contemporaneous", "leadlag", "eventstudy", "all"])
    parser.add_argument("--prices-db", default=str(market.DEFAULT_PRICES_DB))
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--level", choices=("macro", "market", "sector", "stock"), default=None)
    parser.add_argument("--method", choices=METHODS, default="shrinkage")
    parser.add_argument("--shrink-k", type=float, default=5.0)
    parser.add_argument("--include-recap", action="store_true")
    args = parser.parse_args()

    conn = get_connection()

    checks = {
        "contemporaneous": cmd_contemporaneous,
        "leadlag": cmd_leadlag,
        "eventstudy": cmd_eventstudy,
    }
    to_run = checks.keys() if args.check == "all" else [args.check]
    for name in to_run:
        checks[name](conn, args)
        print()


if __name__ == "__main__":
    main()
