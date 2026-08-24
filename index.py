"""Aggregate item-level scores into a daily sentiment index.

Replaces the old "just average sentiment_score" approach with three
methods side by side, because a plain mean has three known problems in
this dataset (see CLAUDE.md issues #2-#4):

1. Shrinkage - a day with 1-3 items swings the mean as hard as a day with
   80. s_hat = n/(n+k) * mean_score pulls low-n days toward 0 instead of
   letting them dominate the chart.
2. Channel-demeaned - one channel supplies roughly half of all scored
   items, so a plain per-day mean is really "mostly that channel's
   opinion, plus a little of everyone else's". Subtracting each channel's
   own all-time mean first removes that channel's fixed effect before
   the daily aggregate is taken.
3. Breadth - (positive count - negative count) / total count, -1..+1.
   Doesn't care about magnitude at all, so one extreme item can't move it
   much; a useful sanity check against the other two.

None of this is computed by an LLM - direction/magnitude/confidence come
from score_llm.py's structured output, but every aggregate here is plain
pandas/SQL arithmetic (see CLAUDE.md: "LLM에게 숫자 계산/집계를 시키지 말 것").

Default filters (see --include-recap to see why they matter):
  - novelty == 'recap' excluded: a same-day market recap post ("코스피
    마감 +2%") restates that day's return, so including it makes the
    index correlate with the market almost by construction rather than by
    saying anything ahead of it.
  - confidence < 0.4 excluded: low-confidence judgments are noise.
  - direction IS NULL excluded: these are "couldn't judge", not "neutral".

level is never blended with an implicit weight - use --weights to opt into
an explicit macro=..,market=..,sector=..,stock=.. composite; each level's
own series is always available unweighted via daily_index(level=...).

Usage
-----
    python index.py                                  # all levels, all methods, printed
    python index.py --level stock --method shrinkage
    python index.py --weights macro=0.4,market=0.3,sector=0.2,stock=0.1
    python index.py --include-recap                  # comparison run
    python index.py --out index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from store import get_connection

LEVELS = ("macro", "market", "sector", "stock")
METHODS = ("shrinkage", "channel_demeaned", "breadth")
DEFAULT_SHRINK_K = 5
MIN_CONFIDENCE = 0.4


def _scored_items(conn, level: str | None, include_recap: bool) -> pd.DataFrame:
    """Raw (date, chat_id, level, score) rows after the default filters.

    Reads the *latest* score per item (whatever rubric_version last touched
    it) via the same join pattern as store.latest_scores(), then joins back
    to items for the date, chat_id, and dedup info.

    Filters to is_cluster_head=1 by default (or NULL, for items scored
    before dedupe.py ever ran on them / a DB where it's never been run -
    that's "not deduplicated yet", not "not a duplicate") - see dedupe.py:
    a story six channels all mention would otherwise count six times in
    every daily aggregate below, dominating the mean.
    """
    query = """
        SELECT i.chat_id AS chat_id, substr(i.date, 1, 10) AS date,
               s.level, s.direction, s.magnitude, s.confidence, s.novelty,
               i.mention_channels
        FROM scores s
        JOIN (
            SELECT chat_id, message_id, MAX(scored_at) AS max_scored_at
            FROM scores GROUP BY chat_id, message_id
        ) latest
          ON s.chat_id = latest.chat_id AND s.message_id = latest.message_id
         AND s.scored_at = latest.max_scored_at
        JOIN items i ON i.chat_id = s.chat_id AND i.message_id = s.message_id
        WHERE s.direction IS NOT NULL
          AND (s.confidence IS NULL OR s.confidence >= ?)
          AND (i.is_cluster_head = 1 OR i.is_cluster_head IS NULL)
    """
    params: list = [MIN_CONFIDENCE]
    if not include_recap:
        query += " AND (s.novelty IS NULL OR s.novelty != 'recap')"
    if level:
        query += " AND s.level = ?"
        params.append(level)

    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["score"] = df["direction"] * df["magnitude"]
    return df


def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    wsum = weights.sum()
    return (values * weights).sum() / wsum if wsum else values.mean()


def _consensus_weight(df: pd.DataFrame, weight_by_consensus: bool) -> pd.Series:
    """1 per item normally; mention_channels per item when
    --weight-by-consensus opts in (a story more channels independently
    covered counts for more, not less - see dedupe.py's module docstring
    on why duplicates are a feature, not noise to discard)."""
    if not weight_by_consensus:
        return pd.Series(1.0, index=df.index)
    return df["mention_channels"].fillna(1).astype(float)


def _shrinkage(df: pd.DataFrame, k: float, weight_by_consensus: bool) -> pd.DataFrame:
    w = _consensus_weight(df, weight_by_consensus)
    d = df.assign(_w=w)
    g = d.groupby("date")
    mean_score = g.apply(lambda x: _weighted_avg(x["score"], x["_w"]), include_groups=False)
    out = mean_score.rename("mean_score").to_frame()
    out["n_items"] = g.size()
    out["n_channels"] = g["chat_id"].nunique()
    out["index_value"] = out["n_items"] / (out["n_items"] + k) * out["mean_score"]
    return out[["index_value", "n_items", "n_channels"]]


def _channel_demeaned(df: pd.DataFrame, weight_by_consensus: bool) -> pd.DataFrame:
    channel_mean = df.groupby("chat_id")["score"].transform("mean")
    w = _consensus_weight(df, weight_by_consensus)
    demeaned = df.assign(demeaned_score=df["score"] - channel_mean, _w=w)
    g = demeaned.groupby("date")
    index_value = g.apply(lambda x: _weighted_avg(x["demeaned_score"], x["_w"]), include_groups=False)
    out = index_value.rename("index_value").to_frame()
    out["n_items"] = g.size()
    out["n_channels"] = g["chat_id"].nunique()
    return out[["index_value", "n_items", "n_channels"]]


def _breadth(df: pd.DataFrame, weight_by_consensus: bool = False) -> pd.DataFrame:
    w = _consensus_weight(df, weight_by_consensus)
    d = df.assign(_w=w)
    g = d.groupby("date")
    pos = g.apply(lambda x: x.loc[x["score"] > 0, "_w"].sum(), include_groups=False)
    neg = g.apply(lambda x: x.loc[x["score"] < 0, "_w"].sum(), include_groups=False)
    total_w = g["_w"].sum()
    out = pd.DataFrame({"index_value": (pos - neg) / total_w, "n_items": g.size()})
    out["n_channels"] = g["chat_id"].nunique()
    return out[["index_value", "n_items", "n_channels"]]


_METHOD_FUNCS = {
    "shrinkage": lambda df, k, wc: _shrinkage(df, k, wc),
    "channel_demeaned": lambda df, k, wc: _channel_demeaned(df, wc),
    "breadth": lambda df, k, wc: _breadth(df, wc),
}


def daily_index(
    conn,
    level: str | None = None,
    method: str = "shrinkage",
    shrink_k: float = DEFAULT_SHRINK_K,
    include_recap: bool = False,
    weight_by_consensus: bool = False,
) -> pd.DataFrame:
    """One row per date. Columns: date, index_value, n_items, n_channels, breadth.

    breadth is always included (even for method != 'breadth') as a cheap
    cross-check - if shrinkage/channel_demeaned and breadth disagree in
    sign, that's worth noticing before trusting either. n_items always
    counts deduplicated items (cluster heads), never raw per-channel posts.
    """
    assert method in METHODS, f"method must be one of {METHODS}, got {method!r}"
    df = _scored_items(conn, level, include_recap)
    if df.empty:
        return pd.DataFrame(columns=["date", "index_value", "n_items", "n_channels", "breadth"])

    result = _METHOD_FUNCS[method](df, shrink_k, weight_by_consensus)
    breadth = _breadth(df, weight_by_consensus)["index_value"].rename("breadth")
    result = result.join(breadth, how="left")
    return result.reset_index().sort_values("date")


def weighted_composite(
    conn,
    weights: dict[str, float],
    method: str = "shrinkage",
    shrink_k: float = DEFAULT_SHRINK_K,
    include_recap: bool = False,
    weight_by_consensus: bool = False,
) -> pd.DataFrame:
    """Explicit-weight blend across levels - never computed implicitly.
    weights must be a dict like {'macro': 0.4, 'market': 0.3, ...}; levels
    not present in `weights` are excluded, not defaulted to some weight.
    """
    unknown = set(weights) - set(LEVELS)
    if unknown:
        raise ValueError(f"unknown level(s) in weights: {unknown}")

    per_level = {}
    for level, w in weights.items():
        per_level[level] = daily_index(
            conn, level=level, method=method, shrink_k=shrink_k,
            include_recap=include_recap, weight_by_consensus=weight_by_consensus,
        )

    all_dates = sorted(set().union(*(df["date"] for df in per_level.values())))
    rows = []
    for date in all_dates:
        value = 0.0
        total_n = 0
        total_channels = set()
        weight_present = 0.0
        for level, w in weights.items():
            day_row = per_level[level][per_level[level]["date"] == date]
            if day_row.empty:
                continue
            value += w * day_row["index_value"].iloc[0]
            weight_present += w
            total_n += int(day_row["n_items"].iloc[0])
        if weight_present == 0:
            continue
        rows.append({"date": date, "index_value": value / weight_present, "n_items": total_n})

    return pd.DataFrame(rows)


def parse_weights(spec: str) -> dict[str, float]:
    weights = {}
    for part in spec.split(","):
        level, _, value = part.partition("=")
        weights[level.strip()] = float(value)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=LEVELS, help="Restrict to one level; default: all levels separately")
    parser.add_argument("--method", choices=METHODS, default="shrinkage")
    parser.add_argument("--shrink-k", type=float, default=DEFAULT_SHRINK_K)
    parser.add_argument("--include-recap", action="store_true", help="Include novelty='recap' items (comparison run)")
    parser.add_argument("--weights", help="Explicit composite, e.g. macro=0.4,market=0.3,sector=0.2,stock=0.1")
    parser.add_argument(
        "--weight-by-consensus", action="store_true",
        help="Weight each (deduplicated) story by mention_channels - see dedupe.py",
    )
    parser.add_argument("--out", help="Write CSV to this path instead of printing")
    args = parser.parse_args()

    conn = get_connection()

    if args.weights:
        df = weighted_composite(
            conn, parse_weights(args.weights), method=args.method,
            shrink_k=args.shrink_k, include_recap=args.include_recap,
            weight_by_consensus=args.weight_by_consensus,
        )
        label = f"weighted composite ({args.weights})"
        _emit(df, label, args.out)
        return

    levels = [args.level] if args.level else list(LEVELS)
    for level in levels:
        df = daily_index(
            conn, level=level, method=args.method, shrink_k=args.shrink_k,
            include_recap=args.include_recap, weight_by_consensus=args.weight_by_consensus,
        )
        out_path = None
        if args.out:
            p = Path(args.out)
            out_path = p if len(levels) == 1 else p.with_stem(f"{p.stem}_{level}")
        _emit(df, f"level={level} method={args.method}", out_path)


def _emit(df: pd.DataFrame, label: str, out_path: str | Path | None) -> None:
    if out_path:
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{label}] wrote {out_path} ({len(df)} row(s))")
    else:
        print(f"=== {label} ({len(df)} day(s)) ===")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
