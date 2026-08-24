"""Calibration check: how well does an LLM rubric_version agree with a
human-labeled gold set?

If calibration/gold.csv doesn't exist yet, running this generates a
template instead of evaluating anything: a random sample of already-scored
items with direction/magnitude/level left blank for a human to fill in.
Re-run once it's filled in (partially-filled rows are fine - unfilled cells
are just skipped per metric) to get actual agreement numbers.

Usage
-----
    python calibration.py                       # generate gold.csv template if missing, else evaluate
    python calibration.py --rubric-version v2
    python calibration.py --rubric-version v1 --compare v2   # side by side
    python calibration.py --sample-size 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from store import get_connection, latest_scores

GOLD_PATH = Path(__file__).parent / "calibration" / "gold.csv"
GOLD_COLUMNS = ["chat_id", "message_id", "direction", "magnitude", "level"]


def make_template(conn, sample_size: int) -> None:
    scored = pd.read_sql_query("SELECT DISTINCT chat_id, message_id FROM scores", conn)
    if scored.empty:
        raise SystemExit("No scored items yet - run score_llm.py first.")

    n = min(sample_size, len(scored))
    sample = scored.sample(n=n, random_state=42).sort_values(["chat_id", "message_id"])

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GOLD_COLUMNS)
        for _, row in sample.iterrows():
            writer.writerow([row["chat_id"], row["message_id"], "", "", ""])

    print(f"Wrote {GOLD_PATH} with {n} row(s).")
    print("Fill in direction (-1/0/1), magnitude (0-3), level (macro/market/sector/stock) by hand,")
    print("then re-run this script to see agreement metrics.")


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_PATH, dtype={"chat_id": "int64", "message_id": "int64"})
    df = df.dropna(subset=["direction"]).copy()
    if df.empty:
        return df
    df["direction"] = df["direction"].astype(int)
    if "magnitude" in df:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    return df


def evaluate(gold: pd.DataFrame, llm: pd.DataFrame, label: str) -> None:
    merged = gold.merge(llm, on=["chat_id", "message_id"], suffixes=("_gold", "_llm"))
    print(f"[{label}] n={len(merged)}")
    if merged.empty:
        print("  no overlap between the gold set and this rubric_version's scores.\n")
        return

    dir_pairs = merged.dropna(subset=["direction_gold", "direction_llm"])
    if not dir_pairs.empty:
        acc = (dir_pairs["direction_gold"] == dir_pairs["direction_llm"]).mean()
        print(f"  direction accuracy: {acc:.1%} (n={len(dir_pairs)})")

    mag_pairs = merged.dropna(subset=["magnitude_gold", "magnitude_llm"])
    if not mag_pairs.empty:
        mae = (mag_pairs["magnitude_gold"] - mag_pairs["magnitude_llm"]).abs().mean()
        print(f"  magnitude MAE: {mae:.2f} (n={len(mag_pairs)})")

    lvl_pairs = merged.dropna(subset=["level_gold", "level_llm"])
    if not lvl_pairs.empty:
        print(f"  level confusion matrix (rows=gold, cols=llm, n={len(lvl_pairs)}):")
        cm = pd.crosstab(lvl_pairs["level_gold"], lvl_pairs["level_llm"])
        print(cm.to_string().replace("\n", "\n  "))

    score_pairs = merged.dropna(subset=["direction_gold", "magnitude_gold", "direction_llm", "magnitude_llm"])
    if len(score_pairs) >= 3:
        gold_score = score_pairs["direction_gold"] * score_pairs["magnitude_gold"]
        llm_score = score_pairs["direction_llm"] * score_pairs["magnitude_llm"]
        rho = gold_score.corr(llm_score, method="spearman")
        print(f"  spearman(gold_score, llm_score): {rho:.2f} (n={len(score_pairs)})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric-version", help="Version to evaluate; default: each item's most recent score")
    parser.add_argument("--compare", help="A second rubric_version to evaluate side by side")
    parser.add_argument("--sample-size", type=int, default=60, help="Template size if gold.csv doesn't exist yet")
    args = parser.parse_args()

    conn = get_connection()

    if not GOLD_PATH.exists():
        make_template(conn, args.sample_size)
        return

    gold = load_gold()
    if gold.empty:
        print(f"{GOLD_PATH} exists but has no filled-in rows yet (direction column empty).")
        return

    versions = [v for v in (args.rubric_version, args.compare) if v] or [None]
    for version in versions:
        llm = latest_scores(conn, rubric_version=version)
        cols = [c for c in ("chat_id", "message_id", "direction", "magnitude", "level") if c in llm.columns]
        evaluate(gold, llm[cols] if cols else llm, version or "latest (per item)")


if __name__ == "__main__":
    main()
