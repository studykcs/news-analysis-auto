"""Score items with KR-FinBERT (snunlp/KR-FinBert-SC) instead of Gemini.

Adds a second, independent rubric_version ('finbert-v1') to the scores
table alongside score_llm.py's Gemini-based 'v2' - see store.py's
SCORES_SCHEMA and rubric.py for why scores are versioned this way. This
does NOT replace score_llm.py (kept, untouched, still the daily default -
see run_pipeline.ps1). Run both and compare them:
    python calibration.py --rubric-version v2 --compare finbert-v1

KR-FinBERT is a plain 3-class classifier (positive/neutral/negative) with
a confidence score - not a structured-extraction model like Gemini, so it
cannot fill level/sector_code/ticker/driver/novelty/horizon. Those stay
NULL for every finbert-v1 row; only direction/magnitude/confidence are
populated from the model, plus a `summary` that's just the truncated
source text (FinBERT doesn't generate text, so there's nothing else to
put there). Practical consequence: finbert-v1 rows are invisible to
dashboard.py's level-based charts (#4 heatmap) and the event study (#6,
needs ticker) - they only contribute to index.py's level=None aggregate.

label -> direction: positive=+1, neutral=0, negative=-1
confidence -> magnitude: bucketed (see confidence_to_magnitude) since
FinBERT's softmax probability isn't itself a 0-3 integer. This is a
hand-picked heuristic, not a calibrated mapping - run calibration.py
before trusting it against a human's judgment.

Model input is store.scoring_input(row) (caption + extracted PDF text,
same as score_llm.py), truncated to the tokenizer's actual 512-token
limit via the pipeline's own tokenizer (truncation=True), not by slicing
the string - a character slice can still overflow the real token budget
for Korean text.

Usage
-----
    python score_finbert.py                       # all unscored-under-this-version items
    python score_finbert.py --date 2026-08-20
    python score_finbert.py --limit 100
    python score_finbert.py --force                 # re-score even if already scored under finbert-v1
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

from store import get_connection, scoring_input, upsert_score

RUBRIC_VERSION = "finbert-v1"
MODEL_NAME = "snunlp/KR-FinBert-SC"

LABEL_TO_DIRECTION = {"positive": 1, "neutral": 0, "negative": -1}


def confidence_to_magnitude(label: str, score: float) -> int:
    """Heuristic bucket, not a calibrated mapping - see module docstring."""
    if label == "neutral":
        return 0
    if score >= 0.9:
        return 3
    if score >= 0.75:
        return 2
    if score >= 0.5:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Only this date (YYYY-MM-DD)")
    parser.add_argument("--since", help="Only dates on/after this (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Max items to process")
    parser.add_argument("--force", action="store_true", help="Re-score items already scored under this rubric_version")
    parser.add_argument("--channel", help="Only this chat_name")
    args = parser.parse_args()

    from transformers import pipeline  # deferred: heavy import, only needed here

    print(f"Loading {MODEL_NAME} (first run downloads the model - this can take a while)...")
    classifier = pipeline("text-classification", model=MODEL_NAME)

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    query = """
        SELECT i.chat_id, i.message_id, i.text, i.extracted_text, substr(i.date, 1, 10) AS d
        FROM items i
        WHERE (i.text IS NOT NULL OR i.extracted_text IS NOT NULL)
    """
    params: list = []
    if not args.force:
        query += """
            AND NOT EXISTS (
                SELECT 1 FROM scores s
                WHERE s.chat_id = i.chat_id AND s.message_id = i.message_id AND s.rubric_version = ?
            )
        """
        params.append(RUBRIC_VERSION)
    if args.date:
        query += " AND substr(i.date, 1, 10) = ?"
        params.append(args.date)
    if args.since:
        query += " AND substr(i.date, 1, 10) >= ?"
        params.append(args.since)
    if args.channel:
        query += " AND i.chat_name = ?"
        params.append(args.channel)
    query += " ORDER BY i.date"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("Nothing to score.")
        return

    # Score in batched passes (pipeline(list, batch_size=...)) rather than one
    # classifier() call per item - far fewer Python/tokenizer round trips for
    # the same CPU work, which matters at 1000+ items. Chunked (not one giant
    # batch) so a failure only costs CHUNK_SIZE items' worth of retry, and so
    # progress/commits happen incrementally on a long run.
    CHUNK_SIZE = 200
    scorable = [(row, scoring_input(row)) for row in rows]
    scorable = [(row, text) for row, text in scorable if text]

    n = 0
    n_error = 0
    by_date: dict[str, int] = defaultdict(int)
    for start in range(0, len(scorable), CHUNK_SIZE):
        chunk = scorable[start : start + CHUNK_SIZE]
        texts = [text for _, text in chunk]
        try:
            results = classifier(texts, truncation=True, max_length=512, batch_size=16)
        except Exception as e:
            print(f"  chunk {start}-{start + len(chunk)} failed entirely: {e}")
            results = [None] * len(chunk)

        for (row, text), result in zip(chunk, results):
            if result is None:
                upsert_score(
                    conn, row["chat_id"], row["message_id"], RUBRIC_VERSION, MODEL_NAME,
                    raw_response="REQUEST_ERROR: batch scoring failed",
                )
                n_error += 1
                continue

            label = result["label"].lower()
            score = float(result["score"])
            direction = LABEL_TO_DIRECTION.get(label)
            magnitude = confidence_to_magnitude(label, score) if direction is not None else None

            upsert_score(
                conn, row["chat_id"], row["message_id"], RUBRIC_VERSION, MODEL_NAME,
                direction=direction, magnitude=magnitude, confidence=score,
                summary=text[:200], raw_response=str(result),
            )
            n += 1
            by_date[row["d"]] += 1

        print(f"  {min(start + CHUNK_SIZE, len(scorable))}/{len(scorable)} processed")

    for date in sorted(by_date):
        print(f"[{date}] scored {by_date[date]} item(s)")
    print(f"Total: {n} scored, {n_error} error(s), under rubric_version={RUBRIC_VERSION!r}")


if __name__ == "__main__":
    main()
