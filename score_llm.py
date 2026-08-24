"""v2 item scoring via Gemini API - direction/magnitude/level/driver/etc.

Supersedes summarize.py's single -5..+5 sentiment_score float (kept in
place, unused, for rollback). Results go into the versioned `scores` table
(store.py) under rubric.RUBRIC_VERSION, so this rubric's output never
overwrites or mixes with an older rubric's rows for the same item - see
migrate_scores.py for how the old scores were preserved as rubric_version
'v1'.

Input is always store.scoring_input(row) - caption plus any extracted PDF
text (extract.py) - never the caption alone, or a caption-less PDF post is
invisible to scoring.

Items are still batched one day at a time in a single API call (cheap, and
rubric.ANCHOR_EXAMPLES in the system instruction is what keeps calibration
stable across separate day-batches - the day boundary itself carries no
special meaning to the model). But failure is handled per item, not per
batch: if the whole call errors, or the whole response fails schema
validation, we still write one row per item - salvaging whatever
individual items *do* parse out of a malformed response, and recording
raw_response with all other fields NULL for the rest. Nothing is silently
dropped, and nothing is left un-rowed (which would make it look
"unprocessed" and get retried forever without --force).

This script does NOT compute any average/index - that would be asking the
LLM (or this script) to aggregate, which belongs in dashboard.py/SQL only.

Usage
-----
    python score_llm.py                          # all unscored-under-this-rubric items
    python score_llm.py --date 2026-08-20
    python score_llm.py --since 2026-08-01
    python score_llm.py --channel "신한 리서치"
    python score_llm.py --force                    # re-score even if already scored under this rubric
    python score_llm.py --rubric-version v2b        # score under an experimental label
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import rubric
from store import get_connection, scoring_input, upsert_score

DEFAULT_MODEL = "gemini-flash-lite-latest"


class ItemAnalysis(BaseModel):
    item_id: str  # "{chat_id}_{message_id}"
    # Gemini's structured-output `enum` only supports STRING type fields, so
    # direction/magnitude can't be declared as an int Literal (it fails schema
    # conversion) - plain int, range enforced in score_day() below instead.
    direction: Optional[int] = None
    magnitude: Optional[int] = None
    confidence: Optional[float] = None
    level: Optional[Literal["macro", "market", "sector", "stock"]] = None
    sector_code: Optional[str] = None
    ticker: Optional[str] = None
    driver: Optional[
        Literal[
            "monetary_policy", "earnings", "flows", "geopolitics", "fx",
            "regulation", "valuation", "supply_chain", "commodity", "other",
        ]
    ] = None
    novelty: Optional[Literal["new", "recap", "repost"]] = None
    horizon: Optional[Literal["intraday", "short", "medium"]] = None
    summary: Optional[str] = None


class BatchAnalysis(BaseModel):
    items: list[ItemAnalysis]


def score_day(client: genai.Client, model: str, rows: list[tuple[str, str, str]]) -> list[dict]:
    """rows: list of (item_id, chat_name, text). Always returns exactly one
    result dict per input row - see module docstring for the partial-failure
    handling this guarantees."""
    expected_ids = [item_id for item_id, _, _ in rows]
    listing = "\n\n".join(f"[{item_id}] ({chat})\n{text[:4000]}" for item_id, chat, text in rows)
    prompt = (
        f"아래 {len(rows)}개 항목을 각각 분석하세요. 모든 item_id에 대해 결과를 반환하되, "
        f"판단 불가 항목은 direction을 null로 반환하세요 (생략 금지).\n\n{listing}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=rubric.SYSTEM_INSTRUCTION,
                temperature=0,
                response_mime_type="application/json",
                response_schema=BatchAnalysis,
            ),
        )
    except Exception as e:
        # The whole call failed - every item in this batch gets an error row
        # rather than silently vanishing.
        err = f"REQUEST_ERROR: {e}"
        return [{"item_id": item_id, "raw_response": err} for item_id in expected_ids]

    raw_text = response.text or ""

    parsed_items: dict[str, ItemAnalysis] = {}
    try:
        parsed_items = {it.item_id: it for it in response.parsed.items}
    except Exception:
        # Whole-response schema validation failed - salvage whatever
        # individual item objects still validate on their own.
        try:
            raw_json = json.loads(raw_text)
            for entry in raw_json.get("items", []):
                try:
                    item = ItemAnalysis.model_validate(entry)
                    parsed_items[item.item_id] = item
                except Exception:
                    continue  # this one item is unrecoverable; falls through below
        except Exception:
            pass  # raw_text isn't valid JSON at all - nothing to salvage

    results = []
    for item_id in expected_ids:
        item = parsed_items.get(item_id)
        if item is None:
            results.append({"item_id": item_id, "raw_response": raw_text})
            continue
        sector_code = item.sector_code if item.sector_code in rubric.SECTOR_CODES else None
        direction = item.direction
        if direction is not None:
            direction = max(-1, min(1, round(direction)))
        magnitude = item.magnitude
        if magnitude is not None:
            magnitude = max(0, min(3, round(magnitude)))
        results.append({
            "item_id": item_id, "direction": direction, "magnitude": magnitude,
            "confidence": item.confidence, "level": item.level, "sector_code": sector_code,
            "ticker": item.ticker, "driver": item.driver, "novelty": item.novelty,
            "horizon": item.horizon, "summary": item.summary, "raw_response": raw_text,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Only this date (YYYY-MM-DD)")
    parser.add_argument("--since", help="Only dates on/after this (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Max items to process")
    parser.add_argument("--force", action="store_true", help="Re-score items already scored under this rubric_version")
    parser.add_argument("--rubric-version", default=rubric.RUBRIC_VERSION)
    parser.add_argument("--channel", help="Only this chat_name")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Add it to .env (see .env.example).")

    client = genai.Client(api_key=api_key)
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    query = """
        SELECT i.chat_id, i.message_id, i.chat_name, i.text, i.extracted_text,
               substr(i.date, 1, 10) AS d
        FROM items i
        WHERE (i.text IS NOT NULL OR i.extracted_text IS NOT NULL)
    """
    params: list = []
    if not args.force:
        query += """
            AND NOT EXISTS (
                SELECT 1 FROM scores s
                WHERE s.chat_id = i.chat_id AND s.message_id = i.message_id
                  AND s.rubric_version = ?
            )
        """
        params.append(args.rubric_version)
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

    by_date: dict[str, list] = defaultdict(list)
    for row in rows:
        item_id = f"{row['chat_id']}_{row['message_id']}"
        by_date[row["d"]].append((item_id, row["chat_name"], scoring_input(row), row["chat_id"], row["message_id"]))

    total = 0
    for date in sorted(by_date):
        day_rows = by_date[date]
        call_rows = [(item_id, chat, text) for item_id, chat, text, _, _ in day_rows]
        ids_map = {item_id: (chat_id, message_id) for item_id, _, _, chat_id, message_id in day_rows}

        results = score_day(client, args.model, call_rows)
        for r in results:
            chat_id, message_id = ids_map[r["item_id"]]
            upsert_score(
                conn, chat_id, message_id, args.rubric_version, args.model,
                direction=r.get("direction"), magnitude=r.get("magnitude"),
                confidence=r.get("confidence"), level=r.get("level"),
                sector_code=r.get("sector_code"), ticker=r.get("ticker"),
                driver=r.get("driver"), novelty=r.get("novelty"), horizon=r.get("horizon"),
                summary=r.get("summary"), raw_response=r.get("raw_response"),
            )
            total += 1
        print(f"[{date}] scored {len(results)} item(s)")
        time.sleep(1)  # be polite to rate limits

    print(f"Total: {total} item(s) scored under rubric_version={args.rubric_version!r}")


if __name__ == "__main__":
    main()
