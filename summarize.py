"""Automate item summarization/scoring using the Gemini API.

Replaces the manual "ask an LLM to read a day's items in chat" workflow
with a scripted call to Gemini, using the same rubric shown in the
dashboard's "채점 기준" card. Processes one day's unscored items per API
call (all channels together), so a day with light volume is a single
request.

Usage
-----
    python summarize.py                       # all unscored days
    python summarize.py --date 2026-08-20
    python summarize.py --model gemini-3.7-flash
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from store import get_connection, update_summary

DEFAULT_MODEL = "gemini-flash-lite-latest"

RUBRIC = """당신은 한국 증권사 리서치/뉴스 메시지를 읽고 그날 시장 심리에 미치는 영향을 평가하는 애널리스트입니다.

각 항목마다 다음을 판단하세요.

sentiment_score (-5..+5 정수): 그 내용이 시장 심리에 얼마나 긍정적/부정적인 재료인지
  -5,-4 매우 부정 (급락/위기)      -3,-2 부정 (하락·우려 지배적)      -1 약한 부정/경계
  0 중립/정보성
  +1 약한 긍정   +2,+3 긍정 (실적 서프라이즈 등)   +4,+5 매우 긍정 (사상 최대 등)

scope: 이 뉴스가 작동하는 층위
  macro  - 거시/통화정책/지정학/환율 등 시장 전체에 영향
  market - 코스피·코스닥 등 시장 전반 시황
  sector - 업종·산업 단위 (반도체, 은행 등)
  stock  - 개별 종목 재료

topic: 자유 텍스트 태그, 짧게 (예: "금리정책", "삼성전자", "반도체")

summary: 한국어 1문장 요약 (60자 이내)

입력으로 주어진 모든 item_id에 대해 빠짐없이 결과를 채워주세요.
"""


class ItemAnalysis(BaseModel):
    item_id: str  # "{chat_id}_{message_id}" - matches the id shown in the prompt
    summary: str
    sentiment_score: int
    scope: Literal["macro", "market", "sector", "stock"]
    topic: str


class DayAnalysis(BaseModel):
    items: list[ItemAnalysis]


def analyze_day(client: genai.Client, model: str, rows: list[tuple[str, str, str]]) -> list[ItemAnalysis]:
    """rows: list of (item_id, chat_name, text)."""
    listing = "\n\n".join(f"[{item_id}] ({chat})\n{text[:3000]}" for item_id, chat, text in rows)
    prompt = f"아래 {len(rows)}개 항목을 각각 분석해 item_id별로 결과를 채워주세요.\n\n{listing}"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RUBRIC,
            response_mime_type="application/json",
            response_schema=DayAnalysis,
        ),
    )
    return response.parsed.items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Only process this date (YYYY-MM-DD); default: all unscored dates")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Add it to .env (see .env.example).")

    client = genai.Client(api_key=api_key)
    conn = get_connection()

    query = "SELECT DISTINCT substr(date,1,10) FROM items WHERE sentiment_score IS NULL AND text IS NOT NULL"
    params: tuple = ()
    if args.date:
        query += " AND substr(date,1,10) = ?"
        params = (args.date,)
    dates = [r[0] for r in conn.execute(query + " ORDER BY 1", params).fetchall()]

    if not dates:
        print("No unscored items with text found.")
        return

    for date in dates:
        rows = conn.execute(
            """
            SELECT chat_id, message_id, chat_name, text FROM items
            WHERE substr(date,1,10) = ? AND sentiment_score IS NULL AND text IS NOT NULL
            """,
            (date,),
        ).fetchall()

        by_item_id = {f"{chat_id}_{message_id}": (chat_id, message_id) for chat_id, message_id, _, _ in rows}
        analysis_rows = [(f"{chat_id}_{message_id}", chat_name, text) for chat_id, message_id, chat_name, text in rows]

        try:
            results = analyze_day(client, args.model, analysis_rows)
        except Exception as e:
            print(f"[{date}] FAILED: {e}")
            continue

        n = 0
        for item in results:
            ids = by_item_id.get(item.item_id)
            if ids is None:
                continue
            chat_id, message_id = ids
            update_summary(conn, chat_id, message_id, item.summary, item.sentiment_score, item.scope, item.topic)
            n += 1
        print(f"[{date}] scored {n}/{len(rows)} items")
        time.sleep(1)  # be polite to rate limits


if __name__ == "__main__":
    main()
