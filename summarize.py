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

RUBRIC = """당신은 한국 증권사 리서치·뉴스 속보를 읽고 그날 시장 심리에 미치는 영향을 정량화하는
매크로/주식 애널리스트입니다. 각 항목을 아래 기준으로 판단하세요.

## sentiment_score (-5.0..+5.0, 소수 첫째 자리까지 연속값)
그 내용이 시장 심리에 얼마나 긍정적/부정적인 재료인지를 판단합니다. 스코어는 "좋은 뉴스냐 나쁜
뉴스냐"가 아니라 "이 재료가 시장 참여자의 위험선호를 얼마나 밀어올리거나 끌어내리는가"를 기준으로
매기세요.

**정수로 뭉치지 말고 소수점까지 세밀하게 구분하세요** (예: -2.3, 0.6, 1.8, 4.2). 같은 방향의
항목이어도 강도가 다르면 반드시 다른 값을 주어야 합니다 — 비슷한 톤이라고 같은 정수로 반올림하지
마세요.

기준점(anchor):
  -5.0 시스템 리스크·패닉 (은행 파산, 전쟁 발발)   -3.0 뚜렷한 하락·우려 지배적
  -1.0 약한 부정/경계                              0.0 중립/정보성 (방향성 판단 불가)
  +1.0 약한 긍정                                    +3.0 실적 서프라이즈·정책 완화 등 뚜렷한 호재
  +5.0 사상 최대 실적/환원 등 최상급 호재

이 기준점 사이 값도 근거의 강도에 비례해 촘촘하게 사용하세요 (예: 소폭 하락은 -1.2, 큰 폭 하락은
-3.7처럼).

**보정 원칙**: 이모지·과장된 문구(🔥, "충격", "대박")에 휘둘리지 말고, 등락률·서프라이즈 폭 같은
정량 정보와 "사상 최대", "역대급", "우려 지배적" 같은 확정적 어휘를 근거로 강도를 매기세요.
같은 방향이어도 근거가 막연하면(예: 단순 코멘트) 절댓값을 낮추고, 수치로 뒷받침되면 절댓값을 높이세요.

## scope
이 뉴스가 작동하는 층위 (같은 "부정적" 재료도 층위에 따라 의미가 다릅니다):
  macro  - 거시/통화정책/지정학/환율 등 시장 전체에 영향
  market - 코스피·코스닥 등 시장 전반 시황·수급
  sector - 업종·산업 단위 (반도체, 은행, 2차전지 등)
  stock  - 개별 종목 고유 재료 (실적, 공시, 이벤트)

## topic
짧은 한국어 자유 태그 1개 (예: "금리정책", "삼성전자", "반도체 수출"). 가능하면 종목명/업종명처럼
나중에 같은 주제를 다시 찾을 수 있는 구체적인 명사로 답하세요.

## summary
한국어 1문장, 60자 이내. 숫자(등락률·금액 등)가 있으면 반드시 포함하고, "~한 소식", "~에 대한 내용"
같은 군더더기 표현은 쓰지 마세요.

## 판단 불가 항목 처리
텍스트만으로 방향성 판단이 불가능한 항목(예: 맥락 없는 URL 한 줄, 자료명뿐인 목차)은 **결과
리스트에서 제외**하세요. 억지로 0점이나 임의 점수를 매기지 말고, 그 item_id는 아예 응답하지 않으면
됩니다 - 그 항목은 결측치로 남습니다.
"""


class ItemAnalysis(BaseModel):
    item_id: str  # "{chat_id}_{message_id}" - matches the id shown in the prompt
    summary: str
    sentiment_score: float
    scope: Literal["macro", "market", "sector", "stock"]
    topic: str


class DayAnalysis(BaseModel):
    items: list[ItemAnalysis]


def analyze_day(client: genai.Client, model: str, rows: list[tuple[str, str, str]]) -> list[ItemAnalysis]:
    """rows: list of (item_id, chat_name, text)."""
    listing = "\n\n".join(f"[{item_id}] ({chat})\n{text[:3000]}" for item_id, chat, text in rows)
    prompt = (
        f"아래 {len(rows)}개 항목을 분석하세요. 판단 가능한 항목만 item_id별로 결과를 채우고, "
        f"판단 불가 항목은 결과에서 제외하세요.\n\n{listing}"
    )

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
    parser.add_argument(
        "--redo", action="store_true",
        help="Clear existing scores first and re-score (all dates, or just --date if given)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Add it to .env (see .env.example).")

    client = genai.Client(api_key=api_key)
    conn = get_connection()

    if args.redo:
        reset_sql = "UPDATE items SET sentiment_score=NULL, summary=NULL, scope=NULL, topic=NULL, summarized_at=NULL"
        reset_params: tuple = ()
        if args.date:
            reset_sql += " WHERE substr(date,1,10) = ?"
            reset_params = (args.date,)
        n_reset = conn.execute(reset_sql, reset_params).rowcount
        conn.commit()
        print(f"--redo: cleared {n_reset} existing score(s)")

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
