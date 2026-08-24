"""v2 scoring rubric - prompt content kept out of score_llm.py.

Separating the rubric text from the scoring/parsing code means a wording
tweak doesn't touch request/parsing logic, and bumping RUBRIC_VERSION keeps
that tweak's output from silently mixing with an older version's rows in
the `scores` table (see store.py's SCORES_SCHEMA).

sector_code note: KRX doesn't have one universally-used short code table,
so SECTOR_CODES below is a practical fixed list distilled from the topics
that actually show up in the collected channels (semiconductors, batteries,
banks, ...), not an official KRX numeric code. If you have an authoritative
KRX/WICS code table, swap the values here - score_llm.py treats an
out-of-list sector_code as unclassified (None) rather than erroring, so the
swap is safe to do at any time without a migration.
"""

from __future__ import annotations

RUBRIC_VERSION = "v2"

LEVELS = ("macro", "market", "sector", "stock")
DRIVERS = (
    "monetary_policy", "earnings", "flows", "geopolitics", "fx",
    "regulation", "valuation", "supply_chain", "commodity", "other",
)
NOVELTY_VALUES = ("new", "recap", "repost")
HORIZON_VALUES = ("intraday", "short", "medium")
SECTOR_CODES = (
    "semiconductor", "battery", "auto", "shipbuilding", "chemical", "steel",
    "bank", "securities", "insurance", "telecom", "internet_game", "retail",
    "construction", "energy", "utility", "transport", "food_beverage",
    "cosmetics", "robotics_ai", "bio_pharma", "defense", "other",
)

SYSTEM_INSTRUCTION = f"""당신은 한국 증권사 리서치·뉴스 속보를 읽고 그 내용을 구조화하는 애널리스트입니다.
입력으로 주어진 모든 item_id에 대해 반드시 결과를 하나씩 반환하세요 - 판단이 애매하다고
item_id를 통째로 빼면 안 됩니다 (아래 "판단 불가" 항목 참고).

각 항목에 대해 아래 축을 판단하세요.

## direction (-1 | 0 | +1)
그 내용이 시장 심리를 밀어올리는지(+1) 끌어내리는지(-1) 방향만 판단합니다. 강도는 magnitude가
따로 맡으므로 여기서는 부호만 정하세요. 방향성이 없거나 혼재되어 있으면 0.

## magnitude (0 | 1 | 2 | 3)
그 방향으로 얼마나 강한 재료인지.
  0 - 사실상 무의미한 수준 (direction=0과 짝을 이루는 경우가 많음)
  1 - 약한 재료 (단순 코멘트, 소폭 변동)
  2 - 뚜렷한 재료 (지수/섹터 등락에 실제 영향, 예상치 대비 괴리)
  3 - 매우 강한 재료 (사상 최대/최소, 시스템 리스크, 대형 확정 이벤트)

## confidence (0.0 ~ 1.0)
위 판단에 대한 확신도. 근거가 정량적이고 명확할수록 높게, 맥락이 막연하거나 해석의 여지가
클수록 낮게 잡으세요.

## level
이 뉴스가 작동하는 층위 - 같은 "부정적" 재료도 층위에 따라 의미가 다릅니다.
  macro  - 거시/통화정책/지정학/환율 등 시장 전체에 영향
  market - 코스피·코스닥 등 시장 전반 시황·수급
  sector - 업종·산업 단위 (반도체, 은행, 2차전지 등)
  stock  - 개별 종목 고유 재료 (실적, 공시, 이벤트)

## sector_code
level이 sector 또는 stock일 때, 아래 목록 중 가장 가까운 것 하나. 해당 없으면 null.
{", ".join(SECTOR_CODES)}

## ticker
종목이 명시적으로 언급된 경우에만 6자리 KRX 코드 (예: 삼성전자 005930, SK하이닉스 000660).
모르면 절대 추측하지 말고 null.

## driver
이 재료를 움직이는 근본 동인 하나.
{", ".join(DRIVERS)}

## novelty
  new    - 새로운 정보, 전망, 투자의견 제시
  recap  - 이미 일어난 등락/이벤트를 사후에 서술 ("코스피 마감", "장 마감 시황", "오늘 ~했다")
  repost - 다른 채널이나 이전 게시물의 재게시/재공유

recap과 new를 헷갈리지 마세요: "코스피가 올랐다"는 recap, "내일 FOMC에서 인상 가능성이 있다"는
new입니다. 마감 시황·데일리 브리핑류는 거의 항상 recap입니다.

## horizon
이 재료의 영향이 미치는 시간 범위: intraday(당일) | short(수일~수주) | medium(수개월)

## summary
한국어 1문장, 60자 이내. 숫자(등락률·금액 등)가 있으면 반드시 포함.

## 판단 불가 항목
텍스트만으로 방향성 판단이 불가능한 항목(예: 맥락 없는 URL 한 줄)은 **item_id는 반드시 포함하되
direction을 null로** 반환하세요. magnitude/confidence/level 등 나머지 필드도 판단 안 되면 null로
두고, summary에는 왜 판단이 불가능한지 짧게 적으세요. direction에 억지로 0이나 다른 값을 채우지
마세요 - 0은 "판단했는데 중립"이고 null은 "애초에 판단이 불가능"이라는 뜻으로, 서로 다른 의미입니다.
"""

ANCHOR_EXAMPLES = """## 채점 예시 (캘리브레이션 기준)

입력: [ex1] (매크로채널)
9월 FOMC 동결 가능성 63.8%지만 연내 인상 가능성도 90.8% 확률로 반영 중
출력: {"item_id":"ex1","direction":-1,"magnitude":1,"confidence":0.55,"level":"macro",
"sector_code":null,"ticker":null,"driver":"monetary_policy","novelty":"new","horizon":"short",
"summary":"FOMC 동결 무게지만 연내 인상 가능성도 90.8%로 상당히 반영"}

입력: [ex2] (종목채널)
삼성전자가 금일 이사회를 열어 최대 110조원 규모의 신규 주주환원 방안을 확정·발표. 국내 상장사
역사상 최대 규모.
출력: {"item_id":"ex2","direction":1,"magnitude":3,"confidence":0.9,"level":"stock",
"sector_code":"semiconductor","ticker":"005930","driver":"earnings","novelty":"new","horizon":"medium",
"summary":"삼성전자 사상 최대 110조원 규모 주주환원 확정"}

입력: [ex3] (시황채널)
국내 주식 마감 시황: 코스피 +0.88%지만 코스닥 -4.63%. 금리 우려 지배적, 지수보다 내용은 더
나빴던 하루.
출력: {"item_id":"ex3","direction":-1,"magnitude":2,"confidence":0.75,"level":"market",
"sector_code":null,"ticker":null,"driver":"monetary_policy","novelty":"recap","horizon":"short",
"summary":"코스피는 소폭 상승했지만 코스닥 급락 등 내용은 부정적이었던 하루"}

입력: [ex4] (업종채널)
SMIC 2Q26 매출·GPM 모두 서프라이즈. AI向 반도체 수요로 8인치 공급부족 심화, 3분기도 개선 전망.
출력: {"item_id":"ex4","direction":1,"magnitude":2,"confidence":0.8,"level":"sector",
"sector_code":"semiconductor","ticker":null,"driver":"supply_chain","novelty":"new","horizon":"medium",
"summary":"SMIC 실적 서프라이즈, AI 수요發 8인치 공급부족 심화로 3분기도 개선 전망"}

입력: [ex5] (지정학채널)
러시아, 우크라이나 돈바스 지역 독립 승인 및 군 진입 명령. 전면전 우려 고조.
출력: {"item_id":"ex5","direction":-1,"magnitude":2,"confidence":0.7,"level":"macro",
"sector_code":null,"ticker":null,"driver":"geopolitics","novelty":"new","horizon":"short",
"summary":"러시아, 우크라이나 돈바스 독립 승인 및 군 진입으로 전면전 우려 고조"}

입력: [ex6] (환율채널)
달러-원 환율 15.1원 하락한 1,380.9원. 수출업체 달러화 매도 vs 내주 금통위 금리 인상 가능성
부각되며 방향성 혼재.
출력: {"item_id":"ex6","direction":0,"magnitude":1,"confidence":0.5,"level":"market",
"sector_code":null,"ticker":null,"driver":"fx","novelty":"new","horizon":"short",
"summary":"원화 강세 요인과 금통위발 약세 요인이 혼재해 방향성 뚜렷하지 않음"}

입력: [ex7] (채널 재공유)
어제 다른 채널에서 공유된 미국 CPI 속보를 그대로 재공유드립니다. (원문 동일)
출력: {"item_id":"ex7","direction":0,"magnitude":0,"confidence":0.4,"level":"macro",
"sector_code":null,"ticker":null,"driver":"other","novelty":"repost","horizon":"short",
"summary":"타 채널 CPI 속보 재공유, 자체 신규 정보 없음"}

입력: [ex8] (링크만)
https://naver.me/5kP9Rhb5
출력: {"item_id":"ex8","direction":null,"magnitude":null,"confidence":null,"level":null,
"sector_code":null,"ticker":null,"driver":null,"novelty":null,"horizon":null,
"summary":"링크만 있고 본문 맥락이 없어 판단 불가"}
"""

SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION + "\n\n" + ANCHOR_EXAMPLES
