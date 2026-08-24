# CLAUDE.md

## 프로젝트 개요
텔레그램 증권사 리서치 채널 6곳에서 게시물을 수집해 SQLite에 저장하고,
LLM으로 논조를 채점한 뒤 자체완결 HTML 대시보드로 렌더링하는 파이프라인.

## 현재 파일
- `collect.py`   : Telethon으로 채널 수집, 파일 다운로드, SQLite upsert
- `store.py`     : SQLite 스키마 및 접근 함수
- `summarize.py` : Gemini API로 하루치 항목 채점
- `dashboard.py` : Plotly 임베드 단일 HTML 리포트 생성
- `channels.py`  : channels.json 로더

## 알려진 문제 (리팩터링 대상)
1. PDF를 다운로드만 하고 파싱하지 않아 채점 커버리지가 47%에 그침
2. 채널별 채점 커버리지 불균등 (한 채널이 전체 점수의 49%를 차지)
3. scope(macro/market/sector/stock)를 단순 평균으로 섞어 지수 해석 불가
4. 일별 평균의 표본 수가 1~3건이라 노이즈를 그대로 렌더링
5. topic이 자유 텍스트라 파편화 ("지정학적 리스크" vs "지정학 리스크")
6. 시장 데이터와의 검증 레이어가 전혀 없음
7. 시황 사후요약 게시물이 채점에 포함돼 지수가 당일 수익률을 그대로 베낌
8. 6개 채널이 같은 뉴스를 올려도 중복 제거 없이 N배 가중됨
9. 대시보드 CSS는 다크모드를 지원하나 Plotly 차트 배경은 라이트로 하드코딩

## 코딩 규약
- Python 3.11+, `from __future__ import annotations`
- 모든 실행 스크립트는 `argparse` + 모듈 docstring에 Usage 섹션
- docstring에는 "무엇을 하는가"뿐 아니라 **"왜 이 방법을 택했는가"**를 적을 것
- 외부 API 키는 `.env` (python-dotenv), 절대 하드코딩 금지
- 기존 파일을 지우지 말고 확장할 것. 스키마 변경은 ALTER TABLE 마이그레이션으로
- 새 의존성을 추가하면 README의 설치 명령도 같이 갱신할 것

## 하지 말 것
- LLM에게 숫자 계산/집계를 시키지 말 것. LLM은 비정형 텍스트 → 구조화 JSON 변환에만 사용
- 검증되지 않은 신호를 "예측력 있음"으로 서술하지 말 것
