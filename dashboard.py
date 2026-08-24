"""Single-file HTML dashboard for the research digest sentiment scoring.

Reads digest.db (see collect.py) and writes one self-contained HTML report:
daily sentiment trend, per-item scores/summaries, and an explanation of how
scores are produced. Charts and the Plotly.js library are embedded, so the
file opens directly in a browser with no server needed.

Usage
-----
    python dashboard.py                    # writes output/dashboard.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from store import get_connection

OUT = Path(__file__).parent / "output"

CHART_PAPER = "#FAF9F4"
CHART_INK = "#211D14"
COLORWAY = ["#9C6B2E", "#3F7A5E", "#B3432B", "#4A5A6B", "#7A5C3E"]

SCOPE_LABELS = {"macro": "매크로", "market": "시장 전반", "sector": "섹터", "stock": "개별종목"}
SCOPE_COLORS = {"macro": "#4A5A6B", "market": "#9C6B2E", "sector": "#7A5C3E", "stock": "#3F7A5E"}

STYLE = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ink: #211D14;
  --ink-soft: #6B6252;
  --paper: #EEEEE7;
  --surface: #FFFFFF;
  --border: #DDDACC;
  --accent: #9C6B2E;
  --accent-soft: #F1E7D6;
  --negative: #B3432B;
  --negative-soft: #F5E4DF;
  --positive: #3F7A5E;
  --positive-soft: #E3EEE8;
  --font-display: "Source Serif 4", Georgia, "Nanum Myeongjo", serif;
  --font-body: "IBM Plex Sans", "Noto Sans KR", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", "Noto Sans Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #EDE9DF;
    --ink-soft: #B3AB99;
    --paper: #15130D;
    --surface: #1E1B13;
    --border: #332E20;
    --accent: #D2A25C;
    --accent-soft: #362B18;
    --negative: #E08066;
    --negative-soft: #3A241F;
    --positive: #7FBBA0;
    --positive-soft: #1F3229;
  }
}
:root[data-theme="dark"] {
  --ink: #EDE9DF;
  --ink-soft: #B3AB99;
  --paper: #15130D;
  --surface: #1E1B13;
  --border: #332E20;
  --accent: #D2A25C;
  --accent-soft: #362B18;
  --negative: #E08066;
  --negative-soft: #3A241F;
  --positive: #7FBBA0;
  --positive-soft: #1F3229;
}
* { box-sizing: border-box; }
body { font-family: var(--font-body); background: var(--paper); color: var(--ink); margin: 0; padding: 2.5rem 1.5rem 4rem; }
.page { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 2.25rem; }
h1 { font-family: var(--font-display); font-weight: 700; font-size: 2rem; margin: 0 0 0.3rem; text-wrap: balance; }
h2 { font-family: var(--font-display); font-weight: 600; font-size: 1.3rem; margin: 0 0 1rem; }
.eyebrow { font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.5rem; }
.meta { color: var(--ink-soft); font-size: 0.95rem; margin: 0; }
.stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stat { background: var(--surface); padding: 1rem 1.2rem; display: flex; flex-direction: column; gap: 0.3rem; }
.stat .label { font-size: 0.78rem; color: var(--ink-soft); }
.stat .value { font-family: var(--font-mono); font-size: 1.4rem; font-variant-numeric: tabular-nums; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card .plot-wrap { background: %(paper)s; border-radius: 8px; padding: 0.5rem; overflow-x: auto; }
table { border-collapse: collapse; width: 100%%; font-variant-numeric: tabular-nums; }
th, td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); text-align: right; font-size: 0.88rem; vertical-align: top; }
th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2), th:nth-child(4), td:nth-child(4), th:nth-child(6), td:nth-child(6) { text-align: left; }
th { font-family: var(--font-mono); font-weight: 500; font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-soft); }
td { font-family: var(--font-body); }
td.mono { font-family: var(--font-mono); }
tr:last-child td { border-bottom: none; }
.chip { display: inline-flex; align-items: center; padding: 0.15rem 0.55rem; border-radius: 999px; font-family: var(--font-mono); font-size: 0.82rem; font-weight: 500; white-space: nowrap; }
.chip.neg { background: var(--negative-soft); color: var(--negative); }
.chip.pos { background: var(--positive-soft); color: var(--positive); }
.chip.neu { background: var(--accent-soft); color: var(--accent); }
.scale { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.scale .band { font-family: var(--font-mono); font-size: 0.8rem; padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--border); }
footer { color: var(--ink-soft); font-size: 0.85rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }
a { color: var(--accent); }
@media (max-width: 640px) { body { padding: 1.5rem 1rem 3rem; } }
</style>
""" % {"paper": CHART_PAPER}


def build_sentiment_fig(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [("#B3432B" if v < 0 else "#3F7A5E") for v in daily["avg_score"]]
    fig.add_bar(x=daily["date"], y=daily["avg_score"], marker_color=colors, name="일별 평균 점수")
    fig.add_hline(y=0, line_color=CHART_INK, line_width=1)
    fig.update_layout(
        title="일별 시장 심리 점수 (평균, -5 ~ +5)",
        paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PAPER,
        font=dict(family="IBM Plex Sans, sans-serif", color=CHART_INK, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[-5, 5], gridcolor="#E3E0D8", zerolinecolor="#E3E0D8"),
        xaxis=dict(gridcolor="#E3E0D8"),
    )
    return fig


def build_scope_fig(scope_daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for scope in ("macro", "market", "sector", "stock"):
        sub = scope_daily[scope_daily["scope"] == scope]
        if sub.empty:
            continue
        fig.add_bar(x=sub["date"], y=sub["avg_score"], name=SCOPE_LABELS[scope], marker_color=SCOPE_COLORS[scope])
    fig.add_hline(y=0, line_color=CHART_INK, line_width=1)
    fig.update_layout(
        title="범위별 일별 심리 점수 (매크로 / 시장전반 / 섹터 / 개별종목)",
        barmode="group",
        paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PAPER,
        font=dict(family="IBM Plex Sans, sans-serif", color=CHART_INK, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[-5, 5], gridcolor="#E3E0D8", zerolinecolor="#E3E0D8"),
        xaxis=dict(gridcolor="#E3E0D8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_content(conn) -> str:
    total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM items WHERE sentiment_score IS NOT NULL").fetchone()[0]

    daily = pd.read_sql_query(
        """
        SELECT substr(date,1,10) AS date, AVG(sentiment_score) AS avg_score, COUNT(*) AS n
        FROM items WHERE sentiment_score IS NOT NULL
        GROUP BY date ORDER BY date
        """,
        conn,
    )

    scope_daily = pd.read_sql_query(
        """
        SELECT substr(date,1,10) AS date, scope, AVG(sentiment_score) AS avg_score, COUNT(*) AS n
        FROM items WHERE sentiment_score IS NOT NULL AND scope IS NOT NULL
        GROUP BY date, scope ORDER BY date
        """,
        conn,
    )

    scope_summary_rows = "".join(
        f"<tr><td>{SCOPE_LABELS.get(scope, scope)}</td><td class='mono'>{n}</td>"
        f"<td><span class='chip {'pos' if avg > 0 else 'neg' if avg < 0 else 'neu'}'>{avg:+.2f}</span></td></tr>"
        for scope, n, avg in conn.execute(
            """
            SELECT scope, COUNT(*), AVG(sentiment_score)
            FROM items WHERE sentiment_score IS NOT NULL AND scope IS NOT NULL
            GROUP BY scope ORDER BY 2 DESC
            """
        ).fetchall()
    )

    topic_rows = "".join(
        f"<tr><td>{topic}</td><td>{SCOPE_LABELS.get(scope, scope)}</td><td class='mono'>{n}</td>"
        f"<td><span class='chip {'pos' if avg > 0 else 'neg' if avg < 0 else 'neu'}'>{avg:+.2f}</span></td></tr>"
        for topic, scope, n, avg in conn.execute(
            """
            SELECT topic, scope, COUNT(*) AS n, AVG(sentiment_score)
            FROM items WHERE sentiment_score IS NOT NULL AND topic IS NOT NULL
            GROUP BY topic, scope ORDER BY n DESC LIMIT 15
            """
        ).fetchall()
    )

    channel_rows = "".join(
        f"<tr><td>{name}</td><td class='mono'>{n}</td><td class='mono'>{s}</td></tr>"
        for name, n, s in conn.execute(
            """
            SELECT chat_name, COUNT(*),
                   SUM(CASE WHEN sentiment_score IS NOT NULL THEN 1 ELSE 0 END)
            FROM items GROUP BY chat_name ORDER BY 2 DESC
            """
        ).fetchall()
    )

    item_rows = "".join(
        f"<tr><td>{date[:10]}</td><td>{chat}</td>"
        f"<td><span class='chip neu'>{SCOPE_LABELS.get(scope, scope or '-')}</span></td>"
        f"<td>{topic or '-'}</td>"
        f"<td><span class='chip {'pos' if score > 0 else 'neg' if score < 0 else 'neu'}'>{score:+.0f}</span></td>"
        f"<td>{summary}</td></tr>"
        for date, chat, scope, topic, score, summary in conn.execute(
            """
            SELECT date, chat_name, scope, topic, sentiment_score, summary
            FROM items WHERE sentiment_score IS NOT NULL
            ORDER BY date DESC, sentiment_score ASC
            """
        ).fetchall()
    )

    latest_score = daily["avg_score"].iloc[-1] if not daily.empty else None
    latest_label = f"{latest_score:+.2f}" if latest_score is not None else "—"
    days_done = len(daily)
    pct = scored / total * 100 if total else 0

    if daily.empty:
        chart_card = '<div class="chart-card"><h2>일별 시장 심리 점수</h2><p>아직 점수가 매겨진 날짜가 없습니다.</p></div>'
        scope_chart_card = ""
    else:
        fig = build_sentiment_fig(daily)
        chart_card = (
            f'<div class="chart-card"><h2>일별 시장 심리 점수</h2><div class="plot-wrap">'
            f'{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div></div>'
        )
        scope_fig = build_scope_fig(scope_daily)
        scope_chart_card = (
            f'<div class="chart-card"><h2>범위별 심리 점수</h2><div class="plot-wrap">'
            f'{pio.to_html(scope_fig, include_plotlyjs=False, full_html=False)}</div></div>'
        )

    return f"""
{STYLE}
<div class="page">
  <div>
    <p class="eyebrow">Research Digest · Sentiment</p>
    <h1>증권사 리서치 다이제스트</h1>
    <p class="meta">텔레그램 채널 6곳에서 수집 · 생성 시각 {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
  </div>

  <div class="stat-strip">
    <div class="stat"><span class="label">전체 수집</span><span class="value">{total:,}건</span></div>
    <div class="stat"><span class="label">점수 매김</span><span class="value">{scored:,}건 ({pct:.1f}%)</span></div>
    <div class="stat"><span class="label">처리된 날짜 수</span><span class="value">{days_done}일</span></div>
    <div class="stat"><span class="label">최근 처리일 점수</span><span class="value">{latest_label}</span></div>
  </div>

  {chart_card}
  {scope_chart_card}

  <div class="card">
    <h2>채점 기준</h2>
    <p>각 메시지를 Claude가 직접 읽고 그 내용이 <strong>그날 시장 심리에 대해 얼마나 긍정적/부정적인 재료인지</strong>를
    -5(매우 부정)부터 +5(매우 긍정)까지 정수로 판단합니다. 공식이나 사전 기반 자동 채점이 아니라,
    지수 등락·실적·정책 발언 등 내용의 맥락을 읽고 매기는 수동 평가입니다. 일별 점수는 그날 처리된
    항목 점수의 <strong>단순 평균</strong>입니다.</p>
    <div class="scale">
      <span class="band">-5·-4 매우 부정 (급락/위기)</span>
      <span class="band">-3·-2 부정 (하락·우려 지배적)</span>
      <span class="band">-1 약한 부정/경계</span>
      <span class="band">0 중립/정보성</span>
      <span class="band">+1 약한 긍정</span>
      <span class="band">+2·+3 긍정 (실적 서프라이즈 등)</span>
      <span class="band">+4·+5 매우 긍정 (사상 최대 등)</span>
    </div>
    <p style="margin-top:1rem">점수와 별개로 각 항목은 <strong>범위(scope)</strong>도 함께 분류합니다 — 뉴스가
    작동하는 층위가 다르면 같은 "부정적" 재료도 의미가 다르기 때문입니다:</p>
    <div class="scale">
      <span class="band">macro 거시/통화정책/지정학/환율 등 시장 전체에 영향</span>
      <span class="band">market 코스피·코스닥 등 시장 전반 시황</span>
      <span class="band">sector 업종·산업 단위 (반도체, 은행 등)</span>
      <span class="band">stock 개별 종목 재료</span>
    </div>
    <p style="margin-top:0.75rem">주제(topic)는 "금리정책", "삼성전자"처럼 자유 태그로 붙여, 같은 주제가 반복
    언급되는지 나중에 추적할 수 있게 합니다.</p>
  </div>

  <div class="card">
    <h2>범위별 요약</h2>
    <table><tr><th>범위</th><th>건수</th><th>평균 점수</th></tr>{scope_summary_rows}</table>
  </div>

  <div class="card">
    <h2>자주 언급된 주제 (상위 15)</h2>
    <table><tr><th>주제</th><th>범위</th><th>건수</th><th>평균 점수</th></tr>{topic_rows}</table>
  </div>

  <div class="card">
    <h2>채널별 진행 현황</h2>
    <table><tr><th>채널</th><th>수집</th><th>점수매김</th></tr>{channel_rows}</table>
  </div>

  <div class="card">
    <h2>처리된 항목 ({scored}건)</h2>
    <table><tr><th>날짜</th><th>채널</th><th>범위</th><th>주제</th><th>점수</th><th>요약</th></tr>{item_rows}</table>
  </div>

  <footer>
    남은 항목은 대화창에서 날짜를 요청하시면(예: "8월 20일치 해줘") 이어서 처리됩니다.
    사진/PDF 캡션이 없는 항목은 아직 이 채점에 포함되지 않았습니다.
  </footer>
</div>
"""


def render_dashboard(conn) -> str:
    content = build_content(conn)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>증권사 리서치 다이제스트</title>
<script>{get_plotlyjs()}</script>
</head>
<body>
{content}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output HTML path (default: output/dashboard.html)")
    args = parser.parse_args()

    conn = get_connection()
    html = render_dashboard(conn)
    conn.close()

    out_path = Path(args.out) if args.out else OUT / "dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
