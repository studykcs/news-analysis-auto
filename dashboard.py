"""Single-file HTML dashboard for the research digest sentiment scoring (v2 schema).

Reads scores/items/extract_status from digest.db and, if a companion KRX
price database is reachable, validates the index against realized returns
(see market.py / validate.py). Self-contained: one HTML file, CSS design
tokens for light/dark, charts embedded via Plotly.

Usage
-----
    python dashboard.py                              # writes output/dashboard.html + docs/index.html
    python dashboard.py --prices-db ../pairs-trading-krx/prices.db
    python dashboard.py --embed-plotly               # inline plotly.js instead of CDN (offline use)
    python dashboard.py --ewma-span 14
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import market
from index import daily_index
from store import get_connection, latest_scores

OUT = Path(__file__).parent / "output"
DOCS_DIR = Path(__file__).parent / "docs"


def _bundled_plotlyjs_version() -> str:
    """The pip package `plotly`'s own version (e.g. 6.9.0) is NOT the
    plotly.js CDN version - cdn.plot.ly hosts plotly.js releases (e.g.
    3.7.0), a separately-numbered artifact the Python package happens to
    bundle. Hardcoding the pip version here 404'd (well, 403'd) on
    cdn.plot.ly and silently killed every chart. Read the real version out
    of the bundled JS's own header comment instead, so this can never drift
    out of sync again after a `pip install -U plotly`."""
    from plotly.offline import get_plotlyjs

    first_line = get_plotlyjs().splitlines()[1]  # "* plotly.js v3.7.0"
    return first_line.rsplit("v", 1)[-1].strip()


PLOTLY_VERSION = _bundled_plotlyjs_version()
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"

LEVELS = ("macro", "market", "sector", "stock")
LEVEL_LABELS = {"macro": "매크로", "market": "시장 전반", "sector": "섹터", "stock": "개별종목"}
LEVEL_COLORS = {"macro": "#4A5A6B", "market": "#9C6B2E", "sector": "#7A5C3E", "stock": "#3F7A5E"}
NEGATIVE = "#B3432B"
POSITIVE = "#3F7A5E"
INK_FALLBACK = "#211D14"  # only used before the theme-sync JS runs

MIN_GROUP_N = 30
LEADLAG_RANGE = range(-5, 6)
EVENT_WINDOW = range(-5, 11)

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
    --ink: #EDE9DF; --ink-soft: #B3AB99; --paper: #15130D; --surface: #1E1B13;
    --border: #332E20; --accent: #D2A25C; --accent-soft: #362B18;
    --negative: #E08066; --negative-soft: #3A241F; --positive: #7FBBA0; --positive-soft: #1F3229;
  }
}
:root[data-theme="dark"] {
  --ink: #EDE9DF; --ink-soft: #B3AB99; --paper: #15130D; --surface: #1E1B13;
  --border: #332E20; --accent: #D2A25C; --accent-soft: #362B18;
  --negative: #E08066; --negative-soft: #3A241F; --positive: #7FBBA0; --positive-soft: #1F3229;
}
* { box-sizing: border-box; }
body { font-family: var(--font-body); background: var(--paper); color: var(--ink); margin: 0; padding: 2.5rem 1.5rem 4rem; }
.page { max-width: 1120px; margin: 0 auto; display: flex; flex-direction: column; gap: 2.25rem; }
h1 { font-family: var(--font-display); font-weight: 700; font-size: 2rem; margin: 0 0 0.3rem; text-wrap: balance; }
h2 { font-family: var(--font-display); font-weight: 600; font-size: 1.3rem; margin: 0 0 1rem; }
.eyebrow { font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.5rem; }
.meta { color: var(--ink-soft); font-size: 0.95rem; margin: 0; }
.stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stat { background: var(--surface); padding: 1rem 1.2rem; display: flex; flex-direction: column; gap: 0.3rem; }
.stat .label { font-size: 0.76rem; color: var(--ink-soft); }
.stat .value { font-family: var(--font-mono); font-size: 1.3rem; font-variant-numeric: tabular-nums; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card .plot-wrap { overflow-x: auto; }
.chart-card .note { color: var(--ink-soft); font-size: 0.85rem; margin: 0.5rem 0 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--border); text-align: right; font-size: 0.86rem; vertical-align: top; }
th.left, td.left { text-align: left; }
th { font-family: var(--font-mono); font-weight: 500; font-size: 0.7rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-soft); }
td { font-family: var(--font-body); }
td.mono { font-family: var(--font-mono); }
tr:last-child td { border-bottom: none; }
.chip { display: inline-flex; align-items: center; padding: 0.15rem 0.55rem; border-radius: 999px; font-family: var(--font-mono); font-size: 0.8rem; font-weight: 500; white-space: nowrap; }
.chip.neg { background: var(--negative-soft); color: var(--negative); }
.chip.pos { background: var(--positive-soft); color: var(--positive); }
.chip.neu { background: var(--accent-soft); color: var(--accent); }
.scale { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.scale .band { font-family: var(--font-mono); font-size: 0.78rem; padding: 0.3rem 0.6rem; border-radius: 8px; border: 1px solid var(--border); }
.limits { margin: 0; padding-left: 1.2rem; color: var(--ink-soft); font-size: 0.92rem; line-height: 1.6; }
.filters { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
.filters select, .filters input { font-family: var(--font-body); font-size: 0.85rem; padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); color: var(--ink); }
footer { color: var(--ink-soft); font-size: 0.85rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }
a { color: var(--accent); }
@media (max-width: 640px) { body { padding: 1.5rem 1rem 3rem; } }
</style>
"""

THEME_SCRIPT = """
<script>
function applyChartTheme() {
  var style = getComputedStyle(document.documentElement);
  var ink = style.getPropertyValue('--ink').trim();
  var grid = style.getPropertyValue('--border').trim();
  document.querySelectorAll('.js-plotly-plot').forEach(function (div) {
    var layout = div.layout || {};
    var update = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', 'font.color': ink, 'legend.font.color': ink };
    Object.keys(layout).forEach(function (key) {
      if (/^(xaxis|yaxis)\\d*$/.test(key)) {
        update[key + '.gridcolor'] = grid;
        update[key + '.zerolinecolor'] = grid;
        update[key + '.tickfont.color'] = ink;
        update[key + '.linecolor'] = grid;
      }
    });
    try { Plotly.relayout(div, update); } catch (e) {}
  });
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyChartTheme);
window.addEventListener('load', applyChartTheme);
applyChartTheme();
</script>
"""


def _fig_layout_base() -> dict:
    """Transparent background so the chart sits on the card's own surface in
    either theme; font/gridline colors are corrected at runtime by
    THEME_SCRIPT (Plotly renders to a static canvas, so it can't read CSS
    variables itself at draw time)."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=INK_FALLBACK, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def _to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


# ---------------------------------------------------------------------------
# Chart 1: sentiment index (EWMA) vs market proxy, with a sample-size subplot
# ---------------------------------------------------------------------------
def build_index_vs_market_fig(idx: pd.DataFrame, market_df: pd.DataFrame | None, ewma_span: int) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    dates = pd.to_datetime(idx["date"])
    ewma = idx["index_value"].ewm(span=ewma_span, min_periods=1).mean()
    fig.add_scatter(x=dates, y=ewma, name=f"논조지수 (EWMA{ewma_span})", mode="lines",
                     line=dict(color="#9C6B2E", width=2.4), row=1, col=1, secondary_y=False)

    if market_df is not None and not market_df.empty:
        m = market_df.copy()
        m["date"] = pd.to_datetime(m["date"])
        m = m[(m["date"] >= dates.min()) & (m["date"] <= dates.max())]
        level = (1 + m["market_return"]).cumprod()
        fig.add_scatter(x=m["date"], y=level, name="시장 대용치 (누적, 시작=1)", mode="lines",
                         line=dict(color="#4A5A6B", width=1.6, dash="dot"), row=1, col=1, secondary_y=True)

    fig.add_bar(x=dates, y=idx["n_items"], name="일별 표본수", marker_color="#DDDACC", row=2, col=1)

    fig.update_layout(
        **_fig_layout_base(),
        title="리서치 채널 논조지수(EWMA) vs 시장 대용치 — 위 라인이 시장보다 먼저 움직이는지 3초 안에 비교",
        height=460, bargap=0.2,
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="논조지수", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="시장 누적수익", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="표본수", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Chart 2: lead-lag correlation with 95% CI (Fisher z)
# ---------------------------------------------------------------------------
def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or pd.isna(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))


def build_leadlag_fig(idx_series: pd.Series, market_series: pd.Series) -> tuple[go.Figure, pd.DataFrame]:
    rows = []
    for k in LEADLAG_RANGE:
        shifted = market_series.shift(-k)
        merged = pd.DataFrame({"index": idx_series, "market": shifted}).dropna()
        n = len(merged)
        r = merged["index"].corr(merged["market"]) if n >= 3 else float("nan")
        lo, hi = _fisher_ci(r, n)
        rows.append({"k": k, "corr": r, "n": n, "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(rows)

    fig = go.Figure()
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["corr"].fillna(0)]
    err_plus = (df["ci_hi"] - df["corr"]).clip(lower=0)
    err_minus = (df["corr"] - df["ci_lo"]).clip(lower=0)
    fig.add_bar(
        x=df["k"], y=df["corr"], marker_color=colors,
        error_y=dict(type="data", array=err_plus, arrayminus=err_minus, visible=True, color="#8A8272"),
        hovertemplate="k=%{x}<br>corr=%{y:+.3f}<extra></extra>",
    )
    fig.add_hline(y=0, line_color="#8A8272", line_width=1)
    fig.add_vline(x=0, line_color="#8A8272", line_width=1, line_dash="dot")
    fig.update_layout(
        **_fig_layout_base(),
        title="Lead-lag 상관: corr(논조지수_t, 시장수익률_t+k) — k&lt;0은 지수가 뒤따라감, k&gt;0은 지수가 앞섬 (95% CI)",
        height=340, xaxis_title="k (거래일)", yaxis_title="상관계수",
    )
    return fig, df


# ---------------------------------------------------------------------------
# Chart 3: score distribution histogram, split by level
# ---------------------------------------------------------------------------
def build_score_hist_fig(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for level in LEVELS:
        vals = scores.loc[scores["level"] == level, "score"].dropna()
        if vals.empty:
            continue
        fig.add_histogram(
            x=vals, name=LEVEL_LABELS[level], marker_color=LEVEL_COLORS[level],
            opacity=0.65, xbins=dict(start=-3.5, end=3.5, size=1),
        )
    fig.update_layout(
        **_fig_layout_base(),
        title="점수 분포 (direction × magnitude, -3..+3) — level별 색 분리",
        barmode="overlay", height=340, xaxis_title="score", yaxis_title="건수",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4: level x week heatmap
# ---------------------------------------------------------------------------
def build_heatmap_fig(scores_with_date: pd.DataFrame) -> go.Figure:
    df = scores_with_date.dropna(subset=["score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    pivot = df.pivot_table(index="level", columns="week", values="score", aggfunc="mean")
    pivot = pivot.reindex(LEVELS)
    week_labels = [d.strftime("%m/%d") for d in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=week_labels, y=[LEVEL_LABELS[l] for l in pivot.index],
        colorscale=[[0, NEGATIVE], [0.5, "#EEEEE7"], [1, POSITIVE]], zmid=0,
        colorbar=dict(title="평균 점수"),
        hovertemplate="주: %{x}<br>%{y}: %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_fig_layout_base(),
        title="level × 주(week) 평균 점수 히트맵",
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 5: per-channel calibration
# ---------------------------------------------------------------------------
def build_channel_fig(channel_stats: pd.DataFrame) -> go.Figure:
    df = channel_stats.sort_values("avg_score")
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["avg_score"]]
    fig = go.Figure()
    fig.add_bar(
        x=df["avg_score"], y=df["channel"], orientation="h", marker_color=colors,
        customdata=np.stack([df["n_scored"], df["coverage_pct"]], axis=-1),
        hovertemplate="%{y}<br>평균 %{x:+.2f} · 채점 %{customdata[0]}건 · 커버리지 %{customdata[1]:.0f}%<extra></extra>",
    )
    fig.add_vline(x=0, line_color="#8A8272", line_width=1)
    fig.update_layout(
        **_fig_layout_base(),
        title="채널별 논조 캘리브레이션 (평균 점수) — 채널 편향은 숨기지 않고 그대로 표시",
        height=120 + 32 * len(df), xaxis_title="평균 점수",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 7: event study CAR (only when prices.db is reachable)
# ---------------------------------------------------------------------------
def build_event_study_fig(conn, prices_db: str) -> tuple[go.Figure | None, str | None]:
    try:
        prices = market.load_prices(prices_db)
        if prices.empty:
            return None, f"{prices_db}에 데이터가 없습니다."
    except Exception as e:
        return None, f"{prices_db}를 읽을 수 없습니다 ({e}). --prices-db로 경로를 지정하세요."

    mkt = market.market_proxy(prices).set_index("date")["market_return"]
    cal = market.trading_calendar(prices)
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = wide.pct_change(fill_method=None)

    scored = pd.read_sql_query(
        """
        SELECT s.ticker, s.direction, i.date AS event_date
        FROM scores s
        JOIN items i ON i.chat_id = s.chat_id AND i.message_id = s.message_id
        WHERE s.ticker IS NOT NULL AND s.direction IS NOT NULL AND s.direction != 0
        """,
        conn,
    )
    if scored.empty:
        return None, "ticker가 태깅된 채점 항목이 없습니다."
    scored["event_date"] = pd.to_datetime(scored["event_date"].str.slice(0, 10))

    records = []
    for _, row in scored.iterrows():
        ticker = row["ticker"]
        if ticker not in returns.columns:
            continue
        t0 = market.next_trading_day(cal, row["event_date"])
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
        records.append({"direction": row["direction"], **car_by_k})

    ev = pd.DataFrame(records)
    if ev.empty:
        return None, "이벤트가 매칭되지 않았습니다 (prices.db에 없는 ticker이거나 거래일 매핑 실패)."

    fig = go.Figure()
    for direction, label, color in ((1, "direction=+1", POSITIVE), (-1, "direction=-1", NEGATIVE)):
        group = ev[ev["direction"] == direction]
        n = len(group)
        if n == 0:
            continue
        means = [group[k].mean() for k in EVENT_WINDOW]
        name = f"{label} (n={n}{'  ⚠n<30' if n < MIN_GROUP_N else ''})"
        fig.add_scatter(x=list(EVENT_WINDOW), y=means, mode="lines+markers", name=name, line=dict(color=color, width=2))

    fig.add_vline(x=0, line_color="#8A8272", line_width=1, line_dash="dot")
    fig.add_hline(y=0, line_color="#8A8272", line_width=1)
    fig.update_layout(
        **_fig_layout_base(),
        title="이벤트 스터디: 종목 태깅 항목의 CAR (t-5..t+10, direction별)",
        height=340, xaxis_title="게시일 기준 거래일 (t=0)", yaxis_title="누적초과수익(CAR)",
        legend=dict(orientation="h", y=1.15),
    )
    return fig, None


# ---------------------------------------------------------------------------
# Gemini (v2) vs KR-FinBERT (finbert-v1) comparison, when both exist
# ---------------------------------------------------------------------------
def build_finbert_comparison_fig(conn) -> tuple[go.Figure | None, str | None, dict | None]:
    df = pd.read_sql_query(
        """
        SELECT g.direction AS g_dir, g.magnitude AS g_mag, f.direction AS f_dir, f.magnitude AS f_mag
        FROM scores g
        JOIN scores f ON f.chat_id = g.chat_id AND f.message_id = g.message_id
        WHERE g.rubric_version = 'v2' AND f.rubric_version = 'finbert-v1'
          AND g.direction IS NOT NULL AND f.direction IS NOT NULL
        """,
        conn,
    )
    if df.empty:
        return None, "두 rubric(v2, finbert-v1)으로 모두 채점된 항목이 아직 없습니다. score_finbert.py를 실행하세요.", None

    df["g_score"] = df["g_dir"] * df["g_mag"]
    df["f_score"] = df["f_dir"] * df["f_mag"]
    n = len(df)
    dir_agree = (df["g_dir"] == df["f_dir"]).mean()
    corr = df["g_score"].corr(df["f_score"]) if n >= 3 else float("nan")

    fig = go.Figure()
    fig.add_scatter(
        x=df["g_score"], y=df["f_score"], mode="markers",
        marker=dict(color="#9C6B2E", size=7, opacity=0.55),
        hovertemplate="Gemini %{x:+.0f} · FinBERT %{y:+.0f}<extra></extra>",
    )
    fig.add_hline(y=0, line_color="#8A8272", line_width=1)
    fig.add_vline(x=0, line_color="#8A8272", line_width=1)
    fig.update_layout(
        **_fig_layout_base(),
        title=f"Gemini(v2) vs FinBERT 점수 비교 — n={n}, 방향 일치율 {dir_agree:.0%}, 상관 {corr:+.2f}",
        height=340, xaxis_title="Gemini score (direction×magnitude)", yaxis_title="FinBERT score",
    )
    return fig, None, {"n": n, "dir_agree": dir_agree, "corr": corr}


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------
def _anon_channels(conn) -> dict[int, str]:
    chat_ids = [r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM items ORDER BY chat_id").fetchall()]
    return {chat_id: f"채널{i + 1}" for i, chat_id in enumerate(chat_ids)}


def build_content(conn, prices_db: str, ewma_span: int) -> str:
    anon = _anon_channels(conn)

    # -- scores with item date/chat, direction not null (broad set, for
    #    histogram/heatmap/channel calibration - NOT the index-eligible set)
    scores_raw = pd.read_sql_query(
        """
        SELECT s.chat_id, s.level, s.direction, s.magnitude, s.confidence, s.novelty,
               s.rubric_version, i.mention_channels, substr(i.date, 1, 10) AS date
        FROM scores s
        JOIN (SELECT chat_id, message_id, MAX(scored_at) AS m FROM scores GROUP BY chat_id, message_id) latest
          ON s.chat_id = latest.chat_id AND s.message_id = latest.message_id AND s.scored_at = latest.m
        JOIN items i ON i.chat_id = s.chat_id AND i.message_id = s.message_id
        WHERE s.direction IS NOT NULL
          AND (i.is_cluster_head = 1 OR i.is_cluster_head IS NULL)
        """,
        conn,
    )
    scores_raw["score"] = scores_raw["direction"] * scores_raw["magnitude"]
    scores_raw["channel"] = scores_raw["chat_id"].map(anon)

    # -- the index-eligible set (index.py's own default filters)
    idx_all = daily_index(conn, level=None, method="shrinkage")

    # -- market proxy, if reachable
    market_df = None
    market_note = None
    try:
        prices = market.load_prices(prices_db)
        if not prices.empty:
            market_df = market.market_proxy(prices)
    except Exception as e:
        market_note = f"시장 데이터를 불러오지 못했습니다 ({e}). --prices-db로 경로를 지정하세요."

    # ---- stats strip ----
    total_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    scorable = conn.execute(
        "SELECT COUNT(*) FROM items WHERE text IS NOT NULL OR extracted_text IS NOT NULL"
    ).fetchone()[0]
    scored_ok = conn.execute(
        """
        SELECT COUNT(*) FROM scores s
        JOIN (SELECT chat_id, message_id, MAX(scored_at) AS m FROM scores GROUP BY chat_id, message_id) latest
          ON s.chat_id = latest.chat_id AND s.message_id = latest.message_id AND s.scored_at = latest.m
        WHERE s.direction IS NOT NULL
        """
    ).fetchone()[0]
    coverage_pct = scored_ok / scorable * 100 if scorable else 0

    extract_incomplete = conn.execute(
        "SELECT COUNT(*) FROM items WHERE type IN ('document','photo') AND (extract_status IS NULL OR extract_status != 'ok')"
    ).fetchone()[0]

    from index import _scored_items  # internal helper, reused here rather than duplicated

    valid_df = _scored_items(conn, None, include_recap=False)
    n_valid = len(valid_df)

    median_n = idx_all["n_items"].median() if not idx_all.empty else 0

    stat_html = f"""
    <div class="stat-strip">
      <div class="stat"><span class="label">채점 커버리지</span><span class="value">{coverage_pct:.1f}%</span></div>
      <div class="stat"><span class="label">추출 미완료(scanned/pending/error)</span><span class="value">{extract_incomplete:,}건</span></div>
      <div class="stat"><span class="label">recap 제외 후 유효 항목</span><span class="value">{n_valid:,}건</span></div>
      <div class="stat"><span class="label">일별 표본수 중앙값</span><span class="value">{median_n:.0f}건</span></div>
    </div>
    """

    # ---- chart cards ----
    cards = []

    fig1 = build_index_vs_market_fig(idx_all, market_df, ewma_span)
    cards.append(f'<div class="chart-card"><h2>1. 논조지수 vs 시장 대용치</h2><div class="plot-wrap">{_to_html(fig1)}</div>'
                 + (f'<p class="note">{market_note}</p>' if market_note else '') + '</div>')

    if market_df is not None and not idx_all.empty:
        idx_s = idx_all.set_index("date")["index_value"]
        idx_s.index = pd.to_datetime(idx_s.index)
        mkt_s = market_df.set_index("date")["market_return"]
        fig2, leadlag_df = build_leadlag_fig(idx_s, mkt_s)
        cards.append(f'<div class="chart-card"><h2>2. Lead-lag 상관</h2><div class="plot-wrap">{_to_html(fig2)}</div>'
                     f'<p class="note">11개 lag를 동시에 검정하는 다중비교입니다 - 낱개 lag의 유의성만으로 결론 내리지 마세요.</p></div>')

    fig3 = build_score_hist_fig(scores_raw)
    n_pos_stock = int(((scores_raw["level"] == "stock") & (scores_raw["score"] > 0)).sum())
    n_neg_stock = int(((scores_raw["level"] == "stock") & (scores_raw["score"] < 0)).sum())
    cards.append(f'<div class="chart-card"><h2>3. 점수 분포</h2><div class="plot-wrap">{_to_html(fig3)}</div>'
                 f'<p class="note">개별종목(stock) 항목: 긍정 {n_pos_stock}건 vs 부정 {n_neg_stock}건 - 발신 채널이 호재 위주로 게시하는 경향을 반영할 수 있습니다.</p></div>')

    fig4 = build_heatmap_fig(scores_raw[["level", "score", "date"]])
    cards.append(f'<div class="chart-card"><h2>4. level × 주간 히트맵</h2><div class="plot-wrap">{_to_html(fig4)}</div></div>')

    channel_stats = (
        scores_raw.groupby("channel")
        .agg(avg_score=("score", "mean"), n_scored=("score", "size"))
        .reset_index()
    )
    total_by_channel = (
        pd.read_sql_query("SELECT chat_id, COUNT(*) AS n_total FROM items GROUP BY chat_id", conn)
        .assign(channel=lambda d: d["chat_id"].map(anon))
    )
    channel_stats = channel_stats.merge(total_by_channel[["channel", "n_total"]], on="channel", how="left")
    channel_stats["coverage_pct"] = channel_stats["n_scored"] / channel_stats["n_total"] * 100
    fig5 = build_channel_fig(channel_stats)
    cards.append(f'<div class="chart-card"><h2>5. 채널별 논조 캘리브레이션</h2><div class="plot-wrap">{_to_html(fig5)}</div></div>')

    fig7, ev_note = build_event_study_fig(conn, prices_db)
    if fig7 is not None:
        cards.append(f'<div class="chart-card"><h2>6. 이벤트 스터디 (CAR)</h2><div class="plot-wrap">{_to_html(fig7)}</div></div>')
    else:
        cards.append(f'<div class="chart-card"><h2>6. 이벤트 스터디 (CAR)</h2><p class="note">{ev_note}</p></div>')

    fig8, fb_note, fb_stats = build_finbert_comparison_fig(conn)
    if fig8 is not None:
        cards.append(f'<div class="chart-card"><h2>Gemini vs FinBERT 비교</h2><div class="plot-wrap">{_to_html(fig8)}</div></div>')
    else:
        cards.append(f'<div class="chart-card"><h2>Gemini vs FinBERT 비교</h2><p class="note">{fb_note}</p></div>')

    # ---- item table: last 150, client-side filtered ----
    items_json = json.loads(
        pd.read_sql_query(
            """
            SELECT substr(i.date,1,10) AS date, s.chat_id, s.level, s.driver, s.novelty,
                   s.direction, s.magnitude, s.summary
            FROM scores s
            JOIN (SELECT chat_id, message_id, MAX(scored_at) AS m FROM scores GROUP BY chat_id, message_id) latest
              ON s.chat_id = latest.chat_id AND s.message_id = latest.message_id AND s.scored_at = latest.m
            JOIN items i ON i.chat_id = s.chat_id AND i.message_id = s.message_id
            WHERE s.direction IS NOT NULL
            ORDER BY i.date DESC LIMIT 150
            """,
            conn,
        ).assign(
            channel=lambda d: d["chat_id"].map(anon),
            score=lambda d: d["direction"] * d["magnitude"],
        )[["date", "channel", "level", "driver", "novelty", "score", "summary"]].to_json(orient="records")
    )
    items_js = json.dumps(items_json, ensure_ascii=False)

    channels_present = sorted(scores_raw["channel"].dropna().unique().tolist())
    channel_options = "".join(f'<option value="{c}">{c}</option>' for c in channels_present)
    level_options = "".join(f'<option value="{lv}">{LEVEL_LABELS[lv]}</option>' for lv in LEVELS)

    table_section = f"""
    <div class="card">
      <h2>최근 채점 항목 (최근 150건)</h2>
      <div class="filters">
        <select id="f-channel"><option value="">채널 전체</option>{channel_options}</select>
        <select id="f-level"><option value="">범위 전체</option>{level_options}</select>
        <input id="f-date" type="text" placeholder="날짜 검색 (YYYY-MM-DD)">
      </div>
      <div style="overflow-x:auto">
        <table id="items-table">
          <thead><tr><th class="left">날짜</th><th class="left">채널</th><th class="left">범위</th><th class="left">동인</th><th>점수</th><th class="left">요약</th></tr></thead>
          <tbody id="items-tbody"></tbody>
        </table>
      </div>
    </div>
    <script>
    const ITEMS = {items_js};
    const LEVEL_LABELS = {json.dumps(LEVEL_LABELS, ensure_ascii=False)};
    function renderItems() {{
      var ch = document.getElementById('f-channel').value;
      var lv = document.getElementById('f-level').value;
      var dt = document.getElementById('f-date').value.trim();
      var rows = ITEMS.filter(function (it) {{
        if (ch && it.channel !== ch) return false;
        if (lv && it.level !== lv) return false;
        if (dt && (it.date || '').indexOf(dt) === -1) return false;
        return true;
      }});
      var body = document.getElementById('items-tbody');
      body.innerHTML = rows.map(function (it) {{
        var score = (it.score === null || it.score === undefined) ? '-' : (it.score > 0 ? '+' : '') + it.score;
        var chip = it.score > 0 ? 'pos' : (it.score < 0 ? 'neg' : 'neu');
        var lvLabel = LEVEL_LABELS[it.level] || (it.level || '-');
        return '<tr><td class="left">' + (it.date || '') + '</td><td class="left">' + it.channel +
               '</td><td class="left">' + lvLabel + '</td><td class="left">' + (it.driver || '-') +
               '</td><td><span class="chip ' + chip + '">' + score + '</span></td><td class="left">' + (it.summary || '') + '</td></tr>';
      }}).join('');
    }}
    document.getElementById('f-channel').addEventListener('change', renderItems);
    document.getElementById('f-level').addEventListener('change', renderItems);
    document.getElementById('f-date').addEventListener('input', renderItems);
    renderItems();
    </script>
    """

    cards_html = "".join(cards)

    return f"""
{STYLE}
<div class="page">
  <div>
    <p class="eyebrow">Research Digest · Research Tone Index</p>
    <h1>리서치 채널 논조 지수 (Research Tone Index)</h1>
    <p class="meta">텔레그램 채널 {len(channels_present)}곳의 논조를 집계한 것으로, 실제 시장 심리를 직접 측정한 값이 아닙니다 ·
    생성 시각 {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
  </div>

  {stat_html}

  {cards_html}

  <div class="card">
    <h2>채점 기준 (v2)</h2>
    <p>각 항목을 Gemini API(<code>score_llm.py</code>)가 읽고 여러 축으로 분해해서 구조화된 JSON으로 반환합니다.
    <strong>direction</strong>(-1/0/+1, 방향)과 <strong>magnitude</strong>(0-3, 강도)를 곱한 값이 최종 점수(-3..+3)입니다.
    <strong>confidence</strong>(0-1)는 판단 확신도, <strong>level</strong>은 매크로/시장/섹터/종목 층위,
    <strong>driver</strong>는 근본 동인(통화정책/실적/수급/지정학/환율/규제/밸류에이션/공급망/원자재/기타),
    <strong>novelty</strong>는 새 정보(new)/사후요약(recap)/재게시(repost) 구분, <strong>horizon</strong>은 영향 시간범위입니다.
    판단이 애매한 항목은 억지로 채점하지 않고 direction을 결측치(null)로 남깁니다.</p>
    <div class="scale">
      <span class="band">score = direction × magnitude (-3..+3)</span>
      <span class="band">recap 항목은 지수 집계에서 기본 제외 (사후요약 순환참조 방지)</span>
      <span class="band">confidence &lt; 0.4 항목도 지수 집계에서 제외</span>
      <span class="band">동일 사안을 여러 채널이 다루면 dedupe.py가 대표 1건만 집계 (mention_channels로 컨센서스 강도 별도 추적)</span>
    </div>
    <p style="margin-top:0.75rem">비교용으로 <strong>KR-FinBERT</strong>(<code>score_finbert.py</code>, rubric_version=<code>finbert-v1</code>)도 같은 항목을 채점합니다 -
    다만 순수 3분류(긍정/중립/부정) 모델이라 level/driver/ticker 등은 만들 수 없어 direction/magnitude/confidence만 채워집니다.
    아래 "Gemini vs FinBERT 비교" 차트 참고.</p>
    <h2 style="margin-top:1.5rem">한계</h2>
    <ul class="limits">
      <li><strong>발신자 편향</strong>: 종목(stock) 레벨 항목은 위 분포 차트에서 보듯 긍정 편향이 뚜렷합니다 - 리서치 채널이 매수의견/호재 위주로 게시하는 경향의 결과로 보입니다. 그래서 이 지수의 이름도 "시장 심리"가 아니라 <strong>"리서치 채널 논조"</strong>입니다 - 실제로 측정하는 건 시장의 심리가 아니라 이 채널들의 발신 논조입니다.</li>
      <li><strong>채널 커버리지 불균등</strong>: 채널별 캘리브레이션 차트에서 보듯 채널마다 게시량·채점률 편차가 큽니다. 특정 채널의 논조가 전체 지수를 과대표현할 수 있습니다.</li>
      <li><strong>시장 대용치의 근사성</strong>: KOSPI 지수 자체가 아니라 가격이력이 충분한 종목들의 동일가중 수익률 근사치입니다 (market.py 참고). 시가총액 가중이 아니므로 실제 지수와 괴리가 있을 수 있습니다.</li>
      <li><strong>표본 수 문제</strong>: 일별 표본수 중앙값이 통계 스트립에 표시됩니다 - 표본이 적은 날의 지수는 shrinkage 방식으로 0에 가깝게 눌러도 여전히 노이즈가 큽니다.</li>
      <li><strong>다중 비교 문제</strong>: lead-lag 차트는 11개 lag를 동시에 검정합니다 - 낱개 lag 하나가 유의해 보여도 우연일 가능성을 배제할 수 없습니다.</li>
      <li>이 대시보드는 어떤 상관관계도 "예측력 있음"으로 단정하지 않습니다 - lead-lag/이벤트 스터디 결과를 있는 그대로 표시할 뿐입니다.</li>
    </ul>
  </div>

  {table_section}

  <footer>
    <code>collect.py → extract.py → score_llm.py → dedupe.py → index.py → validate.py → dashboard.py</code>
    순서로 매일 갱신됩니다. 채점 기준 원문은 <code>rubric.py</code>, 검증 로직은 <code>validate.py</code>, 중복 판정은
    <code>dedupe.py</code>를 참고하세요.
  </footer>
</div>
{THEME_SCRIPT}
"""


def render_dashboard(conn, prices_db: str, ewma_span: int, embed_plotly: bool) -> str:
    content = build_content(conn, prices_db, ewma_span)
    if embed_plotly:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>리서치 채널 논조 지수</title>
{plotly_tag}
</head>
<body>
{content}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output HTML path (default: output/dashboard.html)")
    parser.add_argument("--no-docs", action="store_true", help="Skip writing docs/index.html (GitHub Pages source)")
    parser.add_argument("--prices-db", default=str(market.DEFAULT_PRICES_DB), help="Path to companion prices.db")
    parser.add_argument("--ewma-span", type=int, default=10, help="EWMA span for the sentiment line (default 10)")
    parser.add_argument("--embed-plotly", action="store_true", help="Inline plotly.js instead of CDN (offline use, larger file)")
    args = parser.parse_args()

    conn = get_connection()
    html = render_dashboard(conn, args.prices_db, args.ewma_span, args.embed_plotly)
    conn.close()

    out_path = Path(args.out) if args.out else OUT / "dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")

    if not args.no_docs:
        DOCS_DIR.mkdir(exist_ok=True)
        docs_path = DOCS_DIR / "index.html"
        docs_path.write_text(html, encoding="utf-8")
        print(f"Wrote {docs_path}")


if __name__ == "__main__":
    main()
