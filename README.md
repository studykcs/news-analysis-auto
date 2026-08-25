# news-analysis-auto

**Dashboard: https://studykcs.github.io/news-analysis-auto/**

Collects PDF/photo/text research posts from Telegram channels into SQLite, scores
each item on multiple axes via an LLM, deduplicates stories multiple channels
cover simultaneously, aggregates the rest into a daily **Research Tone Index**,
and validates that index against realized KRX returns - all rendered into one
self-contained HTML dashboard.

**Why "Research Tone Index" and not "sentiment index"**: what this pipeline
actually measures is the tone of what a handful of brokerage research channels
choose to publish, not the market's sentiment. Those aren't the same thing -
see **Limitations** below - so the name says what it is, not what it sounds like.

## Where each piece lives

| Component | File |
|---|---|
| Telegram collection (PDF / photo / text) into SQLite | [`collect.py`](collect.py) |
| **PDF text extraction** (PyMuPDF) | [`extract.py`](extract.py) |
| **Gemini API** structured multi-axis scoring | [`score_llm.py`](score_llm.py) · [`rubric.py`](rubric.py) |
| **KR-FinBERT** local scorer (comparison rubric) | [`score_finbert.py`](score_finbert.py) |
| Near-duplicate clustering (SimHash + rapidfuzz, Union-Find) | [`dedupe.py`](dedupe.py) |
| Daily index: **shrinkage**, channel-demeaning, breadth | [`index.py`](index.py) |
| KRX price loading / market proxy | [`market.py`](market.py) |
| **Lead-lag** correlation + Newey-West (HAC) regression, **CAR event study** | [`validate.py`](validate.py) |
| Versioned scoring schema (`rubric_version`) | [`store.py`](store.py) |
| Self-contained HTML dashboard | [`dashboard.py`](dashboard.py) |

Scores are stored **versioned** by `rubric_version`, so the Gemini (`v2`) and
KR-FinBERT (`finbert-v1`) rubrics coexist and stay comparable instead of
overwriting each other — see **Alternative scorer: KR-FinBERT** below.

## Setup

1. **Install dependencies**
   ```
   pip install telethon python-dotenv pandas plotly google-genai pymupdf statsmodels scipy numpy rapidfuzz
   ```
   Optional, only for `score_finbert.py` (see **Alternative scorer**):
   ```
   pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Get a Telegram API key** (personal account, not a bot) from <https://my.telegram.org> → "API development tools". Free, takes a minute.

3. **Create `.env`** (copy `.env.example`):
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_PHONE=+1xxxxxxxxxx
   GEMINI_API_KEY=your_gemini_api_key
   ```
   Get a Gemini key free at <https://aistudio.google.com/apikey>.

4. **Create `channels.json`** (copy `channels.example.json`) with the channels you want to collect from. To find a channel's numeric ID, log in once and list your dialogs:
   ```python
   from telethon.sync import TelegramClient
   from dotenv import load_dotenv
   import os
   load_dotenv()
   client = TelegramClient("telegram_digest", int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"])
   with client:
       for d in client.iter_dialogs(limit=30):
           print(d.id, d.name)
   ```

5. **First run** — `python collect.py` will ask for a login code sent to your Telegram app (one-time; a `telegram_digest.session` file is saved after). By default the first sync only goes back 30 days per channel (`INITIAL_BACKFILL_DAYS` in `collect.py`); later runs are incremental.

## Pipeline

```
python collect.py         # fetch new items, save files/, upsert SQLite
python extract.py         # pull text out of downloaded PDFs (first 2 pages)
python score_llm.py       # score unscored items via Gemini API (v2 rubric)
python dedupe.py          # cluster near-duplicate cross-channel posts
python index.py           # print the daily Research Tone Index (or --out a CSV)
python validate.py all    # check the index against realized KRX returns
python dashboard.py       # render output/dashboard.html + docs/index.html
```

(`dedupe.py` has to run before `index.py` - the aggregation reads
`items.is_cluster_head`, which only `dedupe.py` sets. It's not in the order the
original request for this pipeline spelled out, but leaving it out would let the
double-counting bug it exists to fix creep back in.)

Or run the daily subset automatically - see **Daily automation** below.

`extract.py` exists because a PDF post with little/no caption is otherwise invisible
to scoring - see its module docstring for the scanned-PDF / photo-OCR caveats.

## Scoring (v2)

Scores live in a separate, **versioned** `scores` table (see `store.py`), not as
columns on `items` - a rubric change never overwrites or mixes with an older
rubric's results for the same item; multiple `rubric_version`s can coexist and
be compared. `migrate_scores.py` is the one-off script that moved the original
inline `items.sentiment_score` float into `scores` as `rubric_version='v1'`.

`rubric.py` holds the actual prompt (`SYSTEM_INSTRUCTION` + `ANCHOR_EXAMPLES`),
kept separate from `score_llm.py` so a wording tweak doesn't touch request/parsing
logic. `score_llm.py` calls Gemini once per unscored day (all channels batched
together - `ANCHOR_EXAMPLES` in the system instruction is what keeps calibration
stable across day-batches) and writes one row per item, even on partial failure.

Each scored item gets:

| Field | Meaning |
|---|---|
| `direction` | -1 / 0 / +1 - which way it pushes tone (null = couldn't judge, not the same as 0) |
| `magnitude` | 0-3 - how strong a signal |
| `confidence` | 0.0-1.0 - how confident the judgment is |
| `level` | `macro` / `market` / `sector` / `stock` |
| `sector_code` | practical fixed tag list, see `rubric.SECTOR_CODES` (not an official KRX code - see its docstring) |
| `ticker` | 6-digit KRX code, only if explicitly named in the text |
| `driver` | `monetary_policy` / `earnings` / `flows` / `geopolitics` / `fx` / `regulation` / `valuation` / `supply_chain` / `commodity` / `other` |
| `novelty` | `new` (fresh info/opinion) / `recap` (post-hoc restates a move that already happened) / `repost` |
| `horizon` | `intraday` / `short` / `medium` |
| `summary` | one Korean sentence |

`score = direction * magnitude` (range -3..+3) is **never stored** - it's derived
on read in `store.latest_scores()` / `index.py`, so a future rescale needs no
backfill migration.

### Alternative scorer: KR-FinBERT (`score_finbert.py`)

A second, independent `rubric_version='finbert-v1'`, scored locally by
`snunlp/KR-FinBert-SC` (a 3-class positive/neutral/negative classifier) instead
of calling Gemini - free, no API cost, but a plain classifier can't produce
`level`/`sector_code`/`ticker`/`driver`/`novelty`/`horizon`, so those columns
stay NULL for every `finbert-v1` row; only `direction`/`magnitude`/`confidence`
are populated (see the module docstring for the confidence→magnitude bucketing,
a hand-picked heuristic, not a calibrated one). This does **not** replace
`score_llm.py` - both coexist in `scores`, and the dashboard's "Gemini vs FinBERT
비교" card plots them against each other (agreement rate, correlation) once both
have scored the same items. `run_pipeline.ps1` does not run it by default -
run it manually, or add it to the pipeline yourself once you've checked the
agreement rate is good enough for your use.

**Agreement, measured on all 1,537 items scored under both rubrics:**

| | |
|---|---|
| items scored under both | 1,373 (both `direction` non-null) |
| **direction agreement** | **41%** |
| **score correlation** (`direction × magnitude`) | **+0.35** |

The two agree far less than the phrase "both are sentiment models" suggests.
That is the point of keeping both: a structured-extraction LLM reading a full
research note and a 3-class classifier reading the same text are not
measuring the same thing, and the index would look materially different
depending on which one you believed. Neither is validated against a human
gold standard here — `calibration.py` exists for exactly that, and until it
is run on a labeled sample, the 41% is a disagreement rate, not evidence that
either one is right.

## Deduplication (`dedupe.py`)

Six channels sometimes cover the same story within a day of each other. Left
uncounted, a story six channels mention would be worth 6x a story one channel
mentions in every aggregate - purely because it was popular to report, not
because it's six times as important. Two-stage near-duplicate matching within a
±1-day window (SimHash to narrow candidates, then a rapidfuzz character-ratio
check to confirm; PDFs additionally get an exact-fingerprint check via extracted
text + char count) clusters matching items and picks one representative
(`is_cluster_head=1`) per cluster; `index.py`'s aggregation only counts cluster
heads by default.

Duplicates aren't discarded, though - **how many channels independently mention
the same story is a real consensus signal**, stored as `mention_channels` on the
cluster head. `python index.py --weight-by-consensus` uses it as a per-item
weight instead of just deduplicating it away.

**What we actually found**: on this dataset, genuine near-duplicate *text*
across different brokerage houses is rare - each house writes its own
independent take even on the identical underlying news event (e.g. Samsung's
₩110T buyback was covered by three different channels the same day, with
character-similarity in the 3-30% range between them - nowhere near the 85%
confirmation threshold). The one true cross-channel duplicate `dedupe.py` found
in 1,662 items was two *sister* channels of the same brokerage house
(한화투자증권 / 한화 Research) posting byte-identical text. Competing houses
apparently don't copy each other's writing, even when covering the same event -
so `--weight-by-consensus` currently has almost no items to act on. The
infrastructure is still worth having: it will catch real reposts (a channel
literally re-sharing another's PDF, a channel re-posting its own earlier item)
whenever they occur, and this finding itself is reported as-is rather than
tuning the similarity thresholds until "duplicates" appear.

## Aggregation (`index.py`)

Never asks the LLM to average anything - all aggregation is plain pandas/SQL.
Default filters: `novelty='recap'` excluded (a same-day recap post restates that
day's return, so including it makes the index correlate with the market almost
by construction - see **Validation results** below), `confidence < 0.4`
excluded, `direction IS NULL` excluded, non-cluster-head duplicates excluded
(`--include-recap` and `--force`-rescoring aside, see `dedupe.py`). Three
aggregation methods, side by side:

- **shrinkage** - `n/(n+k) * mean_score` pulls low-sample days toward 0 instead
  of letting a 1-item day swing as hard as an 80-item day
- **channel_demeaned** - subtracts each channel's own all-time mean before
  aggregating, since one channel supplies roughly half of all scored items
- **breadth** - `(positive - negative) / total`, -1..+1, a magnitude-blind
  cross-check against the other two

Levels are never blended with an implicit weight - `daily_index(level=...)` gives
each level's own series; `--weights macro=0.4,market=0.3,sector=0.2,stock=0.1`
opts into an explicit composite. `--weight-by-consensus` weights each
(deduplicated) story by `mention_channels` instead of counting every story once
- see **Deduplication** above for why its effect is currently small.

## Validation (`market.py` + `validate.py`)

The index means nothing until it's checked against what the market actually did.
`market.py` builds an equal-weighted return proxy from a **companion** KRX price
database (`--prices-db`, schema `prices(date, ticker, close)` / `tickers(ticker,
name)`) - **not** a true market-cap-weighted index; see its docstring for why
(no shares-outstanding data available). `validate.py` runs three checks and
reports results as computed, including null results:

```
python validate.py contemporaneous   # corr(index_t, market_t), recap in vs out
python validate.py leadlag           # corr(index_t, market_t+k) k=-5..5 + NW regression
python validate.py eventstudy        # ticker-level CAR t-5..t+10 by direction
python validate.py all
```

### Validation results (level=market, first real run)

Reported as computed - a negative or null result here is not a bug to fix, it's
the answer.

- **Contemporaneous**: `corr(index_t, market_t)` was **+0.608** (n=21) with
  `novelty='recap'` posts included, and collapsed to **+0.115** (n=19) with them
  excluded. Both are small samples. The size of that drop is itself the finding:
  most of the naive correlation was the index echoing that day's own recap
  posts, not saying anything ahead of the market - confirming the reason recap
  is excluded from the index by default.
- **Lead-lag**: across k=-5..+5, correlations ranged roughly -0.38 to +0.24 with
  no consistent sign or monotonic pattern (n=16-19 per lag - 11 lags tested at
  once, a multiple-comparisons exposure, so no single lag here should be read as
  "the" answer). The Newey-West-adjusted regression of `return_t+1` on
  `index_t` and `return_t` found `index_t`'s coefficient not significant
  (t=-1.22, p=0.224, n=19); `return_t`'s own coefficient was significant
  (t=+2.61, p=0.009) - i.e. today's return predicted tomorrow's better than the
  tone index did, on this small sample.
- **Event study**: ticker-tagged items with `direction=+1` (n=144) showed no
  significant CAR at any horizon from t=0 to t+10 (all |t|<1.4, p>0.17).
  `direction=-1` items (n=20, flagged small-sample) showed a significant,
  persistent *negative* CAR through t+10 (p<0.05 at every horizon shown) - i.e.
  negative-toned items were followed by real underperformance in this sample,
  but positive-toned items were not followed by outperformance. Take the size
  of that n=20 group seriously before reading much into it.

None of this establishes predictive power at this sample size - it establishes
that the recap-driven circularity hypothesis (see **Limitations**) is real and
now filtered out by default, and that the *negative*-tone signal is the one
worth watching as more data accumulates, not the positive one.

## Calibration (`calibration.py`)

`python calibration.py` generates `calibration/gold.csv` (a random sample of
scored items with blank answer columns) the first time it's run, if the file
doesn't exist yet. Fill in `direction`/`magnitude`/`level` by hand, then re-run
to get direction accuracy, magnitude MAE, a level confusion matrix, and Spearman
correlation - optionally comparing two `rubric_version`s side by side
(`--rubric-version v1 --compare v2`, or `finbert-v1`). `calibration/gold.csv` is
gitignored (a personal, hand-labeled file, not project data).

## Dashboard

Single HTML file, CSS design tokens for light/dark, Plotly charts loaded from
CDN by default (`--embed-plotly` inlines plotly.js instead, for fully offline
use - much larger file; note the CDN version must be plotly.js's own version,
e.g. 3.7.0, not the pip package's version, e.g. 6.9.0 - they're numbered
independently, and `dashboard.py` reads the real one out of the bundled JS
rather than hardcoding it). Chart backgrounds are transparent and re-colored at
runtime by a small theme-sync script (Plotly renders to a static canvas, so it
can't read CSS variables at draw time). Channel names are anonymized
(채널1, 채널2, ...) since `chat_name` often contains a real analyst's name and
this dashboard is published.

## Limitations

- **Sender bias**: research channels skew toward publishing bullish/buy-rated
  content on individual stocks - the score-distribution chart makes this
  visible rather than hiding it (roughly 4-5x more positive than negative
  stock-level items in this dataset).
- **Channel coverage imbalance**: one channel supplies a large share of all
  scored items; `channel_demeaned` exists specifically to reduce this, but
  doesn't erase it, and the raw per-channel calibration chart shows it directly.
- **Market proxy approximation**: `market.py`'s equal-weighted proxy is not
  KOSPI/KOSDAQ - see its docstring for why a true cap-weighted index isn't
  buildable from the available data.
- **Sample size**: many days have single-digit item counts even after
  deduplication; `shrinkage` softens but does not eliminate the noise this
  creates. The dashboard's median daily sample size stat exists so this isn't
  invisible.
- **Multiple comparisons**: `validate.py leadlag` tests 11 lags at once; nothing
  in this pipeline corrects for that beyond stating it plainly wherever the
  result is shown.
- This pipeline measures **research channel tone**, not market sentiment - see
  the note at the top of this README.

## Daily automation

`run_pipeline.ps1` runs collect → extract → score_llm → dedupe → dashboard →
git commit/push in sequence, so a Windows Task Scheduler job can drive the
pipeline unattended. `score_finbert.py` and `validate.py` are not part of the
daily run (the former needs a one-time multi-GB torch/transformers install and
adds real runtime for a comparison-only score; the latter is a diagnostic you
read, not a state that needs updating every day) - run them by hand when you
want them. See `run_pipeline.ps1` for what it assumes (a `python` on PATH, a
git remote already configured).
