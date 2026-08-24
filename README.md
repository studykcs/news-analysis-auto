# news-analysis-auto

Collects PDF/photo/text research posts from Telegram channels into SQLite, scores
each item on multiple axes via an LLM, aggregates that into a daily sentiment
index, and (optionally) validates the index against realized KRX returns - all
rendered into one self-contained HTML dashboard.

## Setup

1. **Install dependencies**
   ```
   pip install telethon python-dotenv pandas plotly google-genai pymupdf statsmodels scipy numpy
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
python dashboard.py       # render output/dashboard.html + docs/index.html
```

Or run all four automatically every day - see **Daily automation** below.

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
| `direction` | -1 / 0 / +1 - which way it pushes sentiment (null = couldn't judge, not the same as 0) |
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

## Aggregation (`index.py`)

Never asks the LLM to average anything - all aggregation is plain pandas/SQL.
Default filters: `novelty='recap'` excluded (a same-day recap post restates that
day's return, so including it makes the index correlate with the market almost
by construction - see `validate.py`'s contemporaneous check), `confidence < 0.4`
excluded, `direction IS NULL` excluded (`--include-recap` re-includes recap for
comparison). Three aggregation methods, side by side:

- **shrinkage** - `n/(n+k) * mean_score` pulls low-sample days toward 0 instead
  of letting a 1-item day swing as hard as an 80-item day
- **channel_demeaned** - subtracts each channel's own all-time mean before
  aggregating, since one channel supplies roughly half of all scored items
- **breadth** - `(positive - negative) / total`, -1..+1, a magnitude-blind
  cross-check against the other two

Levels are never blended with an implicit weight - `daily_index(level=...)` gives
each level's own series; `--weights macro=0.4,market=0.3,sector=0.2,stock=0.1`
opts into an explicit composite.

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

## Calibration (`calibration.py`)

`python calibration.py` generates `calibration/gold.csv` (a random sample of
scored items with blank answer columns) the first time it's run, if the file
doesn't exist yet. Fill in `direction`/`magnitude`/`level` by hand, then re-run
to get direction accuracy, magnitude MAE, a level confusion matrix, and Spearman
correlation - optionally comparing two `rubric_version`s side by side
(`--rubric-version v1 --compare v2`). `calibration/gold.csv` is gitignored (a
personal, hand-labeled file, not project data).

## Dashboard

Single HTML file, CSS design tokens for light/dark, Plotly charts loaded from
CDN by default (`--embed-plotly` inlines plotly.js instead, for fully offline
use - much larger file). Chart backgrounds are transparent and re-colored at
runtime by a small theme-sync script (Plotly renders to a static canvas, so it
can't read CSS variables at draw time). Channel names are anonymized
(채널1, 채널2, ...) since `chat_name` often contains a real analyst's name and
this dashboard is published.

## Daily automation

`run_pipeline.ps1` runs collect → extract → score_llm → dashboard → git commit/push
in sequence, so a Windows Task Scheduler job can drive the whole pipeline
unattended. See the script for what it assumes (a `python` on PATH, a git remote
already configured).
