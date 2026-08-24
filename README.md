# news-analysis-auto

Collects PDF/photo/text research posts from Telegram channels into SQLite, and renders a self-contained HTML dashboard with sentiment scoring.

## Setup

1. **Install dependencies**
   ```
   pip install telethon python-dotenv pandas plotly
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

## Usage

```
python collect.py         # fetch new items, save files/, upsert SQLite
python summarize.py       # score unscored items via Gemini API
python dashboard.py       # render output/dashboard.html
```

Or run all three automatically every day - see **Daily automation** below.

## Sentiment scoring

Every text-bearing item gets three fields filled in by `summarize.py` (Gemini API, one call per unscored day, all channels batched together):

| Field | Meaning |
|---|---|
| `sentiment_score` | Integer -5..+5: how bullish/bearish the item is for that day's market |
| `scope` | `macro` (거시/통화정책/지정학/환율) · `market` (코스피·코스닥 등 시장 전반) · `sector` (업종 단위) · `stock` (개별 종목) |
| `topic` | Free-text tag, e.g. "금리정책", "삼성전자" |

**`sentiment_score` scale:**

| Score | Meaning |
|---|---|
| -5, -4 | 매우 부정 (급락/위기) |
| -3, -2 | 부정 (하락·우려 지배적) |
| -1 | 약한 부정/경계 |
| 0 | 중립/정보성 |
| +1 | 약한 긍정 |
| +2, +3 | 긍정 (실적 서프라이즈 등) |
| +4, +5 | 매우 긍정 (사상 최대 등) |

A day's overall score is the simple average of that day's item scores (see `dashboard.py`). The exact rubric text sent to Gemini lives in `summarize.py` (`RUBRIC`) and is also shown on the dashboard's "채점 기준" card.

Items without text (pure photo/document, no caption) are not scored - `summarize.py` only processes rows where `text IS NOT NULL`.

## Daily automation

`run_pipeline.ps1` runs collect → summarize → dashboard → git commit/push in sequence, so a Windows Task Scheduler job can drive the whole pipeline unattended. See the script for what it assumes (a `python` on PATH, a git remote already configured).
