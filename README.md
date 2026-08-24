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
   ```

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
python collect.py        # fetch new items, save files/, upsert SQLite
python dashboard.py       # render output/dashboard.html
```

## Sentiment scoring

`sentiment_score` (-5..+5) and `scope` (`macro`/`market`/`sector`/`stock`) are **not computed automatically** by this code - they're filled in by manually asking an LLM (e.g. Claude) to read a day's items and call `store.update_summary(...)` for each. See the dashboard's "채점 기준" card for the exact rubric used.
