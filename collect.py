"""Fetch new items from configured Telegram research channels and store them locally.

Downloads PDFs/photos/documents into files/<channel>/<YYYY-MM>/, extracts
message text and links, and upserts everything into SQLite (see store.py).
Summarization is a separate step (not run here) - new rows are inserted with
summary=NULL.

Channels to collect from are listed in channels.py. Message IDs are only
unique within a chat, so each channel is synced independently.

First run requires an interactive login: Telegram sends a one-time code to
your app, which you type into the terminal. After that a local session file
(telegram_digest.session) keeps you logged in - do not delete it, and never
commit it (see .gitignore). Re-running only fetches messages newer than the
last one already stored, per channel.

Usage
-----
    python collect.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl

from channels import CHANNELS
from store import get_connection, insert_item, last_message_id

FILES_DIR = Path(__file__).parent / "files"
URL_RE = re.compile(r"https?://\S+")

# First-ever sync per channel only goes back this far. Later runs are
# incremental (via min_id) and ignore this.
INITIAL_BACKFILL_DAYS = 30


def extract_url(message) -> str | None:
    if message.entities:
        for entity in message.entities:
            if isinstance(entity, MessageEntityTextUrl):
                return entity.url
            if isinstance(entity, MessageEntityUrl) and message.text:
                offset, length = entity.offset, entity.length
                return message.text[offset : offset + length]
    if message.text:
        found = URL_RE.search(message.text)
        if found:
            return found.group(0)
    return None


def classify_and_download(client: TelegramClient, message, chat_id: int, chat_name: str) -> dict:
    month_dir = FILES_DIR / chat_name / message.date.strftime("%Y-%m")
    file_path = None
    file_name = None
    item_type = "text"

    if message.photo:
        item_type = "photo"
        month_dir.mkdir(parents=True, exist_ok=True)
        file_path = client.download_media(message, file=str(month_dir / f"{message.id}.jpg"))
        file_name = Path(file_path).name if file_path else None
    elif message.document:
        item_type = "document"
        month_dir.mkdir(parents=True, exist_ok=True)
        file_path = client.download_media(message, file=str(month_dir) + "/")
        file_name = Path(file_path).name if file_path else None

    url = extract_url(message)
    if item_type == "text" and url:
        item_type = "link"

    return {
        "chat_id": chat_id,
        "chat_name": chat_name,
        "message_id": message.id,
        "date": message.date.astimezone(timezone.utc).isoformat(),
        "type": item_type,
        "text": message.text or None,
        "file_path": file_path,
        "file_name": file_name,
        "url": url,
    }


def main() -> None:
    load_dotenv()
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    phone = os.environ.get("TELEGRAM_PHONE")
    if not (api_id and api_hash and phone):
        raise SystemExit(
            "Missing TELEGRAM_API_ID/TELEGRAM_API_HASH/TELEGRAM_PHONE. "
            "Copy .env.example to .env and fill it in."
        )

    conn = get_connection()
    client = TelegramClient("telegram_digest", int(api_id), api_hash)

    with client:
        client.start(phone=phone)

        for chat_id, chat_name in CHANNELS.items():
            since_id = last_message_id(conn, chat_id)
            # Only bound the very first sync of a channel; later syncs rely on min_id.
            offset_date = (
                datetime.now(timezone.utc) - timedelta(days=INITIAL_BACKFILL_DAYS)
                if since_id == 0
                else None
            )
            count = 0
            for message in client.iter_messages(
                chat_id, min_id=since_id, offset_date=offset_date, reverse=True
            ):
                if not (message.text or message.photo or message.document):
                    continue
                item = classify_and_download(client, message, chat_id, chat_name)
                insert_item(conn, item)
                count += 1
            conn.commit()
            print(f"[{chat_name}] +{count} new item(s) since message_id {since_id}")


if __name__ == "__main__":
    main()
