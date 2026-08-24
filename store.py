"""SQLite storage for collected Telegram research items."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "digest.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    chat_id INTEGER NOT NULL,
    chat_name TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    type TEXT NOT NULL,          -- 'photo' | 'document' | 'text' | 'link'
    text TEXT,                   -- message text or caption
    file_path TEXT,              -- local saved path, if any media
    file_name TEXT,              -- original filename, if any
    url TEXT,                    -- extracted link, if any
    summary TEXT,                -- filled in later by the summarization step
    summarized_at TEXT,
    sentiment_score REAL,        -- -5 (very bearish) .. +5 (very bullish), per item
    scope TEXT,                  -- 'macro' | 'market' | 'sector' | 'stock'
    topic TEXT,                  -- free-text tag, e.g. '금리정책', '반도체', '삼성전자'
    PRIMARY KEY (chat_id, message_id)
);
"""

SCOPES = ("macro", "market", "sector", "stock")


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    if "sentiment_score" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN sentiment_score REAL")
    if "scope" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN scope TEXT")
    if "topic" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN topic TEXT")
    return conn


def update_summary(
    conn: sqlite3.Connection,
    chat_id: int,
    message_id: int,
    summary: str,
    sentiment_score: float,
    scope: str,
    topic: str,
) -> None:
    assert scope in SCOPES, f"scope must be one of {SCOPES}, got {scope!r}"
    conn.execute(
        """
        UPDATE items SET summary = ?, sentiment_score = ?, scope = ?, topic = ?,
                          summarized_at = datetime('now')
        WHERE chat_id = ? AND message_id = ?
        """,
        (summary, sentiment_score, scope, topic, chat_id, message_id),
    )
    conn.commit()


def last_message_id(conn: sqlite3.Connection, chat_id: int) -> int:
    """Highest message_id already stored for this chat, or 0 if none yet."""
    row = conn.execute(
        "SELECT MAX(message_id) FROM items WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row[0] or 0


def insert_item(conn: sqlite3.Connection, item: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO items
            (chat_id, chat_name, message_id, date, type, text, file_path, file_name, url)
        VALUES (:chat_id, :chat_name, :message_id, :date, :type, :text, :file_path, :file_name, :url)
        """,
        item,
    )


def items_without_summary(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM items WHERE summary IS NULL ORDER BY date"
    ).fetchall()
