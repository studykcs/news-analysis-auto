"""SQLite storage for collected Telegram research items."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

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
    extracted_text TEXT,         -- text pulled from a PDF/document file, if any
    extract_status TEXT,         -- 'ok' | 'scanned' | 'image_pending' | 'no_file' | 'error'
    extracted_at TEXT,
    extract_chars INTEGER,       -- len(extracted_text), for quality monitoring
    PRIMARY KEY (chat_id, message_id)
);
"""

SCOPES = ("macro", "market", "sector", "stock")
EXTRACT_STATUSES = ("ok", "scanned", "image_pending", "no_file", "error")

# scores is a separate, versioned table (not columns on items) so a rubric
# change never overwrites what an older rubric said about the same item -
# old and new scores stay comparable side by side. score itself (-3..+3) is
# never stored: it's always direction*magnitude, derived on read in
# latest_scores(), so rescaling the rubric later needs no backfill.
SCORES_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    rubric_version TEXT NOT NULL,   -- e.g. 'v1'
    model TEXT NOT NULL,            -- e.g. 'gemini-2.5-flash'
    direction INTEGER,              -- -1 | 0 | +1
    magnitude INTEGER,              -- 0..3
    confidence REAL,                -- 0..1
    level TEXT,                     -- 'macro' | 'market' | 'sector' | 'stock'
    sector_code TEXT,               -- KRX sector code (fixed enum), NULL if n/a
    ticker TEXT,                    -- 6-digit code, only if explicitly named
    driver TEXT,                    -- fixed enum, see DRIVERS
    novelty TEXT,                   -- 'new' | 'recap' | 'repost'
    horizon TEXT,                   -- 'intraday' | 'short' | 'medium'
    summary TEXT,
    raw_response TEXT,              -- raw LLM response, verbatim
    scored_at TEXT NOT NULL,
    PRIMARY KEY (chat_id, message_id, rubric_version)
);
"""

DRIVERS = (
    "monetary_policy", "earnings", "flows", "geopolitics", "fx",
    "regulation", "valuation", "supply_chain", "commodity", "other",
)
NOVELTY_VALUES = ("new", "recap", "repost")
HORIZON_VALUES = ("intraday", "short", "medium")


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.execute(SCORES_SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    if "sentiment_score" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN sentiment_score REAL")
    if "scope" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN scope TEXT")
    if "topic" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN topic TEXT")
    if "extracted_text" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN extracted_text TEXT")
    if "extract_status" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN extract_status TEXT")
    if "extracted_at" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN extracted_at TEXT")
    if "extract_chars" not in existing_cols:
        conn.execute("ALTER TABLE items ADD COLUMN extract_chars INTEGER")
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


def update_extraction(
    conn: sqlite3.Connection,
    chat_id: int,
    message_id: int,
    extracted_text: str | None,
    extract_status: str,
) -> None:
    assert extract_status in EXTRACT_STATUSES, f"extract_status must be one of {EXTRACT_STATUSES}, got {extract_status!r}"
    conn.execute(
        """
        UPDATE items SET extracted_text = ?, extract_status = ?, extract_chars = ?,
                          extracted_at = datetime('now')
        WHERE chat_id = ? AND message_id = ?
        """,
        (extracted_text, extract_status, len(extracted_text) if extracted_text else 0, chat_id, message_id),
    )
    conn.commit()


def scoring_input(row: sqlite3.Row) -> str:
    """The text to actually hand an LLM for scoring: caption + any extracted
    document text. Row needs 'text' and 'extracted_text' keys (sqlite3.Row or
    a dict). Using row['text'] alone silently drops PDF-only posts that have
    no caption - always score through this function instead."""
    parts = [row["text"] or "", row["extracted_text"] or ""]
    return "\n\n".join(p for p in parts if p).strip()


def upsert_score(
    conn: sqlite3.Connection,
    chat_id: int,
    message_id: int,
    rubric_version: str,
    model: str,
    *,
    direction: int | None = None,
    magnitude: int | None = None,
    confidence: float | None = None,
    level: str | None = None,
    sector_code: str | None = None,
    ticker: str | None = None,
    driver: str | None = None,
    novelty: str | None = None,
    horizon: str | None = None,
    summary: str | None = None,
    raw_response: str | None = None,
    scored_at: str | None = None,
) -> None:
    if driver is not None:
        assert driver in DRIVERS, f"driver must be one of {DRIVERS}, got {driver!r}"
    if level is not None:
        assert level in SCOPES, f"level must be one of {SCOPES}, got {level!r}"
    if novelty is not None:
        assert novelty in NOVELTY_VALUES, f"novelty must be one of {NOVELTY_VALUES}, got {novelty!r}"
    if horizon is not None:
        assert horizon in HORIZON_VALUES, f"horizon must be one of {HORIZON_VALUES}, got {horizon!r}"

    conn.execute(
        """
        INSERT OR REPLACE INTO scores
            (chat_id, message_id, rubric_version, model, direction, magnitude,
             confidence, level, sector_code, ticker, driver, novelty, horizon,
             summary, raw_response, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')))
        """,
        (
            chat_id, message_id, rubric_version, model, direction, magnitude,
            confidence, level, sector_code, ticker, driver, novelty, horizon,
            summary, raw_response, scored_at,
        ),
    )
    conn.commit()


def latest_scores(conn: sqlite3.Connection, rubric_version: str | None = None) -> pd.DataFrame:
    """Scores as a DataFrame, with `score` (direction*magnitude, -3..+3)
    computed on read - the scale is never persisted, so a future rubric
    tweak needs no backfill migration.

    rubric_version=None returns each item's most recently scored row (by
    scored_at) regardless of which rubric produced it - use this for "what
    do we currently believe"; pass an explicit version to compare rubrics
    against each other for the same items.
    """
    if rubric_version:
        df = pd.read_sql_query(
            "SELECT * FROM scores WHERE rubric_version = ?", conn, params=(rubric_version,)
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT s.* FROM scores s
            JOIN (
                SELECT chat_id, message_id, MAX(scored_at) AS max_scored_at
                FROM scores GROUP BY chat_id, message_id
            ) latest
              ON s.chat_id = latest.chat_id
             AND s.message_id = latest.message_id
             AND s.scored_at = latest.max_scored_at
            """,
            conn,
        )
    if not df.empty:
        df["score"] = df["direction"] * df["magnitude"]
    return df


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
