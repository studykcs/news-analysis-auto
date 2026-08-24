"""Extract text from downloaded PDF research reports.

collect.py only downloads PDFs/photos - it never reads what's inside a PDF,
which is a big reason sentiment scoring only reaches a fraction of items:
summarize.py currently scores off the Telegram caption alone, so a PDF post
with little or no caption is effectively invisible to it. This script fills
items.extracted_text so store.scoring_input() can combine caption + PDF body
for scoring.

Only the first --max-pages pages are read by default: a broker research
report's title/rating/target price/summary all live on page 1-2, and reading
the whole document burns far more tokens later (at scoring time) for little
extra signal.

A PDF yielding under 200 characters is assumed to be a scanned image (no
text layer) and flagged 'scanned' rather than force-fed near-empty text.
OCR is out of scope here - this only flags it for a future pass. Photos
(.jpg) are likewise out of scope for this task and flagged 'image_pending'.

Usage
-----
    python extract.py                      # all unprocessed PDFs/photos
    python extract.py --limit 50
    python extract.py --channel "신한 리서치"
    python extract.py --force               # re-extract even if already done
    python extract.py --max-pages 3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF; `import fitz` still works but is deprecated

from store import get_connection, update_extraction

SCANNED_CHAR_THRESHOLD = 200
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def extract_pdf_text(path: Path, max_pages: int) -> str:
    with fitz.open(path) as doc:
        n = min(max_pages, doc.page_count)
        return "\n".join(doc[i].get_text() for i in range(n)).strip()


def process_item(file_path: str | None, item_type: str, max_pages: int) -> tuple[str | None, str]:
    """Returns (extracted_text, extract_status)."""
    if not file_path:
        return None, "no_file"

    path = Path(file_path)
    if not path.exists():
        return None, "no_file"

    if item_type == "photo" or path.suffix.lower() in IMAGE_SUFFIXES:
        return None, "image_pending"

    if path.suffix.lower() != ".pdf":
        return None, "error"

    try:
        text = extract_pdf_text(path, max_pages)
    except Exception:
        return None, "error"

    if len(text) < SCANNED_CHAR_THRESHOLD:
        return (text or None), "scanned"
    return text, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max items to process")
    parser.add_argument("--force", action="store_true", help="Re-extract even items that already have a status")
    parser.add_argument("--max-pages", type=int, default=2, help="Pages to read per PDF (default: 2)")
    parser.add_argument("--channel", help="Only process items from this chat_name")
    args = parser.parse_args()

    conn = get_connection()

    query = "SELECT chat_id, message_id, chat_name, type, file_path FROM items WHERE type IN ('document', 'photo')"
    params: list = []
    if not args.force:
        query += " AND extract_status IS NULL"
    if args.channel:
        query += " AND chat_name = ?"
        params.append(args.channel)
    query += " ORDER BY date"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("Nothing to extract.")
        return

    by_channel_type: Counter = Counter()
    overall: Counter = Counter()

    for chat_id, message_id, chat_name, item_type, file_path in rows:
        text, status = process_item(file_path, item_type, args.max_pages)
        update_extraction(conn, chat_id, message_id, text, status)
        by_channel_type[(chat_name, item_type, status)] += 1
        overall[status] += 1

    print(f"Processed {len(rows)} item(s)\n")
    print(f"{'channel':<20} {'type':<10} {'status':<14} count")
    for (chat_name, item_type, status), n in sorted(by_channel_type.items()):
        print(f"{chat_name:<20} {item_type:<10} {status:<14} {n}")

    print("\nOverall:")
    for status, n in overall.most_common():
        print(f"  {status}: {n}")


if __name__ == "__main__":
    main()
