"""One-off migration: copy the old items-inline scores into the scores table.

items.sentiment_score/scope/topic/summary predate the scores table and used
a single continuous -5..+5 float, not direction*magnitude. This copies them
into scores as rubric_version='v1' so the old scoring and any future rubric
can be compared side by side without re-scoring anything. items' own
columns are left untouched - this is additive, not a destructive migration,
so it's safe to re-run (INSERT OR REPLACE on the same primary key) and easy
to roll back (just don't read from scores).

direction/magnitude are reverse-engineered from the old float: direction is
its sign, magnitude is round(abs(score)) clipped into the 0..3 range the
new schema uses - the old scale ran to +-5, so anything >=2.5 collapses to
the new scale's max, 3.

Usage
-----
    python migrate_scores.py             # migrate everything
    python migrate_scores.py --dry-run   # preview, write nothing
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from store import get_connection, upsert_score

RUBRIC_VERSION = "v1"
# What summarize.py's DEFAULT_MODEL was when these items.sentiment_score
# values were produced - not recorded per-row in the old schema, so this is
# a single best-effort label for the whole batch, not a per-item fact.
MODEL = "gemini-flash-lite-latest"


def to_direction_magnitude(score: float) -> tuple[int, int]:
    direction = 0 if score == 0 else (1 if score > 0 else -1)
    magnitude = min(3, round(abs(score)))
    return direction, magnitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be migrated, write nothing")
    args = parser.parse_args()

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT chat_id, message_id, sentiment_score, scope, summary, summarized_at
        FROM items WHERE sentiment_score IS NOT NULL
        """
    ).fetchall()

    print(f"{len(rows)} v1-scored item(s) to migrate")

    if args.dry_run:
        for chat_id, message_id, score, scope, summary, summarized_at in rows[:5]:
            direction, magnitude = to_direction_magnitude(score)
            print(f"  {chat_id}_{message_id}: score={score} -> direction={direction} magnitude={magnitude} level={scope}")
        if len(rows) > 5:
            print(f"  ... and {len(rows) - 5} more")
        print("(dry run - nothing written)")
        return

    n = 0
    for chat_id, message_id, score, scope, summary, summarized_at in rows:
        direction, magnitude = to_direction_magnitude(score)
        upsert_score(
            conn, chat_id, message_id, RUBRIC_VERSION, MODEL,
            direction=direction, magnitude=magnitude, level=scope, summary=summary,
            scored_at=summarized_at or datetime.now(timezone.utc).isoformat(),
        )
        n += 1

    print(f"Migrated {n} item(s) into scores (rubric_version='{RUBRIC_VERSION}')")


if __name__ == "__main__":
    main()
