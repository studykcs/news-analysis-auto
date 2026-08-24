"""Cluster near-duplicate items posted by multiple channels.

Six channels regularly cover the same story (e.g. a Fed statement, a
CPI print) within a day of each other. Left alone, that story gets
counted once per channel in every daily aggregate - a story six houses
mention is worth 6x a story one house mentions, purely because it was
popular to report, not because it's six times as important. Duplicates
are not discarded here, though - they're turned into a feature:
`mention_channels` (how many distinct channels covered the same story)
is a real consensus signal, and index.py can optionally weight by it
(--weight-by-consensus) instead of just deduplicating it away.

Two-stage matching, restricted to items within +-1 day of each other
(duplicates of the same story don't span weeks):

  1. SimHash on normalized text (URLs/phone numbers stripped, common
     compliance-notice boilerplate lines dropped so two *different*
     Shinhan reports don't look similar just because they share the same
     disclaimer footer) narrows the field: only pairs within
     SIMHASH_HAMMING_THRESHOLD bits of each other become candidates.
     (Implemented as direct pairwise comparison within the +-1-day
     window rather than LSH banding - at this dataset's size, a window
     rarely has more than a few hundred items, so banding would add
     complexity without a measurable speedup. Revisit if that changes.)
  2. Candidate pairs are confirmed (or rejected) with rapidfuzz's
     character-level ratio on the same normalized text.

PDFs get an independent, stricter check: same file size + same PyMuPDF
page count + same hash of the already-extracted first-page text (from
extract.py's extracted_text - this does not re-open the PDF) counts as
a duplicate regardless of caption similarity, since that combination is
essentially never coincidental.

Clusters are unioned (transitively - A~B and B~C merges all three) and
one representative per cluster (the earliest-posted item) is marked
is_cluster_head=1; every item in the cluster gets the same
dup_cluster_id and mention_channels (= number of distinct chat_id in
the cluster). index.py's default aggregation only uses cluster heads.

Usage
-----
    python dedupe.py                         # cluster all items, all dates
    python dedupe.py --date 2026-08-20
    python dedupe.py --since 2026-08-01
    python dedupe.py --hamming 8 --fuzzy-threshold 85
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from datetime import timedelta

import pandas as pd
from rapidfuzz import fuzz

from store import get_connection

SIMHASH_BITS = 64
SIMHASH_HAMMING_THRESHOLD = 10  # candidate if hamming distance <= this
FUZZY_THRESHOLD = 85  # rapidfuzz ratio (0-100) to confirm a text duplicate

BOILERPLATE_PATTERNS = [
    r"위\s*내용은.*?승인이\s*이뤄진\s*내용입니다\.?",
    r"위\s*내용은.*?승인이\s*이루어진\s*내용입니다\.?",
    r"제공해\s*드린\s*조사분석자료는.*?없습니다\.?",
    r"본\s*조사자료는.*?없습니다\.?",
    r"위\s*문자의\s*내용은\s*컴플라이언스의\s*승인을\s*득하였음",
    r"\*텔레그램\s*채널:\s*\S+",
    r"☎️?\s*[\d\-]+",
]


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\d{2,3}-\d{3,4}-\d{4}", " ", text)
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.S)
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split() if len(t) > 1]


def simhash(text: str, bits: int = SIMHASH_BITS) -> int:
    tokens = _tokenize(text)
    if not tokens:
        return 0
    weights = [0] * bits
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            weights[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i in range(bits):
        if weights[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _pdf_fingerprint(row: pd.Series) -> str | None:
    """file_size + page_count aren't columns we store, so this uses what we
    do have: extract_chars (proxy for content length) + a hash of the
    already-extracted first-page(s) text. Coarser than a true page-count
    check, but doesn't require re-opening the PDF."""
    if row["type"] != "document" or not row["extracted_text"]:
        return None
    prefix_hash = hashlib.md5(row["extracted_text"][:1000].encode("utf-8")).hexdigest()
    return f"{row['extract_chars']}:{prefix_hash}"


def cluster_items(df: pd.DataFrame, hamming_threshold: int, fuzzy_threshold: int) -> dict[str, str]:
    """Returns {item_key: cluster_id} for every row in df.
    item_key is '{chat_id}_{message_id}'."""
    df = df.copy()
    df["item_key"] = df["chat_id"].astype(str) + "_" + df["message_id"].astype(str)
    df["norm_text"] = (df["text"].fillna("") + " " + df["extracted_text"].fillna("")).map(normalize_text)
    df["simhash"] = df["norm_text"].map(simhash)
    df["pdf_fp"] = df.apply(_pdf_fingerprint, axis=1)
    df["date_dt"] = pd.to_datetime(df["date"].str.slice(0, 10))

    uf = UnionFind(df["item_key"].tolist())

    # PDF fingerprint matches: exact match = duplicate, no threshold needed
    for fp, group in df[df["pdf_fp"].notna()].groupby("pdf_fp"):
        keys = group["item_key"].tolist()
        for k in keys[1:]:
            uf.union(keys[0], k)

    # Text-based matching within a +-1 day window
    df_sorted = df.sort_values("date_dt").reset_index(drop=True)
    n = len(df_sorted)
    for i in range(n):
        row_i = df_sorted.iloc[i]
        if not row_i["norm_text"]:
            continue
        for j in range(i + 1, n):
            row_j = df_sorted.iloc[j]
            if row_j["date_dt"] - row_i["date_dt"] > timedelta(days=1):
                break  # sorted by date - nothing further can be within window
            if not row_j["norm_text"]:
                continue
            if uf.find(row_i["item_key"]) == uf.find(row_j["item_key"]):
                continue
            if hamming_distance(row_i["simhash"], row_j["simhash"]) > hamming_threshold:
                continue
            score = fuzz.ratio(row_i["norm_text"], row_j["norm_text"])
            if score >= fuzzy_threshold:
                uf.union(row_i["item_key"], row_j["item_key"])

    return {k: uf.find(k) for k in df["item_key"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Only cluster items on this date (YYYY-MM-DD)")
    parser.add_argument("--since", help="Only cluster items on/after this date (YYYY-MM-DD)")
    parser.add_argument("--hamming", type=int, default=SIMHASH_HAMMING_THRESHOLD)
    parser.add_argument("--fuzzy-threshold", type=int, default=FUZZY_THRESHOLD)
    args = parser.parse_args()

    conn = get_connection()

    query = "SELECT chat_id, message_id, type, text, extracted_text, extract_chars, date FROM items WHERE 1=1"
    params: list = []
    if args.date:
        query += " AND substr(date,1,10) = ?"
        params.append(args.date)
    if args.since:
        query += " AND substr(date,1,10) >= ?"
        params.append(args.since)

    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        print("No items to cluster.")
        return

    clusters = cluster_items(df, args.hamming, args.fuzzy_threshold)

    df["item_key"] = df["chat_id"].astype(str) + "_" + df["message_id"].astype(str)
    df["cluster_id"] = df["item_key"].map(clusters)

    cluster_sizes = df.groupby("cluster_id")["chat_id"].nunique()
    head_key = df.sort_values(["date", "message_id"]).groupby("cluster_id")["item_key"].first()
    head_keys = set(head_key.values)

    n_updated = 0
    n_multi = 0
    for _, row in df.iterrows():
        mention_channels = int(cluster_sizes[row["cluster_id"]])
        is_head = 1 if row["item_key"] in head_keys and head_key[row["cluster_id"]] == row["item_key"] else 0
        conn.execute(
            "UPDATE items SET dup_cluster_id = ?, is_cluster_head = ?, mention_channels = ? "
            "WHERE chat_id = ? AND message_id = ?",
            (row["cluster_id"], is_head, mention_channels, row["chat_id"], row["message_id"]),
        )
        n_updated += 1
        if mention_channels > 1 and is_head:
            n_multi += 1
    conn.commit()

    n_clusters = df["cluster_id"].nunique()
    print(f"{n_updated} item(s) clustered into {n_clusters} cluster(s)")
    print(f"{n_multi} cluster(s) span >1 channel (real cross-channel duplicates)")
    dist = cluster_sizes.value_counts().sort_index()
    print("\nmention_channels distribution (per cluster):")
    print(dist.to_string())


if __name__ == "__main__":
    main()
