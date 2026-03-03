#!/usr/bin/env python3
"""
Build vibe features from existing database tables.

Inputs:
- music.track_catalog (genre metadata)
- music.listen_events (30-day popularity signals)

Output:
- music.track_vibe_features
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from psycopg import connect
from psycopg.rows import dict_row


VIBE_RULES: Dict[str, List[str]] = {
    "chill": ["chill", "ambient", "downtempo", "trip-hop", "lofi", "lounge", "new age"],
    "focus": ["instrumental", "classical", "minimal", "soundtrack", "piano", "jazz"],
    "energetic": ["rock", "metal", "punk", "edm", "electro", "techno", "drum"],
    "party": ["dance", "house", "club", "hip-hop", "rap", "disco", "reggaeton"],
    "happy": ["pop", "funk", "soul", "indie pop"],
    "sad": ["blues", "acoustic", "emo", "sadcore", "singer-songwriter"],
    "romantic": ["rnb", "soul", "love", "ballad", "smooth jazz"],
}

# Tie-break priority when keyword matches are equal.
VIBE_PRIORITY = ["chill", "focus", "energetic", "party", "happy", "sad", "romantic"]


CANDIDATES_SQL = """
WITH anchor AS (
    SELECT MAX(event_ts) AS max_event_ts
    FROM music.listen_events
),
pop AS (
    SELECT
        COALESCE(track_id, '__unknown__') AS track_id,
        COUNT(*)::BIGINT AS plays_30d,
        COUNT(DISTINCT user_id)::BIGINT AS unique_listeners_30d
    FROM music.listen_events le
    CROSS JOIN anchor a
    WHERE a.max_event_ts IS NOT NULL
      AND le.event_ts >= a.max_event_ts - INTERVAL '30 days'
      AND le.event_type = 'play'
    GROUP BY COALESCE(track_id, '__unknown__')
),
catalog AS (
    SELECT
        COALESCE(track_id, '__unknown__') AS track_id,
        MAX(NULLIF(track_name, '')) AS track_name,
        MAX(NULLIF(artist_name, '')) AS artist_name,
        MAX(NULLIF(genre, '')) AS genre
    FROM music.track_catalog
    GROUP BY COALESCE(track_id, '__unknown__')
),
event_names AS (
    SELECT
        COALESCE(track_id, '__unknown__') AS track_id,
        MAX(NULLIF(track_name, '')) AS track_name,
        MAX(NULLIF(artist_name, '')) AS artist_name
    FROM music.listen_events
    WHERE track_id IS NOT NULL
    GROUP BY COALESCE(track_id, '__unknown__')
),
combined AS (
    SELECT
        COALESCE(c.track_id, e.track_id, p.track_id) AS track_id,
        COALESCE(c.track_name, e.track_name) AS track_name,
        COALESCE(c.artist_name, e.artist_name) AS artist_name,
        c.genre,
        COALESCE(p.plays_30d, 0)::BIGINT AS plays_30d,
        COALESCE(p.unique_listeners_30d, 0)::BIGINT AS unique_listeners_30d
    FROM catalog c
    FULL OUTER JOIN event_names e
      ON e.track_id = c.track_id
    FULL OUTER JOIN pop p
      ON p.track_id = COALESCE(c.track_id, e.track_id)
)
SELECT
    track_id,
    track_name,
    artist_name,
    genre,
    plays_30d,
    unique_listeners_30d
FROM combined
WHERE track_id IS NOT NULL
  AND track_id <> '__unknown__'
ORDER BY plays_30d DESC, track_id
"""


UPSERT_SQL = """
INSERT INTO music.track_vibe_features (
    track_id,
    vibe_label,
    confidence,
    label_source,
    rule_keywords,
    genre,
    plays_30d,
    unique_listeners_30d,
    updated_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (track_id) DO UPDATE SET
    vibe_label = EXCLUDED.vibe_label,
    confidence = EXCLUDED.confidence,
    label_source = EXCLUDED.label_source,
    rule_keywords = EXCLUDED.rule_keywords,
    genre = EXCLUDED.genre,
    plays_30d = EXCLUDED.plays_30d,
    unique_listeners_30d = EXCLUDED.unique_listeners_30d,
    updated_at = NOW()
"""


def build_conn_kwargs() -> Mapping[str, str]:
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        return {"conninfo": dsn}

    host = os.environ.get("PGHOST")
    dbname = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError(
            "Postgres connection is required. Set POSTGRES_DSN or PGHOST/PGDATABASE/PGUSER/PGPASSWORD."
        )

    return {
        "host": host,
        "port": os.environ.get("PGPORT", "5432"),
        "dbname": dbname,
        "user": user,
        "password": password,
        "sslmode": os.environ.get("PGSSLMODE", "require"),
    }


def _find_matches(text: Optional[str]) -> List[Tuple[str, List[str]]]:
    if not text:
        return []
    lowered = text.lower()
    matches: List[Tuple[str, List[str]]] = []
    for vibe, keywords in VIBE_RULES.items():
        hit = [kw for kw in keywords if kw in lowered]
        if hit:
            matches.append((vibe, hit))
    return matches


def choose_vibe(genre: Optional[str], track_name: Optional[str], artist_name: Optional[str]) -> Tuple[Optional[str], List[str], str]:
    genre_matches = _find_matches(genre)
    if genre_matches:
        matches = genre_matches
        source = "genre_rule"
    else:
        # Fallback to lexical clues when genre is missing.
        text_matches = _find_matches(" ".join([track_name or "", artist_name or ""]).strip())
        if not text_matches:
            return None, [], "no_match"
        matches = text_matches
        source = "text_rule"

    if not matches:
        return None, [], "no_match"

    matches.sort(
        key=lambda x: (
            len(x[1]),
            -VIBE_PRIORITY.index(x[0]) if x[0] in VIBE_PRIORITY else -999,
        ),
        reverse=True,
    )
    best_vibe, best_keywords = matches[0]
    return best_vibe, best_keywords, source


def score_confidence(match_count: int, plays_30d: int, unique_listeners_30d: int, matched: bool, source: str) -> float:
    if not matched:
        return 0.150

    base = 0.450 if source == "genre_rule" else 0.320
    score = base + min(match_count * 0.120, 0.300)

    if plays_30d >= 50:
        score += 0.080
    elif plays_30d >= 10:
        score += 0.040

    if unique_listeners_30d >= 20:
        score += 0.080
    elif unique_listeners_30d >= 5:
        score += 0.040

    return float(min(score, 0.950))


def batch_iter(rows: Sequence[Tuple], size: int) -> Iterable[Sequence[Tuple]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build music.track_vibe_features from genre + behavior signals.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on candidate tracks.")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Also store tracks with no matched vibe as vibe_label='unknown'.",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per DB upsert batch.")
    parser.add_argument("--dry-run", action="store_true", help="Compute summary only, do not write DB.")
    args = parser.parse_args()

    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)

    try:
        with conn.cursor() as cur:
            sql = CANDIDATES_SQL
            if args.limit:
                sql += "\nLIMIT %s"
                cur.execute(sql, (args.limit,))
            else:
                cur.execute(sql)
            rows = cur.fetchall()

        out_rows: List[Tuple] = []
        matched_count = 0
        unknown_count = 0

        for r in rows:
            track_id = r["track_id"]
            track_name = r["track_name"]
            artist_name = r["artist_name"]
            genre = r["genre"]
            plays_30d = int(r["plays_30d"] or 0)
            unique_listeners_30d = int(r["unique_listeners_30d"] or 0)

            vibe, keywords, source = choose_vibe(genre, track_name, artist_name)
            if vibe is None:
                if not args.include_unknown:
                    continue
                vibe = "unknown"
                keywords = []
                label_source = "no_genre_match"
                unknown_count += 1
            else:
                label_source = source
                matched_count += 1

            confidence = score_confidence(
                match_count=len(keywords),
                plays_30d=plays_30d,
                unique_listeners_30d=unique_listeners_30d,
                matched=(vibe != "unknown"),
                source=label_source,
            )

            out_rows.append(
                (
                    track_id,
                    vibe,
                    confidence,
                    label_source,
                    keywords if keywords else None,
                    genre,
                    plays_30d,
                    unique_listeners_30d,
                )
            )

        print(
            f"candidates={len(rows)} prepared={len(out_rows)} "
            f"matched={matched_count} unknown={unknown_count} dry_run={args.dry_run}"
        )

        if args.dry_run or not out_rows:
            return

        with conn.cursor() as cur:
            for chunk in batch_iter(out_rows, args.batch_size):
                cur.executemany(UPSERT_SQL, chunk)
        conn.commit()
        print(f"upserted_rows={len(out_rows)} into music.track_vibe_features")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
