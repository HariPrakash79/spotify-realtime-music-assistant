#!/usr/bin/env python3
"""
Lightweight serving API for music recommendations.

Endpoints:
- GET /metrics/model
- GET /metrics/recsource
- GET /trending?limit=20
- GET /recs/{user_id}?limit=20
- GET /favorites/{user_id}?limit=20
- GET /search/tracks?query=...&limit=10
- GET /vibe?vibe=...&limit=10
- POST /feedback/vibe
- POST /feedback/interaction
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Dict, List, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from psycopg import connect
from psycopg.rows import dict_row

from text_cleanup import clean_display_text, clean_text

CONSENSUS_MIN_USERS = int(os.environ.get("VIBE_FEEDBACK_MIN_USERS", "15"))
CONSENSUS_MIN_TOP_SHARE = float(os.environ.get("VIBE_FEEDBACK_MIN_TOP_SHARE", "0.70"))
CONSENSUS_MIN_MARGIN = float(os.environ.get("VIBE_FEEDBACK_MIN_MARGIN", "0.15"))
READABLE_SCAN_MIN = int(os.environ.get("READABLE_SCAN_MIN", "500"))
READABLE_SCAN_MULTIPLIER = int(os.environ.get("READABLE_SCAN_MULTIPLIER", "80"))
USE_HYBRID_RECS = os.environ.get("USE_HYBRID_RECS", "true").strip().lower() in {"1", "true", "yes", "on"}
USE_ML_RECS = os.environ.get("USE_ML_RECS", "true").strip().lower() in {"1", "true", "yes", "on"}
USE_DENSE_RECS = os.environ.get("USE_DENSE_RECS", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_DEMO_INTERACTION_LOGGING = os.environ.get("ENABLE_DEMO_INTERACTION_LOGGING", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


MODEL_METRICS_SQL = """
SELECT
    events,
    users,
    tracks,
    events_per_user
FROM music.v_model_metrics_1000_ready
"""


TRENDING_SQL = """
SELECT
    global_rank_7d,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM music.v_global_trending_tracks_7d_ready
WHERE global_rank_7d <= %s
ORDER BY global_rank_7d
"""

NAMED_TRENDING_SQL = """
WITH
named AS (
    SELECT
        MIN(NULLIF(track_id, '')) AS track_id,
        MIN(NULLIF(artist_id, '')) AS artist_id,
        track_name,
        artist_name,
        COUNT(*)::BIGINT AS plays_all_time,
        COUNT(DISTINCT user_id)::BIGINT AS unique_listeners_all_time,
        MAX(event_ts) AS last_play_ts_all_time
    FROM music.v_listen_events_recommendation_ready le
    WHERE le.event_type = 'play'
      AND COALESCE(le.track_name, '') <> ''
      AND COALESCE(le.artist_name, '') <> ''
      AND le.artist_name <> '__unknown_artist__'
      AND (le.track_id IS NULL OR le.track_name <> le.track_id)
      AND le.track_name !~ '^[0-9]+$'
    GROUP BY track_name, artist_name
),
ranked AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY plays_all_time DESC, unique_listeners_all_time DESC, last_play_ts_all_time DESC, track_name, artist_name
        ) AS global_rank_7d,
        COALESCE(track_id, track_name) AS track_id,
        track_name,
        artist_id,
        artist_name,
        plays_all_time AS plays_7d,
        unique_listeners_all_time AS unique_listeners_7d
    FROM named
)
SELECT
    global_rank_7d,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM ranked
WHERE global_rank_7d <= %s
ORDER BY global_rank_7d
"""

VIBE_NAMED_FALLBACK_SQL = """
WITH
named AS (
    SELECT
        MIN(NULLIF(le.track_id, '')) AS track_id,
        MIN(NULLIF(le.artist_id, '')) AS artist_id,
        le.track_name,
        le.artist_name,
        COUNT(*)::BIGINT AS plays_all_time,
        COUNT(DISTINCT le.user_id)::BIGINT AS unique_listeners_all_time
    FROM music.v_listen_events_recommendation_ready le
    WHERE le.event_type = 'play'
      AND COALESCE(le.track_name, '') <> ''
      AND COALESCE(le.artist_name, '') <> ''
      AND le.artist_name <> '__unknown_artist__'
      AND (le.track_id IS NULL OR le.track_name <> le.track_id)
      AND le.track_name !~ '^[0-9]+$'
    GROUP BY le.track_name, le.artist_name
),
catalog_norm AS (
    SELECT
        lower(COALESCE(track_name, '')) AS track_name_l,
        lower(COALESCE(artist_name, '')) AS artist_name_l,
        lower(COALESCE(genre, '')) AS genre_l
    FROM music.track_catalog
    WHERE COALESCE(track_name, '') <> ''
),
enriched AS (
    SELECT
        n.track_id,
        n.track_name,
        n.artist_id,
        n.artist_name,
        n.plays_all_time,
        n.unique_listeners_all_time,
        COALESCE(MAX(NULLIF(cn.genre_l, '')), '') AS genre_l,
        lower(n.track_name) AS track_name_l,
        lower(n.artist_name) AS artist_name_l
    FROM named n
    LEFT JOIN catalog_norm cn
      ON cn.track_name_l = lower(n.track_name)
     AND (
         cn.artist_name_l = lower(n.artist_name)
         OR cn.artist_name_l = ''
         OR lower(n.artist_name) = ''
     )
    GROUP BY
        n.track_id,
        n.track_name,
        n.artist_id,
        n.artist_name,
        n.plays_all_time,
        n.unique_listeners_all_time
),
scored AS (
    SELECT
        e.track_id,
        e.track_name,
        e.artist_id,
        e.artist_name,
        e.plays_all_time,
        e.unique_listeners_all_time,
        (
            SELECT MAX(
                CASE
                    WHEN e.genre_l LIKE concat(chr(37), kw, chr(37)) THEN 3
                    WHEN e.track_name_l LIKE concat(chr(37), kw, chr(37)) THEN 2
                    WHEN e.artist_name_l LIKE concat(chr(37), kw, chr(37)) THEN 1
                    ELSE 0
                END
            )
            FROM unnest(%s::text[]) kw
        ) AS vibe_match_score
    FROM enriched e
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY vibe_match_score DESC, plays_all_time DESC, unique_listeners_all_time DESC, track_name, artist_name
    ) AS global_rank_7d,
    COALESCE(track_id, track_name) AS track_id,
    track_name,
    artist_id,
    artist_name,
    plays_all_time AS plays_7d,
    unique_listeners_all_time AS unique_listeners_7d
FROM scored
WHERE vibe_match_score > 0
ORDER BY global_rank_7d
LIMIT %s
"""


USER_RECS_DENSE_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d_dense_1000_ready
WHERE user_id = %s
  AND recommendation_rank <= %s
ORDER BY recommendation_rank
"""

USER_RECS_MF_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_mf_ready
WHERE user_id = %s
  AND recommendation_rank <= %s
ORDER BY recommendation_rank
"""

USER_RECS_HYBRID_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_hybrid_ready
WHERE user_id = %s
  AND recommendation_rank <= %s
ORDER BY recommendation_rank
"""

USER_FAVORITES_SQL = """
WITH ranked AS (
    SELECT
        le.user_id,
        COALESCE(NULLIF(le.track_id, ''), COALESCE(NULLIF(le.track_name, ''), '__unknown_track__')) AS track_id,
        COALESCE(NULLIF(le.track_name, ''), COALESCE(NULLIF(le.track_id, ''), '__unknown_track__')) AS track_name,
        MIN(NULLIF(le.artist_id, '')) AS artist_id,
        COALESCE(MAX(NULLIF(le.artist_name, '')), '__unknown_artist__') AS artist_name,
        COUNT(*)::BIGINT AS plays,
        MAX(le.event_ts) AS last_played_at
    FROM music.listen_events le
    WHERE le.user_id = %s
    GROUP BY
        le.user_id,
        COALESCE(NULLIF(le.track_id, ''), COALESCE(NULLIF(le.track_name, ''), '__unknown_track__')),
        COALESCE(NULLIF(le.track_name, ''), COALESCE(NULLIF(le.track_id, ''), '__unknown_track__'))
)
SELECT
    user_id,
    ROW_NUMBER() OVER (ORDER BY plays DESC, last_played_at DESC, track_name, artist_name) AS favorite_rank,
    track_id,
    track_name,
    artist_id,
    artist_name,
    plays
FROM ranked
ORDER BY favorite_rank
LIMIT %s
"""

USER_EXISTS_SQL = """
SELECT 1
FROM music.v_model_users_1000_ready
WHERE user_id = %s
LIMIT 1
"""

USER_FROM_DISPLAY_SQL = """
SELECT
    user_id,
    display_name
FROM music.user_profiles
WHERE lower(display_name) = lower(%s)
LIMIT 1
"""

USER_FROM_DISPLAY_PARTIAL_SQL = """
SELECT
    user_id,
    display_name
FROM music.user_profiles
WHERE lower(display_name) LIKE lower(%s)
ORDER BY display_name
LIMIT 2
"""

USER_FROM_DISPLAY_CANDIDATES_SQL = """
SELECT
    up.user_id,
    up.display_name
FROM music.user_profiles up
JOIN music.v_model_users_1000_ready mu
  ON mu.user_id = up.user_id
WHERE COALESCE(up.display_name, '') <> ''
ORDER BY up.display_name
"""

DISPLAY_FROM_USER_SQL = """
SELECT
    display_name
FROM music.user_profiles
WHERE user_id = %s
LIMIT 1
"""

REC_SOURCE_LATEST_HYBRID_SQL = """
SELECT
    model_version,
    generated_at
FROM music.user_recommendations_hybrid_ready
ORDER BY generated_at DESC
LIMIT 1
"""

REC_SOURCE_LATEST_MF_SQL = """
SELECT
    model_version,
    generated_at
FROM music.user_recommendations_mf_ready
ORDER BY generated_at DESC
LIMIT 1
"""

REC_SOURCE_COUNTS_HYBRID_SQL = """
SELECT
    COUNT(*)::BIGINT AS rows,
    COUNT(DISTINCT user_id)::BIGINT AS users
FROM music.v_user_recommendations_hybrid_ready
"""

REC_SOURCE_COUNTS_MF_SQL = """
SELECT
    COUNT(*)::BIGINT AS rows,
    COUNT(DISTINCT user_id)::BIGINT AS users
FROM music.v_user_recommendations_mf_ready
"""

REC_SOURCE_COUNTS_DENSE_SQL = """
SELECT
    COUNT(*)::BIGINT AS rows,
    COUNT(DISTINCT user_id)::BIGINT AS users
FROM music.v_user_recommendations_30d_dense_1000_ready
"""

REC_SOURCE_COUNTS_TRENDING_SQL = """
SELECT
    COUNT(*)::BIGINT AS rows
FROM music.v_global_trending_tracks_7d_ready
"""

SEARCH_TRACKS_SQL = """
WITH q AS (
    SELECT
        %s::text AS query_text,
        lower(%s::text) AS query_lower
),
le AS (
    SELECT
        COALESCE(track_id, '__unknown__') AS track_id,
        COALESCE(NULLIF(track_name, ''), COALESCE(track_id, '__unknown__')) AS track_name,
        COALESCE(MAX(NULLIF(artist_name, '')), '__unknown_artist__') AS artist_name,
        NULL::text AS genre,
        COUNT(*)::BIGINT AS popularity_30d
    FROM music.v_listen_events_recommendation_ready le
    WHERE track_name IS NOT NULL
      AND lower(track_name) LIKE concat(chr(37), (SELECT query_lower FROM q), chr(37))
    GROUP BY COALESCE(track_id, '__unknown__'), COALESCE(NULLIF(track_name, ''), COALESCE(track_id, '__unknown__'))
),
tc AS (
    SELECT
        COALESCE(track_id, '__unknown__') AS track_id,
        COALESCE(NULLIF(track_name, ''), COALESCE(track_id, '__unknown__')) AS track_name,
        COALESCE(NULLIF(artist_name, ''), '__unknown_artist__') AS artist_name,
        genre,
        0::BIGINT AS popularity_30d
    FROM music.track_catalog tc
    WHERE track_name IS NOT NULL
      AND lower(track_name) LIKE concat(chr(37), (SELECT query_lower FROM q), chr(37))
),
combined AS (
    SELECT * FROM le
    UNION ALL
    SELECT * FROM tc
),
ranked AS (
    SELECT
        track_id,
        track_name,
        artist_name,
        genre,
        popularity_30d,
        CASE
            WHEN lower(track_name) = (SELECT query_lower FROM q) THEN 300
            WHEN lower(track_name) LIKE concat((SELECT query_lower FROM q), chr(37)) THEN 200
            WHEN lower(track_name) LIKE concat(chr(37), (SELECT query_lower FROM q), chr(37)) THEN 100
            ELSE 0
        END AS match_score,
        ROW_NUMBER() OVER (
            PARTITION BY lower(track_name), lower(COALESCE(artist_name, ''))
            ORDER BY popularity_30d DESC, track_id
        ) AS dedup_rn
    FROM combined
)
SELECT
    track_id,
    track_name,
    artist_name,
    genre,
    popularity_30d,
    match_score
FROM ranked
WHERE dedup_rn = 1
ORDER BY match_score DESC, popularity_30d DESC, track_name
LIMIT %s
"""

VIBE_FEATURE_TRACKS_SQL = """
SELECT
    vte.track_id,
    COALESCE(MAX(NULLIF(tc.track_name, '')), MAX(vte.track_id), '__unknown_track__') AS track_name,
    COALESCE(MAX(NULLIF(tc.artist_name, '')), '__unknown_artist__') AS artist_name,
    vte.genre,
    vte.vibe_label,
    vte.confidence,
    COALESCE(vte.plays_30d, 0) AS plays_30d
FROM music.v_track_vibe_effective vte
LEFT JOIN music.track_catalog tc
  ON tc.track_id = vte.track_id
WHERE vte.vibe_label = ANY(%s::text[])
  AND vte.vibe_label <> 'unknown'
GROUP BY
    vte.track_id,
    vte.genre,
    vte.vibe_label,
    vte.confidence,
    vte.plays_30d
ORDER BY vte.confidence DESC, COALESCE(vte.plays_30d, 0) DESC, vte.track_id
LIMIT %s
"""

FEEDBACK_UPSERT_SQL = """
INSERT INTO music.track_vibe_feedback (
    user_id,
    track_id,
    predicted_vibe,
    user_selected_vibe,
    feedback_count,
    first_seen_at,
    updated_at
)
VALUES (%s, %s, %s, %s, 1, NOW(), NOW())
ON CONFLICT (user_id, track_id) DO UPDATE SET
    predicted_vibe = EXCLUDED.predicted_vibe,
    user_selected_vibe = EXCLUDED.user_selected_vibe,
    feedback_count = music.track_vibe_feedback.feedback_count + 1,
    updated_at = NOW()
"""

DEMO_INTERACTION_INSERT_SQL = """
INSERT INTO music.demo_interactions (
    user_id,
    track_id,
    action,
    source_endpoint,
    model_mode,
    model_version,
    recommendation_rank,
    context_vibe,
    session_id,
    signal_strength,
    metadata,
    event_ts
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
"""

TRACK_FEEDBACK_CONSENSUS_SQL = """
WITH votes AS (
    SELECT
        user_selected_vibe AS vibe_label,
        COUNT(DISTINCT user_id)::BIGINT AS unique_users
    FROM music.track_vibe_feedback
    WHERE track_id = %s
      AND user_selected_vibe IS NOT NULL
      AND user_selected_vibe <> ''
    GROUP BY user_selected_vibe
),
ranked AS (
    SELECT
        vibe_label,
        unique_users,
        SUM(unique_users) OVER () AS total_unique_users,
        ROW_NUMBER() OVER (ORDER BY unique_users DESC, vibe_label) AS rn
    FROM votes
),
top1 AS (
    SELECT
        vibe_label,
        unique_users,
        total_unique_users
    FROM ranked
    WHERE rn = 1
),
top2 AS (
    SELECT unique_users
    FROM ranked
    WHERE rn = 2
)
SELECT
    top1.vibe_label,
    top1.unique_users AS top_unique_users,
    top1.total_unique_users,
    (top1.unique_users::NUMERIC / NULLIF(top1.total_unique_users, 0)) AS top_share,
    ((top1.unique_users - COALESCE(top2.unique_users, 0))::NUMERIC / NULLIF(top1.total_unique_users, 0)) AS margin
FROM top1
LEFT JOIN top2 ON TRUE
"""

UPSERT_OVERRIDE_SQL = """
INSERT INTO music.track_vibe_overrides (
    track_id,
    vibe_label,
    confidence,
    unique_users,
    top_share,
    margin,
    threshold_users,
    source,
    updated_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, 'user_feedback_consensus', NOW())
ON CONFLICT (track_id) DO UPDATE SET
    vibe_label = EXCLUDED.vibe_label,
    confidence = EXCLUDED.confidence,
    unique_users = EXCLUDED.unique_users,
    top_share = EXCLUDED.top_share,
    margin = EXCLUDED.margin,
    threshold_users = EXCLUDED.threshold_users,
    source = EXCLUDED.source,
    updated_at = NOW()
"""

DELETE_OVERRIDE_SQL = "DELETE FROM music.track_vibe_overrides WHERE track_id = %s"

INTERACTION_ACTION_ALIASES = {
    "thumbs_up": "like",
    "thumbsup": "like",
    "liked": "like",
    "thumbs_down": "dislike",
    "thumbsdown": "dislike",
    "played": "play",
    "listen": "play",
    "favorited": "favorite",
    "favourite": "favorite",
    "added_to_playlist": "add_to_playlist",
}

ALLOWED_INTERACTION_ACTIONS = {
    "impression",
    "play",
    "like",
    "dislike",
    "skip",
    "favorite",
    "add_to_playlist",
    "query",
    "search",
    "session_start",
    "session_end",
}

TRACK_REQUIRED_ACTIONS = {
    "impression",
    "play",
    "like",
    "dislike",
    "skip",
    "favorite",
    "add_to_playlist",
}

VIBE_CATALOG_FALLBACK_SQL = """
WITH pop AS (
    SELECT
        track_id,
        COUNT(*)::BIGINT AS plays_30d
    FROM music.v_listen_events_recommendation_ready
    WHERE event_type = 'play'
    GROUP BY track_id
)
SELECT
    COALESCE(tc.track_id, '__unknown__') AS track_id,
    COALESCE(NULLIF(tc.track_name, ''), COALESCE(tc.track_id, '__unknown__')) AS track_name,
    COALESCE(NULLIF(tc.artist_name, ''), '__unknown_artist__') AS artist_name,
    tc.genre,
    COALESCE(pop.plays_30d, 0) AS plays_30d
FROM music.track_catalog tc
LEFT JOIN pop
  ON pop.track_id = tc.track_id
WHERE tc.track_name IS NOT NULL
  AND tc.genre IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM unnest(%s::text[]) kw
      WHERE lower(tc.genre) LIKE concat(chr(37), kw, chr(37))
  )
ORDER BY COALESCE(pop.plays_30d, 0) DESC, tc.track_name
LIMIT %s
"""


def vibe_keywords(vibe: str) -> List[str]:
    base = vibe.strip().lower()
    mapping = {
        "chill": ["chill", "ambient", "downtempo", "lofi", "trip-hop"],
        "focus": ["instrumental", "classical", "ambient", "minimal"],
        "happy": ["pop", "dance", "funk", "disco"],
        "sad": ["sad", "melancholy", "heartbreak", "breakup", "blues", "ballad", "emo"],
        "party": ["dance", "electronic", "house", "hip-hop"],
        "energetic": ["rock", "edm", "electronic", "metal", "drum"],
        "romantic": ["soul", "rnb", "jazz", "love"],
    }
    for key, values in mapping.items():
        if key in base:
            return values
    return [base]


def vibe_labels(vibe: str) -> List[str]:
    base = vibe.strip().lower()
    mapping = {
        "chill": ["chill"],
        "focus": ["focus", "chill"],
        "happy": ["happy", "party"],
        "sad": ["sad", "romantic"],
        "party": ["party", "energetic"],
        "energetic": ["energetic", "party"],
        "romantic": ["romantic", "chill"],
        "study": ["focus", "chill"],
        "calm": ["chill"],
    }
    for key, values in mapping.items():
        if key in base:
            return values
    return [base]


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


def fetch_rows(sql: str, params: tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    def _clean_metadata_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        # Output guardrail: keep API metadata human-readable even if stale mojibake rows exist in DB.
        if "track_name" in row:
            cleaned_track = clean_display_text(row.get("track_name"), repair_mojibake=True)
            if cleaned_track:
                row["track_name"] = cleaned_track
        if "artist_name" in row:
            cleaned_artist = clean_display_text(row.get("artist_name"), repair_mojibake=True)
            if cleaned_artist:
                row["artist_name"] = cleaned_artist
        return row

    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [_clean_metadata_fields(dict(row)) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> Dict[str, Any] | None:
    rows = fetch_rows(sql, params)
    return rows[0] if rows else None


def _normalize_name_key(value: str) -> str:
    cleaned = clean_text(value)
    if cleaned is None:
        return ""
    lowered = cleaned.lower()
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _fuzzy_match_user_display_name(ref: str) -> tuple[str, str] | None:
    # Fuzzy matching is only for human name-like inputs.
    if len(ref) < 4 or not any(ch.isalpha() for ch in ref):
        return None

    query_norm = _normalize_name_key(ref)
    if not query_norm:
        return None

    try:
        rows = fetch_rows(USER_FROM_DISPLAY_CANDIDATES_SQL)
    except Exception:
        return None
    if not rows:
        return None

    query_tokens = set(query_norm.split())
    scored: List[tuple[float, str, str]] = []
    for row in rows:
        user_id = str(row.get("user_id") or "").strip()
        display_name = str(row.get("display_name") or "").strip()
        if not user_id or not display_name:
            continue

        display_norm = _normalize_name_key(display_name)
        if not display_norm:
            continue

        full_ratio = difflib.SequenceMatcher(None, query_norm, display_norm).ratio()
        token_ratio = max(
            (difflib.SequenceMatcher(None, query_norm, tok).ratio() for tok in display_norm.split()),
            default=0.0,
        )
        overlap_bonus = 0.03 if query_tokens.intersection(display_norm.split()) else 0.0
        score = max(full_ratio, token_ratio) + overlap_bonus
        scored.append((score, user_id, display_name))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_user_id, best_display = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    score_gap = best_score - second_score

    # Guardrails: require strong match, or very strong best score with clear gap.
    if best_score >= 0.84 and (score_gap >= 0.05 or best_score >= 0.92):
        return best_user_id, best_display
    return None


def resolve_user_reference(user_ref: str) -> tuple[str, str | None]:
    ref = user_ref.strip()
    if not ref:
        return user_ref, None

    direct = fetch_one(USER_EXISTS_SQL, (ref,))
    if direct:
        try:
            display = fetch_one(DISPLAY_FROM_USER_SQL, (ref,))
            return ref, (display["display_name"] if display else None)
        except Exception:
            return ref, None

    try:
        mapped = fetch_one(USER_FROM_DISPLAY_SQL, (ref,))
        if mapped:
            return mapped["user_id"], mapped["display_name"]
    except Exception:
        return ref, None

    # Partial display name fallback for user-friendly prompts like "for aarav".
    # Apply only for non-trivial alpha inputs and only if unique.
    if len(ref) >= 3 and any(ch.isalpha() for ch in ref):
        try:
            rows = fetch_rows(USER_FROM_DISPLAY_PARTIAL_SQL, (f"%{ref}%",))
            if len(rows) == 1:
                row = rows[0]
                return str(row["user_id"]), str(row["display_name"])
        except Exception:
            return ref, None

        fuzzy = _fuzzy_match_user_display_name(ref)
        if fuzzy is not None:
            return fuzzy

    return ref, None


def is_human_readable_track(row: Dict[str, Any]) -> bool:
    artist = str(row.get("artist_name") or "").strip().lower()
    track_name = str(row.get("track_name") or "").strip()
    track_id = str(row.get("track_id") or "").strip()

    if not track_name:
        return False
    if track_name.isdigit():
        return False
    if artist in {"", "__unknown_artist__"}:
        return False
    # If name is literally the id token, it's not user-friendly metadata.
    if track_id and track_name == track_id:
        return False
    return True


def readable_only(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return [r for r in rows if is_human_readable_track(r)][:limit]


def scan_limit(limit: int, multiplier: int | None = None) -> int:
    mul = multiplier if multiplier is not None else READABLE_SCAN_MULTIPLIER
    return max(limit * max(mul, 1), READABLE_SCAN_MIN, limit)


class VibeFeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    user_selected_vibe: str = Field(min_length=1, max_length=64)
    predicted_vibe: str | None = Field(default=None, max_length=64)


class InteractionFeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    action: str = Field(min_length=1, max_length=64)
    track_id: str | None = Field(default=None, max_length=256)
    source_endpoint: str | None = Field(default=None, max_length=64)
    model_mode: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    recommendation_rank: int | None = Field(default=None, ge=1, le=10000)
    context_vibe: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=128)
    signal_strength: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: Dict[str, Any] | None = None


def normalize_vibe(v: str | None) -> str | None:
    if v is None:
        return None
    cleaned = v.strip().lower()
    return cleaned or None


def normalize_interaction_action(action: str | None) -> str | None:
    if action is None:
        return None
    cleaned = action.strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return None
    return INTERACTION_ACTION_ALIASES.get(cleaned, cleaned)


def default_signal_strength(action: str) -> float:
    mapping = {
        "impression": 0.05,
        "query": 0.05,
        "search": 0.05,
        "skip": 0.20,
        "play": 1.00,
        "like": 2.00,
        "favorite": 2.50,
        "add_to_playlist": 3.00,
        "dislike": -1.50,
    }
    return mapping.get(action, 1.00)


def _safe_text(value: Any, max_len: int) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def write_demo_interactions(
    rows: Sequence[tuple[Any, ...]],
    *,
    fail_on_error: bool,
) -> int:
    if not ENABLE_DEMO_INTERACTION_LOGGING or not rows:
        return 0

    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)
    try:
        with conn.cursor() as cur:
            cur.executemany(DEMO_INTERACTION_INSERT_SQL, rows)
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        if fail_on_error:
            raise
        return 0
    finally:
        conn.close()


def log_impression_rows(
    *,
    user_id: str,
    source_endpoint: str,
    response_mode: str,
    items: Sequence[Dict[str, Any]],
    context_vibe: str | None = None,
    session_id: str | None = None,
) -> None:
    if not ENABLE_DEMO_INTERACTION_LOGGING:
        return
    if not user_id or not items:
        return

    rows: List[tuple[Any, ...]] = []
    for item in items:
        track_id = _safe_text(item.get("track_id"), 256)
        if not track_id:
            continue
        rec_rank = item.get("recommendation_rank") or item.get("favorite_rank") or item.get("global_rank_7d")
        try:
            rank = int(rec_rank) if rec_rank is not None else None
        except Exception:
            rank = None
        metadata_obj = {
            "track_name": _safe_text(item.get("track_name"), 512),
            "artist_name": _safe_text(item.get("artist_name"), 512),
            "source": "api_auto_impression",
        }
        metadata_json = json.dumps(metadata_obj, ensure_ascii=False)
        rows.append(
            (
                user_id,
                track_id,
                "impression",
                source_endpoint,
                response_mode or None,
                None,
                rank,
                context_vibe,
                session_id,
                default_signal_strength("impression"),
                metadata_json,
            )
        )

    if rows:
        write_demo_interactions(rows, fail_on_error=False)


def maybe_apply_feedback_override(track_id: str) -> Dict[str, Any]:
    consensus = fetch_one(TRACK_FEEDBACK_CONSENSUS_SQL, (track_id,))
    if not consensus:
        return {
            "applied": False,
            "reason": "no_feedback_for_track",
            "track_id": track_id,
        }

    top_vibe = normalize_vibe(str(consensus.get("vibe_label")))
    top_unique_users = int(consensus.get("top_unique_users") or 0)
    total_unique_users = int(consensus.get("total_unique_users") or 0)
    top_share = float(consensus.get("top_share") or 0.0)
    margin = float(consensus.get("margin") or 0.0)

    passes = (
        top_vibe is not None
        and total_unique_users >= CONSENSUS_MIN_USERS
        and top_share >= CONSENSUS_MIN_TOP_SHARE
        and margin >= CONSENSUS_MIN_MARGIN
    )

    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)
    try:
        with conn.cursor() as cur:
            if passes:
                confidence = min(0.980, max(0.700, top_share + 0.100))
                cur.execute(
                    UPSERT_OVERRIDE_SQL,
                    (
                        track_id,
                        top_vibe,
                        confidence,
                        total_unique_users,
                        top_share,
                        margin,
                        CONSENSUS_MIN_USERS,
                    ),
                )
            else:
                cur.execute(DELETE_OVERRIDE_SQL, (track_id,))
        conn.commit()
    finally:
        conn.close()

    return {
        "applied": bool(passes),
        "track_id": track_id,
        "consensus_vibe": top_vibe,
        "top_unique_users": top_unique_users,
        "total_unique_users": total_unique_users,
        "top_share": round(top_share, 4),
        "margin": round(margin, 4),
        "thresholds": {
            "min_users": CONSENSUS_MIN_USERS,
            "min_top_share": CONSENSUS_MIN_TOP_SHARE,
            "min_margin": CONSENSUS_MIN_MARGIN,
        },
    }


app = FastAPI(title="Music Recommendation API", version="0.1.0")


DEMO_CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Spotify Recommendation Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-0: #07142b;
      --bg-1: #0f2748;
      --card: rgba(9, 25, 48, 0.78);
      --ink: #edf5ff;
      --ink-dim: #b5c7e1;
      --accent: #2ae8b4;
      --accent-2: #6fc3ff;
      --border: rgba(130, 167, 214, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(circle at 10% 10%, rgba(42, 232, 180, 0.14), transparent 35%),
        radial-gradient(circle at 85% 15%, rgba(111, 195, 255, 0.18), transparent 42%),
        linear-gradient(165deg, var(--bg-0), var(--bg-1) 62%, #12315b);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 22px 14px 18px;
    }
    .title {
      margin: 0 0 6px;
      font-size: clamp(1.45rem, 2.8vw, 2.1rem);
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .sub {
      margin: 0 0 14px;
      color: var(--ink-dim);
      font-size: 0.94rem;
      line-height: 1.45;
    }
    .panel {
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 16px;
      backdrop-filter: blur(10px);
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
      overflow: hidden;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 12px 0;
    }
    .chip {
      border: 1px solid var(--border);
      color: var(--ink);
      background: rgba(26, 53, 92, 0.75);
      border-radius: 999px;
      font-size: 0.82rem;
      padding: 6px 10px;
      cursor: pointer;
    }
    .chip:hover {
      border-color: rgba(111, 195, 255, 0.75);
    }
    .chat {
      height: min(64vh, 620px);
      padding: 12px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .bubble {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      white-space: pre-wrap;
      line-height: 1.45;
      max-width: 92%;
      font-size: 0.93rem;
    }
    .u {
      align-self: flex-end;
      background: rgba(28, 89, 154, 0.55);
      border-color: rgba(111, 195, 255, 0.5);
    }
    .a {
      align-self: flex-start;
      background: rgba(18, 48, 84, 0.85);
    }
    .meta {
      color: var(--ink-dim);
      font-size: 0.79rem;
      margin-top: 7px;
    }
    .composer {
      display: flex;
      gap: 8px;
      border-top: 1px solid var(--border);
      padding: 10px;
    }
    .composer input {
      flex: 1;
      border: 1px solid var(--border);
      background: rgba(6, 21, 41, 0.7);
      color: var(--ink);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      outline: none;
    }
    .composer input:focus {
      border-color: rgba(42, 232, 180, 0.6);
      box-shadow: 0 0 0 2px rgba(42, 232, 180, 0.18);
    }
    .composer button {
      border: 0;
      border-radius: 10px;
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: #042138;
      font-weight: 700;
      padding: 10px 14px;
      cursor: pointer;
    }
    .composer button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .foot {
      margin-top: 10px;
      color: var(--ink-dim);
      font-size: 0.79rem;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Music Assistant Demo</h1>
    <p class="sub">Ask naturally for recommendations and moods. This UI hits the live recommendation API directly.</p>
    <div class="panel">
      <div class="chip-row">
        <button class="chip" data-prompt="Recommend songs for Aarav Edwards">Aarav recs</button>
        <button class="chip" data-prompt="party songs for Aarav Edwards">Aarav party</button>
        <button class="chip" data-prompt="sad songs for Abigail Johnson">Abigail sad</button>
        <button class="chip" data-prompt="romantic songs for Camila Lopez">Camila romantic</button>
        <button class="chip" data-prompt="favorites of Ariana Reed">Ariana favs</button>
      </div>
      <div id="chat" class="chat"></div>
      <div class="composer">
        <input id="input" type="text" placeholder="Type: Recommend songs for Aarav Edwards" />
        <button id="send">Send</button>
      </div>
    </div>
    <div class="foot">Supported vibe labels: chill, focus, happy, sad, party, energetic, romantic.</div>
  </div>

  <script>
    const VIBES = ["chill", "focus", "happy", "sad", "party", "energetic", "romantic"];
    const qs = new URLSearchParams(window.location.search);
    const apiParam = qs.get("api_base");
    const API_BASE = (apiParam && apiParam.trim()) ? apiParam.trim().replace(/\/+$/, "") : window.location.origin;
    const timeoutParam = Number(qs.get("timeout_ms"));
    const FETCH_TIMEOUT_MS = Number.isFinite(timeoutParam) && timeoutParam >= 5000 ? timeoutParam : 25000;
    const state = {
      sessionId: "demo_web_" + Date.now(),
      lastUserRef: null,
      lastVibe: null,
      lastItems: [],
      lastOffset: 0,
      lastMode: null,
    };

    const chatEl = document.getElementById("chat");
    const inputEl = document.getElementById("input");
    const sendEl = document.getElementById("send");

    function addBubble(text, role = "a", meta = "") {
      const bubble = document.createElement("div");
      bubble.className = "bubble " + role;
      bubble.textContent = text;
      if (meta) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = meta;
        bubble.appendChild(m);
      }
      chatEl.appendChild(bubble);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function formatItems(items, limit = 10) {
      const rows = [];
      const sliced = items.slice(0, Math.max(1, limit));
      for (let i = 0; i < sliced.length; i++) {
        const row = sliced[i] || {};
        const name = row.track_name || row.track_id || "Unknown track";
        const artist = row.artist_name || row.artist_id || "Unknown artist";
        rows.push((i + 1) + ". " + name + " - " + artist);
      }
      return rows.join("\n");
    }

    function keyOf(row) {
      const track = String(row.track_id || "").trim().toLowerCase();
      const name = String(row.track_name || "").trim().toLowerCase();
      const artist = String(row.artist_name || "").trim().toLowerCase();
      return track ? ("id:" + track) : ("na:" + name + "|" + artist);
    }

    function parseLimit(text, fallback = 10) {
      const m = text.match(/\b([1-9]|1[0-9]|20)\b/);
      return m ? Number(m[1]) : fallback;
    }

    function extractVibe(text) {
      const lower = text.toLowerCase();
      for (const vibe of VIBES) {
        if (lower.includes(vibe)) {
          return vibe;
        }
      }
      return null;
    }

    function extractUserRef(text) {
      const lower = text.toLowerCase();
      const mx = text.match(/\b(?:for|to)\s+([A-Za-z][A-Za-z .'\-]{1,80})$/);
      if (mx) {
        let candidate = mx[1].trim();
        candidate = candidate.replace(/\b(now|please|pls|then)\b$/i, "").trim();
        if (/^(him|her|them)$/i.test(candidate)) {
          return state.lastUserRef || null;
        }
        if (/^abi(\b|gail)/i.test(candidate)) return "Abigail Johnson";
        if (/^aarav\b/i.test(candidate)) return "Aarav Edwards";
        if (/^ariana\b/i.test(candidate)) return "Ariana Reed";
        if (/^camila\b/i.test(candidate)) return "Camila Lopez";
        if (/^(caleb|rogers)\b/i.test(candidate)) return "Caleb Rogers";
        return candidate;
      }
      if (/aarav/.test(lower)) return "Aarav Edwards";
      if (/abigail|abi\b/.test(lower)) return "Abigail Johnson";
      if (/ariana/.test(lower)) return "Ariana Reed";
      if (/camila/.test(lower)) return "Camila Lopez";
      if (/caleb|rogers/.test(lower)) return "Caleb Rogers";
      if (/\b(him|her|them|for him|for her)\b/.test(lower) && state.lastUserRef) {
        return state.lastUserRef;
      }
      return null;
    }

    function isMore(text) {
      const lower = text.toLowerCase().trim();
      return (
        /^(?:\d+\s+more|more)$/.test(lower) ||
        /\bgive\s+\d+\s+more\b/.test(lower) ||
        /\banother\s+\d+\b/.test(lower)
      );
    }

    async function fetchWithTimeout(url, options = {}, timeoutMs = FETCH_TIMEOUT_MS) {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, { ...options, signal: controller.signal });
      } catch (err) {
        if (err && err.name === "AbortError") {
          throw new Error(
            "Request timed out after " + Math.round(timeoutMs / 1000) + "s. " +
            "API base: " + API_BASE + ". " +
            "If local backend cannot reach DB, open /demo?api_base=https://uq3i5irvfr.us-east-2.awsapprunner.com"
          );
        }
        throw err;
      } finally {
        clearTimeout(id);
      }
    }

    async function apiGet(path, params = {}) {
      const url = new URL(path, API_BASE);
      for (const [k, v] of Object.entries(params)) {
        if (v !== null && v !== undefined && String(v).length > 0) {
          url.searchParams.set(k, String(v));
        }
      }
      const resp = await fetchWithTimeout(url.toString(), { method: "GET" });
      if (!resp.ok) {
        let detail = resp.statusText;
        try {
          const payload = await resp.json();
          detail = payload.detail || JSON.stringify(payload);
        } catch (_err) {}
        throw new Error(resp.status + " " + detail);
      }
      return await resp.json();
    }

    async function apiPost(path, payload) {
      const url = new URL(path, API_BASE);
      const resp = await fetchWithTimeout(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        let detail = resp.statusText;
        try {
          const data = await resp.json();
          detail = data.detail || JSON.stringify(data);
        } catch (_err) {}
        throw new Error(resp.status + " " + detail);
      }
      return await resp.json();
    }

    async function handleMessage(text) {
      const raw = text.trim();
      const lower = raw.toLowerCase();
      const norm = lower.replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
      if (!raw) return;

      if (isMore(raw)) {
        const pageSize = parseLimit(raw, 5);
        const start = state.lastOffset;
        const end = start + pageSize;
        const page = state.lastItems.slice(start, end);
        if (!page.length) {
          addBubble("No more unseen tracks in the current list. Ask for a different user or vibe.");
          return;
        }
        state.lastOffset = end;
        addBubble(formatItems(page, page.length));
        return;
      }

      const likeMatch = lower.match(/^like\s+([1-9]|1[0-9]|20)$/);
      if (likeMatch && state.lastItems.length && state.lastUserRef) {
        const idx = Number(likeMatch[1]) - 1;
        const row = state.lastItems[idx];
        if (!row) {
          addBubble("That item number is out of range.");
          return;
        }
        const payload = {
          user_id: state.lastUserRef,
          track_id: row.track_id,
          action: "like",
          source_endpoint: "/demo",
          recommendation_rank: idx + 1,
          session_id: state.sessionId,
        };
        const result = await apiPost("/feedback/interaction", payload);
        addBubble("Feedback saved as '" + result.status + "'.");
        return;
      }

      const asksGenreCapability =
        (/\bgenres?\b/.test(norm) && /\b(what|which|can|supported|give)\b/.test(norm)) ||
        /\b(what\s+genres|genres?\s+can\s+you|what\s+vibes?|supported\s+vibes?)\b/.test(norm);
      if (asksGenreCapability) {
        addBubble(
          "This demo is grounded on vibe labels (not full genre metadata).\n" +
          "Supported vibe labels: chill, focus, happy, sad, party, energetic, romantic.\n\n" +
          "Examples:\n" +
          "- romantic songs for Aarav Edwards\n" +
          "- party songs for Abigail Johnson\n" +
          "- favorites of Ariana Reed"
        );
        return;
      }

      if (/(help|what can you do|vibes)/.test(lower)) {
        addBubble(
          "You can ask for:\n" +
          "- Recommend songs for Aarav Edwards\n" +
          "- party songs for Abigail Johnson\n" +
          "- chill songs\n" +
          "- favorites of Ariana Reed\n" +
          "- like 1"
        );
        return;
      }

      const asksTrackCount = /\bhow many\s+tracks?\b|\btotal\s+tracks?\b|\btracks?\s+in\s+total\b|\btrack\s+count\b/.test(norm);
      const asksUserCount = /\bhow many\s+users?\b|\btotal\s+users?\b|\buser\s+count\b/.test(norm);
      const asksEventCount = /\bhow many\s+events?\b|\btotal\s+events?\b|\bevent\s+count\b/.test(norm);
      const asksMetrics = /\bmetrics?\b|\bhealth\b|\bstatus\b/.test(norm) || asksTrackCount || asksUserCount || asksEventCount;
      if (asksMetrics) {
        const m = await apiGet("/metrics/model");
        if (asksTrackCount && !asksUserCount && !asksEventCount) {
          addBubble("Total tracks in catalog: " + (m.tracks || 0));
          return;
        }
        if (asksUserCount && !asksTrackCount && !asksEventCount) {
          addBubble("Total users in model: " + (m.users || 0));
          return;
        }
        if (asksEventCount && !asksTrackCount && !asksUserCount) {
          addBubble("Total events in model: " + (m.events || 0));
          return;
        }
        addBubble(
          "System status:\n" +
          "events: " + (m.events || 0) + "\n" +
          "users: " + (m.users || 0) + "\n" +
          "tracks: " + (m.tracks || 0)
        );
        return;
      }

      let vibe = extractVibe(raw);
      let userRef = extractUserRef(raw);
      const wantsMostListened = /\bmost\s+listened\b|\btop\s+listened\b|\blistened\s+the\s+most\b|\btop\s+tracks?\b/.test(norm);
      const wantsFavorites = /\bfavs?\b|\bfavorites?\b|\bfavourites?\b/.test(norm) || wantsMostListened;
      let wantsRecs = /\brecommend\b|\brecs?\b|\bsuggest\b|\bsongs?\b|\bmusic\b|\bgive\s+some\b/.test(norm);
      const pageSize = parseLimit(raw, 10);
      const hasPronounUser = /\b(him|her|them)\b/.test(norm);

      if (wantsFavorites && !userRef && state.lastUserRef) {
        userRef = state.lastUserRef;
      }

      if (!userRef && hasPronounUser) {
        addBubble("Tell me which user you mean, for example: songs for Aarav Edwards.");
        return;
      }

      // If user asks "for abi now" after a vibe request, continue with that vibe.
      if (!vibe && userRef && state.lastVibe && !wantsFavorites) {
        const followupSamePattern = /\b(for|give|some|songs?|music|now|then)\b/.test(norm);
        if (followupSamePattern) {
          vibe = state.lastVibe;
        }
      }

      // If a user is provided but intent words are missing, assume recommendations.
      if (userRef && !vibe && !wantsFavorites && !wantsRecs) {
        wantsRecs = true;
      }

      if (vibe && userRef) {
        const [recsData, vibeData] = await Promise.all([
          apiGet("/recs/" + encodeURIComponent(userRef), { limit: 40, fallback_to_trending: true, session_id: state.sessionId }),
          apiGet("/vibe", { vibe: vibe, limit: 100, user_id: userRef, session_id: state.sessionId }),
        ]);
        const recItems = Array.isArray(recsData.items) ? recsData.items : [];
        const vibeItems = Array.isArray(vibeData.items) ? vibeData.items : [];
        const vibeKeys = new Set(vibeItems.map(keyOf));
        const overlap = recItems.filter((r) => vibeKeys.has(keyOf(r)));
        const overlapKeys = new Set(overlap.map(keyOf));
        const vibeFill = vibeItems.filter((r) => !overlapKeys.has(keyOf(r)));
        const combined = overlap.concat(vibeFill);
        const shownCount = Math.min(pageSize, combined.length);
        state.lastUserRef = userRef;
        state.lastVibe = vibe;
        state.lastMode = "personalized_vibe_blended";
        state.lastItems = combined.length ? combined : recItems;
        state.lastOffset = Math.min(pageSize, state.lastItems.length);
        if (overlap.length >= pageSize) {
          addBubble("Here are " + pageSize + " personalized '" + vibe + "' songs for " + userRef + ":\n\n" + formatItems(combined, pageSize));
          return;
        }
        if (overlap.length > 0 && combined.length > overlap.length) {
          const added = Math.max(0, shownCount - overlap.length);
          addBubble(
            "I found " + overlap.length + " personalized '" + vibe + "' songs for " + userRef +
            ". Added " + added + " more popular '" + vibe + "' tracks to complete the list:\n\n" +
            formatItems(combined, pageSize)
          );
          return;
        }
        if (combined.length) {
          addBubble(
            "I could not find personalized '" + vibe + "' songs for " + userRef +
            " yet. Here are " + shownCount + " popular '" + vibe + "' tracks:\n\n" +
            formatItems(combined, pageSize)
          );
          return;
        }
        if (recItems.length) {
          addBubble(
            "I could not find enough '" + vibe + "' tracks right now. Here are personalized songs for " +
            userRef + " instead:\n\n" + formatItems(recItems, pageSize)
          );
          return;
        }
        addBubble("I couldn't find enough tracks for that request yet. Try another vibe or user.");
        return;
      }

      if (userRef && wantsFavorites) {
        const favData = await apiGet("/favorites/" + encodeURIComponent(userRef), {
          limit: Math.max(pageSize, 10),
          fallback_to_recs: false,
          session_id: state.sessionId,
        });
        const items = Array.isArray(favData.items) ? favData.items : [];
        state.lastUserRef = userRef;
        state.lastVibe = null;
        state.lastMode = "favorites";
        state.lastItems = items;
        state.lastOffset = Math.min(pageSize, state.lastItems.length);
        const requestedLabel = wantsMostListened ? "most listened tracks" : "favorites";
        const prefix = String(favData.message || "").trim();
        if (items.length) {
          addBubble(
            "Here are " + Math.min(pageSize, items.length) + " " + requestedLabel + " for " + userRef + ":\n\n" +
            formatItems(items, pageSize)
          );
        } else {
          if (prefix) {
            addBubble(prefix.replace(/\byour\b/ig, userRef + "'s"));
          } else {
            addBubble("No " + requestedLabel + " available yet for " + userRef + ".");
          }
        }
        return;
      }

      if (wantsFavorites && !userRef) {
        addBubble("Tell me which user you mean, for example: favorites for Aarav Edwards or most listened for Abigail Johnson.");
        return;
      }

      if (userRef && wantsRecs) {
        const recsData = await apiGet("/recs/" + encodeURIComponent(userRef), {
          limit: Math.max(pageSize, 10),
          fallback_to_trending: true,
          session_id: state.sessionId,
        });
        const items = Array.isArray(recsData.items) ? recsData.items : [];
        state.lastUserRef = userRef;
        state.lastVibe = null;
        state.lastMode = "recs";
        state.lastItems = items;
        state.lastOffset = Math.min(pageSize, state.lastItems.length);
        addBubble("Here are " + Math.min(pageSize, items.length) + " recommendations for " + userRef + ":\n\n" + formatItems(items, pageSize));
        return;
      }

      if (vibe) {
        const vibeData = await apiGet("/vibe", { vibe: vibe, limit: Math.max(pageSize, 10), session_id: state.sessionId });
        const items = Array.isArray(vibeData.items) ? vibeData.items : [];
        state.lastVibe = vibe;
        state.lastMode = "vibe";
        state.lastItems = items;
        state.lastOffset = Math.min(pageSize, state.lastItems.length);
        addBubble("Here are " + Math.min(pageSize, items.length) + " '" + vibe + "' tracks:\n\n" + formatItems(items, pageSize));
        return;
      }

      const trending = await apiGet("/trending", { limit: Math.max(pageSize, 10), session_id: state.sessionId });
      const items = Array.isArray(trending.items) ? trending.items : [];
      state.lastMode = "trending";
      state.lastItems = items;
      state.lastOffset = Math.min(pageSize, state.lastItems.length);
      addBubble("I interpreted that as a general music request. Here are popular tracks:\n\n" + formatItems(items, pageSize));
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = "";
      addBubble(text, "u");
      sendEl.disabled = true;
      try {
        await handleMessage(text);
      } catch (err) {
        addBubble("Request failed: " + (err && err.message ? err.message : String(err)));
      } finally {
        sendEl.disabled = false;
        inputEl.focus();
      }
    }

    sendEl.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        sendMessage();
      }
    });

    document.querySelectorAll(".chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        inputEl.value = btn.getAttribute("data-prompt") || "";
        sendMessage();
      });
    });

    addBubble(
      "Live demo assistant is ready.\nAPI base: " + API_BASE + "\n\nTry:\n- Recommend songs for Aarav Edwards\n- party songs for Abigail Johnson\n- favorites of Ariana Reed\n- like 1"
    );
  </script>
</body>
</html>
"""


@app.middleware("http")
async def force_json_utf8_charset(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type.lower():
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.get("/", include_in_schema=False)
def root_redirect_to_demo() -> RedirectResponse:
    return RedirectResponse(url="/demo")


@app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    return HTMLResponse(content=DEMO_CHAT_HTML)


@app.get("/metrics/model")
def get_model_metrics() -> Dict[str, Any]:
    try:
        rows = fetch_rows(MODEL_METRICS_SQL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"metrics query failed: {exc}")
    if not rows:
        raise HTTPException(status_code=404, detail="No model metrics found.")
    return rows[0]


@app.get("/metrics/recsource")
def get_recsource_metrics() -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "serving_priority": ["hybrid", "mf", "trending"],
        "feature_flags": {
            "USE_HYBRID_RECS": USE_HYBRID_RECS,
            "USE_ML_RECS": USE_ML_RECS,
            "USE_DENSE_RECS": USE_DENSE_RECS,
        },
        "sources": {},
    }

    for name, latest_sql, count_sql in [
        ("hybrid", REC_SOURCE_LATEST_HYBRID_SQL, REC_SOURCE_COUNTS_HYBRID_SQL),
        ("mf", REC_SOURCE_LATEST_MF_SQL, REC_SOURCE_COUNTS_MF_SQL),
        ("dense", None, REC_SOURCE_COUNTS_DENSE_SQL),
        ("trending", None, REC_SOURCE_COUNTS_TRENDING_SQL),
    ]:
        try:
            latest = fetch_one(latest_sql) if latest_sql else None
            counts = fetch_one(count_sql) or {}
            rows = int(counts.get("rows") or 0)
            users = int(counts.get("users") or 0)
            response["sources"][name] = {
                "available": rows > 0,
                "rows": rows,
                "users": users if name != "trending" else None,
                "model_version": (latest.get("model_version") if latest else None),
                "generated_at": (latest.get("generated_at") if latest else None),
            }
        except Exception as exc:
            response["sources"][name] = {
                "available": False,
                "error": str(exc),
            }

    return response


@app.get("/trending")
def get_trending(
    limit: int = Query(default=20, ge=1, le=200),
    user_id: str | None = Query(default=None, max_length=128),
    session_id: str | None = Query(default=None, max_length=128),
) -> Dict[str, Any]:
    limit_scan = scan_limit(limit)
    resolved_user_id: str | None = None
    if user_id:
        resolved_user_id, _display_name = resolve_user_reference(user_id)
    try:
        rows = fetch_rows(TRENDING_SQL, (limit_scan,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trending query failed: {exc}")

    rows = readable_only(rows, limit)
    mode = "trending_7d"
    message = None
    if not rows:
        try:
            rows = fetch_rows(NAMED_TRENDING_SQL, (limit,))
            mode = "trending_7d_named_fallback"
            if rows:
                message = "Primary trending was ID-heavy; returned readable all-time metadata fallback."
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"trending named fallback failed: {exc}")

    resp: Dict[str, Any] = {
        "mode": mode,
        "count": len(rows),
        "items": rows,
    }
    if message:
        resp["message"] = message
    if not rows:
        resp["message"] = "No human-readable tracks available in current result slice."
    if resolved_user_id:
        log_impression_rows(
            user_id=resolved_user_id,
            source_endpoint="/trending",
            response_mode=mode,
            items=rows,
            session_id=session_id,
        )
    return resp


@app.get("/search/tracks")
def search_tracks(query: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="query is required.")
    limit_scan = scan_limit(limit, multiplier=30)
    try:
        rows = fetch_rows(SEARCH_TRACKS_SQL, (cleaned, cleaned, limit_scan))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"track search failed: {exc}")
    rows = readable_only(rows, limit)
    resp: Dict[str, Any] = {
        "query": cleaned,
        "count": len(rows),
        "items": rows,
    }
    if not rows:
        resp["message"] = "No human-readable track metadata found for this query."
    return resp


@app.get("/vibe")
def get_vibe_tracks(
    vibe: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
    user_id: str | None = Query(default=None, max_length=128),
    session_id: str | None = Query(default=None, max_length=128),
) -> Dict[str, Any]:
    cleaned = vibe.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="vibe is required.")
    resolved_user_id: str | None = None
    if user_id:
        resolved_user_id, _display_name = resolve_user_reference(user_id)
    labels = vibe_labels(cleaned)
    keywords = vibe_keywords(cleaned)
    rows: List[Dict[str, Any]] = []
    limit_scan = scan_limit(limit, multiplier=100)

    # Preferred path: engineered vibe features table
    try:
        rows = fetch_rows(VIBE_FEATURE_TRACKS_SQL, (labels, limit_scan))
    except Exception as exc:
        # If engineered table is not built yet, continue with catalog/genre fallback.
        print(f"vibe feature query unavailable, falling back to catalog rules: {exc}")
        rows = []
    readable = readable_only(rows, limit)
    if readable:
        if resolved_user_id:
            log_impression_rows(
                user_id=resolved_user_id,
                source_endpoint="/vibe",
                response_mode="vibe_match_features",
                items=readable,
                context_vibe=cleaned,
                session_id=session_id,
            )
        return {
            "mode": "vibe_match_features",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": len(readable),
            "items": readable,
        }

    # Fallback path: direct genre keyword search in track catalog.
    try:
        fallback = fetch_rows(VIBE_CATALOG_FALLBACK_SQL, (keywords, limit_scan))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vibe catalog fallback failed: {exc}")
    fallback = readable_only(fallback, limit)
    if fallback:
        if resolved_user_id:
            log_impression_rows(
                user_id=resolved_user_id,
                source_endpoint="/vibe",
                response_mode="vibe_match_catalog",
                items=fallback,
                context_vibe=cleaned,
                session_id=session_id,
            )
        return {
            "mode": "vibe_match_catalog",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": len(fallback),
            "items": fallback,
        }

    # Last-resort fallback: trending tracks
    try:
        vibe_named = fetch_rows(VIBE_NAMED_FALLBACK_SQL, (keywords, limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vibe named keyword fallback failed: {exc}")
    if vibe_named:
        if resolved_user_id:
            log_impression_rows(
                user_id=resolved_user_id,
                source_endpoint="/vibe",
                response_mode="vibe_fallback_named_keyword",
                items=vibe_named,
                context_vibe=cleaned,
                session_id=session_id,
            )
        return {
            "mode": "vibe_fallback_named_keyword",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": len(vibe_named),
            "items": vibe_named,
            "message": f"Here are the closest '{cleaned}' tracks from your catalog.",
        }

    # Last-resort fallback: generic named trending tracks
    try:
        trending = fetch_rows(TRENDING_SQL, (limit_scan,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vibe final fallback failed: {exc}")
    trending = readable_only(trending, limit)
    if not trending:
        try:
            trending = fetch_rows(NAMED_TRENDING_SQL, (limit,))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"vibe named fallback failed: {exc}")
    if not trending:
        return {
            "mode": "vibe_no_readable_metadata",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": 0,
            "items": [],
            "message": f"I couldn't find enough readable '{cleaned}' tracks yet. Try another vibe.",
        }
    if resolved_user_id:
        log_impression_rows(
            user_id=resolved_user_id,
            source_endpoint="/vibe",
            response_mode="vibe_fallback_named_trending",
            items=trending,
            context_vibe=cleaned,
            session_id=session_id,
        )
    return {
        "mode": "vibe_fallback_named_trending",
        "vibe": cleaned,
        "labels": labels,
        "keywords": keywords,
        "count": len(trending),
        "items": trending,
        "message": f"Here are popular tracks closest to the '{cleaned}' vibe.",
    }


@app.post("/feedback/vibe")
def submit_vibe_feedback(payload: VibeFeedbackRequest) -> Dict[str, Any]:
    user_id = payload.user_id.strip()
    track_id = payload.track_id.strip()
    user_selected_vibe = normalize_vibe(payload.user_selected_vibe)
    predicted_vibe = normalize_vibe(payload.predicted_vibe)

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    if not track_id:
        raise HTTPException(status_code=400, detail="track_id is required.")
    if not user_selected_vibe:
        raise HTTPException(status_code=400, detail="user_selected_vibe is required.")

    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)
    try:
        with conn.cursor() as cur:
            cur.execute(
                FEEDBACK_UPSERT_SQL,
                (user_id, track_id, predicted_vibe, user_selected_vibe),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"feedback write failed: {exc}")
    finally:
        conn.close()

    try:
        override_result = maybe_apply_feedback_override(track_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"override update failed: {exc}")

    return {
        "status": "accepted",
        "feedback": {
            "user_id": user_id,
            "track_id": track_id,
            "user_selected_vibe": user_selected_vibe,
            "predicted_vibe": predicted_vibe,
        },
        "override_update": override_result,
    }


@app.post("/feedback/interaction")
def submit_interaction_feedback(payload: InteractionFeedbackRequest) -> Dict[str, Any]:
    requested_user_ref = payload.user_id.strip()
    resolved_user_id, display_name = resolve_user_reference(requested_user_ref)
    action = normalize_interaction_action(payload.action)
    if action is None:
        raise HTTPException(status_code=400, detail="action is required.")
    if action not in ALLOWED_INTERACTION_ACTIONS:
        allowed = ", ".join(sorted(ALLOWED_INTERACTION_ACTIONS))
        raise HTTPException(status_code=400, detail=f"unsupported action '{action}'. Allowed: {allowed}")

    track_id = _safe_text(payload.track_id, 256)
    if action in TRACK_REQUIRED_ACTIONS and not track_id:
        raise HTTPException(status_code=400, detail=f"track_id is required for action '{action}'.")

    source_endpoint = _safe_text(payload.source_endpoint, 64) or "/feedback/interaction"
    model_mode = _safe_text(payload.model_mode, 128)
    model_version = _safe_text(payload.model_version, 128)
    context_vibe = normalize_vibe(payload.context_vibe)
    session_id = _safe_text(payload.session_id, 128)
    signal_strength = float(payload.signal_strength) if payload.signal_strength is not None else default_signal_strength(action)
    recommendation_rank = payload.recommendation_rank

    metadata_obj: Dict[str, Any] = dict(payload.metadata or {})
    metadata_obj["source"] = "api_feedback"
    if requested_user_ref != resolved_user_id:
        metadata_obj["requested_user_ref"] = requested_user_ref
    metadata_json = json.dumps(metadata_obj, ensure_ascii=False)

    row = (
        resolved_user_id,
        track_id,
        action,
        source_endpoint,
        model_mode,
        model_version,
        recommendation_rank,
        context_vibe,
        session_id,
        signal_strength,
        metadata_json,
    )

    try:
        write_demo_interactions([row], fail_on_error=True)
    except Exception as exc:
        msg = str(exc)
        if "demo_interactions" in msg and "does not exist" in msg:
            raise HTTPException(
                status_code=500,
                detail="interaction feedback table missing: apply sql/postgres_schema.sql first.",
            )
        raise HTTPException(status_code=500, detail=f"interaction feedback write failed: {exc}")

    response: Dict[str, Any] = {
        "status": "accepted",
        "interaction": {
            "user_id": resolved_user_id,
            "track_id": track_id,
            "action": action,
            "source_endpoint": source_endpoint,
            "context_vibe": context_vibe,
            "recommendation_rank": recommendation_rank,
            "signal_strength": signal_strength,
            "session_id": session_id,
        },
    }
    if display_name:
        response["user_display_name"] = display_name
    return response


@app.get("/favorites/{user_id}")
def get_user_favorites(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    fallback_to_recs: bool = Query(default=True),
    session_id: str | None = Query(default=None, max_length=128),
) -> Dict[str, Any]:
    requested_user_ref = user_id.strip()
    looks_like_display_name = " " in requested_user_ref
    resolved_user_id, display_name = resolve_user_reference(user_id)
    unresolved_display_name = looks_like_display_name and display_name is None and resolved_user_id == requested_user_ref
    limit_scan = scan_limit(limit, multiplier=50)

    try:
        favorites = fetch_rows(USER_FAVORITES_SQL, (resolved_user_id, limit_scan))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"favorites query failed: {exc}")
    favorites = readable_only(favorites, limit)
    if favorites:
        resp: Dict[str, Any] = {
            "user_id": resolved_user_id,
            "mode": "user_favorites",
            "count": len(favorites),
            "items": favorites,
            "message": "Here are your top listened tracks.",
        }
        if display_name:
            resp["user_display_name"] = display_name
        if unresolved_display_name:
            resp["user_reference_note"] = (
                "Display name was not found in music.user_profiles; interpreted input as raw user_id."
            )
        log_impression_rows(
            user_id=resolved_user_id,
            source_endpoint="/favorites",
            response_mode="user_favorites",
            items=favorites,
            session_id=session_id,
        )
        return resp

    if not fallback_to_recs:
        resp: Dict[str, Any] = {
            "user_id": resolved_user_id,
            "mode": "no_favorites",
            "count": 0,
            "items": [],
            "message": "No readable top-listened tracks available for this user yet.",
        }
        if display_name:
            resp["user_display_name"] = display_name
        if unresolved_display_name:
            resp["user_reference_note"] = (
                "Display name was not found in music.user_profiles; interpreted input as raw user_id."
            )
        return resp

    fallback = get_recs(
        user_id=resolved_user_id,
        limit=limit,
        fallback_to_trending=True,
        session_id=session_id,
    )
    if isinstance(fallback, dict):
        fallback_mode = str(fallback.get("mode") or "")
        if fallback_mode.startswith("personalized"):
            fallback["mode"] = "favorites_fallback_personalized"
            fallback["message"] = "No readable favorites found; returned personalized recommendations."
        elif fallback_mode.startswith("fallback"):
            fallback["mode"] = "favorites_fallback_trending"
            fallback["message"] = "No readable favorites found; returned trending tracks."
    if display_name:
        fallback["user_display_name"] = display_name
    return fallback


@app.get("/recs/{user_id}")
def get_recs(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    fallback_to_trending: bool = Query(default=True),
    session_id: str | None = Query(default=None, max_length=128),
) -> Dict[str, Any]:
    requested_user_ref = user_id.strip()
    looks_like_display_name = " " in requested_user_ref
    resolved_user_id, display_name = resolve_user_reference(user_id)
    unresolved_display_name = looks_like_display_name and display_name is None and resolved_user_id == requested_user_ref
    limit_scan = scan_limit(limit, multiplier=50)
    recs: List[Dict[str, Any]] = []
    mode = "personalized_hybrid_ready"
    source_errors: List[str] = []
    attempted_sources: List[str] = []

    if USE_HYBRID_RECS:
        attempted_sources.append("hybrid")
        try:
            recs = fetch_rows(USER_RECS_HYBRID_SQL, (resolved_user_id, limit_scan))
            mode = "personalized_hybrid_ready"
        except Exception as exc:
            source_errors.append(f"hybrid:{exc}")
            recs = []

    if not recs and USE_ML_RECS:
        attempted_sources.append("mf")
        try:
            recs = fetch_rows(USER_RECS_MF_SQL, (resolved_user_id, limit_scan))
            mode = "personalized_mf_ready"
        except Exception as exc:
            source_errors.append(f"mf:{exc}")
            recs = []

    if not recs and USE_DENSE_RECS:
        attempted_sources.append("dense")
        try:
            recs = fetch_rows(USER_RECS_DENSE_SQL, (resolved_user_id, limit_scan))
            mode = "personalized_dense_1000"
        except Exception as exc:
            source_errors.append(f"dense:{exc}")
            recs = []

    recs = readable_only(recs, limit)

    if recs:
        resp: Dict[str, Any] = {
            "user_id": resolved_user_id,
            "mode": mode,
            "count": len(recs),
            "items": recs,
        }
        if display_name:
            resp["user_display_name"] = display_name
        if source_errors:
            resp["note"] = "Higher-priority source unavailable in this run; served next recommendation source."
        if attempted_sources:
            resp["sources_attempted"] = attempted_sources
        if unresolved_display_name:
            resp["user_reference_note"] = (
                "Display name was not found in music.user_profiles; interpreted input as raw user_id."
            )
        log_impression_rows(
            user_id=resolved_user_id,
            source_endpoint="/recs",
            response_mode=mode,
            items=recs,
            session_id=session_id,
        )
        return resp

    if not fallback_to_trending:
        resp: Dict[str, Any] = {
            "user_id": user_id,
            "mode": "no_results",
            "count": 0,
            "items": [],
        }
        if attempted_sources:
            resp["sources_attempted"] = attempted_sources
        if source_errors:
            resp["source_errors"] = source_errors
        return resp

    try:
        fallback = fetch_rows(TRENDING_SQL, (limit_scan,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"fallback query failed: {exc}")
    fallback = readable_only(fallback, limit)
    mode = "fallback_trending"
    message = "No personalized readable recommendations available; returned global trending tracks."
    if not fallback:
        try:
            fallback = fetch_rows(NAMED_TRENDING_SQL, (limit,))
            if fallback:
                mode = "fallback_named_trending"
                message = "Personalized slice is ID-heavy; returned readable all-time named fallback."
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"named fallback query failed: {exc}")

    resp: Dict[str, Any] = {
        "user_id": resolved_user_id,
        "mode": mode,
        "message": message,
        "count": len(fallback),
        "items": fallback,
    }
    if attempted_sources:
        resp["sources_attempted"] = attempted_sources
    if display_name:
        resp["user_display_name"] = display_name
    if unresolved_display_name:
        resp["user_reference_note"] = (
            "Display name was not found in music.user_profiles; interpreted input as raw user_id."
        )
    if not fallback:
        resp["message"] = "No human-readable recommendations available for this user in current dataset slice."
    log_impression_rows(
        user_id=resolved_user_id,
        source_endpoint="/recs",
        response_mode=mode,
        items=fallback,
        session_id=session_id,
    )
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
