#!/usr/bin/env python3
"""
Lightweight serving API for music recommendations.

Endpoints:
- GET /metrics/model
- GET /trending?limit=20
- GET /recs/{user_id}?limit=20
- GET /search/tracks?query=...&limit=10
- GET /vibe?vibe=...&limit=10
- POST /feedback/vibe
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from psycopg import connect
from psycopg.rows import dict_row

CONSENSUS_MIN_USERS = int(os.environ.get("VIBE_FEEDBACK_MIN_USERS", "15"))
CONSENSUS_MIN_TOP_SHARE = float(os.environ.get("VIBE_FEEDBACK_MIN_TOP_SHARE", "0.70"))
CONSENSUS_MIN_MARGIN = float(os.environ.get("VIBE_FEEDBACK_MIN_MARGIN", "0.15"))
READABLE_SCAN_MIN = int(os.environ.get("READABLE_SCAN_MIN", "500"))
READABLE_SCAN_MULTIPLIER = int(os.environ.get("READABLE_SCAN_MULTIPLIER", "80"))
USE_ML_RECS = os.environ.get("USE_ML_RECS", "true").strip().lower() in {"1", "true", "yes", "on"}


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

DISPLAY_FROM_USER_SQL = """
SELECT
    display_name
FROM music.user_profiles
WHERE user_id = %s
LIMIT 1
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
    conn_kwargs = build_conn_kwargs()
    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> Dict[str, Any] | None:
    rows = fetch_rows(sql, params)
    return rows[0] if rows else None


def resolve_user_reference(user_ref: str) -> tuple[str, str | None]:
    direct = fetch_one(USER_EXISTS_SQL, (user_ref,))
    if direct:
        try:
            display = fetch_one(DISPLAY_FROM_USER_SQL, (user_ref,))
            return user_ref, (display["display_name"] if display else None)
        except Exception:
            return user_ref, None

    try:
        mapped = fetch_one(USER_FROM_DISPLAY_SQL, (user_ref,))
        if mapped:
            return mapped["user_id"], mapped["display_name"]
    except Exception:
        return user_ref, None

    return user_ref, None


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


def normalize_vibe(v: str | None) -> str | None:
    if v is None:
        return None
    cleaned = v.strip().lower()
    return cleaned or None


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


@app.get("/metrics/model")
def get_model_metrics() -> Dict[str, Any]:
    try:
        rows = fetch_rows(MODEL_METRICS_SQL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"metrics query failed: {exc}")
    if not rows:
        raise HTTPException(status_code=404, detail="No model metrics found.")
    return rows[0]


@app.get("/trending")
def get_trending(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    limit_scan = scan_limit(limit)
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
def get_vibe_tracks(vibe: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    cleaned = vibe.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="vibe is required.")
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
        return {
            "mode": "vibe_fallback_named_keyword",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": len(vibe_named),
            "items": vibe_named,
            "message": "No direct vibe metadata found; returned best keyword-matched named tracks.",
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
            "message": "No human-readable tracks available for this vibe in current dataset slice.",
        }
    return {
        "mode": "vibe_fallback_named_trending",
        "vibe": cleaned,
        "labels": labels,
        "keywords": keywords,
        "count": len(trending),
        "items": trending,
        "message": "No vibe-specific named tracks found; returned readable general tracks.",
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


@app.get("/recs/{user_id}")
def get_recs(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    fallback_to_trending: bool = Query(default=True),
) -> Dict[str, Any]:
    requested_user_ref = user_id.strip()
    looks_like_display_name = " " in requested_user_ref
    resolved_user_id, display_name = resolve_user_reference(user_id)
    unresolved_display_name = looks_like_display_name and display_name is None and resolved_user_id == requested_user_ref
    limit_scan = scan_limit(limit, multiplier=50)
    recs: List[Dict[str, Any]] = []
    mode = "personalized_dense_1000"
    source_errors: List[str] = []

    if USE_ML_RECS:
        try:
            recs = fetch_rows(USER_RECS_MF_SQL, (resolved_user_id, limit_scan))
            mode = "personalized_mf_ready"
        except Exception as exc:
            # Keep serving from rule-based model if ML table/view is not ready yet.
            source_errors.append(f"mf:{exc}")
            recs = []

    if not recs:
        try:
            recs = fetch_rows(USER_RECS_DENSE_SQL, (resolved_user_id, limit_scan))
            mode = "personalized_dense_1000"
        except Exception as exc:
            source_errors.append(f"dense:{exc}")
            raise HTTPException(status_code=500, detail=f"recommendation query failed: {' | '.join(source_errors)}")

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
            resp["note"] = "ML source unavailable in this run; served fallback recommendation source."
        if unresolved_display_name:
            resp["user_reference_note"] = (
                "Display name was not found in music.user_profiles; interpreted input as raw user_id."
            )
        return resp

    if not fallback_to_trending:
        return {
            "user_id": user_id,
            "mode": "no_results",
            "count": 0,
            "items": [],
        }

    try:
        fallback = fetch_rows(TRENDING_SQL, (limit_scan,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"fallback query failed: {exc}")
    fallback = readable_only(fallback, limit)
    mode = "fallback_trending"
    message = "User not in dense personalization slice; returned global trending tracks."
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
    if display_name:
        resp["user_display_name"] = display_name
    if unresolved_display_name:
        resp["user_reference_note"] = (
            "Display name was not found in music.user_profiles; interpreted input as raw user_id."
        )
    if not fallback:
        resp["message"] = "No human-readable recommendations available for this user in current dataset slice."
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
