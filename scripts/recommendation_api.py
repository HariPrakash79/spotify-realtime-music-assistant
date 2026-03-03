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


MODEL_METRICS_SQL = """
SELECT
    events,
    users,
    tracks,
    events_per_user
FROM music.v_model_metrics_1000
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
FROM music.v_global_trending_tracks_7d
WHERE global_rank_7d <= %s
ORDER BY global_rank_7d
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
FROM music.v_user_recommendations_30d_dense_1000
WHERE user_id = %s
  AND recommendation_rank <= %s
ORDER BY recommendation_rank
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
    FROM music.listen_events le
    WHERE track_name IS NOT NULL
      AND lower(track_name) LIKE '%' || (SELECT query_lower FROM q) || '%'
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
      AND lower(track_name) LIKE '%' || (SELECT query_lower FROM q) || '%'
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
            WHEN lower(track_name) LIKE (SELECT query_lower FROM q) || '%%' THEN 200
            WHEN lower(track_name) LIKE '%%' || (SELECT query_lower FROM q) || '%%' THEN 100
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
    FROM music.listen_events
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
      WHERE lower(tc.genre) LIKE '%%' || kw || '%%'
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
        "sad": ["acoustic", "singer-songwriter", "blues", "piano"],
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
    try:
        rows = fetch_rows(TRENDING_SQL, (limit,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trending query failed: {exc}")

    return {
        "mode": "trending_7d",
        "count": len(rows),
        "items": rows,
    }


@app.get("/search/tracks")
def search_tracks(query: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    cleaned = query.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="query is required.")
    try:
        rows = fetch_rows(SEARCH_TRACKS_SQL, (cleaned, cleaned, limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"track search failed: {exc}")
    return {
        "query": cleaned,
        "count": len(rows),
        "items": rows,
    }


@app.get("/vibe")
def get_vibe_tracks(vibe: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    cleaned = vibe.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="vibe is required.")
    labels = vibe_labels(cleaned)
    keywords = vibe_keywords(cleaned)
    rows: List[Dict[str, Any]] = []

    # Preferred path: engineered vibe features table
    try:
        rows = fetch_rows(VIBE_FEATURE_TRACKS_SQL, (labels, limit))
    except Exception as exc:
        # If engineered table is not built yet, continue with catalog/genre fallback.
        print(f"vibe feature query unavailable, falling back to catalog rules: {exc}")
        rows = []
    if rows:
        return {
            "mode": "vibe_match_features",
            "vibe": cleaned,
            "labels": labels,
            "keywords": keywords,
            "count": len(rows),
            "items": rows,
        }

    # Fallback path: direct genre keyword search in track catalog.
    try:
        fallback = fetch_rows(VIBE_CATALOG_FALLBACK_SQL, (keywords, limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vibe catalog fallback failed: {exc}")
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
        trending = fetch_rows(TRENDING_SQL, (limit,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vibe final fallback failed: {exc}")
    return {
        "mode": "vibe_fallback_trending",
        "vibe": cleaned,
        "labels": labels,
        "keywords": keywords,
        "count": len(trending),
        "items": trending,
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
    try:
        recs = fetch_rows(USER_RECS_DENSE_SQL, (user_id, limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recommendation query failed: {exc}")

    if recs:
        return {
            "user_id": user_id,
            "mode": "personalized_dense_1000",
            "count": len(recs),
            "items": recs,
        }

    if not fallback_to_trending:
        return {
            "user_id": user_id,
            "mode": "no_results",
            "count": 0,
            "items": [],
        }

    try:
        fallback = fetch_rows(TRENDING_SQL, (limit,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"fallback query failed: {exc}")

    return {
        "user_id": user_id,
        "mode": "fallback_trending",
        "message": "User not in dense personalization slice; returned global trending tracks.",
        "count": len(fallback),
        "items": fallback,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
