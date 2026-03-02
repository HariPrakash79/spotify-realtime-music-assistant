#!/usr/bin/env python3
"""
Lightweight serving API for music recommendations.

Endpoints:
- GET /metrics/model
- GET /trending?limit=20
- GET /recs/{user_id}?limit=20
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping

from fastapi import FastAPI, HTTPException, Query
from psycopg import connect
from psycopg.rows import dict_row


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

    uvicorn.run("scripts.recommendation_api:app", host="0.0.0.0", port=8000, reload=False)
