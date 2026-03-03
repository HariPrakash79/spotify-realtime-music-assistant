#!/usr/bin/env python3
"""
Build and persist hybrid recommendations for serving.

Hybrid score:
  hybrid = w_mf * normalized_mf + w_pop * normalized_popularity + w_artist * artist_affinity_bonus

Input:
  - Latest MF recommendations from music.v_user_recommendations_mf_ready
  - User history from music.v_listen_events_recommendation_ready

Output:
  - Writes top-K per user into music.user_recommendations_hybrid_ready
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from psycopg import connect
from psycopg.rows import dict_row


FETCH_HISTORY_SQL = """
WITH model_users AS (
    SELECT user_id
    FROM music.v_model_users_1000_ready
    ORDER BY plays DESC, user_id
    LIMIT %s
)
SELECT
    le.user_id,
    le.track_id,
    MAX(le.track_name) AS track_name,
    MAX(le.artist_name) AS artist_name,
    COUNT(*)::INTEGER AS plays
FROM music.v_listen_events_recommendation_ready le
JOIN model_users mu
  ON mu.user_id = le.user_id
GROUP BY le.user_id, le.track_id
HAVING COUNT(*) >= %s
ORDER BY le.user_id, COUNT(*) DESC, le.track_id
"""

FETCH_MF_RECS_SQL = """
WITH model_users AS (
    SELECT user_id
    FROM music.v_model_users_1000_ready
    ORDER BY plays DESC, user_id
    LIMIT %s
)
SELECT
    r.user_id,
    r.recommendation_rank,
    r.track_id,
    r.track_name,
    r.artist_name,
    r.recommendation_score
FROM music.v_user_recommendations_mf_ready r
JOIN model_users mu
  ON mu.user_id = r.user_id
WHERE r.recommendation_rank <= %s
ORDER BY r.user_id, r.recommendation_rank
"""

DELETE_MODEL_SQL = "DELETE FROM music.user_recommendations_hybrid_ready WHERE model_version = %s"

INSERT_RECS_SQL = """
INSERT INTO music.user_recommendations_hybrid_ready (
    model_version,
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    mf_score,
    pop_score,
    artist_bonus,
    recommendation_score
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def connect_postgres():
    kwargs = build_conn_kwargs()
    if "conninfo" in kwargs:
        return connect(kwargs["conninfo"], row_factory=dict_row)
    return connect(row_factory=dict_row, **kwargs)


def fetch_rows(sql: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def prepare_history_structures(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[
    List[str],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, str]],
    Dict[str, int],
    Dict[str, Dict[str, int]],
]:
    user_track_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    track_meta: Dict[str, Dict[str, str]] = {}
    global_pop: Dict[str, int] = defaultdict(int)
    user_artist_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        user_id = str(row["user_id"])
        track_id = str(row["track_id"])
        plays = int(row["plays"])
        track_name = str(row.get("track_name") or track_id)
        artist_name = str(row.get("artist_name") or "__unknown_artist__")

        user_track_counts[user_id][track_id] = plays
        global_pop[track_id] += plays
        user_artist_counts[user_id][artist_name] += plays

        if track_id not in track_meta:
            track_meta[track_id] = {
                "track_name": track_name,
                "artist_name": artist_name,
            }

    user_ids = sorted(user_track_counts.keys())
    return user_ids, user_track_counts, track_meta, global_pop, user_artist_counts


def build_hybrid_rows(
    *,
    user_ids: Sequence[str],
    user_track_counts: Mapping[str, Mapping[str, int]],
    track_meta: Dict[str, Dict[str, str]],
    global_pop: Mapping[str, int],
    user_artist_counts: Mapping[str, Mapping[str, int]],
    mf_rows: Sequence[Dict[str, Any]],
    top_k: int,
    mf_scan: int,
    candidate_top_n: int,
    artist_top_n: int,
    weight_mf: float,
    weight_pop: float,
    weight_artist: float,
    model_version: str,
) -> List[Tuple[str, str, int, str, str, str, float, float, float, float]]:
    mf_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in mf_rows:
        user_id = str(row["user_id"])
        mf_by_user[user_id].append(row)
        track_id = str(row["track_id"])
        if track_id not in track_meta:
            track_meta[track_id] = {
                "track_name": str(row.get("track_name") or track_id),
                "artist_name": str(row.get("artist_name") or "__unknown_artist__"),
            }

    top_pop_tracks = [
        track_id
        for track_id, _ in sorted(global_pop.items(), key=lambda x: (-int(x[1]), str(x[0])))[: max(candidate_top_n, top_k)]
    ]
    max_pop = max(global_pop.values()) if global_pop else 1

    weight_sum = max(weight_mf + weight_pop + weight_artist, 1e-9)
    w_mf = weight_mf / weight_sum
    w_pop = weight_pop / weight_sum
    w_artist = weight_artist / weight_sum

    output: List[Tuple[str, str, int, str, str, str, float, float, float, float]] = []
    for user_id in user_ids:
        seen = set(user_track_counts[user_id].keys())
        artist_counts = user_artist_counts.get(user_id, {})
        top_artists = {
            artist
            for artist, _ in sorted(
                artist_counts.items(),
                key=lambda x: (-int(x[1]), str(x[0])),
            )[: max(artist_top_n, 1)]
        }

        user_mf = mf_by_user.get(user_id, [])[: max(mf_scan, top_k)]
        mf_scores = [float(r.get("recommendation_score", 0.0)) for r in user_mf]
        mf_min = min(mf_scores) if mf_scores else 0.0
        mf_max = max(mf_scores) if mf_scores else 1.0
        mf_den = (mf_max - mf_min) if mf_max != mf_min else 1.0

        candidates: Dict[str, Dict[str, Any]] = {}
        for row in user_mf:
            track_id = str(row["track_id"])
            if track_id in seen:
                continue
            meta = track_meta.get(track_id, {})
            candidates[track_id] = {
                "track_id": track_id,
                "track_name": str(row.get("track_name") or meta.get("track_name") or track_id),
                "artist_name": str(row.get("artist_name") or meta.get("artist_name") or "__unknown_artist__"),
                "mf_score": float(row.get("recommendation_score", 0.0)),
            }

        candidate_cap = max(top_k * 25, mf_scan * 3, 500)
        for track_id in top_pop_tracks:
            if track_id in seen:
                continue
            if track_id not in candidates:
                meta = track_meta.get(track_id, {})
                candidates[track_id] = {
                    "track_id": track_id,
                    "track_name": str(meta.get("track_name", track_id)),
                    "artist_name": str(meta.get("artist_name", "__unknown_artist__")),
                    "mf_score": mf_min,
                }
            if len(candidates) >= candidate_cap:
                break

        scored_rows: List[Dict[str, Any]] = []
        for track_id, c in candidates.items():
            mf_norm = (float(c["mf_score"]) - mf_min) / mf_den
            pop_norm = float(global_pop.get(track_id, 0)) / float(max_pop)
            artist_bonus = 1.0 if str(c["artist_name"]) in top_artists else 0.0
            hybrid_score = (w_mf * mf_norm) + (w_pop * pop_norm) + (w_artist * artist_bonus)
            scored_rows.append(
                {
                    "track_id": track_id,
                    "track_name": c["track_name"],
                    "artist_name": c["artist_name"],
                    "mf_score": float(mf_norm),
                    "pop_score": float(pop_norm),
                    "artist_bonus": float(artist_bonus),
                    "hybrid_score": float(hybrid_score),
                }
            )

        ranked = sorted(
            scored_rows,
            key=lambda x: (-float(x["hybrid_score"]), -float(x["pop_score"]), str(x["track_id"])),
        )[: max(top_k, 1)]

        for rank, row in enumerate(ranked, start=1):
            output.append(
                (
                    model_version,
                    user_id,
                    rank,
                    str(row["track_id"]),
                    str(row["track_name"]),
                    str(row["artist_name"]),
                    float(row["mf_score"]),
                    float(row["pop_score"]),
                    float(row["artist_bonus"]),
                    float(row["hybrid_score"]),
                )
            )

    return output


def write_recommendations(
    model_version: str,
    rows: Sequence[Tuple[str, str, int, str, str, str, float, float, float, float]],
) -> None:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(DELETE_MODEL_SQL, (model_version,))
            cur.executemany(INSERT_RECS_SQL, rows)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and store hybrid recommendations for serving.")
    parser.add_argument("--max-users", type=int, default=1000, help="Max users from ready model slice.")
    parser.add_argument("--min-user-track-plays", type=int, default=1, help="Min user-track plays in history.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k recommendations per user to store.")
    parser.add_argument("--mf-scan", type=int, default=500, help="MF rows scanned per user before rerank.")
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=10000,
        help="Global popularity candidate pool size.",
    )
    parser.add_argument("--artist-top-n", type=int, default=5, help="Top artists per user for artist bonus.")
    parser.add_argument("--weight-mf", type=float, default=0.45, help="MF score weight.")
    parser.add_argument("--weight-pop", type=float, default=0.45, help="Popularity prior weight.")
    parser.add_argument("--weight-artist", type=float, default=0.10, help="Artist affinity bonus weight.")
    parser.add_argument("--dry-run", action="store_true", help="Build recs but skip DB write.")
    parser.add_argument(
        "--model-version",
        default=f"hybrid_ready_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        help="Model version tag stored with recommendations.",
    )
    args = parser.parse_args()

    if args.weight_mf < 0 or args.weight_pop < 0 or args.weight_artist < 0:
        raise ValueError("Hybrid weights must be non-negative.")
    if (args.weight_mf + args.weight_pop + args.weight_artist) <= 0:
        raise ValueError("At least one hybrid weight must be positive.")

    history_rows = fetch_rows(FETCH_HISTORY_SQL, (args.max_users, args.min_user_track_plays))
    if not history_rows:
        raise ValueError("No recommendation-ready user history found.")

    mf_rows = fetch_rows(FETCH_MF_RECS_SQL, (args.max_users, args.mf_scan))
    if not mf_rows:
        raise ValueError("No MF recommendations found. Run scripts/train_personalized_mf.py first.")

    user_ids, user_track_counts, track_meta, global_pop, user_artist_counts = prepare_history_structures(history_rows)
    print(
        f"history_rows={len(history_rows)} users={len(user_ids)} tracks={len(global_pop)} "
        f"mf_rows={len(mf_rows)} top_k={args.top_k} model_version={args.model_version}"
    )

    rec_rows = build_hybrid_rows(
        user_ids=user_ids,
        user_track_counts=user_track_counts,
        track_meta=track_meta,
        global_pop=global_pop,
        user_artist_counts=user_artist_counts,
        mf_rows=mf_rows,
        top_k=args.top_k,
        mf_scan=args.mf_scan,
        candidate_top_n=args.candidate_top_n,
        artist_top_n=args.artist_top_n,
        weight_mf=args.weight_mf,
        weight_pop=args.weight_pop,
        weight_artist=args.weight_artist,
        model_version=args.model_version,
    )
    print(f"generated_recommendations={len(rec_rows)}")

    if args.dry_run:
        print("dry_run=True, skipping write.")
        return

    write_recommendations(args.model_version, rec_rows)
    print("write_complete=True")


if __name__ == "__main__":
    main()
