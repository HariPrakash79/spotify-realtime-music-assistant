#!/usr/bin/env python3
"""
Train a lightweight matrix-factorization recommender on the recommendation-ready slice.

Writes top-K per-user recommendations into:
  music.user_recommendations_mf_ready

Usage:
  python scripts/train_personalized_mf.py
  python scripts/train_personalized_mf.py --epochs 8 --factors 48 --top-k 100
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from psycopg import connect
from psycopg.rows import dict_row

from text_cleanup import clean_text


FETCH_INTERACTIONS_SQL = """
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

FETCH_INTERACTIONS_WITH_FEEDBACK_SQL = """
WITH model_users AS (
    SELECT user_id
    FROM music.v_model_users_1000_ready
    ORDER BY plays DESC, user_id
    LIMIT %s
),
base AS (
    SELECT
        le.user_id,
        le.track_id,
        MAX(le.track_name) AS track_name,
        MAX(le.artist_name) AS artist_name,
        COUNT(*)::NUMERIC AS base_plays
    FROM music.v_listen_events_recommendation_ready le
    JOIN model_users mu
      ON mu.user_id = le.user_id
    GROUP BY le.user_id, le.track_id
),
feedback AS (
    SELECT
        di.user_id,
        di.track_id,
        COALESCE(
            MAX(NULLIF(tc.track_name, '')),
            MAX(NULLIF(le.track_name, '')),
            di.track_id
        ) AS track_name,
        COALESCE(
            MAX(NULLIF(tc.artist_name, '')),
            MAX(NULLIF(le.artist_name, '')),
            '__unknown_artist__'
        ) AS artist_name,
        SUM(
            CASE
                WHEN di.action IN ('play', 'like', 'favorite', 'add_to_playlist') THEN GREATEST(di.signal_strength::NUMERIC, 0.10)
                WHEN di.action = 'impression' THEN GREATEST(di.signal_strength::NUMERIC, 0.01)
                WHEN di.action IN ('skip', 'dislike') THEN -1 * GREATEST(ABS(di.signal_strength::NUMERIC), 0.10)
                ELSE 0::NUMERIC
            END
        ) AS feedback_score
    FROM music.demo_interactions di
    JOIN model_users mu
      ON mu.user_id = di.user_id
    LEFT JOIN music.track_catalog tc
      ON tc.track_id = di.track_id
    LEFT JOIN music.v_listen_events_recommendation_ready le
      ON le.user_id = di.user_id
     AND le.track_id = di.track_id
    WHERE COALESCE(di.track_id, '') <> ''
    GROUP BY di.user_id, di.track_id
),
merged AS (
    SELECT
        COALESCE(b.user_id, f.user_id) AS user_id,
        COALESCE(b.track_id, f.track_id) AS track_id,
        COALESCE(
            NULLIF(b.track_name, ''),
            NULLIF(f.track_name, ''),
            COALESCE(b.track_id, f.track_id)
        ) AS track_name,
        COALESCE(
            NULLIF(b.artist_name, ''),
            NULLIF(f.artist_name, ''),
            '__unknown_artist__'
        ) AS artist_name,
        GREATEST(
            0::NUMERIC,
            COALESCE(b.base_plays, 0::NUMERIC) + (%s::NUMERIC * COALESCE(f.feedback_score, 0::NUMERIC))
        ) AS plays_score
    FROM base b
    FULL OUTER JOIN feedback f
      ON f.user_id = b.user_id
     AND f.track_id = b.track_id
)
SELECT
    user_id,
    track_id,
    track_name,
    artist_name,
    CEIL(plays_score)::INTEGER AS plays
FROM merged
WHERE plays_score >= %s::NUMERIC
ORDER BY user_id, plays DESC, track_id
"""

DELETE_MODEL_SQL = "DELETE FROM music.user_recommendations_mf_ready WHERE model_version = %s"

INSERT_RECS_SQL = """
INSERT INTO music.user_recommendations_mf_ready (
    model_version,
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass
class Interaction:
    user_idx: int
    item_idx: int
    weight: float


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


def fetch_interactions(
    max_users: int,
    min_user_track_plays: int,
    *,
    include_demo_feedback: bool,
    demo_feedback_boost: float,
) -> List[Dict[str, object]]:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            if include_demo_feedback:
                try:
                    cur.execute(
                        FETCH_INTERACTIONS_WITH_FEEDBACK_SQL,
                        (max_users, demo_feedback_boost, min_user_track_plays),
                    )
                except Exception as exc:
                    print(f"demo_feedback_integration_unavailable={exc}")
                    conn.rollback()
                    cur.execute(FETCH_INTERACTIONS_SQL, (max_users, min_user_track_plays))
            else:
                cur.execute(FETCH_INTERACTIONS_SQL, (max_users, min_user_track_plays))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def prepare_training_data(
    rows: Sequence[Dict[str, object]],
) -> Tuple[
    List[Interaction],
    List[str],
    List[str],
    Dict[str, str],
    Dict[str, str],
    List[set[int]],
]:
    user_ids = sorted({str(r["user_id"]) for r in rows})
    item_ids = sorted({str(r["track_id"]) for r in rows})

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {t: i for i, t in enumerate(item_ids)}

    track_name_by_id: Dict[str, str] = {}
    artist_name_by_id: Dict[str, str] = {}
    per_user_seen: List[set[int]] = [set() for _ in user_ids]

    max_log = 0.0
    raw: List[Tuple[int, int, float]] = []
    for r in rows:
        u = user_to_idx[str(r["user_id"])]
        i = item_to_idx[str(r["track_id"])]
        plays = int(r["plays"])
        logp = math.log1p(max(plays, 1))
        raw.append((u, i, logp))
        if logp > max_log:
            max_log = logp

        per_user_seen[u].add(i)
        track_name_by_id[str(r["track_id"])] = (
            clean_text(r.get("track_name"), repair_mojibake=True) or str(r["track_id"])
        )
        artist_name_by_id[str(r["track_id"])] = (
            clean_text(r.get("artist_name"), repair_mojibake=True) or "__unknown_artist__"
        )

    if max_log <= 0.0:
        max_log = 1.0

    interactions = [
        Interaction(user_idx=u, item_idx=i, weight=max(0.10, min(1.0, logp / max_log)))
        for (u, i, logp) in raw
    ]
    return interactions, user_ids, item_ids, track_name_by_id, artist_name_by_id, per_user_seen


def train_mf(
    interactions: Sequence[Interaction],
    n_users: int,
    n_items: int,
    user_seen: Sequence[set[int]],
    *,
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
    neg_ratio: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_f = (rng.standard_normal((n_users, factors)).astype(np.float32)) * 0.05
    item_f = (rng.standard_normal((n_items, factors)).astype(np.float32)) * 0.05
    user_b = np.zeros(n_users, dtype=np.float32)
    item_b = np.zeros(n_items, dtype=np.float32)

    order = np.arange(len(interactions), dtype=np.int64)
    all_items = np.arange(n_items, dtype=np.int32)

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        sq_err = 0.0
        updates = 0

        for idx in order:
            inter = interactions[int(idx)]
            u = inter.user_idx
            i = inter.item_idx
            target = inter.weight

            pred = float(np.dot(user_f[u], item_f[i]) + user_b[u] + item_b[i])
            err = target - pred

            u_old = user_f[u].copy()
            i_old = item_f[i].copy()
            user_f[u] += lr * (err * i_old - reg * u_old)
            item_f[i] += lr * (err * u_old - reg * i_old)
            user_b[u] += lr * (err - reg * user_b[u])
            item_b[i] += lr * (err - reg * item_b[i])

            sq_err += err * err
            updates += 1

            for _ in range(max(neg_ratio, 0)):
                # Sample a negative item not seen by this user.
                for _attempt in range(40):
                    j = int(rng.choice(all_items))
                    if j not in user_seen[u]:
                        break
                else:
                    continue

                target_n = 0.0
                pred_n = float(np.dot(user_f[u], item_f[j]) + user_b[u] + item_b[j])
                err_n = target_n - pred_n

                u_old = user_f[u].copy()
                j_old = item_f[j].copy()
                user_f[u] += lr * (err_n * j_old - reg * u_old)
                item_f[j] += lr * (err_n * u_old - reg * j_old)
                user_b[u] += lr * (err_n - reg * user_b[u])
                item_b[j] += lr * (err_n - reg * item_b[j])

                sq_err += err_n * err_n
                updates += 1

        rmse = math.sqrt(sq_err / max(updates, 1))
        print(f"epoch={epoch}/{epochs} updates={updates} rmse={rmse:.4f}")

    return user_f, item_f, user_b, item_b


def build_recommendations(
    user_f: np.ndarray,
    item_f: np.ndarray,
    user_b: np.ndarray,
    item_b: np.ndarray,
    user_ids: Sequence[str],
    item_ids: Sequence[str],
    user_seen: Sequence[set[int]],
    track_name_by_id: Mapping[str, str],
    artist_name_by_id: Mapping[str, str],
    top_k: int,
) -> List[Tuple[str, int, str, str, str, float]]:
    recs: List[Tuple[str, int, str, str, str, float]] = []
    n_items = item_f.shape[0]
    k = min(top_k, n_items)

    for u_idx, user_id in enumerate(user_ids):
        scores = item_f @ user_f[u_idx] + item_b + user_b[u_idx]
        if user_seen[u_idx]:
            scores[list(user_seen[u_idx])] = -1e12

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        rank = 1
        for item_idx in top_idx:
            if scores[item_idx] <= -1e11:
                continue
            track_id = item_ids[int(item_idx)]
            track_name = track_name_by_id.get(track_id, track_id)
            artist_name = artist_name_by_id.get(track_id, "__unknown_artist__")
            recs.append((user_id, rank, track_id, track_name, artist_name, float(scores[item_idx])))
            rank += 1

    return recs


def write_recommendations(model_version: str, recs: Sequence[Tuple[str, int, str, str, str, float]]) -> None:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(DELETE_MODEL_SQL, (model_version,))
            rows = [
                (model_version, user_id, rank, track_id, track_name, artist_name, score)
                for (user_id, rank, track_id, track_name, artist_name, score) in recs
            ]
            cur.executemany(INSERT_RECS_SQL, rows)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MF recommender and write recommendations to Postgres.")
    parser.add_argument("--max-users", type=int, default=1000, help="Max users from ready model slice.")
    parser.add_argument("--min-user-track-plays", type=int, default=1, help="Min plays for user-track interaction.")
    parser.add_argument("--factors", type=int, default=48, help="Latent factors.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--reg", type=float, default=0.01, help="L2 regularization.")
    parser.add_argument("--neg-ratio", type=int, default=2, help="Negative samples per positive.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k recommendations per user to store.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include-demo-feedback",
        dest="include_demo_feedback",
        action="store_true",
        default=True,
        help="Blend demo_interactions signals into MF training data (default: enabled).",
    )
    parser.add_argument(
        "--no-demo-feedback",
        dest="include_demo_feedback",
        action="store_false",
        help="Disable demo_interactions blending and train only from listen_events.",
    )
    parser.add_argument(
        "--demo-feedback-boost",
        type=float,
        default=4.0,
        help="Multiplier applied to aggregated demo interaction signal before blending with plays.",
    )
    parser.add_argument(
        "--model-version",
        default=f"mf_ready_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}",
        help="Model version tag stored with recommendations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Train and print metrics without DB write.")
    args = parser.parse_args()

    rows = fetch_interactions(
        max_users=args.max_users,
        min_user_track_plays=args.min_user_track_plays,
        include_demo_feedback=args.include_demo_feedback,
        demo_feedback_boost=args.demo_feedback_boost,
    )
    if not rows:
        raise ValueError("No interactions found in recommendation-ready slice.")

    interactions, user_ids, item_ids, track_name_by_id, artist_name_by_id, user_seen = prepare_training_data(rows)
    print(
        f"train_rows={len(interactions)} users={len(user_ids)} items={len(item_ids)} "
        f"top_k={args.top_k} model_version={args.model_version} "
        f"demo_feedback={args.include_demo_feedback} feedback_boost={args.demo_feedback_boost:.2f}"
    )

    user_f, item_f, user_b, item_b = train_mf(
        interactions=interactions,
        n_users=len(user_ids),
        n_items=len(item_ids),
        user_seen=user_seen,
        factors=args.factors,
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    recs = build_recommendations(
        user_f=user_f,
        item_f=item_f,
        user_b=user_b,
        item_b=item_b,
        user_ids=user_ids,
        item_ids=item_ids,
        user_seen=user_seen,
        track_name_by_id=track_name_by_id,
        artist_name_by_id=artist_name_by_id,
        top_k=args.top_k,
    )
    print(f"generated_recommendations={len(recs)}")

    if args.dry_run:
        print("dry_run=True, skipping write.")
        return

    write_recommendations(args.model_version, recs)
    print("write_complete=True")


if __name__ == "__main__":
    main()
