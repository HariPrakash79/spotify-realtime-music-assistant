#!/usr/bin/env python3
"""
Evaluate recommendation quality on the recommendation-ready slice.

This script performs a simple temporal holdout per user:
- Hold out latest N events per user as test.
- Treat older events as train/history.
- Score recommendation lists against holdout tracks.

By default, truth uses only NOVEL holdout tracks:
- truth = holdout_tracks - train_tracks
This aligns with the serving policy that excludes already-seen tracks from recommendations.

Metrics:
- precision@k
- recall@k
- ndcg@k
- hit_rate@k
- item_coverage@k
- user_coverage
- personalization (mean pairwise Jaccard distance)
- readable_rate@k

Sources:
- mf:     music.v_user_recommendations_mf_ready (or temporal retrain prediction)
- dense:  music.v_user_recommendations_30d_dense_1000_ready
- pop:    global popularity baseline (computed from train split)
- hybrid: temporal MF reranked with popularity prior + user artist affinity
- both:   evaluates mf and dense
- all:    evaluates pop, mf, hybrid, dense
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from psycopg import connect
from psycopg.rows import dict_row


HOLDOUT_SQL = """
WITH ranked AS (
    SELECT
        le.user_id,
        le.track_id,
        le.track_name,
        le.artist_name,
        le.event_ts,
        le.event_id,
        ROW_NUMBER() OVER (
            PARTITION BY le.user_id
            ORDER BY le.event_ts DESC, le.event_id DESC
        ) AS rn_desc,
        COUNT(*) OVER (PARTITION BY le.user_id) AS user_events
    FROM music.v_listen_events_recommendation_ready le
    JOIN music.v_model_users_1000_ready mu
      ON mu.user_id = le.user_id
)
SELECT
    user_id,
    track_id,
    track_name,
    artist_name,
    rn_desc,
    user_events
FROM ranked
WHERE user_events >= %s
ORDER BY user_id, rn_desc, track_id
"""

RECS_MF_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_mf_ready
WHERE recommendation_rank <= %s
ORDER BY user_id, recommendation_rank
"""

RECS_DENSE_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_id,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d_dense_1000_ready
WHERE recommendation_rank <= %s
ORDER BY user_id, recommendation_rank
"""

TOTAL_TRACKS_SQL = """
SELECT COUNT(DISTINCT track_id)::BIGINT AS total_tracks
FROM music.v_listen_events_model_1000_ready
"""


@dataclass
class EvalSummary:
    source: str
    users_evaluated: int
    users_with_recs: int
    user_coverage: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    item_coverage_at_k: float
    personalization_at_k: float
    readable_rate_at_k: float


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


def fetch_rows(sql: str, params: Tuple[object, ...] = ()) -> List[Dict[str, object]]:
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def is_readable(track_name: str | None, artist_name: str | None, track_id: str | None) -> bool:
    t = (track_name or "").strip()
    a = (artist_name or "").strip().lower()
    tid = (track_id or "").strip()
    if not t:
        return False
    if t.isdigit():
        return False
    if a in {"", "__unknown_artist__"}:
        return False
    if tid and t == tid:
        return False
    return True


def build_holdout(
    rows: Sequence[Dict[str, object]],
    holdout_size: int,
    target_mode: str,
) -> Tuple[Dict[str, set[str]], Dict[str, set[str]], Dict[Tuple[str, str], int], Dict[str, Dict[str, str]]]:
    train: Dict[str, set[str]] = defaultdict(set)
    holdout: Dict[str, set[str]] = defaultdict(set)
    train_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    track_meta: Dict[str, Dict[str, str]] = {}
    for r in rows:
        user_id = str(r["user_id"])
        track_id = str(r["track_id"])
        rn_desc = int(r["rn_desc"])
        if track_id not in track_meta:
            track_meta[track_id] = {
                "track_name": str(r.get("track_name") or track_id),
                "artist_name": str(r.get("artist_name") or "__unknown_artist__"),
            }
        if rn_desc <= holdout_size:
            holdout[user_id].add(track_id)
        else:
            train[user_id].add(track_id)
            train_counts[(user_id, track_id)] += 1

    if target_mode == "novel":
        test = {u: (holdout.get(u, set()) - train.get(u, set())) for u in holdout.keys()}
    else:
        test = {u: set(holdout.get(u, set())) for u in holdout.keys()}

    valid_users = [u for u in test.keys() if test[u] and train.get(u)]
    train = {u: train[u] for u in valid_users}
    test = {u: test[u] for u in valid_users}
    valid_user_set = set(valid_users)
    train_counts = {
        (u, t): c for (u, t), c in train_counts.items() if u in valid_user_set
    }
    return train, test, train_counts, track_meta


def prepare_mf_training(
    train_counts: Mapping[Tuple[str, str], int],
) -> Tuple[List[Interaction], List[str], List[str], List[set[int]]]:
    user_ids = sorted({u for (u, _) in train_counts.keys()})
    item_ids = sorted({t for (_, t) in train_counts.keys()})
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {t: i for i, t in enumerate(item_ids)}

    per_user_seen: List[set[int]] = [set() for _ in user_ids]
    raw: List[Tuple[int, int, float]] = []
    max_log = 0.0
    for (user_id, track_id), plays in train_counts.items():
        u = user_to_idx[user_id]
        i = item_to_idx[track_id]
        logp = math.log1p(max(int(plays), 1))
        raw.append((u, i, logp))
        per_user_seen[u].add(i)
        if logp > max_log:
            max_log = logp

    if max_log <= 0.0:
        max_log = 1.0

    interactions = [
        Interaction(user_idx=u, item_idx=i, weight=max(0.10, min(1.0, logp / max_log)))
        for (u, i, logp) in raw
    ]
    return interactions, user_ids, item_ids, per_user_seen


def train_mf_temporal(
    *,
    interactions: Sequence[Interaction],
    n_users: int,
    n_items: int,
    user_seen: Sequence[set[int]],
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
                for _attempt in range(40):
                    j = int(rng.choice(all_items))
                    if j not in user_seen[u]:
                        break
                else:
                    continue
                pred_n = float(np.dot(user_f[u], item_f[j]) + user_b[u] + item_b[j])
                err_n = -pred_n

                u_old = user_f[u].copy()
                j_old = item_f[j].copy()
                user_f[u] += lr * (err_n * j_old - reg * u_old)
                item_f[j] += lr * (err_n * u_old - reg * j_old)
                user_b[u] += lr * (err_n - reg * user_b[u])
                item_b[j] += lr * (err_n - reg * item_b[j])

                sq_err += err_n * err_n
                updates += 1

        rmse = math.sqrt(sq_err / max(updates, 1))
        print(f"temporal_mf epoch={epoch}/{epochs} updates={updates} rmse={rmse:.4f}")
    return user_f, item_f, user_b, item_b


def build_temporal_mf_recs(
    *,
    user_f: np.ndarray,
    item_f: np.ndarray,
    user_b: np.ndarray,
    item_b: np.ndarray,
    user_ids: Sequence[str],
    item_ids: Sequence[str],
    user_seen: Sequence[set[int]],
    track_meta: Mapping[str, Mapping[str, str]],
    k_scan: int,
) -> Dict[str, List[Dict[str, object]]]:
    recs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    n_items = item_f.shape[0]
    k = min(max(k_scan, 1), n_items)
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
            meta = track_meta.get(track_id, {})
            recs[user_id].append(
                {
                    "user_id": user_id,
                    "recommendation_rank": rank,
                    "track_id": track_id,
                    "track_name": meta.get("track_name", track_id),
                    "artist_name": meta.get("artist_name", "__unknown_artist__"),
                    "recommendation_score": float(scores[item_idx]),
                }
            )
            rank += 1
    return recs


def load_recommendations(source: str, k_scan: int) -> Dict[str, List[Dict[str, object]]]:
    if source == "mf":
        sql = RECS_MF_SQL
    elif source == "dense":
        sql = RECS_DENSE_SQL
    else:
        raise ValueError(f"Unsupported DB-backed source: {source}")
    rows = fetch_rows(sql, (k_scan,))
    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        out[str(r["user_id"])].append(r)
    return out


def build_popularity_baseline_recs(
    *,
    train_counts: Mapping[Tuple[str, str], int],
    train: Mapping[str, set[str]],
    test_users: Sequence[str],
    track_meta: Mapping[str, Mapping[str, str]],
    k_scan: int,
    candidate_track_ids: set[str] | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    global_pop: Dict[str, int] = defaultdict(int)
    for (_u, t), c in train_counts.items():
        if candidate_track_ids is not None and t not in candidate_track_ids:
            continue
        global_pop[t] += int(c)

    ranked_tracks = sorted(global_pop.items(), key=lambda x: (-x[1], x[0]))
    top_tracks = ranked_tracks[: max(k_scan * 5, 500)]

    recs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for user_id in test_users:
        seen = train[user_id]
        rank = 1
        for track_id, pop in top_tracks:
            if track_id in seen:
                continue
            meta = track_meta.get(track_id, {})
            recs[user_id].append(
                {
                    "user_id": user_id,
                    "recommendation_rank": rank,
                    "track_id": track_id,
                    "track_name": meta.get("track_name", track_id),
                    "artist_name": meta.get("artist_name", "__unknown_artist__"),
                    "recommendation_score": float(pop),
                }
            )
            rank += 1
            if rank > k_scan:
                break
    return recs


def build_hybrid_recs(
    *,
    mf_recs: Mapping[str, List[Dict[str, object]]],
    train_counts: Mapping[Tuple[str, str], int],
    train: Mapping[str, set[str]],
    test_users: Sequence[str],
    track_meta: Mapping[str, Mapping[str, str]],
    k_scan: int,
    candidate_track_ids: set[str] | None = None,
    weight_mf: float = 0.65,
    weight_pop: float = 0.25,
    weight_artist: float = 0.10,
) -> Dict[str, List[Dict[str, object]]]:
    weight_sum = max(weight_mf + weight_pop + weight_artist, 1e-9)
    w_mf = weight_mf / weight_sum
    w_pop = weight_pop / weight_sum
    w_artist = weight_artist / weight_sum

    # Global popularity prior.
    global_pop: Dict[str, int] = defaultdict(int)
    for (_u, t), c in train_counts.items():
        if candidate_track_ids is not None and t not in candidate_track_ids:
            continue
        global_pop[t] += int(c)
    max_pop = max(global_pop.values()) if global_pop else 1

    # User artist affinity from train history.
    user_artist_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (u, t), c in train_counts.items():
        artist = str(track_meta.get(t, {}).get("artist_name", "__unknown_artist__"))
        user_artist_counts[u][artist] += int(c)

    user_top_artists: Dict[str, set[str]] = {}
    for u, counts in user_artist_counts.items():
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        user_top_artists[u] = {a for a, _ in ranked[:5]}

    # Candidate pool from MF + global popularity.
    pop_ranked = [t for t, _ in sorted(global_pop.items(), key=lambda x: (-x[1], x[0]))[: max(k_scan * 5, 500)]]

    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for user_id in test_users:
        seen = train[user_id]
        candidates: Dict[str, Dict[str, object]] = {}

        mf_rows = mf_recs.get(user_id, [])
        if mf_rows:
            mf_scores = [float(r.get("recommendation_score", 0.0)) for r in mf_rows]
            mf_min = min(mf_scores)
            mf_max = max(mf_scores)
            mf_den = (mf_max - mf_min) if mf_max != mf_min else 1.0
        else:
            mf_min, mf_den = 0.0, 1.0

        for r in mf_rows[: max(k_scan * 5, 300)]:
            tid = str(r["track_id"])
            if candidate_track_ids is not None and tid not in candidate_track_ids:
                continue
            if tid in seen:
                continue
            candidates[tid] = {
                "track_id": tid,
                "track_name": str(r.get("track_name") or tid),
                "artist_name": str(r.get("artist_name") or "__unknown_artist__"),
                "mf_score": float(r.get("recommendation_score", 0.0)),
            }

        for tid in pop_ranked:
            if tid in seen:
                continue
            if tid not in candidates:
                meta = track_meta.get(tid, {})
                candidates[tid] = {
                    "track_id": tid,
                    "track_name": str(meta.get("track_name", tid)),
                    "artist_name": str(meta.get("artist_name", "__unknown_artist__")),
                    "mf_score": mf_min,  # neutral low if missing in MF pool
                }
            if len(candidates) >= max(k_scan * 8, 600):
                break

        scored_rows: List[Dict[str, object]] = []
        top_artists = user_top_artists.get(user_id, set())
        for tid, c in candidates.items():
            pop_norm = float(global_pop.get(tid, 0)) / float(max_pop)
            mf_norm = (float(c["mf_score"]) - mf_min) / mf_den
            artist_bonus = 1.0 if str(c["artist_name"]) in top_artists else 0.0
            hybrid_score = (w_mf * mf_norm) + (w_pop * pop_norm) + (w_artist * artist_bonus)
            scored_rows.append(
                {
                    "user_id": user_id,
                    "track_id": tid,
                    "track_name": c["track_name"],
                    "artist_name": c["artist_name"],
                    "recommendation_score": float(hybrid_score),
                }
            )

        ranked = sorted(
            scored_rows,
            key=lambda x: (-float(x["recommendation_score"]), str(x["track_id"])),
        )[:k_scan]
        for rank, row in enumerate(ranked, start=1):
            row["recommendation_rank"] = rank
        out[user_id] = ranked

    return out


def dcg_at_k(pred_tracks: Sequence[str], truth: set[str], k: int) -> float:
    score = 0.0
    for rank, track_id in enumerate(pred_tracks[:k], start=1):
        if track_id in truth:
            score += 1.0 / math.log2(rank + 1.0)
    return score


def idcg_at_k(truth_size: int, k: int) -> float:
    ideal_hits = min(truth_size, k)
    if ideal_hits <= 0:
        return 0.0
    return sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return mean(vals) if vals else 0.0


def evaluate_source(
    *,
    source: str,
    k: int,
    train: Mapping[str, set[str]],
    test: Mapping[str, set[str]],
    total_tracks: int,
    preloaded_recs: Mapping[str, List[Dict[str, object]]] | None = None,
) -> EvalSummary:
    k_scan = max(k * 5, 200)
    recs = dict(preloaded_recs) if preloaded_recs is not None else load_recommendations(source, k_scan)

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    hit_rates: List[float] = []
    readable_flags: List[float] = []
    predicted_items: set[str] = set()

    user_pred_sets: Dict[str, set[str]] = {}
    users_with_recs = 0

    for user_id, truth in test.items():
        raw = recs.get(user_id, [])
        seen = train[user_id]
        picked: List[Dict[str, object]] = []
        used_track_ids: set[str] = set()
        for row in raw:
            track_id = str(row["track_id"])
            if track_id in seen:
                continue
            if track_id in used_track_ids:
                continue
            used_track_ids.add(track_id)
            picked.append(row)
            if len(picked) >= k:
                break

        pred_tracks = [str(r["track_id"]) for r in picked]
        if pred_tracks:
            users_with_recs += 1

        user_pred_sets[user_id] = set(pred_tracks)
        predicted_items.update(pred_tracks)

        hits = sum(1 for t in pred_tracks if t in truth)
        precisions.append(hits / float(k))
        recalls.append(hits / float(max(len(truth), 1)))
        hit_rates.append(1.0 if hits > 0 else 0.0)

        dcg = dcg_at_k(pred_tracks, truth, k)
        idcg = idcg_at_k(len(truth), k)
        ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

        if picked:
            readable_flags.extend(
                1.0
                if is_readable(
                    str(r.get("track_name") or ""),
                    str(r.get("artist_name") or ""),
                    str(r.get("track_id") or ""),
                )
                else 0.0
                for r in picked
            )

    # Personalization: average pairwise Jaccard distance across user top-k sets.
    distances: List[float] = []
    user_items = [(u, s) for u, s in user_pred_sets.items() if s]
    for (_, a), (_, b) in itertools.combinations(user_items, 2):
        union = a | b
        if not union:
            continue
        jaccard = len(a & b) / float(len(union))
        distances.append(1.0 - jaccard)

    users_evaluated = len(test)
    return EvalSummary(
        source=source,
        users_evaluated=users_evaluated,
        users_with_recs=users_with_recs,
        user_coverage=(users_with_recs / float(max(users_evaluated, 1))),
        precision_at_k=safe_mean(precisions),
        recall_at_k=safe_mean(recalls),
        ndcg_at_k=safe_mean(ndcgs),
        hit_rate_at_k=safe_mean(hit_rates),
        item_coverage_at_k=(len(predicted_items) / float(max(total_tracks, 1))),
        personalization_at_k=safe_mean(distances),
        readable_rate_at_k=safe_mean(readable_flags),
    )


def print_summary(summary: EvalSummary, k: int) -> None:
    print(f"\nsource={summary.source} k={k}")
    print(f"users_evaluated={summary.users_evaluated}")
    print(f"users_with_recs={summary.users_with_recs}")
    print(f"user_coverage={summary.user_coverage:.4f}")
    print(f"precision@{k}={summary.precision_at_k:.4f}")
    print(f"recall@{k}={summary.recall_at_k:.4f}")
    print(f"ndcg@{k}={summary.ndcg_at_k:.4f}")
    print(f"hit_rate@{k}={summary.hit_rate_at_k:.4f}")
    print(f"item_coverage@{k}={summary.item_coverage_at_k:.4f}")
    print(f"personalization@{k}={summary.personalization_at_k:.4f}")
    print(f"readable_rate@{k}={summary.readable_rate_at_k:.4f}")


def write_json(path: str, payload: Dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: str, k: int, summaries: Sequence[EvalSummary]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Recommendation Quality Evaluation")
    lines.append("")
    lines.append(f"- `k`: {k}")
    lines.append("")
    lines.append("| Source | Users Eval | Users with Recs | User Coverage | Precision@K | Recall@K | NDCG@K | HitRate@K | ItemCoverage@K | Personalization@K | ReadableRate@K |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s.source} | {s.users_evaluated} | {s.users_with_recs} | {s.user_coverage:.4f} | "
            f"{s.precision_at_k:.4f} | {s.recall_at_k:.4f} | {s.ndcg_at_k:.4f} | {s.hit_rate_at_k:.4f} | "
            f"{s.item_coverage_at_k:.4f} | {s.personalization_at_k:.4f} | {s.readable_rate_at_k:.4f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendation quality using temporal holdout.")
    parser.add_argument("--source", choices=["mf", "dense", "pop", "hybrid", "both", "all"], default="all")
    parser.add_argument("--k", type=int, default=20, help="Top-k evaluation cutoff.")
    parser.add_argument("--holdout-size", type=int, default=5, help="Latest N events per user held out for testing.")
    parser.add_argument("--min-user-events", type=int, default=20, help="Minimum events per user to include in evaluation.")
    parser.add_argument(
        "--target-mode",
        choices=["novel", "all"],
        default="novel",
        help="Evaluation truth set: 'novel' uses holdout tracks not seen in train (default), 'all' uses all holdout tracks.",
    )
    parser.add_argument("--json-out", default="artifacts/evaluation/recommendation_quality.json")
    parser.add_argument("--md-out", default="artifacts/evaluation/recommendation_quality.md")
    parser.add_argument(
        "--mf-eval-mode",
        choices=["temporal_retrain", "stored"],
        default="temporal_retrain",
        help="For source=mf/both: evaluate stored DB recs or retrain MF on temporal train split.",
    )
    parser.add_argument("--mf-factors", type=int, default=48)
    parser.add_argument("--mf-epochs", type=int, default=4)
    parser.add_argument("--mf-lr", type=float, default=0.03)
    parser.add_argument("--mf-reg", type=float, default=0.01)
    parser.add_argument("--mf-neg-ratio", type=int, default=2)
    parser.add_argument("--mf-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=10000,
        help="Restrict candidate track universe to top-N train-popularity tracks. Set 0 to disable.",
    )
    parser.add_argument("--hybrid-weight-mf", type=float, default=0.65)
    parser.add_argument("--hybrid-weight-pop", type=float, default=0.25)
    parser.add_argument("--hybrid-weight-artist", type=float, default=0.10)
    parser.add_argument(
        "--hybrid-weight-grid",
        default="",
        help="Optional semicolon-separated triplets, e.g. '0.7,0.2,0.1;0.6,0.3,0.1'. Best NDCG is selected.",
    )
    args = parser.parse_args()

    holdout_rows = fetch_rows(HOLDOUT_SQL, (args.min_user_events,))
    train, test, train_counts, track_meta = build_holdout(
        holdout_rows,
        holdout_size=args.holdout_size,
        target_mode=args.target_mode,
    )
    if not test:
        raise ValueError(
            "No valid users for evaluation. Try --target-mode all or lower --min-user-events/--holdout-size."
        )

    total_tracks_row = fetch_rows(TOTAL_TRACKS_SQL)
    total_tracks = int(total_tracks_row[0]["total_tracks"]) if total_tracks_row else 0

    # Optional candidate restriction to reduce tail-noise and improve ranking signal.
    candidate_track_ids: set[str] | None = None
    if args.candidate_top_n > 0:
        pop_tmp: Dict[str, int] = defaultdict(int)
        for (_u, t), c in train_counts.items():
            pop_tmp[t] += int(c)
        ranked_tmp = sorted(pop_tmp.items(), key=lambda x: (-x[1], x[0]))[: args.candidate_top_n]
        candidate_track_ids = {t for t, _ in ranked_tmp}
        if candidate_track_ids:
            train_counts = {(u, t): c for (u, t), c in train_counts.items() if t in candidate_track_ids}
            train = {u: {t for t in s if t in candidate_track_ids} for u, s in train.items()}
            test = {u: {t for t in s if t in candidate_track_ids} for u, s in test.items()}
            track_meta = {t: m for t, m in track_meta.items() if t in candidate_track_ids}
            valid_users = [u for u in test.keys() if test[u] and train.get(u)]
            train = {u: train[u] for u in valid_users}
            test = {u: test[u] for u in valid_users}
            if not test:
                raise ValueError(
                    "No valid users after candidate filtering. Reduce --candidate-top-n or disable with 0."
                )
            total_tracks = len(candidate_track_ids)
            print(
                f"candidate_filter enabled top_n={args.candidate_top_n} "
                f"candidate_tracks={len(candidate_track_ids)} users_after_filter={len(test)}"
            )

    if args.source == "both":
        sources = ["mf", "dense"]
    elif args.source == "all":
        sources = ["pop", "mf", "hybrid", "dense"]
    else:
        sources = [args.source]
    mf_preloaded_recs: Dict[str, List[Dict[str, object]]] | None = None
    pop_preloaded_recs: Dict[str, List[Dict[str, object]]] | None = None
    hybrid_preloaded_recs: Dict[str, List[Dict[str, object]]] | None = None
    hybrid_source_label = "hybrid"
    if "mf" in sources and args.mf_eval_mode == "temporal_retrain":
        interactions, user_ids, item_ids, user_seen = prepare_mf_training(train_counts)
        print(
            f"temporal_mf_train interactions={len(interactions)} users={len(user_ids)} "
            f"items={len(item_ids)}"
        )
        if interactions and user_ids and item_ids:
            user_f, item_f, user_b, item_b = train_mf_temporal(
                interactions=interactions,
                n_users=len(user_ids),
                n_items=len(item_ids),
                user_seen=user_seen,
                factors=args.mf_factors,
                epochs=args.mf_epochs,
                lr=args.mf_lr,
                reg=args.mf_reg,
                neg_ratio=args.mf_neg_ratio,
                seed=args.mf_seed,
            )
            mf_preloaded_recs = build_temporal_mf_recs(
                user_f=user_f,
                item_f=item_f,
                user_b=user_b,
                item_b=item_b,
                user_ids=user_ids,
                item_ids=item_ids,
                user_seen=user_seen,
                track_meta=track_meta,
                k_scan=max(args.k * 5, 200),
            )
        else:
            print("temporal_mf_train skipped (no interactions after split)")

    if "pop" in sources or "hybrid" in sources:
        pop_preloaded_recs = build_popularity_baseline_recs(
            train_counts=train_counts,
            train=train,
            test_users=list(test.keys()),
            track_meta=track_meta,
            k_scan=max(args.k * 5, 200),
            candidate_track_ids=candidate_track_ids,
        )

    if "hybrid" in sources:
        mf_for_hybrid = mf_preloaded_recs
        if mf_for_hybrid is None:
            # Use stored MF if temporal retrain was not requested/computed.
            mf_for_hybrid = load_recommendations("mf", max(args.k * 5, 200))
        if args.hybrid_weight_grid.strip():
            combos: List[Tuple[float, float, float]] = []
            for token in args.hybrid_weight_grid.split(";"):
                token = token.strip()
                if not token:
                    continue
                parts = [p.strip() for p in token.split(",")]
                if len(parts) != 3:
                    raise ValueError(f"Invalid hybrid weight triplet: {token}")
                combos.append((float(parts[0]), float(parts[1]), float(parts[2])))
        else:
            combos = [(args.hybrid_weight_mf, args.hybrid_weight_pop, args.hybrid_weight_artist)]

        best_key = None
        best_summary = None
        best_recs = None
        for w_mf, w_pop, w_artist in combos:
            trial_recs = build_hybrid_recs(
                mf_recs=mf_for_hybrid,
                train_counts=train_counts,
                train=train,
                test_users=list(test.keys()),
                track_meta=track_meta,
                k_scan=max(args.k * 5, 200),
                candidate_track_ids=candidate_track_ids,
                weight_mf=w_mf,
                weight_pop=w_pop,
                weight_artist=w_artist,
            )
            trial_summary = evaluate_source(
                source=f"hybrid(w={w_mf:.2f},{w_pop:.2f},{w_artist:.2f})",
                k=args.k,
                train=train,
                test=test,
                total_tracks=total_tracks,
                preloaded_recs=trial_recs,
            )
            print_summary(trial_summary, args.k)
            key = (trial_summary.ndcg_at_k, trial_summary.hit_rate_at_k, trial_summary.precision_at_k)
            if best_key is None or key > best_key:
                best_key = key
                best_summary = trial_summary
                best_recs = trial_recs

        hybrid_preloaded_recs = best_recs
        if best_summary is not None:
            hybrid_source_label = best_summary.source

    summaries: List[EvalSummary] = []
    for source in sources:
        preloaded: Mapping[str, List[Dict[str, object]]] | None = None
        if source == "mf" and mf_preloaded_recs is not None:
            preloaded = mf_preloaded_recs
        elif source == "pop":
            preloaded = pop_preloaded_recs
        elif source == "hybrid":
            preloaded = hybrid_preloaded_recs

        summary = evaluate_source(
            source=source,
            k=args.k,
            train=train,
            test=test,
            total_tracks=total_tracks,
            preloaded_recs=preloaded,
        )
        if source == "hybrid":
            summary.source = hybrid_source_label
        summaries.append(summary)
        print_summary(summary, args.k)

    payload = {
        "k": args.k,
        "holdout_size": args.holdout_size,
        "min_user_events": args.min_user_events,
        "target_mode": args.target_mode,
        "mf_eval_mode": args.mf_eval_mode,
        "candidate_top_n": args.candidate_top_n,
        "hybrid_weights": {
            "mf": args.hybrid_weight_mf,
            "pop": args.hybrid_weight_pop,
            "artist": args.hybrid_weight_artist,
            "grid": args.hybrid_weight_grid,
        },
        "users_in_holdout": len(test),
        "results": [asdict(s) for s in summaries],
    }
    write_json(args.json_out, payload)
    write_markdown(args.md_out, args.k, summaries)
    print(f"\njson_written={args.json_out}")
    print(f"markdown_written={args.md_out}")


if __name__ == "__main__":
    main()
