#!/usr/bin/env python3
"""
Run common recommendation queries against Postgres/RDS.

Examples:
  python scripts/query_recommendations.py --query top-users
  python scripts/query_recommendations.py --query recs --user-id user_000002
  python scripts/query_recommendations.py --query trending --limit 10
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Mapping, Sequence

from psycopg import connect
from psycopg.rows import dict_row


TOP_USERS_SQL = """
SELECT
    user_id,
    COUNT(*) AS plays
FROM music.listen_events
GROUP BY user_id
ORDER BY plays DESC
LIMIT %s
"""


USER_RECS_SQL = """
SELECT
    user_id,
    recommendation_rank,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d
WHERE user_id = %s
  AND recommendation_rank <= %s
ORDER BY recommendation_rank
"""


TRENDING_SQL = """
SELECT
    global_rank_7d,
    track_name,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM music.v_global_trending_tracks_7d
WHERE global_rank_7d <= %s
ORDER BY global_rank_7d
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


def print_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        print("No rows returned.")
        return

    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print("-" * (len(" | ".join(headers)) + 8))
    for row in rows:
        print(" | ".join(str(row.get(h, "")) for h in headers))


def run_query(query_name: str, limit: int, user_id: str | None) -> tuple[str, Iterable[object]]:
    if query_name == "top-users":
        return TOP_USERS_SQL, (limit,)
    if query_name == "trending":
        return TRENDING_SQL, (limit,)
    if query_name == "recs":
        if not user_id:
            raise ValueError("--user-id is required when --query recs is used.")
        return USER_RECS_SQL, (user_id, limit)
    raise ValueError(f"Unsupported query: {query_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recommendation queries on Postgres.")
    parser.add_argument(
        "--query",
        choices=["top-users", "recs", "trending"],
        default="recs",
        help="Which query to run.",
    )
    parser.add_argument("--user-id", default=None, help="User id (required for --query recs).")
    parser.add_argument("--limit", type=int, default=20, help="Limit rows/rank returned.")
    args = parser.parse_args()

    conn_kwargs = build_conn_kwargs()
    sql, params = run_query(args.query, args.limit, args.user_id)

    if "conninfo" in conn_kwargs:
        conn = connect(conn_kwargs["conninfo"], row_factory=dict_row)
    else:
        conn = connect(row_factory=dict_row, **conn_kwargs)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    print_rows(rows)


if __name__ == "__main__":
    main()
