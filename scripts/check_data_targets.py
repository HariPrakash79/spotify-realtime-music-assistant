#!/usr/bin/env python3
"""
Check project data volume targets in Postgres.

Default targets:
  - events >= 500000
  - distinct tracks >= 50000
  - distinct users >= 1000

Examples:
  python scripts/check_data_targets.py
  python scripts/check_data_targets.py --target-events 300000 --target-users 100
"""

from __future__ import annotations

import argparse
import os

from psycopg import connect
from psycopg.rows import dict_row


METRICS_SQL = """
SELECT
    COUNT(*)::BIGINT AS total_events,
    COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
    COUNT(DISTINCT track_id)::BIGINT FILTER (WHERE track_id IS NOT NULL) AS distinct_tracks
FROM music.listen_events
"""


def build_conn_kwargs() -> dict:
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


def status(actual: int, target: int) -> str:
    return "PASS" if actual >= target else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset progress against project targets.")
    parser.add_argument("--target-events", type=int, default=500000, help="Target minimum total events.")
    parser.add_argument("--target-tracks", type=int, default=50000, help="Target minimum distinct tracks.")
    parser.add_argument("--target-users", type=int, default=1000, help="Target minimum distinct users.")
    args = parser.parse_args()

    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(METRICS_SQL)
            row = cur.fetchone()
    finally:
        conn.close()

    total_events = int(row["total_events"])
    distinct_users = int(row["distinct_users"])
    distinct_tracks = int(row["distinct_tracks"])

    print("metric\tactual\ttarget\tstatus")
    print(f"events\t{total_events}\t{args.target_events}\t{status(total_events, args.target_events)}")
    print(f"users\t{distinct_users}\t{args.target_users}\t{status(distinct_users, args.target_users)}")
    print(f"tracks\t{distinct_tracks}\t{args.target_tracks}\t{status(distinct_tracks, args.target_tracks)}")

    all_pass = (
        total_events >= args.target_events
        and distinct_users >= args.target_users
        and distinct_tracks >= args.target_tracks
    )

    if all_pass:
        print("overall\tPASS\tAll targets satisfied.")
    else:
        print("overall\tIN_PROGRESS\tOne or more targets not yet reached.")


if __name__ == "__main__":
    main()
