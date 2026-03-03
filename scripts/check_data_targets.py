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
  python scripts/check_data_targets.py --scope raw
  python scripts/check_data_targets.py --scope both
"""

from __future__ import annotations

import argparse
import os

from psycopg import connect
from psycopg.rows import dict_row


METRICS_SQL_RAW = """
SELECT
    COUNT(*)::BIGINT AS total_events,
    COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
    COUNT(DISTINCT track_id)::BIGINT AS distinct_tracks
FROM music.listen_events
"""

METRICS_SQL_READY = """
SELECT
    COUNT(*)::BIGINT AS total_events,
    COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
    COUNT(DISTINCT track_id)::BIGINT AS distinct_tracks
FROM music.v_listen_events_model_1000_ready
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


def query_metrics(scope: str) -> dict:
    sql = METRICS_SQL_READY if scope == "ready" else METRICS_SQL_RAW
    conn = connect_postgres()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
    finally:
        conn.close()
    return {
        "events": int(row["total_events"]),
        "users": int(row["distinct_users"]),
        "tracks": int(row["distinct_tracks"]),
    }


def print_scope(scope: str, metrics: dict, target_events: int, target_tracks: int, target_users: int) -> bool:
    total_events = metrics["events"]
    distinct_users = metrics["users"]
    distinct_tracks = metrics["tracks"]

    print(f"\nscope\t{scope}")
    print("metric\tactual\ttarget\tstatus")
    print(f"events\t{total_events}\t{target_events}\t{status(total_events, target_events)}")
    print(f"users\t{distinct_users}\t{target_users}\t{status(distinct_users, target_users)}")
    print(f"tracks\t{distinct_tracks}\t{target_tracks}\t{status(distinct_tracks, target_tracks)}")

    all_pass = (
        total_events >= target_events
        and distinct_users >= target_users
        and distinct_tracks >= target_tracks
    )
    if all_pass:
        print("overall\tPASS\tAll targets satisfied.")
    else:
        print("overall\tIN_PROGRESS\tOne or more targets not yet reached.")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset progress against project targets.")
    parser.add_argument("--target-events", type=int, default=500000, help="Target minimum total events.")
    parser.add_argument("--target-tracks", type=int, default=50000, help="Target minimum distinct tracks.")
    parser.add_argument("--target-users", type=int, default=1000, help="Target minimum distinct users.")
    parser.add_argument(
        "--scope",
        choices=["ready", "raw", "both"],
        default="ready",
        help="Which dataset slice to validate. 'ready' uses recommendation-ready filtered model slice.",
    )
    args = parser.parse_args()

    if args.scope == "both":
        raw_ok = print_scope(
            scope="raw",
            metrics=query_metrics("raw"),
            target_events=args.target_events,
            target_tracks=args.target_tracks,
            target_users=args.target_users,
        )
        ready_ok = print_scope(
            scope="ready",
            metrics=query_metrics("ready"),
            target_events=args.target_events,
            target_tracks=args.target_tracks,
            target_users=args.target_users,
        )
        if raw_ok and ready_ok:
            print("\ncombined_overall\tPASS\tBoth raw and ready scopes satisfy targets.")
        else:
            print("\ncombined_overall\tIN_PROGRESS\tAt least one scope is below target.")
    else:
        print_scope(
            scope=args.scope,
            metrics=query_metrics(args.scope),
            target_events=args.target_events,
            target_tracks=args.target_tracks,
            target_users=args.target_users,
        )


if __name__ == "__main__":
    main()
