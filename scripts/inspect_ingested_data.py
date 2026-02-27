#!/usr/bin/env python3
"""
Export ingestion quality + dataset snapshots from Postgres into local artifacts.

Outputs:
  - <output_dir>/summary.json
  - <output_dir>/source_breakdown.csv
  - <output_dir>/top_users.csv
  - <output_dir>/catalog_genres.csv
  - <output_dir>/event_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from psycopg import connect
from psycopg.rows import dict_row


SUMMARY_SQL = """
SELECT
    COUNT(*)::BIGINT AS total_events,
    COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
    COUNT(DISTINCT track_id)::BIGINT AS distinct_tracks,
    MIN(event_ts) AS min_event_ts,
    MAX(event_ts) AS max_event_ts
FROM music.listen_events
"""


SOURCE_BREAKDOWN_SQL = """
SELECT
    source,
    COUNT(*)::BIGINT AS events,
    COUNT(DISTINCT user_id)::BIGINT AS users,
    COUNT(DISTINCT track_id)::BIGINT AS tracks,
    MIN(event_ts) AS min_event_ts,
    MAX(event_ts) AS max_event_ts
FROM music.listen_events
GROUP BY source
ORDER BY events DESC, source
"""


TOP_USERS_SQL = """
SELECT
    user_id,
    COUNT(*)::BIGINT AS plays,
    COUNT(DISTINCT track_id)::BIGINT AS distinct_tracks,
    MIN(event_ts) AS first_seen_ts,
    MAX(event_ts) AS last_seen_ts
FROM music.listen_events
GROUP BY user_id
ORDER BY plays DESC, user_id
LIMIT %s
"""


CATALOG_GENRES_SQL = """
SELECT
    COALESCE(NULLIF(genre, ''), '__unknown__') AS genre,
    COUNT(*)::BIGINT AS tracks
FROM music.track_catalog
GROUP BY COALESCE(NULLIF(genre, ''), '__unknown__')
ORDER BY tracks DESC, genre
LIMIT %s
"""


EVENT_SAMPLE_SQL = """
SELECT
    event_id,
    source,
    user_id,
    track_id,
    artist_id,
    event_ts,
    event_type,
    track_name,
    artist_name,
    ingestion_ts
FROM music.listen_events
ORDER BY event_ts DESC, event_id DESC
LIMIT %s
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


def run_sql(conn, sql: str, params: Iterable[Any] | None = None) -> Sequence[Mapping[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def to_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_json_safe(v) for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export snapshots from ingested Postgres data.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/inspection",
        help="Where to write summary files.",
    )
    parser.add_argument(
        "--sample-events",
        type=int,
        default=1000,
        help="Number of latest listen_events rows to export.",
    )
    parser.add_argument(
        "--top-users",
        type=int,
        default=100,
        help="Number of top users to export.",
    )
    parser.add_argument(
        "--top-genres",
        type=int,
        default=100,
        help="Number of catalog genres to export.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_postgres()
    try:
        summary_rows = run_sql(conn, SUMMARY_SQL)
        source_rows = run_sql(conn, SOURCE_BREAKDOWN_SQL)
        top_users_rows = run_sql(conn, TOP_USERS_SQL, (args.top_users,))
        catalog_genres_rows = run_sql(conn, CATALOG_GENRES_SQL, (args.top_genres,))
        sample_rows = run_sql(conn, EVENT_SAMPLE_SQL, (args.sample_events,))
    finally:
        conn.close()

    summary = dict(summary_rows[0]) if summary_rows else {}
    summary_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: to_json_safe(v) for k, v in summary.items()},
        "targets": {"events": 500000, "users": 1000, "tracks": 50000},
        "target_status": {
            "events": "PASS" if int(summary.get("total_events", 0) or 0) >= 500000 else "FAIL",
            "users": "PASS" if int(summary.get("distinct_users", 0) or 0) >= 1000 else "FAIL",
            "tracks": "PASS" if int(summary.get("distinct_tracks", 0) or 0) >= 50000 else "FAIL",
        },
        "notes": [
            "listen_events may exclude invalid rows (missing user_id or event_ts).",
            "producer-side limits (max-records, max-records-per-user) reduce ingest volume by design.",
            "duplicate events are removed using event_hash uniqueness in Postgres schema.",
        ],
        "files": {
            "source_breakdown": str((out_dir / "source_breakdown.csv").as_posix()),
            "top_users": str((out_dir / "top_users.csv").as_posix()),
            "catalog_genres": str((out_dir / "catalog_genres.csv").as_posix()),
            "event_sample": str((out_dir / "event_sample.csv").as_posix()),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    write_csv(out_dir / "source_breakdown.csv", source_rows)
    write_csv(out_dir / "top_users.csv", top_users_rows)
    write_csv(out_dir / "catalog_genres.csv", catalog_genres_rows)
    write_csv(out_dir / "event_sample.csv", sample_rows)

    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'source_breakdown.csv'} rows={len(source_rows)}")
    print(f"wrote {out_dir / 'top_users.csv'} rows={len(top_users_rows)}")
    print(f"wrote {out_dir / 'catalog_genres.csv'} rows={len(catalog_genres_rows)}")
    print(f"wrote {out_dir / 'event_sample.csv'} rows={len(sample_rows)}")


if __name__ == "__main__":
    main()
