#!/usr/bin/env python3
"""
Repair mojibake/corrupted UTF-8 text in Postgres metadata columns.

Targets:
- music.listen_events.track_name
- music.listen_events.artist_name
- music.track_catalog.track_name
- music.track_catalog.artist_name

Usage:
  python scripts/repair_metadata_text.py --dry-run
  python scripts/repair_metadata_text.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from psycopg import connect
from psycopg import sql as psql
from psycopg.rows import dict_row

from text_cleanup import clean_text


TARGETS: Sequence[Tuple[str, str, str]] = (
    ("music", "listen_events", "track_name"),
    ("music", "listen_events", "artist_name"),
    ("music", "track_catalog", "track_name"),
    ("music", "track_catalog", "artist_name"),
    ("music", "user_recommendations_mf_ready", "track_name"),
    ("music", "user_recommendations_mf_ready", "artist_name"),
    ("music", "user_recommendations_hybrid_ready", "track_name"),
    ("music", "user_recommendations_hybrid_ready", "artist_name"),
)


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


def fetch_value_counts(conn, schema: str, table: str, column: str) -> List[Dict[str, Any]]:
    query = psql.SQL(
        """
        SELECT
            {col} AS value,
            COUNT(*)::BIGINT AS n
        FROM {schema}.{table}
        WHERE {col} IS NOT NULL
          AND {col} <> ''
        GROUP BY {col}
        ORDER BY n DESC
        """
    ).format(
        schema=psql.Identifier(schema),
        table=psql.Identifier(table),
        col=psql.Identifier(column),
    )
    with conn.cursor() as cur:
        cur.execute(query)
        return [dict(r) for r in cur.fetchall()]


def build_mapping(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    mapping: List[Tuple[str, str, int]] = []
    for row in rows:
        old = str(row["value"])
        n = int(row["n"])
        new = clean_text(old, repair_mojibake=True)
        if new is None:
            continue
        if new != old:
            mapping.append((old, new, n))
    return mapping


def apply_mapping(conn, schema: str, table: str, column: str, mapping: Sequence[Tuple[str, str, int]]) -> int:
    if not mapping:
        return 0
    query = psql.SQL(
        "UPDATE {schema}.{table} SET {col} = %s WHERE {col} = %s"
    ).format(
        schema=psql.Identifier(schema),
        table=psql.Identifier(table),
        col=psql.Identifier(column),
    )
    params = [(new, old) for (old, new, _n) in mapping]
    updated_total = 0
    with conn.cursor() as cur:
        for new, old in params:
            cur.execute(query, (new, old))
            updated_total += max(cur.rowcount, 0)
    return updated_total


def summarize_examples(mapping: Sequence[Tuple[str, str, int]], limit: int = 10) -> List[Dict[str, Any]]:
    examples = sorted(mapping, key=lambda x: -x[2])[: max(limit, 0)]
    return [
        {
            "old": old,
            "new": new,
            "rows_affected": n,
        }
        for old, new, n in examples
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair mojibake metadata text in Postgres.")
    parser.add_argument("--dry-run", action="store_true", help="Preview fixes without writing updates.")
    parser.add_argument(
        "--summary-out",
        default="artifacts/cleanup/metadata_repair_summary.json",
        help="Where to write JSON summary.",
    )
    parser.add_argument("--example-limit", type=int, default=12, help="Number of example replacements per column.")
    args = parser.parse_args()

    conn = connect_postgres()
    summary: Dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "targets": [],
    }
    overall_rows_changed = 0
    overall_value_changes = 0

    try:
        for schema, table, column in TARGETS:
            rows = fetch_value_counts(conn, schema, table, column)
            mapping = build_mapping(rows)
            rows_changed = sum(n for (_old, _new, n) in mapping)
            value_changes = len(mapping)
            updated_rows = 0

            if not args.dry_run and mapping:
                updated_rows = apply_mapping(conn, schema, table, column, mapping)

            target_item = {
                "target": f"{schema}.{table}.{column}",
                "distinct_values_scanned": len(rows),
                "distinct_values_changed": value_changes,
                "rows_that_would_change": rows_changed,
                "rows_updated": updated_rows if not args.dry_run else 0,
                "examples": summarize_examples(mapping, limit=args.example_limit),
            }
            summary["targets"].append(target_item)

            overall_rows_changed += rows_changed
            overall_value_changes += value_changes
            print(
                f"{schema}.{table}.{column}: "
                f"distinct_changed={value_changes} rows_affected={rows_changed}"
                + (f" rows_updated={updated_rows}" if not args.dry_run else "")
            )

        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        conn.close()

    summary["overall"] = {
        "distinct_values_changed": overall_value_changes,
        "rows_that_would_change": overall_rows_changed,
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_written={out}")
    if args.dry_run:
        print("dry_run=True no database rows were changed.")


if __name__ == "__main__":
    main()
