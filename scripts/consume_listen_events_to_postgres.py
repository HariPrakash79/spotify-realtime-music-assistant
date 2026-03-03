#!/usr/bin/env python3
"""
Consume Kafka listen events and write them into Postgres.

Topic:
  listen_events_raw

Target table:
  music.listen_events
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from kafka import KafkaConsumer
from psycopg import connect
from psycopg.rows import dict_row

from text_cleanup import clean_text

INSERT_SQL = """
INSERT INTO music.listen_events (
    source,
    user_id,
    track_id,
    artist_id,
    event_ts,
    event_type,
    track_name,
    artist_name,
    event_hash
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_hash) DO NOTHING
"""

def build_event_hash(
    source: str,
    user_id: str,
    track_id: Optional[str],
    artist_id: Optional[str],
    event_ts: str,
    event_type: str,
    track_name: Optional[str],
    artist_name: Optional[str],
) -> str:
    parts = [
        source,
        user_id,
        track_id or "",
        artist_id or "",
        event_ts,
        event_type,
        track_name or "",
        artist_name or "",
    ]
    payload = "\x1f".join(parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def canonicalize_event_ts(event_ts: str) -> str:
    text = event_ts.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds")


def parse_event(
    msg_value: object,
) -> Optional[Tuple[str, str, Optional[str], Optional[str], str, str, Optional[str], Optional[str], str]]:
    if not isinstance(msg_value, dict):
        return None

    source = clean_text(msg_value.get("source"))
    user_id = clean_text(msg_value.get("user_id"))
    track_id = clean_text(msg_value.get("track_id"))
    artist_id = clean_text(msg_value.get("artist_id"))
    event_ts = clean_text(msg_value.get("event_ts"))
    event_type = clean_text(msg_value.get("event_type")) or "play"
    track_name = clean_text(msg_value.get("track_name"))
    artist_name = clean_text(msg_value.get("artist_name"))

    if not source or not user_id or not event_ts:
        return None

    event_ts = canonicalize_event_ts(event_ts)

    event_hash = build_event_hash(
        source=source,
        user_id=user_id,
        track_id=track_id,
        artist_id=artist_id,
        event_ts=event_ts,
        event_type=event_type,
        track_name=track_name,
        artist_name=artist_name,
    )

    return (
        source,
        user_id,
        track_id,
        artist_id,
        event_ts,
        event_type,
        track_name,
        artist_name,
        event_hash,
    )


def make_consumer(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    auto_offset_reset: str,
    poll_max_records: int,
) -> KafkaConsumer:
    servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("No Kafka bootstrap servers provided.")

    return KafkaConsumer(
        topic,
        bootstrap_servers=servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
        max_poll_records=poll_max_records,
    )


def build_dsn_from_parts(args) -> Optional[str]:
    if args.db_dsn:
        return args.db_dsn

    host = args.db_host or os.environ.get("PGHOST")
    port = args.db_port or os.environ.get("PGPORT", "5432")
    dbname = args.db_name or os.environ.get("PGDATABASE")
    user = args.db_user or os.environ.get("PGUSER")
    password = args.db_password or os.environ.get("PGPASSWORD")
    sslmode = args.db_sslmode or os.environ.get("PGSSLMODE", "require")

    if not (host and dbname and user and password):
        return None

    return (
        f"host={host} port={port} dbname={dbname} "
        f"user={user} password={password} sslmode={sslmode}"
    )


def insert_batch(conn, rows: List[Tuple]) -> None:
    with conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows)
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume listen_events_raw from Kafka and write to Postgres.")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument("--topic", default="listen_events_raw", help="Kafka topic name.")
    parser.add_argument(
        "--group-id",
        default=os.environ.get("KAFKA_GROUP_ID", "music-listen-events-sink-v1"),
        help="Consumer group id.",
    )
    parser.add_argument(
        "--auto-offset-reset",
        default="earliest",
        choices=["earliest", "latest"],
        help="Where to start if no committed offsets.",
    )
    parser.add_argument(
        "--poll-max-records",
        type=int,
        default=1000,
        help="Max records fetched per poll.",
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=1000,
        help="Rows per DB insert transaction.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional stop after N consumed messages.",
    )
    parser.add_argument(
        "--db-dsn",
        default=os.environ.get("POSTGRES_DSN"),
        help="Postgres DSN (preferred), e.g. postgresql://user:pass@host:5432/db?sslmode=require",
    )
    parser.add_argument("--db-host", default=None, help="Postgres host (fallback if no --db-dsn).")
    parser.add_argument("--db-port", default=None, help="Postgres port.")
    parser.add_argument("--db-name", default=None, help="Postgres database name.")
    parser.add_argument("--db-user", default=None, help="Postgres user.")
    parser.add_argument("--db-password", default=None, help="Postgres password.")
    parser.add_argument("--db-sslmode", default=None, help="Postgres SSL mode.")
    parser.add_argument("--dry-run", action="store_true", help="Consume and validate but do not write DB.")
    args = parser.parse_args()

    dsn = build_dsn_from_parts(args)
    if not args.dry_run and not dsn:
        raise ValueError("Postgres connection is required. Set --db-dsn or PG* vars.")

    consumer = make_consumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
        poll_max_records=args.poll_max_records,
    )

    conn = None
    if not args.dry_run:
        conn = connect(dsn, autocommit=False, row_factory=dict_row)

    total_seen = 0
    total_written = 0
    total_invalid = 0
    batch: List[Tuple] = []

    print(
        f"consuming topic={args.topic} group_id={args.group_id} "
        f"bootstrap={args.bootstrap_servers} dry_run={args.dry_run}"
    )

    try:
        while True:
            records_map = consumer.poll(timeout_ms=1000, max_records=args.poll_max_records)
            if not records_map:
                if args.max_messages and total_seen >= args.max_messages:
                    break
                continue

            for _tp, records in records_map.items():
                for record in records:
                    total_seen += 1
                    parsed = parse_event(record.value)
                    if parsed is None:
                        total_invalid += 1
                    else:
                        batch.append(parsed)

                    if args.max_messages and total_seen >= args.max_messages:
                        break
                if args.max_messages and total_seen >= args.max_messages:
                    break

            if len(batch) >= args.insert_batch_size or (args.max_messages and total_seen >= args.max_messages):
                if args.dry_run:
                    total_written += len(batch)
                    batch = []
                else:
                    insert_batch(conn, batch)
                    total_written += len(batch)
                    batch = []
                consumer.commit()
                print(
                    f"progress seen={total_seen} written={total_written} invalid={total_invalid}"
                )

            if args.max_messages and total_seen >= args.max_messages:
                break

        if batch:
            if args.dry_run:
                total_written += len(batch)
            else:
                insert_batch(conn, batch)
                total_written += len(batch)
                consumer.commit()
            print(
                f"progress seen={total_seen} written={total_written} invalid={total_invalid}"
            )
    finally:
        try:
            consumer.close()
        finally:
            if conn is not None:
                conn.close()

    print(f"done seen={total_seen} written={total_written} invalid={total_invalid}")


if __name__ == "__main__":
    main()
