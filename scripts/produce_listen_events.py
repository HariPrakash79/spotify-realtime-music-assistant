#!/usr/bin/env python3
"""
Produce staged listen event parquet data from S3 into Kafka topic listen_events_raw.

Source:
  s3://<bucket>/<prefix>/stage/listen_events/source=<source>/run_date=<YYYY-MM-DD>/part-*.parquet
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq
from kafka import KafkaProducer

from text_cleanup import clean_text


def qualify_key(root_prefix: str, object_key: str) -> str:
    object_key = object_key.lstrip("/")
    root_prefix = root_prefix.strip("/")
    return f"{root_prefix}/{object_key}" if root_prefix else object_key


def build_stage_prefix(root_prefix: str, source: str, run_date: Optional[str]) -> str:
    base = qualify_key(root_prefix, f"stage/listen_events/source={source}")
    if run_date:
        return f"{base}/run_date={run_date}/"
    return f"{base}/"


def get_s3_client(profile: Optional[str], region: str):
    if profile:
        session = boto3.session.Session(profile_name=profile, region_name=region)
        return session.client("s3")
    return boto3.client("s3", region_name=region)


def discover_latest_run_date(s3, bucket: str, root_prefix: str, source: str) -> str:
    base_prefix = build_stage_prefix(root_prefix, source, None)
    paginator = s3.get_paginator("list_objects_v2")

    run_dates: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp.get("Prefix", "")
            marker = "run_date="
            if marker in p:
                value = p.split(marker, 1)[1].strip("/").split("/", 1)[0]
                run_dates.append(value)

    if not run_dates:
        raise RuntimeError(
            f"No run_date prefixes found under s3://{bucket}/{base_prefix}. "
            "Run normalize_to_stage.py first."
        )
    return sorted(run_dates)[-1]


def list_parquet_keys(
    s3,
    bucket: str,
    root_prefix: str,
    source: str,
    run_date: str,
    max_files: Optional[int],
) -> List[str]:
    stage_prefix = build_stage_prefix(root_prefix, source, run_date)
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=stage_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            keys.append(key)
            if max_files and len(keys) >= max_files:
                return keys

    if not keys:
        raise RuntimeError(
            f"No parquet files found under s3://{bucket}/{stage_prefix}. "
            "Verify normalization output."
        )
    return keys


def to_iso_utc(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            value = value.tz_localize(timezone.utc)
        else:
            value = value.tz_convert(timezone.utc)
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    text = str(value).strip()
    return text or None


def make_producer(
    bootstrap_servers: str,
    acks: str,
    linger_ms: int,
    batch_size: int,
    compression_type: str,
):
    servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("No Kafka bootstrap servers provided.")
    return KafkaProducer(
        bootstrap_servers=servers,
        acks=acks,
        linger_ms=linger_ms,
        batch_size=batch_size,
        compression_type=compression_type,
        value_serializer=lambda v: json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: (k or "").encode("utf-8"),
    )


def iter_event_records_from_s3_object(
    s3,
    bucket: str,
    key: str,
    batch_rows: int,
    randomize_rows: bool = False,
    rng: Optional[random.Random] = None,
) -> Iterable[Dict[str, Optional[str]]]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    buf = io.BytesIO(body)
    pf = pq.ParquetFile(buf)

    for batch in pf.iter_batches(batch_size=batch_rows):
        df = batch.to_pandas()
        if randomize_rows and len(df) > 1:
            idx = list(range(len(df)))
            if rng is not None:
                rng.shuffle(idx)
            else:
                random.shuffle(idx)
            df = df.iloc[idx]
        for _, row in df.iterrows():
            yield {
                "source": clean_text(row.get("source")),
                "user_id": clean_text(row.get("user_id")),
                "track_id": clean_text(row.get("track_id")),
                "artist_id": clean_text(row.get("artist_id")),
                "event_ts": to_iso_utc(row.get("event_ts")),
                "event_type": clean_text(row.get("event_type")) or "play",
                "track_name": clean_text(row.get("track_name")),
                "artist_name": clean_text(row.get("artist_name")),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce staged listen events from S3 to Kafka.")
    parser.add_argument("--bucket", default=os.environ.get("DATA_BUCKET"), help="S3 bucket name.")
    parser.add_argument("--root-prefix", default=os.environ.get("S3_PREFIX", ""), help="S3 root key prefix.")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile name.")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region.")
    parser.add_argument("--source", default="lastfm_1k", help="Staged source name.")
    parser.add_argument("--run-date", default=None, help="Run date (YYYY-MM-DD). Defaults to latest available.")
    parser.add_argument("--topic", default="listen_events_raw", help="Kafka topic name.")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of parquet files.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of produced records.")
    parser.add_argument(
        "--max-records-per-user",
        type=int,
        default=None,
        help="Optional cap per user to improve diversity (e.g., 3000 for ~100 users in 300k records).",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random"],
        default="sequential",
        help="Sequential preserves S3 file order; random shuffles file order for more user diversity.",
    )
    parser.add_argument(
        "--randomize-rows",
        action="store_true",
        help="Shuffle row order within each parquet batch when sample strategy is random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-strategy random is enabled.",
    )
    parser.add_argument("--parquet-batch-rows", type=int, default=10000, help="Rows per parquet read batch.")
    parser.add_argument("--acks", default="all", choices=["0", "1", "all"], help="Kafka acks mode.")
    parser.add_argument("--linger-ms", type=int, default=20, help="Kafka producer linger_ms.")
    parser.add_argument("--batch-size", type=int, default=65536, help="Kafka producer batch_size in bytes.")
    parser.add_argument(
        "--compression-type",
        default="gzip",
        choices=["gzip", "snappy", "lz4", "zstd", "none"],
        help="Kafka producer compression type.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Read and count records without sending to Kafka.")
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("Bucket is required. Pass --bucket or set DATA_BUCKET.")

    s3 = get_s3_client(args.profile, args.region)
    run_date = args.run_date or discover_latest_run_date(s3, args.bucket, args.root_prefix, args.source)
    keys = list_parquet_keys(
        s3=s3,
        bucket=args.bucket,
        root_prefix=args.root_prefix,
        source=args.source,
        run_date=run_date,
        max_files=args.max_files,
    )
    rng: Optional[random.Random] = None
    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(keys)

    print(f"source={args.source} run_date={run_date} files={len(keys)}")
    print(f"s3://{args.bucket}/{build_stage_prefix(args.root_prefix, args.source, run_date)}")

    producer = None
    if not args.dry_run:
        compression = None if args.compression_type == "none" else args.compression_type
        producer = make_producer(
            bootstrap_servers=args.bootstrap_servers,
            acks=args.acks,
            linger_ms=args.linger_ms,
            batch_size=args.batch_size,
            compression_type=compression,
        )
        print(f"kafka_topic={args.topic} bootstrap={args.bootstrap_servers}")

    produced = 0
    invalid = 0
    skipped_user_cap = 0
    per_user_counts: Dict[str, int] = {}

    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{len(keys)}] reading {key}")
        for event in iter_event_records_from_s3_object(
            s3=s3,
            bucket=args.bucket,
            key=key,
            batch_rows=args.parquet_batch_rows,
            randomize_rows=(args.sample_strategy == "random" and args.randomize_rows),
            rng=rng,
        ):
            user_id = event.get("user_id")
            if not user_id or not event.get("event_ts"):
                invalid += 1
                continue

            if args.max_records_per_user is not None:
                seen_for_user = per_user_counts.get(user_id, 0)
                if seen_for_user >= args.max_records_per_user:
                    skipped_user_cap += 1
                    continue

            if args.dry_run:
                produced += 1
                if produced <= 3:
                    print("sample", event)
            else:
                producer.send(args.topic, key=user_id, value=event)
                produced += 1

            per_user_counts[user_id] = per_user_counts.get(user_id, 0) + 1

            if args.max_records and produced >= args.max_records:
                break

        if args.max_records and produced >= args.max_records:
            break

    if producer is not None:
        producer.flush(timeout=120)
        producer.close()

    print(
        "done "
        f"produced={produced} "
        f"invalid_skipped={invalid} "
        f"user_cap_skipped={skipped_user_cap} "
        f"unique_users={len(per_user_counts)} "
        f"topic={args.topic} "
        f"run_date={run_date}"
    )


if __name__ == "__main__":
    main()
