#!/usr/bin/env python3
"""
Profile user diversity from staged listen event parquet files in S3.

Examples:
  python scripts/profile_user_diversity.py --source lastfm_1k --sample-files 30
  python scripts/profile_user_diversity.py --source lastfm_1k --all-files
"""

from __future__ import annotations

import argparse
import io
import os
import random
from collections import Counter
from typing import List, Optional

import boto3
import pyarrow.parquet as pq


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
            prefix = cp.get("Prefix", "")
            marker = "run_date="
            if marker not in prefix:
                continue
            run_date = prefix.split(marker, 1)[1].strip("/").split("/", 1)[0]
            run_dates.append(run_date)

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
) -> List[str]:
    stage_prefix = build_stage_prefix(root_prefix, source, run_date)
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=stage_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

    if not keys:
        raise RuntimeError(
            f"No parquet files found under s3://{bucket}/{stage_prefix}. "
            "Verify normalization output."
        )
    return keys


def pick_keys(
    keys: List[str],
    sample_files: int,
    sample_strategy: str,
    seed: int,
    all_files: bool,
) -> List[str]:
    if all_files or sample_files >= len(keys):
        return list(keys)

    if sample_strategy == "sequential":
        return keys[:sample_files]

    rng = random.Random(seed)
    sampled = list(keys)
    rng.shuffle(sampled)
    return sampled[:sample_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile user diversity in staged listen events.")
    parser.add_argument("--bucket", default=os.environ.get("DATA_BUCKET"), help="S3 bucket name.")
    parser.add_argument("--root-prefix", default=os.environ.get("S3_PREFIX", ""), help="S3 root key prefix.")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile name.")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region.")
    parser.add_argument("--source", default="lastfm_1k", help="Staged source name.")
    parser.add_argument("--run-date", default=None, help="Run date (YYYY-MM-DD). Defaults to latest.")
    parser.add_argument("--sample-files", type=int, default=30, help="Number of files to scan in sample mode.")
    parser.add_argument(
        "--sample-strategy",
        choices=["random", "sequential"],
        default="random",
        help="How sampled files are selected.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used in random sample mode.")
    parser.add_argument("--all-files", action="store_true", help="Scan all files (exact distinct users).")
    parser.add_argument("--top-users", type=int, default=10, help="Show top users by sampled row count.")
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("Bucket is required. Pass --bucket or set DATA_BUCKET.")

    s3 = get_s3_client(args.profile, args.region)
    run_date = args.run_date or discover_latest_run_date(s3, args.bucket, args.root_prefix, args.source)
    all_keys = list_parquet_keys(
        s3=s3,
        bucket=args.bucket,
        root_prefix=args.root_prefix,
        source=args.source,
        run_date=run_date,
    )
    scan_keys = pick_keys(
        keys=all_keys,
        sample_files=args.sample_files,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        all_files=args.all_files,
    )

    print(f"source={args.source} run_date={run_date}")
    print(f"s3://{args.bucket}/{build_stage_prefix(args.root_prefix, args.source, run_date)}")
    print(f"files_total={len(all_keys)} files_scanned={len(scan_keys)} mode={'all' if args.all_files else 'sample'}")

    unique_users = set()
    user_counts: Counter[str] = Counter()
    sampled_rows = 0

    for i, key in enumerate(scan_keys, start=1):
        print(f"[{i}/{len(scan_keys)}] scanning {key}")
        obj = s3.get_object(Bucket=args.bucket, Key=key)
        table = pq.read_table(io.BytesIO(obj["Body"].read()), columns=["user_id"])
        sampled_rows += table.num_rows

        col = table.column("user_id")
        for chunk in col.chunks:
            values = chunk.to_pylist()
            for user_id in values:
                if user_id is None:
                    continue
                text = str(user_id).strip()
                if not text:
                    continue
                unique_users.add(text)
                user_counts[text] += 1

    print(
        "summary "
        f"sampled_rows={sampled_rows} "
        f"sampled_unique_users={len(unique_users)}"
    )

    if not args.all_files:
        print("note sampled mode gives a lower bound. Use --all-files for exact distinct users.")

    if user_counts:
        print(f"top_users_sampled (top {args.top_users})")
        for user_id, cnt in user_counts.most_common(args.top_users):
            print(f"{user_id}\t{cnt}")


if __name__ == "__main__":
    main()
