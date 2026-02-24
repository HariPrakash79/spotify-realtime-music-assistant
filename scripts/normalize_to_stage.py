#!/usr/bin/env python3
"""
Normalize raw S3 datasets into SQL-friendly staged parquet files.

Current coverage:
- lastfm_1k (listen events)
- fma_metadata (track catalog)

Output prefixes:
- <root_prefix>/stage/listen_events/source=lastfm_1k/run_date=YYYY-MM-DD/*.parquet
- <root_prefix>/stage/track_catalog/source=fma_metadata/run_date=YYYY-MM-DD/*.parquet
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

MB = 1024 * 1024

LISTEN_EVENTS_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("user_id", pa.string()),
        ("track_id", pa.string()),
        ("artist_id", pa.string()),
        ("event_ts", pa.timestamp("us", tz="UTC")),
        ("event_type", pa.string()),
        ("track_name", pa.string()),
        ("artist_name", pa.string()),
    ]
)

TRACK_CATALOG_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("track_id", pa.string()),
        ("artist_id", pa.string()),
        ("track_name", pa.string()),
        ("artist_name", pa.string()),
        ("genre", pa.string()),
        ("duration_sec", pa.float32()),
    ]
)


def load_datasets(config_path: str) -> Dict[str, Dict[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Invalid config format: expected top-level 'datasets' map.")
    return datasets


def qualify_key(root_prefix: str, object_key: str) -> str:
    object_key = object_key.lstrip("/")
    root_prefix = root_prefix.strip("/")
    return f"{root_prefix}/{object_key}" if root_prefix else object_key


def parse_iso_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def safe_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null"}:
        return None
    return text


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if pd.isna(f):
        return None
    return f


class S3ParquetWriter:
    def __init__(
        self,
        s3,
        bucket: str,
        base_key: str,
        schema: pa.Schema,
        dry_run: bool = False,
    ) -> None:
        self.s3 = s3
        self.bucket = bucket
        self.base_key = base_key.strip("/")
        self.schema = schema
        self.dry_run = dry_run
        self.part = 0

    def write_rows(self, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        key = f"{self.base_key}/part-{self.part:05d}.parquet"
        self.part += 1
        if self.dry_run:
            print(f"[dry-run] would write s3://{self.bucket}/{key} rows={len(rows)}")
            return

        table = pa.Table.from_pylist(rows, schema=self.schema)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        print(f"wrote s3://{self.bucket}/{key} rows={table.num_rows}")


def normalize_lastfm(
    s3,
    bucket: str,
    root_prefix: str,
    raw_object_key: str,
    run_date: str,
    batch_rows: int,
    dry_run: bool,
) -> None:
    raw_key = qualify_key(root_prefix, raw_object_key)
    out_base = qualify_key(
        root_prefix,
        f"stage/listen_events/source=lastfm_1k/run_date={run_date}",
    )
    writer = S3ParquetWriter(
        s3=s3,
        bucket=bucket,
        base_key=out_base,
        schema=LISTEN_EVENTS_SCHEMA,
        dry_run=dry_run,
    )

    print(f"normalizing lastfm_1k from s3://{bucket}/{raw_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_key)
    body = obj["Body"]

    rows: List[Dict[str, object]] = []
    total = 0
    bad = 0
    target_name = "userid-timestamp-artid-artname-traid-traname.tsv"

    with tarfile.open(fileobj=body, mode="r|gz") as tar:
        found = False
        for member in tar:
            if not member.isfile() or not member.name.endswith(target_name):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            found = True
            for raw_line in stream:
                line = raw_line.decode("utf-8", "ignore").rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 6:
                    bad += 1
                    continue

                user_id, ts_text, artist_id, artist_name, track_id, track_name = parts
                ts = parse_iso_ts(ts_text)
                if ts is None:
                    bad += 1
                    continue

                rows.append(
                    {
                        "source": "lastfm_1k",
                        "user_id": safe_text(user_id),
                        "track_id": safe_text(track_id),
                        "artist_id": safe_text(artist_id),
                        "event_ts": ts,
                        "event_type": "play",
                        "track_name": safe_text(track_name),
                        "artist_name": safe_text(artist_name),
                    }
                )
                total += 1
                if len(rows) >= batch_rows:
                    writer.write_rows(rows)
                    rows = []
            break

        if not found:
            raise RuntimeError(f"Could not locate {target_name} in {raw_key}")

    if rows:
        writer.write_rows(rows)

    print(f"lastfm_1k done. rows={total}, bad_rows={bad}")


def flatten_columns(columns: Iterable[object]) -> List[str]:
    out: List[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = []
            for part in col:
                text = str(part).strip()
                if not text or text.startswith("Unnamed"):
                    continue
                parts.append(text)
            flat = "_".join(parts).lower().replace(" ", "_")
        else:
            flat = str(col).strip().lower().replace(" ", "_")
        out.append(flat)
    return out


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fma_chunks(zipf: zipfile.ZipFile, member: str, chunksize: int):
    try:
        with zipf.open(member) as f:
            yield from pd.read_csv(f, header=[0, 1], chunksize=chunksize, low_memory=False)
        return
    except Exception:
        pass

    with zipf.open(member) as f:
        yield from pd.read_csv(f, header=0, chunksize=chunksize, low_memory=False)


def normalize_fma_catalog(
    s3,
    bucket: str,
    root_prefix: str,
    raw_object_key: str,
    run_date: str,
    batch_rows: int,
    dry_run: bool,
) -> None:
    raw_key = qualify_key(root_prefix, raw_object_key)
    out_base = qualify_key(
        root_prefix,
        f"stage/track_catalog/source=fma_metadata/run_date={run_date}",
    )
    writer = S3ParquetWriter(
        s3=s3,
        bucket=bucket,
        base_key=out_base,
        schema=TRACK_CATALOG_SCHEMA,
        dry_run=dry_run,
    )

    print(f"normalizing fma_metadata from s3://{bucket}/{raw_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_key)
    body = obj["Body"]

    total = 0
    with tempfile.SpooledTemporaryFile(max_size=512 * MB) as tmp:
        for chunk in body.iter_chunks(chunk_size=8 * MB):
            if chunk:
                tmp.write(chunk)
        tmp.seek(0)

        with zipfile.ZipFile(tmp) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith("tracks.csv")]
            if not members:
                raise RuntimeError("tracks.csv not found in fma_metadata.zip")
            tracks_member = members[0]

            for chunk_df in fma_chunks(zf, tracks_member, batch_rows):
                chunk_df.columns = flatten_columns(chunk_df.columns)
                if "track_id" not in chunk_df.columns:
                    chunk_df = chunk_df.rename(columns={chunk_df.columns[0]: "track_id"})

                c_track_id = first_existing(chunk_df, ["track_id"])
                c_artist_id = first_existing(chunk_df, ["artist_id"])
                c_track_name = first_existing(chunk_df, ["track_title", "track_name", "title"])
                c_artist_name = first_existing(chunk_df, ["artist_name", "artist"])
                c_genre = first_existing(chunk_df, ["track_genre_top", "genre_top", "genre"])
                c_duration = first_existing(chunk_df, ["track_duration", "duration", "duration_sec"])

                rows: List[Dict[str, object]] = []
                for _, row in chunk_df.iterrows():
                    rows.append(
                        {
                            "source": "fma_metadata",
                            "track_id": safe_text(row.get(c_track_id)) if c_track_id else None,
                            "artist_id": safe_text(row.get(c_artist_id)) if c_artist_id else None,
                            "track_name": safe_text(row.get(c_track_name)) if c_track_name else None,
                            "artist_name": safe_text(row.get(c_artist_name)) if c_artist_name else None,
                            "genre": safe_text(row.get(c_genre)) if c_genre else None,
                            "duration_sec": safe_float(row.get(c_duration)) if c_duration else None,
                        }
                    )

                writer.write_rows(rows)
                total += len(rows)

    print(f"fma_metadata done. rows={total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw S3 datasets into stage parquet.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Datasets to normalize: lastfm_1k, fma_metadata, all",
    )
    parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_BUCKET"),
        help="S3 bucket (or set DATA_BUCKET env var).",
    )
    parser.add_argument(
        "--root-prefix",
        default=os.environ.get("S3_PREFIX", ""),
        help="Root key prefix (example: music-assistant).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=100000,
        help="Rows per output parquet part.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing stage parquet files.",
    )
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("Bucket is required. Pass --bucket or set DATA_BUCKET.")

    datasets = load_datasets(args.config)
    requested = args.dataset
    if requested == ["all"]:
        requested = ["lastfm_1k", "fma_metadata"]

    unknown = [d for d in requested if d not in datasets]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(sorted(datasets.keys()))}"
        )

    s3 = boto3.client("s3", region_name=args.region)
    run_date = datetime.now(timezone.utc).date().isoformat()

    for dataset in requested:
        if dataset == "lastfm_1k":
            normalize_lastfm(
                s3=s3,
                bucket=args.bucket,
                root_prefix=args.root_prefix,
                raw_object_key=datasets[dataset]["object_key"],
                run_date=run_date,
                batch_rows=args.batch_rows,
                dry_run=args.dry_run,
            )
            continue

        if dataset == "fma_metadata":
            normalize_fma_catalog(
                s3=s3,
                bucket=args.bucket,
                root_prefix=args.root_prefix,
                raw_object_key=datasets[dataset]["object_key"],
                run_date=run_date,
                batch_rows=args.batch_rows,
                dry_run=args.dry_run,
            )
            continue

        print(f"dataset {dataset} is not implemented yet in normalize_to_stage.py; skipping")


if __name__ == "__main__":
    main()
