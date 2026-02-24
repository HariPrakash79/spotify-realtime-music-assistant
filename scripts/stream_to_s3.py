#!/usr/bin/env python3
"""
Stream public datasets directly to S3-compatible object storage.
No local raw files are created.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from typing import Dict, Iterable, List, Tuple

import boto3
import requests
import yaml
from botocore.exceptions import ClientError

MB = 1024 * 1024
MIN_S3_PART_SIZE = 5 * MB
DEFAULT_PART_SIZE = 64 * MB


def load_datasets(config_path: str) -> Dict[str, Dict[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Invalid config format: expected top-level 'datasets' map.")
    return datasets


def resolve_targets(
    datasets: Dict[str, Dict[str, str]], dataset_args: List[str]
) -> Iterable[Tuple[str, Dict[str, str]]]:
    if not dataset_args or dataset_args == ["all"]:
        return datasets.items()

    missing = [d for d in dataset_args if d not in datasets]
    if missing:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(missing)}. "
            f"Available: {', '.join(sorted(datasets.keys()))}"
        )
    return [(name, datasets[name]) for name in dataset_args]


def get_s3_client(endpoint_url: str | None, region: str) -> boto3.client:
    session = boto3.session.Session()
    return session.client("s3", endpoint_url=endpoint_url, region_name=region)


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_part(s3, bucket: str, key: str, upload_id: str, part_no: int, data: bytes) -> str:
    resp = s3.upload_part(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        PartNumber=part_no,
        Body=data,
    )
    return resp["ETag"]


def stream_url_to_s3(
    s3,
    bucket: str,
    key: str,
    url: str,
    part_size: int,
    timeout_seconds: int,
) -> None:
    if part_size < MIN_S3_PART_SIZE:
        raise ValueError(f"part_size must be >= {MIN_S3_PART_SIZE} bytes")

    with requests.get(url, stream=True, timeout=timeout_seconds) as r:
        r.raise_for_status()
        upload = s3.create_multipart_upload(Bucket=bucket, Key=key)
        upload_id = upload["UploadId"]
        parts = []
        part_no = 1
        bytes_uploaded = 0
        start = time.time()
        buffer = io.BytesIO()

        try:
            for chunk in r.iter_content(chunk_size=8 * MB):
                if not chunk:
                    continue
                buffer.write(chunk)

                if buffer.tell() >= part_size:
                    data = buffer.getvalue()
                    etag = upload_part(s3, bucket, key, upload_id, part_no, data)
                    parts.append({"ETag": etag, "PartNumber": part_no})
                    bytes_uploaded += len(data)
                    elapsed = max(time.time() - start, 1e-6)
                    speed = bytes_uploaded / MB / elapsed
                    print(
                        f"  Uploaded part {part_no}, total={bytes_uploaded / MB:.2f} MB, "
                        f"speed={speed:.2f} MB/s"
                    )
                    part_no += 1
                    buffer = io.BytesIO()

            remaining = buffer.getvalue()
            if remaining:
                etag = upload_part(s3, bucket, key, upload_id, part_no, remaining)
                parts.append({"ETag": etag, "PartNumber": part_no})
                bytes_uploaded += len(remaining)
                print(f"  Uploaded final part {part_no}, total={bytes_uploaded / MB:.2f} MB")

            s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream public datasets directly into S3-compatible storage."
    )
    parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to dataset manifest YAML.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Dataset name(s) from config or 'all'.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_BUCKET"),
        help="S3 bucket name. Can also come from DATA_BUCKET env var.",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("S3_PREFIX", ""),
        help="Optional key prefix inside bucket (example: music-assistant).",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("S3_ENDPOINT_URL"),
        help="S3 endpoint URL for R2/MinIO (optional).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--part-size-mb",
        type=int,
        default=64,
        help="Multipart upload part size in MB (>=5).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout for source download request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite object if key already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without uploading.",
    )
    args = parser.parse_args()

    datasets = load_datasets(args.config)
    if args.list:
        print("Available datasets:")
        for name in sorted(datasets.keys()):
            item = datasets[name]
            print(f"- {name}: {item.get('url')} -> {item.get('object_key')}")
        return

    if not args.bucket:
        raise ValueError("Bucket is required. Pass --bucket or set DATA_BUCKET.")

    part_size = args.part_size_mb * MB
    s3 = get_s3_client(args.endpoint_url, args.region)

    for name, item in resolve_targets(datasets, args.dataset):
        src = item["url"]
        object_key = item["object_key"].lstrip("/")
        if args.prefix:
            object_key = f"{args.prefix.strip('/')}/{object_key}"

        print(f"\nDataset: {name}")
        print(f"  Source: {src}")
        print(f"  Target: s3://{args.bucket}/{object_key}")

        if args.dry_run:
            continue

        if object_exists(s3, args.bucket, object_key) and not args.force:
            print("  Skipped (already exists). Use --force to overwrite.")
            continue

        stream_url_to_s3(
            s3=s3,
            bucket=args.bucket,
            key=object_key,
            url=src,
            part_size=part_size,
            timeout_seconds=args.timeout_seconds,
        )
        print("  Upload completed.")


if __name__ == "__main__":
    main()
