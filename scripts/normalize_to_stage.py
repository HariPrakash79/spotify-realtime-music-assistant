#!/usr/bin/env python3
"""
Normalize raw S3 datasets into SQL-friendly staged parquet files.

Current coverage:
- lastfm_1k (listen events)
- fma_metadata (track catalog)
- deezer_recsys25_archive (listen events; inferred columns)

Output prefixes:
- <root_prefix>/stage/listen_events/source=lastfm_1k/run_date=YYYY-MM-DD/*.parquet
- <root_prefix>/stage/listen_events/source=deezer_recsys25_archive/run_date=YYYY-MM-DD/*.parquet
- <root_prefix>/stage/track_catalog/source=fma_metadata/run_date=YYYY-MM-DD/*.parquet
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import lzma
import os
import re
import tarfile
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from text_cleanup import clean_text

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

USER_COL_CANDIDATES = [
    "user_id",
    "userid",
    "user",
    "session_id",
    "session",
    "msno",
    "member_id",
    "u",
    "usr",
    "customer_id",
    "listener_id",
    "uid",
    "user_idx",
    "user_index",
    "account_id",
    "profile_id",
]

TRACK_COL_CANDIDATES = [
    "track_id",
    "trackid",
    "item_idx",
    "track_idx",
    "song_idx",
    "trackid_hash",
    "song",
    "sng_id",
    "songid_hash",
    "media_id",
    "mediaid",
    "itemid_hash",
    "item_id",
    "itemid",
    "i",
    "sid",
    "isrc",
    "song_id",
    "songid",
    "track",
    "item",
]

ARTIST_ID_COL_CANDIDATES = [
    "artist_id",
    "artistid",
    "artist",
    "arid",
    "creator_id",
    "performer_id",
    "author_id",
]

TS_COL_CANDIDATES = [
    "event_ts",
    "event_timestamp",
    "event_time",
    "time",
    "t",
    "unix_ts",
    "ts_ms",
    "timestamp_ms",
    "timestamp_us",
    "utc_ts",
    "listen_time",
    "played_ts",
    "play_ts",
    "played_at",
    "play_timestamp",
    "listen_ts",
    "listened_at",
    "timestamp",
    "ts",
    "datetime",
    "date_time",
    "created_at",
]

TRACK_NAME_COL_CANDIDATES = [
    "track_name",
    "track_title",
    "song_name",
    "song_title",
    "media_name",
    "item_name",
    "title",
]

ARTIST_NAME_COL_CANDIDATES = [
    "artist_name",
    "creator_name",
    "performer_name",
    "author_name",
]

EVENT_TYPE_COL_CANDIDATES = [
    "event_type",
    "event",
    "event_name",
    "interaction",
    "action",
    "interaction_type",
]

EMBEDDING_HINTS = [
    "embedding",
    "vector",
    "feature",
    "audio",
    "spectrogram",
    "mfcc",
]

COMPRESSED_SUFFIXES = [".gz", ".bz2", ".xz"]


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
    return clean_text(value, repair_mojibake=True)


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


def normalize_col_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def pick_candidate_column(columns: Iterable[object], candidates: List[str]) -> Optional[str]:
    norm_to_orig: Dict[str, str] = {}
    for col in columns:
        norm = normalize_col_name(col)
        if norm and norm not in norm_to_orig:
            norm_to_orig[norm] = str(col)

    for candidate in candidates:
        if candidate in norm_to_orig:
            return norm_to_orig[candidate]

    for candidate in candidates:
        for norm, original in norm_to_orig.items():
            if norm.startswith(f"{candidate}_") or norm.endswith(f"_{candidate}"):
                return original
    return None


def infer_event_columns(columns: Iterable[object]) -> Optional[Dict[str, Optional[str]]]:
    user_col = pick_candidate_column(columns, USER_COL_CANDIDATES)
    track_col = pick_candidate_column(columns, TRACK_COL_CANDIDATES)
    ts_col = pick_candidate_column(columns, TS_COL_CANDIDATES)
    if not user_col or not track_col:
        return None

    return {
        "user_id": user_col,
        "track_id": track_col,
        "artist_id": pick_candidate_column(columns, ARTIST_ID_COL_CANDIDATES),
        "event_ts": ts_col,
        "track_name": pick_candidate_column(columns, TRACK_NAME_COL_CANDIDATES),
        "artist_name": pick_candidate_column(columns, ARTIST_NAME_COL_CANDIDATES),
        "event_type": pick_candidate_column(columns, EVENT_TYPE_COL_CANDIDATES),
    }


def infer_event_columns_from_df(df: pd.DataFrame) -> Optional[Dict[str, Optional[str]]]:
    col_map = infer_event_columns(df.columns)
    if col_map is not None:
        return col_map

    # Fallback when schema names are unfamiliar: pick the first two non-embedding columns.
    cols = [str(c) for c in df.columns]
    if len(cols) < 2:
        return None

    usable_cols = [
        c
        for c in cols
        if not any(hint in normalize_col_name(c) for hint in EMBEDDING_HINTS)
    ]
    if len(usable_cols) < 2:
        return None

    user_col = pick_candidate_column(usable_cols, USER_COL_CANDIDATES) or usable_cols[0]
    track_col = (
        pick_candidate_column([c for c in usable_cols if c != user_col], TRACK_COL_CANDIDATES)
        or next((c for c in usable_cols if c != user_col), None)
    )
    if not track_col:
        return None

    return {
        "user_id": user_col,
        "track_id": track_col,
        "artist_id": pick_candidate_column(cols, ARTIST_ID_COL_CANDIDATES),
        "event_ts": pick_candidate_column(cols, TS_COL_CANDIDATES),
        "track_name": pick_candidate_column(cols, TRACK_NAME_COL_CANDIDATES),
        "artist_name": pick_candidate_column(cols, ARTIST_NAME_COL_CANDIDATES),
        "event_type": pick_candidate_column(cols, EVENT_TYPE_COL_CANDIDATES),
    }


def parse_event_ts(value: object) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            value = value.tz_localize(timezone.utc)
        else:
            value = value.tz_convert(timezone.utc)
        return value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        number = float(value)
        abs_number = abs(number)

        if abs_number >= 1e18:
            seconds = number / 1e9
        elif abs_number >= 1e15:
            seconds = number / 1e6
        elif abs_number >= 1e12:
            seconds = number / 1e3
        else:
            seconds = number

        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception:
            return None

    text = safe_text(value)
    if text is None:
        return None

    try:
        numeric = float(text)
        return parse_event_ts(numeric)
    except Exception:
        pass

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def is_embedding_like_member(member_name: str) -> bool:
    lowered = member_name.lower()
    return any(hint in lowered for hint in EMBEDDING_HINTS)


def strip_compression_suffix(member_name: str) -> str:
    lowered = member_name.lower()
    changed = True
    while changed:
        changed = False
        for suffix in COMPRESSED_SUFFIXES:
            if lowered.endswith(suffix):
                lowered = lowered[: -len(suffix)]
                changed = True
    return lowered


def detect_tabular_member_type(member_name: str) -> Optional[str]:
    base = strip_compression_suffix(member_name)
    leaf = Path(base).name
    if base.endswith(".parquet") or base.endswith(".pq"):
        return "parquet"
    if base.endswith(".csv"):
        return "csv"
    if base.endswith(".tsv"):
        return "tsv"
    if base.endswith(".psv") or base.endswith(".pipe"):
        return "pipe"
    if base.endswith(".jsonl") or base.endswith(".ndjson"):
        return "jsonl"
    if base.endswith(".txt") or base.endswith(".dat"):
        return "text"
    if base.endswith(".json"):
        return "json"
    # Deezer RecSys sessions files are often extensionless, e.g. user_sessions/sessions_000000000123
    if leaf.startswith("sessions_") and "/user_sessions/" in base:
        return "session_like"
    return None


def open_maybe_compressed_stream(stream, member_name: str):
    lowered = member_name.lower()
    if lowered.endswith(".gz"):
        return gzip.GzipFile(fileobj=stream)
    if lowered.endswith(".bz2"):
        return bz2.BZ2File(stream)
    if lowered.endswith(".xz"):
        return lzma.LZMAFile(stream)
    return stream


def rows_from_event_df(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    source_name: str,
    writer: "S3ParquetWriter",
    rows_buffer: List[Dict[str, object]],
    batch_rows: int,
    total: int,
    bad: int,
    max_records: Optional[int],
    allow_missing_ts: bool = False,
    fallback_base_ts: Optional[datetime] = None,
) -> Tuple[int, int, bool]:
    c_user = column_map["user_id"]
    c_track = column_map["track_id"]
    c_artist_id = column_map["artist_id"]
    c_ts = column_map["event_ts"]
    c_track_name = column_map["track_name"]
    c_artist_name = column_map["artist_name"]
    c_event_type = column_map["event_type"]

    for _, row in df.iterrows():
        user_id = safe_text(row.get(c_user)) if c_user else None
        event_ts = parse_event_ts(row.get(c_ts)) if c_ts else None
        if user_id is None:
            bad += 1
            continue
        if event_ts is None:
            if allow_missing_ts:
                base_ts = fallback_base_ts or datetime.now(timezone.utc)
                event_ts = base_ts + timedelta(microseconds=total + 1)
            else:
                bad += 1
                continue

        track_id = safe_text(row.get(c_track)) if c_track else None
        artist_id = safe_text(row.get(c_artist_id)) if c_artist_id else None
        track_name = safe_text(row.get(c_track_name)) if c_track_name else None
        artist_name = safe_text(row.get(c_artist_name)) if c_artist_name else None
        event_type = safe_text(row.get(c_event_type)) if c_event_type else None
        rows_buffer.append(
            {
                "source": source_name,
                "user_id": user_id,
                "track_id": track_id,
                "artist_id": artist_id,
                "event_ts": event_ts,
                "event_type": event_type or "play",
                "track_name": track_name,
                "artist_name": artist_name,
            }
        )
        total += 1

        if len(rows_buffer) >= batch_rows:
            writer.write_rows(rows_buffer)
            rows_buffer.clear()

        if max_records is not None and total >= max_records:
            return total, bad, True

    return total, bad, False


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


def normalize_deezer_listen_events(
    s3,
    bucket: str,
    root_prefix: str,
    raw_object_key: str,
    run_date: str,
    batch_rows: int,
    dry_run: bool,
    max_records: Optional[int],
    max_members: Optional[int],
    tmp_dir: Optional[str],
    audit_dir: Optional[str],
) -> None:
    source_name = "deezer_recsys25_archive"
    raw_key = qualify_key(root_prefix, raw_object_key)
    out_base = qualify_key(
        root_prefix,
        f"stage/listen_events/source={source_name}/run_date={run_date}",
    )
    writer = S3ParquetWriter(
        s3=s3,
        bucket=bucket,
        base_key=out_base,
        schema=LISTEN_EVENTS_SCHEMA,
        dry_run=dry_run,
    )

    print(f"normalizing {source_name} from s3://{bucket}/{raw_key}")
    obj = s3.get_object(Bucket=bucket, Key=raw_key)
    body = obj["Body"]

    rows: List[Dict[str, object]] = []
    total = 0
    bad = 0
    run_start_ts = datetime.now(timezone.utc)
    scanned_members = 0
    used_members = 0
    should_stop = False
    detected_mappings: List[Dict[str, object]] = []
    skipped_sample: List[Dict[str, str]] = []
    skip_counts: Dict[str, int] = {}
    seen_file_members: List[str] = []
    unsupported_extension_counts: Dict[str, int] = {}

    def note_skip(member_name: str, reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if len(skipped_sample) < 50:
            skipped_sample.append({"member": member_name, "reason": reason})

    def note_file_member(member_name: str) -> None:
        if len(seen_file_members) < 200:
            seen_file_members.append(member_name)

    def note_unsupported_extension(member_name: str) -> None:
        base = strip_compression_suffix(member_name)
        ext = Path(base).suffix or "<no_ext>"
        unsupported_extension_counts[ext] = unsupported_extension_counts.get(ext, 0) + 1

    def add_mapping(member_name: str, columns: List[str], col_map: Dict[str, Optional[str]]) -> None:
        detected_mappings.append(
            {
                "member": member_name,
                "columns": columns,
                "mapped_columns": {k: v for k, v in col_map.items() if v},
                "missing_standard_columns": [k for k, v in col_map.items() if not v],
                "timestamp_mode": "source" if col_map.get("event_ts") else "generated_utc",
            }
        )

    with tarfile.open(fileobj=body, mode="r|gz") as tar:
        for member in tar:
            if should_stop:
                break
            if not member.isfile():
                continue

            member_name = member.name
            note_file_member(member_name)
            lower_name = member_name.lower()
            member_kind = detect_tabular_member_type(lower_name)
            if member_kind is None:
                note_skip(member_name, "unsupported_extension")
                note_unsupported_extension(member_name)
                continue

            scanned_members += 1
            if is_embedding_like_member(lower_name):
                print(f"skipping {member_name} (embedding/features file)")
                note_skip(member_name, "embedding_or_features")
                continue

            if max_members is not None and used_members >= max_members:
                print(f"{source_name}: reached --deezer-max-members={max_members}, stopping.")
                break

            stream = tar.extractfile(member)
            if stream is None:
                note_skip(member_name, "cannot_extract_stream")
                continue
            try:
                read_stream = open_maybe_compressed_stream(stream, member_name)
            except Exception as exc:
                print(f"skipping {member_name} (cannot open compressed stream: {exc})")
                note_skip(member_name, "bad_compressed_stream")
                continue

            if member_kind in {"parquet", "session_like"}:
                with tempfile.SpooledTemporaryFile(
                    max_size=256 * MB,
                    dir=tmp_dir or None,
                ) as tmp:
                    while True:
                        chunk = read_stream.read(8 * MB)
                        if not chunk:
                            break
                        tmp.write(chunk)
                    tmp.seek(0)

                    try:
                        pf = pq.ParquetFile(tmp)
                    except Exception as exc:
                        if member_kind == "parquet":
                            print(f"skipping {member_name} (cannot read parquet: {exc})")
                            note_skip(member_name, "bad_parquet")
                            continue

                        # session_like fallback: try to parse as text/jsonl when extensionless parquet detection fails
                        def _open_session_fallback_stream():
                            tmp.seek(0)
                            header = tmp.read(6)
                            tmp.seek(0)
                            if header.startswith(b"\x1f\x8b"):
                                return gzip.GzipFile(fileobj=tmp)
                            if header.startswith(b"BZh"):
                                return bz2.BZ2File(tmp)
                            if header.startswith(b"\xfd7zXZ\x00"):
                                return lzma.LZMAFile(tmp)
                            return tmp

                        chunk_iter = None
                        try:
                            chunk_iter = pd.read_csv(
                                _open_session_fallback_stream(),
                                sep=None,
                                engine="python",
                                chunksize=batch_rows,
                                low_memory=False,
                            )
                        except Exception:
                            try:
                                chunk_iter = pd.read_json(
                                    _open_session_fallback_stream(),
                                    lines=True,
                                    chunksize=batch_rows,
                                )
                            except Exception as fallback_exc:
                                print(
                                    f"skipping {member_name} (cannot read session file as parquet/text/jsonl: {fallback_exc})"
                                )
                                note_skip(member_name, "bad_session_like_file")
                                continue

                        col_map: Optional[Dict[str, Optional[str]]] = None
                        for df in chunk_iter:
                            if col_map is None:
                                col_map = infer_event_columns_from_df(df)
                                if col_map is None:
                                    print(f"skipping {member_name} (no usable user/track columns)")
                                    note_skip(member_name, "no_user_track_columns")
                                    break
                                used_members += 1
                                add_mapping(member_name, [str(c) for c in df.columns], col_map)
                                print(
                                    f"reading {member_name} user={col_map['user_id']} "
                                    f"track={col_map['track_id']} ts={col_map['event_ts']}"
                                )

                            total, bad, should_stop = rows_from_event_df(
                                df=df,
                                column_map=col_map,
                                source_name=source_name,
                                writer=writer,
                                rows_buffer=rows,
                                batch_rows=batch_rows,
                                total=total,
                                bad=bad,
                                max_records=max_records,
                                allow_missing_ts=True,
                                fallback_base_ts=run_start_ts,
                            )
                            if should_stop:
                                print(f"{source_name}: reached --max-records={max_records}, stopping.")
                                break
                        continue

                    batch_iter = pf.iter_batches(batch_size=batch_rows)
                    try:
                        first_batch = next(batch_iter)
                    except StopIteration:
                        note_skip(member_name, "empty_parquet")
                        continue

                    first_df = first_batch.to_pandas()
                    col_map = infer_event_columns_from_df(first_df)
                    if col_map is None:
                        print(f"skipping {member_name} (no usable user/track columns)")
                        note_skip(member_name, "no_user_track_columns")
                        continue

                    used_members += 1
                    add_mapping(member_name, [str(c) for c in pf.schema_arrow.names], col_map)
                    print(
                        f"reading {member_name} rows={pf.metadata.num_rows} "
                        f"user={col_map['user_id']} track={col_map['track_id']} ts={col_map['event_ts']}"
                    )

                    total, bad, should_stop = rows_from_event_df(
                        df=first_df,
                        column_map=col_map,
                        source_name=source_name,
                        writer=writer,
                        rows_buffer=rows,
                        batch_rows=batch_rows,
                        total=total,
                        bad=bad,
                        max_records=max_records,
                        allow_missing_ts=True,
                        fallback_base_ts=run_start_ts,
                    )
                    if should_stop:
                        print(f"{source_name}: reached --max-records={max_records}, stopping.")
                        break

                    for batch in batch_iter:
                        df = batch.to_pandas()
                        total, bad, should_stop = rows_from_event_df(
                            df=df,
                            column_map=col_map,
                            source_name=source_name,
                            writer=writer,
                            rows_buffer=rows,
                            batch_rows=batch_rows,
                            total=total,
                            bad=bad,
                            max_records=max_records,
                            allow_missing_ts=True,
                            fallback_base_ts=run_start_ts,
                        )
                        if should_stop:
                            print(f"{source_name}: reached --max-records={max_records}, stopping.")
                            break
                continue

            if member_kind == "jsonl":
                try:
                    chunk_iter = pd.read_json(
                        read_stream,
                        lines=True,
                        chunksize=batch_rows,
                    )
                except Exception as exc:
                    print(f"skipping {member_name} (cannot read jsonl table: {exc})")
                    note_skip(member_name, "bad_jsonl_table")
                    continue
            elif member_kind == "json":
                try:
                    json_df = pd.read_json(read_stream, lines=False)
                    chunk_iter = [json_df]
                except Exception as exc:
                    print(f"skipping {member_name} (cannot read json table: {exc})")
                    note_skip(member_name, "bad_json_table")
                    continue
            else:
                sep = ","
                read_kwargs = {
                    "chunksize": batch_rows,
                    "low_memory": False,
                }
                if member_kind == "tsv":
                    sep = "\t"
                    read_kwargs["sep"] = sep
                elif member_kind == "pipe":
                    sep = "|"
                    read_kwargs["sep"] = sep
                elif member_kind == "text":
                    # Unknown text files can still be parseable tables; let pandas sniff.
                    read_kwargs["sep"] = None
                    read_kwargs["engine"] = "python"
                else:
                    read_kwargs["sep"] = sep

                try:
                    chunk_iter = pd.read_csv(read_stream, **read_kwargs)
                except Exception as exc:
                    print(f"skipping {member_name} (cannot read text table: {exc})")
                    note_skip(member_name, "bad_text_table")
                    continue

            col_map: Optional[Dict[str, Optional[str]]] = None
            for df in chunk_iter:
                if col_map is None:
                    col_map = infer_event_columns_from_df(df)
                    if col_map is None:
                        print(f"skipping {member_name} (no usable user/track columns)")
                        note_skip(member_name, "no_user_track_columns")
                        break
                    used_members += 1
                    add_mapping(member_name, [str(c) for c in df.columns], col_map)
                    print(
                        f"reading {member_name} user={col_map['user_id']} "
                        f"track={col_map['track_id']} ts={col_map['event_ts']}"
                    )

                total, bad, should_stop = rows_from_event_df(
                    df=df,
                    column_map=col_map,
                    source_name=source_name,
                    writer=writer,
                    rows_buffer=rows,
                    batch_rows=batch_rows,
                    total=total,
                    bad=bad,
                    max_records=max_records,
                    allow_missing_ts=True,
                    fallback_base_ts=run_start_ts,
                )
                if should_stop:
                    print(f"{source_name}: reached --max-records={max_records}, stopping.")
                    break

    if rows:
        writer.write_rows(rows)

    audit_root = Path(audit_dir or "artifacts/deezer_audit")
    audit_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = audit_root / f"deezer_column_audit_{run_date}_{stamp}.json"
    audit = {
        "dataset": source_name,
        "run_date": run_date,
        "bucket": bucket,
        "raw_key": raw_key,
        "started_at_utc": run_start_ts.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_written": total,
        "bad_rows": bad,
        "scanned_members": scanned_members,
        "used_members": used_members,
        "max_records": max_records,
        "deezer_max_members": max_members,
        "skip_counts": skip_counts,
        "skipped_members_sample": skipped_sample,
        "seen_file_members_sample": seen_file_members,
        "unsupported_extension_counts": unsupported_extension_counts,
        "detected_mappings": detected_mappings,
        "status": "ok" if total > 0 else "no_rows",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote mapping audit: {audit_path}")

    if total == 0:
        raise RuntimeError(
            f"{source_name}: no rows normalized from archive. "
            f"Check mapping audit: {audit_path}"
        )

    print(
        f"{source_name} done. rows={total}, bad_rows={bad}, "
        f"scanned_members={scanned_members}, used_members={used_members}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw S3 datasets into stage parquet.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Datasets to normalize: lastfm_1k, fma_metadata, deezer_recsys25_archive, all",
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
        "--max-records",
        type=int,
        default=None,
        help="Optional max rows to output per dataset run (useful for very large sources).",
    )
    parser.add_argument(
        "--deezer-max-members",
        type=int,
        default=None,
        help="Optional max archive members to read for deezer_recsys25_archive.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=os.environ.get("NORMALIZE_TMP_DIR"),
        help="Optional temp directory for large intermediate files (example: D:\\temp).",
    )
    parser.add_argument(
        "--audit-dir",
        default=os.environ.get("NORMALIZE_AUDIT_DIR", "artifacts/deezer_audit"),
        help="Directory for Deezer column-mapping audit JSON files.",
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
        requested = ["lastfm_1k", "fma_metadata", "deezer_recsys25_archive"]

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

        if dataset == "deezer_recsys25_archive":
            normalize_deezer_listen_events(
                s3=s3,
                bucket=args.bucket,
                root_prefix=args.root_prefix,
                raw_object_key=datasets[dataset]["object_key"],
                run_date=run_date,
                batch_rows=args.batch_rows,
                dry_run=args.dry_run,
                max_records=args.max_records,
                max_members=args.deezer_max_members,
                tmp_dir=args.tmp_dir,
                audit_dir=args.audit_dir,
            )
            continue

        print(f"dataset {dataset} is not implemented yet in normalize_to_stage.py; skipping")


if __name__ == "__main__":
    main()
