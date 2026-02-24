# Spotify Assistant Data Intake (Cloud-First)

This repo is set up to ingest datasets without storing raw files on your laptop.

## What this does

- Streams dataset files from public URLs.
- Uploads them directly to S3-compatible object storage (AWS S3, Cloudflare R2, MinIO).
- Skips local raw file persistence.

## Datasets configured

- `lastfm_1k`: listening events + users + tags
- `fma_metadata`: track/genre metadata
- `deezer_recsys25_archive`: large-scale session data archive (RecSys25)

Dataset URLs are defined in `configs/datasets.yaml`.

## Setup

```powershell
cd d:\Spotify
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set storage credentials:

```powershell
$env:AWS_ACCESS_KEY_ID = "YOUR_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET"
$env:AWS_REGION = "us-east-1"
$env:DATA_BUCKET = "your-bucket-name"
```

If using Cloudflare R2 / MinIO, also set endpoint:

```powershell
$env:S3_ENDPOINT_URL = "https://<account>.r2.cloudflarestorage.com"
```

Optional prefix in bucket:

```powershell
$env:S3_PREFIX = "music-assistant"
```

## Usage

List datasets:

```powershell
python scripts/stream_to_s3.py --list
```

Dry run (no upload):

```powershell
python scripts/stream_to_s3.py --dataset lastfm_1k fma_metadata --dry-run
```

Upload selected datasets:

```powershell
python scripts/stream_to_s3.py --dataset lastfm_1k fma_metadata
```

Upload all datasets:

```powershell
python scripts/stream_to_s3.py --dataset all
```

Upload Deezer RecSys25 archive:

```powershell
python scripts/stream_to_s3.py --dataset deezer_recsys25_archive
```

## Normalize to SQL-friendly stage files

This step converts raw files into staged parquet tables that are easier to query and load into Postgres.

Install dependencies (if not already installed):

```powershell
pip install -r requirements.txt
```

Dry run:

```powershell
python scripts/normalize_to_stage.py --dataset all --dry-run
```

Write stage parquet:

```powershell
python scripts/normalize_to_stage.py --dataset all
```

Outputs:

- `s3://<bucket>/<prefix>/stage/listen_events/source=lastfm_1k/run_date=YYYY-MM-DD/*.parquet`
- `s3://<bucket>/<prefix>/stage/track_catalog/source=fma_metadata/run_date=YYYY-MM-DD/*.parquet`

## Produce listen events to Kafka

Set your runtime variables:

```powershell
$env:AWS_PROFILE="spotify"
$env:AWS_REGION="us-east-1"
$env:DATA_BUCKET="spotify-project-hari-2026"
$env:S3_PREFIX="music-assistant"
$env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
```

Dry run (reads staged parquet and prints a few sample events):

```powershell
python scripts/produce_listen_events.py --source lastfm_1k --dry-run --max-files 1 --max-records 10
```

Produce to Kafka topic `listen_events_raw`:

```powershell
python scripts/produce_listen_events.py --source lastfm_1k --topic listen_events_raw
```

Produce only a bounded sample:

```powershell
python scripts/produce_listen_events.py --source lastfm_1k --topic listen_events_raw --max-records 50000
```

Force overwrite existing objects:

```powershell
python scripts/stream_to_s3.py --dataset lastfm_1k --force
```
