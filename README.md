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

## Profile user diversity before ingest

Quick sample scan (fast lower-bound estimate):

```powershell
python scripts/profile_user_diversity.py --source lastfm_1k --sample-files 30
```

Exact distinct users (scans all stage files):

```powershell
python scripts/profile_user_diversity.py --source lastfm_1k --all-files
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

Produce a randomized bounded sample (better user diversity):

```powershell
python scripts/produce_listen_events.py --source lastfm_1k --topic listen_events_raw --sample-strategy random --randomize-rows --seed 42 --max-records 50000
```

Produce ~300k events with user diversity guardrail (target >=100 users):

```powershell
python scripts/produce_listen_events.py --source lastfm_1k --topic listen_events_raw --sample-strategy random --randomize-rows --seed 42 --max-records 300000 --max-records-per-user 3000
```

## Consume Kafka events into Postgres

Set a Postgres DSN:

```powershell
$env:POSTGRES_DSN="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
```

Dry run (consume and validate payloads, no DB write):

```powershell
python scripts/consume_listen_events_to_postgres.py --topic listen_events_raw --dry-run --max-messages 10000
```

Write to Postgres:

```powershell
python scripts/consume_listen_events_to_postgres.py --topic listen_events_raw
```

Bounded run (for testing):

```powershell
python scripts/consume_listen_events_to_postgres.py --topic listen_events_raw --max-messages 50000
```

## Recommendation SQL layer

Apply latest schema/views:

```powershell
python -c "import pathlib; from psycopg import connect; sql=pathlib.Path('sql/postgres_schema.sql').read_text(); conn=connect(); cur=conn.cursor(); cur.execute(sql); conn.commit(); conn.close(); print('schema updated')"
```

Get top personalized recommendations for one user (replace `USER_ID_HERE`):

```sql
SELECT
    user_id,
    recommendation_rank,
    track_name,
    artist_name,
    recommendation_score
FROM music.v_user_recommendations_30d
WHERE user_id = 'USER_ID_HERE'
  AND recommendation_rank <= 20
ORDER BY recommendation_rank;
```

Global fallback tracks (trending in last 7 days):

```sql
SELECT
    global_rank_7d,
    track_name,
    artist_name,
    plays_7d,
    unique_listeners_7d
FROM music.v_global_trending_tracks_7d
WHERE global_rank_7d <= 20
ORDER BY global_rank_7d;
```

Use the helper script instead of long one-liners:

```powershell
python scripts/query_recommendations.py --query top-users
python scripts/query_recommendations.py --query recs --user-id user_000002
python scripts/query_recommendations.py --query trending --limit 10
```

Or use ready SQL snippets in:

`queries/recommendations.sql`

## Data targets and progress check

Current project goals:

- `>= 500000` listen events
- `>= 50000` distinct tracks
- `>= 1000` distinct users

Check progress against targets:

```powershell
python scripts/check_data_targets.py
```

If you want custom targets for a smaller milestone:

```powershell
python scripts/check_data_targets.py --target-events 300000 --target-tracks 20000 --target-users 100
```

Force overwrite existing objects:

```powershell
python scripts/stream_to_s3.py --dataset lastfm_1k --force
```
