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

## Run From VS Code (No Manual Terminal Commands)

Use Run and Debug with:

- `Normalize Deezer (Limited)`
- `Normalize Deezer (Scale)`
- `Produce Deezer -> Kafka (300k)`
- `Consume Kafka -> Postgres (300k)`
- `Check Data Targets (500k/1k/50k)`
- `Inspect Ingested Data (Export CSV/JSON)`

These are preconfigured in `.vscode/launch.json`.

Inspection outputs are written to:

- `artifacts/inspection/summary.json`
- `artifacts/inspection/source_breakdown.csv`
- `artifacts/inspection/top_users.csv`
- `artifacts/inspection/catalog_genres.csv`
- `artifacts/inspection/event_sample.csv`

Before running inspection from VS Code, create `.env` from `.env.example` and set `POSTGRES_DSN`.

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

Normalize Deezer with safe limits (recommended first run):

```powershell
$env:NORMALIZE_TMP_DIR="D:\temp"
python scripts/normalize_to_stage.py --dataset deezer_recsys25_archive --max-records 800000 --deezer-max-members 20
```

This run also writes a column-mapping audit JSON to:

- `artifacts/deezer_audit/`
- Audit is written even if Deezer normalization fails, so you can inspect detected columns.

Outputs:

- `s3://<bucket>/<prefix>/stage/listen_events/source=lastfm_1k/run_date=YYYY-MM-DD/*.parquet`
- `s3://<bucket>/<prefix>/stage/listen_events/source=deezer_recsys25_archive/run_date=YYYY-MM-DD/*.parquet`
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

Produce Deezer staged events to the same topic:

```powershell
python scripts/produce_listen_events.py --source deezer_recsys25_archive --topic listen_events_raw --max-records 300000 --sample-strategy random --randomize-rows --seed 42 --max-records-per-user 500
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
python scripts/query_recommendations.py --query model-metrics
python scripts/query_recommendations.py --query top-users-model --limit 20
python scripts/query_recommendations.py --query recs-model --user-id user_000002 --limit 20
```

Dense personalization views (auto-derived from existing data, no re-ingestion):

- `music.v_model_users_gt20` -> strict dense users (`>20` plays)
- `music.v_model_users_1000` -> balanced slice: top 1000 users by plays
- `music.v_listen_events_model_1000` -> events for only those users
- `music.v_model_metrics_1000` -> events/users/tracks/events_per_user for the dense slice
- `music.v_user_recommendations_30d_dense_1000` -> personalized recommendations only for dense slice users

Or use ready SQL snippets in:

`queries/recommendations.sql`

## Recommendation-ready ML workflow

The project now supports:

- recommendation-ready filtering (`music.v_listen_events_recommendation_ready`)
- human-friendly user names (`music.user_profiles`)
- ML recommendations via matrix factorization (`music.v_user_recommendations_mf_ready`)
- hybrid recommendations for serving (`music.v_user_recommendations_hybrid_ready`)

Repair metadata text artifacts (recommended before retraining):

```powershell
python scripts/repair_metadata_text.py --dry-run
python scripts/repair_metadata_text.py
```

Summary output:

- `artifacts/cleanup/metadata_repair_summary.json`

Generate display names (writes `artifacts/user_profiles/user_names.txt`):

```powershell
python scripts/generate_user_profiles.py --max-users 1000
```

Train MF recommendations (writes to `music.user_recommendations_mf_ready`):

```powershell
python scripts/train_personalized_mf.py --epochs 8 --factors 48 --top-k 100
```

Train hybrid recommendations (writes to `music.user_recommendations_hybrid_ready`):

```powershell
python scripts/train_hybrid_recs.py --top-k 100 --mf-scan 500 --candidate-top-n 10000 --weight-mf 0.45 --weight-pop 0.45 --weight-artist 0.10
```

Evaluate recommendation quality (temporal holdout):

```powershell
python scripts/evaluate_recommendation_quality.py --source both --k 20 --holdout-size 5 --min-user-events 20
```

Evaluation outputs:

- `artifacts/evaluation/recommendation_quality.json`
- `artifacts/evaluation/recommendation_quality.md`

Metrics reported:

- `precision@k`
- `recall@k`
- `ndcg@k`
- `hit_rate@k`
- `item_coverage@k`
- `user_coverage`
- `personalization@k`
- `readable_rate@k`

## Recommendation API (serving layer)

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run API:

```powershell
python scripts/recommendation_api.py
```

Base URL: `http://localhost:8000`

Endpoints:

- `GET /metrics/model`
- `GET /metrics/recsource`
- `GET /trending?limit=20`
- `GET /recs/{user_id}?limit=20`
- `GET /favorites/{user_id}?limit=20`
- `GET /search/tracks?query=<song>&limit=10`
- `GET /vibe?vibe=<chill|focus|happy|sad|party|energetic|romantic>&limit=10`
- `POST /feedback/vibe`
- `POST /feedback/interaction`

Examples:

```powershell
curl "http://localhost:8000/metrics/model"
curl "http://localhost:8000/trending?limit=10"
curl "http://localhost:8000/recs/user_000002?limit=20"
curl "http://localhost:8000/favorites/user_000002?limit=20"
curl "http://localhost:8000/search/tracks?query=Morning%20Child&limit=5"
curl "http://localhost:8000/vibe?vibe=chill&limit=10"
curl -X POST "http://localhost:8000/feedback/vibe" -H "Content-Type: application/json" -d "{\"user_id\":\"101617\",\"track_id\":\"45659\",\"user_selected_vibe\":\"energetic\",\"predicted_vibe\":\"chill\"}"
curl -X POST "http://localhost:8000/feedback/interaction" -H "Content-Type: application/json" -d "{\"user_id\":\"101617\",\"track_id\":\"45659\",\"action\":\"like\",\"source_endpoint\":\"/recs\"}"
```

Python client helper:

```powershell
python scripts/recommendation_client.py --query metrics
python scripts/recommendation_client.py --query trending --limit 10
python scripts/recommendation_client.py --query recs --user-id 101617 --limit 20 --fallback-to-trending
python scripts/recommendation_client.py --query favorites --user-id 101617 --limit 20
python scripts/recommendation_client.py --query search --text "Morning Child" --limit 5
python scripts/recommendation_client.py --query vibe --text chill --limit 10
python scripts/recommendation_client.py --query feedback --user-id 101617 --track-id 45659 --vibe energetic --predicted-vibe chill
python scripts/recommendation_client.py --query interaction --user-id 101617 --track-id 45659 --action like --source-endpoint /recs --rank 1
```

Optional base URL override:

```powershell
$env:RECOMMENDATION_API_BASE_URL="http://localhost:8000"
```

### Demo feedback loop for retraining

Every recommendation response can now be captured as interaction data for later retraining.

1. Apply latest schema so `music.demo_interactions` exists:

```powershell
python -c "import pathlib; from psycopg import connect; sql=pathlib.Path('sql/postgres_schema.sql').read_text(); conn=connect(); cur=conn.cursor(); cur.execute(sql); conn.commit(); conn.close(); print('schema updated')"
```

2. API auto-logs `impression` rows for:
- `/recs/{user_id}`
- `/favorites/{user_id}`
- `/trending` (if `user_id` query param is provided)
- `/vibe` (if `user_id` query param is provided)

3. Explicit user actions can be posted via:
- `POST /feedback/interaction` (`play`, `like`, `favorite`, `add_to_playlist`, `skip`, `dislike`, etc.)

4. MF training can blend these demo interactions:

```powershell
python scripts/train_personalized_mf.py --include-demo-feedback --demo-feedback-boost 4.0
```

Disable blending if needed:

```powershell
python scripts/train_personalized_mf.py --no-demo-feedback
```

### Vibe Feature Engineering

Build/update vibe labels from `track_catalog.genre` plus 30-day popularity signals:

```powershell
python scripts/build_track_vibe_features.py --include-unknown
```

This populates `music.track_vibe_features` used by `/vibe` endpoint.

### Feedback-driven vibe override

`POST /feedback/vibe` stores user corrections and can override a track vibe only when there is consensus.

Default consensus thresholds:
- minimum unique users per track: `15`
- top vibe share: `>= 70%`
- margin over second vibe: `>= 15%`

If thresholds are met, an override is written to `music.track_vibe_overrides`.

Effective labels used by `/vibe` are read from:
- `music.v_track_vibe_effective` (override first, engineered label second)

### Interactive Chat Assistant

Run:

```powershell
python scripts/chat_assistant.py
```

Behavior:
- If user asks recommendations with `user_id`, assistant calls personalized `/recs/{user_id}`.
- If a requested song is not found, assistant asks for vibe and returns closest vibe-based songs.
- If there is no user context, assistant falls back to trending.

### LLM Music Assistant (Tool-Calling)

Run the API first:

```powershell
python scripts/recommendation_api.py
```

Set LLM env vars:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Run LLM assistant:

```powershell
python scripts/llm_music_assistant.py
```

What it does:
- Uses OpenAI chat with tool-calling.
- Calls your real endpoints (`/recs`, `/trending`, `/search/tracks`, `/vibe`, `/feedback/vibe`).
- Returns grounded answers from your project data instead of made-up songs.

### Bedrock Music Assistant (Tool-Calling)

If you prefer AWS credits over OpenAI billing, use Bedrock assistant.

Prerequisite:
- Enable model access in Amazon Bedrock for your account/region.

Set env vars:

```powershell
$env:AWS_PROFILE="spotify"
$env:AWS_REGION="us-east-1"
$env:BEDROCK_REGION="us-east-1"
$env:BEDROCK_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"
```

Run:

```powershell
python scripts/bedrock_music_assistant.py
```

Notes:
- Keep `python scripts/recommendation_api.py` running in another terminal.
- If Bedrock returns access errors, grant model access in Bedrock console first.

### LangChain + Bedrock Assistant

If you want LangChain orchestration, run:

```powershell
python scripts/langchain_bedrock_assistant.py
```

It uses:
- `langchain`
- `langchain-aws`
- Bedrock chat model + tool calls mapped to your live API-backed functions.

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
