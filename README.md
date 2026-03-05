# Spotify Realtime Music Assistant

An end-to-end music recommendation system with real personalization, vibe-aware discovery, and a live chat-style demo UI.

## Overview

This project serves recommendations from PostgreSQL-backed models through a FastAPI service and exposes a non-technical chatbot demo (`/demo`) for live walkthroughs.

It supports:
- Personalized recommendations (`/recs/{user}`)
- User favorites / top listened tracks (`/favorites/{user}`)
- Vibe-based tracks (`/vibe`)
- Feedback capture for continuous retraining (`/feedback/interaction`, `/feedback/vibe`)

## Architecture

1. Data ingestion and normalization scripts prepare listening + catalog data.
2. PostgreSQL stores events, profiles, model outputs, and feedback.
3. Training scripts build MF and hybrid recommendation tables.
4. FastAPI serves recommendation endpoints.
5. `/demo` provides a chat UI that calls the same production APIs.

## Tech Stack

- Python 3.11
- PostgreSQL (RDS-compatible)
- FastAPI + Uvicorn
- psycopg
- pandas / numpy
- Optional assistants: OpenAI / Bedrock / LangChain

## Repository Map

- `scripts/recommendation_api.py`: API + demo UI (`/demo`)
- `scripts/train_personalized_mf.py`: matrix factorization training
- `scripts/train_hybrid_recs.py`: hybrid recommendation generation
- `scripts/evaluate_recommendation_quality.py`: offline evaluation
- `sql/postgres_schema.sql`: schema, views, and serving tables
- `demo_prompts_and_user_names.txt`: sample prompts + demo user list

## Local Setup

1. Create environment and install dependencies

```powershell
cd d:\Spotify
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Load environment variables (DB + feature flags)

```powershell
. .\.local\secrets.ps1
```

3. Apply schema

```powershell
@'
import pathlib
from psycopg import connect
sql = pathlib.Path("sql/postgres_schema.sql").read_text(encoding="utf-8")
with connect() as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
print("schema ready")
'@ | .\.venv\Scripts\python.exe
```

4. Train recommendation outputs (if needed)

```powershell
python scripts/train_personalized_mf.py --factors 128 --epochs 15 --lr 0.01 --reg 0.001 --neg-ratio 0 --top-k 100
python scripts/train_hybrid_recs.py --top-k 100 --mf-scan 500 --candidate-top-n 10000 --weight-mf 0.45 --weight-pop 0.45 --weight-artist 0.10
```

5. Start API

```powershell
python scripts/recommendation_api.py
```

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Demo UI: `http://127.0.0.1:8000/demo`

## Quick API Validation

```powershell
curl "http://127.0.0.1:8000/metrics/model"
curl "http://127.0.0.1:8000/recs/Aarav%20Edwards?limit=5"
curl "http://127.0.0.1:8000/favorites/Abigail%20Johnson?limit=5&fallback_to_recs=false"
curl "http://127.0.0.1:8000/vibe?vibe=romantic&user_id=Aarav%20Edwards&limit=5"
```

## Demo Prompt Examples

Use these in `/demo`:
- `Recommend songs for Aarav Edwards`
- `party songs for Abigail Johnson`
- `romantic songs for Camila Lopez`
- `favorites of Ariana Reed`
- `most listened tracks of Abigail Johnson`
- `how many users`
- `how many tracks are there in total`
- `like 1`

Full prompt/user list is in `demo_prompts_and_user_names.txt`.

## Evaluation

Run offline quality evaluation:

```powershell
python scripts/evaluate_recommendation_quality.py --source both --k 20 --holdout-size 5 --min-user-events 20
```

Outputs:
- `artifacts/evaluation/recommendation_quality.json`
- `artifacts/evaluation/recommendation_quality.md`

## Deployment (AWS App Runner)

- Runtime: Python 3.11
- Port: `8000`
- Start command:

```bash
python3 -m uvicorn recommendation_api:app --host 0.0.0.0 --port 8000 --app-dir scripts
```

Required environment variables:
- `PGHOST`
- `PGPORT`
- `PGDATABASE`
- `PGUSER`
- `PGPASSWORD`
- `PGSSLMODE=require`
- `USE_HYBRID_RECS=true`
- `USE_ML_RECS=true`
- `USE_DENSE_RECS=false`

If your DB is VPC-routed, App Runner needs:
- a VPC connector in the DB VPC/subnets
- DB SG inbound `5432` from the App Runner connector SG

## Troubleshooting

- `Postgres connection is required...`:
  - missing DB env vars in runtime.
- `password authentication failed for user "postgres"`:
  - wrong `PGPASSWORD` value.
- `No module named uvicorn`:
  - dependencies not installed in runtime image.
- Request timeouts on App Runner:
  - usually VPC connector / SG / route mismatch.
- `WinError 10048` locally:
  - port 8000 already in use; stop old process or change port.

## Live Demo App

https://uq3i5irvfr.us-east-2.awsapprunner.com/demo
