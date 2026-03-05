# Spotify Realtime Music Assistant

Personalized music recommendation system with:
- user-based recommendations (`/recs`)
- favorites / most-listened retrieval (`/favorites`)
- vibe-based discovery (`/vibe`)
- interaction feedback loop for retraining (`/feedback/interaction`)
- non-technical chatbot demo UI (`/demo`)

## What This Project Delivers

- End-to-end recommendation workflow from data to serving
- Hybrid recommendation strategy (MF + popularity + artist signals)
- FastAPI serving layer with model/source observability
- Live demo chat interface for non-technical stakeholders
- Optional LLM assistants (OpenAI / Bedrock / LangChain Bedrock)

## Tech Stack

- Python 3.11
- PostgreSQL (RDS-compatible)
- FastAPI + Uvicorn
- psycopg
- pandas / numpy
- Optional: OpenAI, AWS Bedrock, LangChain

## Quick Start (Local)

1. Create environment and install dependencies

```powershell
cd d:\Spotify
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Load DB environment variables (example)

```powershell
. .\.local\secrets.ps1
```

3. Ensure schema/tables/views are up to date

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

4. Train recommendation tables (if needed)

```powershell
python scripts/train_personalized_mf.py --factors 128 --epochs 15 --lr 0.01 --reg 0.001 --neg-ratio 0 --top-k 100
python scripts/train_hybrid_recs.py --top-k 100 --mf-scan 500 --candidate-top-n 10000 --weight-mf 0.45 --weight-pop 0.45 --weight-artist 0.10
```

5. Run API

```powershell
python scripts/recommendation_api.py
```

- API docs: `http://127.0.0.1:8000/docs`
- Demo UI: `http://127.0.0.1:8000/demo`

## Core Endpoints

- `GET /metrics/model`
- `GET /metrics/recsource`
- `GET /recs/{user_id}?limit=20`
- `GET /favorites/{user_id}?limit=20&fallback_to_recs=false`
- `GET /vibe?vibe=sad&user_id=<user>&limit=10`
- `GET /search/tracks?query=<text>&limit=10`
- `POST /feedback/interaction`
- `POST /feedback/vibe`

## Demo Chat Behavior

The `/demo` page supports natural prompts such as:
- `Recommend songs for Aarav Edwards`
- `party songs for Abigail Johnson`
- `favorites of Ariana Reed`
- `how many users`
- `like 1`

Follow-up prompts like `5 more` continue from the previous result set.

## Evaluation

Run offline recommendation evaluation:

```powershell
python scripts/evaluate_recommendation_quality.py --source both --k 20 --holdout-size 5 --min-user-events 20
```

Outputs:
- `artifacts/evaluation/recommendation_quality.json`
- `artifacts/evaluation/recommendation_quality.md`

## Deployment Notes (AWS App Runner)

- Runtime: Python 3.11
- Port: `8000`
- Start command:

```bash
python3 -m uvicorn recommendation_api:app --host 0.0.0.0 --port 8000 --app-dir scripts
```

- Required env vars include:
  - `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGSSLMODE=require`
  - `USE_HYBRID_RECS=true`, `USE_ML_RECS=true`, `USE_DENSE_RECS=false`
- If DB is in VPC/private routing, configure App Runner VPC connector + SG rules for Postgres `5432`.

## Live Demo App

https://uq3i5irvfr.us-east-2.awsapprunner.com/demo
