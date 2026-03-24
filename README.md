# CommitGuard — ML-Based Commit Risk Scoring for CI/CD

An end-to-end machine learning system that scores every Git commit by bug risk in real time, enabling CI/CD pipelines to dynamically select the right test strategy — cutting unnecessary test time while catching high-risk changes.

---

## How It Works

```
git push → GitHub Webhook → FastAPI → Feature Extraction → XGBoost Model → Decision Engine → PostgreSQL → Dashboard
```

Every commit pushed to GitHub is automatically:
1. Received via webhook
2. Scored by the ML model (0.0 → 1.0 risk probability)
3. Mapped to a CI test strategy
4. Stored and visualized on the dashboard

---

## Risk Levels & Decisions

| Risk Score | Level | CI Strategy | Time |
|---|---|---|---|
| ≥ 0.70 | HIGH | Full test suite | 45 min |
| 0.40 – 0.69 | MEDIUM | Extended tests | 15 min |
| < 0.40 | LOW | Smoke tests | 5 min |

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 75% | 76.5% | 68.4% | 72.2% | 0.777 |
| **XGBoost (prod)** | **80%** | **82.4%** | **73.7%** | **77.8%** | **0.805** |

**22 features** across 4 categories:
- Commit metrics (lines added/deleted, files changed, churn ratio)
- Code scope (touches core modules, touches test files)
- Developer history (total commits, bug rate, recent frequency)
- Temporal (hour of day, day of week, weekend flag)

---

## Project Structure

```
├── src/
│   ├── ingestion/             # GitHub API commit extraction
│   ├── features/              # Feature engineering pipeline
│   ├── training/              # Model training & evaluation
│   ├── inference/             # FastAPI + model loader + predictor
│   ├── webhook/               # Real-time webhook handler, parser, DB writer
│   └── monitoring/            # Metrics collector
├── models/
│   └── advanced_xgboost.pkl   # Production model
├── dashboard.py               # Streamlit monitoring dashboard
├── config/                    # YAML configs
├── scripts/                   # CLI runners for each pipeline stage
├── Dockerfile                 # API container
├── Dockerfile.dashboard       # Dashboard container
├── docker-compose.yml         # Local multi-container setup
└── render.yaml                # Render deployment config
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the API
```bash
python src/inference/run_api.py
```
API at `http://localhost:8002` — Docs at `http://localhost:8002/docs`

### 3. Run the dashboard
```bash
streamlit run dashboard.py
```
Dashboard at `http://localhost:8501`

### 4. Run with Docker
```bash
docker compose up -d
```
- API → `http://localhost:8001`
- Dashboard → `http://localhost:8502`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + model status |
| POST | `/predict` | Score a single commit |
| POST | `/predict/batch` | Score multiple commits |
| GET | `/model/info` | Model metadata + feature list |
| POST | `/webhook/github` | GitHub push event receiver |

### Example — Score a commit
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "commit_hash": "abc123",
    "lines_added": 250,
    "lines_deleted": 80,
    "files_changed": 10,
    "touches_core": 1,
    "touches_tests": 0,
    "total_commits": 60,
    "buggy_commits": 20,
    "recent_frequency": 8
  }'
```

Response:
```json
{
  "commit_hash": "abc123",
  "risk_score": 0.743,
  "risk_level": "HIGH",
  "recommendation": "Run full test suite (45 min) - High bug risk detected"
}
```

---

## GitHub Webhook Setup

1. Go to your repo → **Settings → Webhooks → Add webhook**
2. Payload URL: `https://your-api-url/webhook/github`
3. Content type: `application/json`
4. Events: **Just the push event**
5. Set `GITHUB_WEBHOOK_SECRET` env var on your server

For local development, expose your API with ngrok:
```bash
ngrok http 8002
```

---

## Run the ML Pipeline

```bash
python scripts/run_ingestion.py               # Fetch commits from GitHub
python scripts/run_feature_engineering.py     # Build features
python scripts/run_labeling.py                # Generate bug labels
python scripts/run_training.py                # Train & evaluate model
```

---

## Deployment

### Render (Dashboard)
Push to GitHub — Render auto-deploys via `render.yaml`.

### Docker Hub
```bash
docker build -t sahojit/ml-commit-risk-api .
docker push sahojit/ml-commit-risk-api
```

---

## Tech Stack

- **ML:** XGBoost, scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn, pydantic
- **Dashboard:** Streamlit, plotly
- **Database:** PostgreSQL, SQLAlchemy
- **Deployment:** Docker, Render, ngrok
- **Webhooks:** GitHub Events API
