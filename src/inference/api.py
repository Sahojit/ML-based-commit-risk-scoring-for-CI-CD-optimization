
import hashlib
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CommitRequest(BaseModel):
    commit_hash: str = Field(..., description="Unique commit identifier")
    lines_added: int = Field(0, ge=0)
    lines_deleted: int = Field(0, ge=0)
    files_changed: int = Field(1, ge=1)
    touches_core: int = Field(0, ge=0, le=1)
    touches_tests: int = Field(0, ge=0, le=1)
    total_commits: int = Field(1, ge=1)
    buggy_commits: int = Field(0, ge=0)
    recent_frequency: int = Field(0, ge=0)
    complexity_score: Optional[float] = None
    avg_lines_added: Optional[float] = None
    avg_lines_deleted: Optional[float] = None
    avg_files_changed: Optional[float] = None
    timestamp: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "commit_hash": "abc123",
                "lines_added": 150,
                "lines_deleted": 50,
                "files_changed": 8,
                "touches_core": 1,
                "touches_tests": 0,
                "total_commits": 100,
                "buggy_commits": 25,
                "recent_frequency": 10,
                "timestamp": "2024-03-01 14:30:00",
            }
        }
    }

class PredictionResponse(BaseModel):
    commit_hash: str
    risk_score: float
    risk_label: int
    risk_level: str
    recommendation: str
    prediction_time: str

class BatchRequest(BaseModel):
    commits: List[CommitRequest]

class BatchResponse(BaseModel):
    results: List[dict]
    total: int
    processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    version: str = "1.0.0"

_state: dict = {"predictor": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.inference.predictor import CommitPredictor

    model_path = os.environ.get("MODEL_PATH", "models/advanced_xgboost.pkl")
    logger.info(f"Loading model from: {model_path}")

    try:
        _state["predictor"] = CommitPredictor(model_path=model_path)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")

    yield

    _state["predictor"] = None
    logger.info("Shutdown complete.")

app = FastAPI(
    title="ML Commit Risk Scoring API",
    description=(
        "Predicts bug risk for Git commits to optimise CI/CD test selection.\n\n"
        "**Risk levels**\n"
        "- `HIGH` (>= 0.7) -> run full test suite (45 min)\n"
        "- `MEDIUM` (0.4-0.7) -> run extended tests (15 min)\n"
        "- `LOW` (< 0.4) -> run smoke tests (5 min)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    predictor = _state["predictor"]
    model_loaded = predictor is not None
    model_type: Optional[str] = None

    if model_loaded:
        try:
            model_type = predictor.get_model_info().get("model_type")
        except Exception:
            pass

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=model_type,
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: CommitRequest):
    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    try:
        result = predictor.predict_commit(request.model_dump())
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        from src.monitoring.metrics_collector import MetricsCollector

        MetricsCollector().log_prediction(
            commit_hash=result["commit_hash"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            features=request.model_dump(),
            response_time_ms=elapsed_ms,
        )
    except Exception:
        pass

    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(request.commits) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 commits.")

    try:
        results = predictor.predict_batch([c.model_dump() for c in request.commits])
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    processed = sum(1 for r in results if "error" not in r)
    return BatchResponse(results=results, total=len(results), processed=processed)

@app.get("/model/info", tags=["Model"])
def model_info():
    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        return predictor.get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _verify_github_signature(body: bytes, signature_header: Optional[str]) -> bool:
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping signature check.")
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        key=secret.encode(), msg=body, digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header)


@app.post("/webhook/github", tags=["Webhook"])
async def github_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not _verify_github_signature(body, signature):
        logger.warning("Webhook rejected: invalid signature.")
        raise HTTPException(status_code=401, detail="Invalid signature.")

    event_type = request.headers.get("X-GitHub-Event", "unknown")
    if event_type != "push":
        logger.info(f"Ignoring non-push event: {event_type}")
        return {"status": "ignored", "event": event_type}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    predictor = _state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from src.webhook.handler import process_push_event
    from src.webhook import db_writer

    db_writer.ensure_predictions_table()
    result = process_push_event(payload, predictor)

    logger.info(f"Webhook processed: {result}")
    return result
