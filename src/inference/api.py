<<<<<<< Updated upstream

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
=======
"""
FastAPI Inference API
REST API for serving commit risk predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging

from src.inference.predictor import CommitPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Commit Risk Scoring API",
    description="Predict bug risk for Git commits and optimize CI/CD testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (load model once at startup)
predictor = None


# ==============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ==============================================================================

class CommitRequest(BaseModel):
    """Request schema for single commit prediction"""
    commit_hash: str = Field(..., description="Git commit hash")
    lines_added: int = Field(0, ge=0, description="Number of lines added")
    lines_deleted: int = Field(0, ge=0, description="Number of lines deleted")
    files_changed: int = Field(1, ge=1, description="Number of files changed")
    touches_core: int = Field(0, ge=0, le=1, description="Touches core module (0 or 1)")
    touches_tests: int = Field(0, ge=0, le=1, description="Touches test files (0 or 1)")
    total_commits: int = Field(1, ge=1, description="Developer's total commits")
    buggy_commits: int = Field(0, ge=0, description="Developer's buggy commits")
    recent_frequency: int = Field(0, ge=0, description="Recent commit frequency")
    timestamp: Optional[str] = Field(None, description="Commit timestamp (ISO format)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "commit_hash": "abc123def456",
                "lines_added": 120,
                "lines_deleted": 45,
                "files_changed": 5,
>>>>>>> Stashed changes
                "touches_core": 1,
                "touches_tests": 0,
                "total_commits": 100,
                "buggy_commits": 25,
                "recent_frequency": 10,
<<<<<<< Updated upstream
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
=======
                "timestamp": "2024-03-01T14:30:00"
            }
        }


class BatchCommitRequest(BaseModel):
    """Request schema for batch prediction"""
    commits: List[CommitRequest] = Field(..., description="List of commits to predict")


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    commit_hash: str
    risk_score: float = Field(..., description="Bug risk probability (0.0 to 1.0)")
    risk_label: int = Field(..., description="Binary prediction (0=clean, 1=buggy)")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    recommendation: str = Field(..., description="Testing recommendation")
    prediction_time: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    status: str
    model_type: str
    model_path: str
    num_features: int
    feature_names: List[str]


# ==============================================================================
# STARTUP/SHUTDOWN EVENTS
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    try:
        logger.info("Loading ML model...")
        predictor = CommitPredictor()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        predictor = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "model_type": predictor.get_model_info()["model_type"] if predictor else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": predictor.get_model_info()["model_type"],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_commit(commit: CommitRequest):
    """Predict bug risk for a single commit"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Convert Pydantic model to dict
        commit_data = commit.dict()
        
        # Make prediction
        result = predictor.predict_commit(commit_data)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log prediction for monitoring
        try:
            from src.monitoring.metrics_collector import MetricsCollector
            collector = MetricsCollector()
            collector.log_prediction(
                commit_hash=result['commit_hash'],
                risk_score=result['risk_score'],
                risk_level=result['risk_level'],
                features=commit_data,
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(batch: BatchCommitRequest):
    """Predict bug risk for multiple commits"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        commits_data = [commit.dict() for commit in batch.commits]
        results = predictor.predict_batch(commits_data)
        return results
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = predictor.get_model_info()
        return info
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# ==============================================================================
# RUN SERVER (for testing)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info("STARTING FASTAPI SERVER")
    logger.info("=" * 70)
    logger.info("API will be available at: http://localhost:8000")
    logger.info("Interactive docs at: http://localhost:8000/docs")
    logger.info("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
>>>>>>> Stashed changes
