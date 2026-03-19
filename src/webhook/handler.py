import logging
import time
from typing import Any, Dict

from src.webhook.payload_parser import parse_github_payload
from src.webhook.feature_builder import build_features
from src.webhook import db_writer

logger = logging.getLogger(__name__)

FALLBACK_DECISION = {
    "risk_score": 1.0,
    "risk_label": 1,
    "risk_level": "HIGH",
    "recommendation": "Run full test suite (45 min) - High bug risk detected",
}


def process_push_event(payload: Dict[str, Any], predictor) -> Dict[str, Any]:
    t0 = time.perf_counter()

    commit = parse_github_payload(payload)
    if commit is None:
        logger.warning("Could not parse payload — no action taken.")
        return {"status": "skipped", "reason": "unparseable_payload"}

    commit_dict = commit.to_dict()
    logger.info(
        f"Processing webhook: repo={commit.repo_name} "
        f"branch={commit.branch} commit={commit.commit_hash[:8]}"
    )

    try:
        features = build_features(commit)

        inference_input = {
            "commit_hash":      commit.commit_hash,
            "lines_added":      commit.lines_added,
            "lines_deleted":    commit.lines_deleted,
            "files_changed":    commit.files_changed,
            "touches_core":     commit.touches_core,
            "touches_tests":    commit.touches_tests,
            "total_commits":    1,
            "buggy_commits":    0,
            "recent_frequency": 0,
            "timestamp":        commit.timestamp,
        }

        prediction = predictor.predict_commit(inference_input)
        logger.info(
            f"Inference done: {commit.commit_hash[:8]} → "
            f"{prediction['risk_level']} ({prediction['risk_score']:.3f})"
        )

    except Exception as exc:
        logger.error(
            f"Inference failed for {commit.commit_hash[:8]}: {exc}. "
            f"Applying safe fallback: FULL TEST."
        )
        prediction = {**FALLBACK_DECISION, "commit_hash": commit.commit_hash}

    db_writer.write_prediction(commit_dict, prediction)

    try:
        from src.monitoring.metrics_collector import MetricsCollector
        elapsed_ms = (time.perf_counter() - t0) * 1000
        MetricsCollector().log_prediction(
            commit_hash=commit.commit_hash,
            risk_score=prediction["risk_score"],
            risk_level=prediction["risk_level"],
            features=commit_dict,
            response_time_ms=elapsed_ms,
        )
    except Exception:
        pass

    return {
        "status":       "processed",
        "commit_hash":  commit.commit_hash,
        "repo":         commit.repo_name,
        "branch":       commit.branch,
        "risk_score":   prediction["risk_score"],
        "risk_level":   prediction["risk_level"],
        "decision":     prediction["recommendation"],
        "latency_ms":   round((time.perf_counter() - t0) * 1000, 2),
    }
