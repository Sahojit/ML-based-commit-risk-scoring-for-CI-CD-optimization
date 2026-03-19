import logging
from typing import Dict, Any

import pandas as pd

from src.webhook.payload_parser import ParsedCommit

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "lines_added", "lines_deleted", "files_changed", "total_churn",
    "churn_ratio", "touches_core", "touches_tests", "complexity_score",
    "total_commits", "buggy_commits", "bug_rate", "recent_frequency",
    "avg_lines_added", "avg_lines_deleted", "avg_files_changed",
    "hour_of_day", "day_of_week", "is_weekend", "month",
]


def build_features(commit: ParsedCommit, author_history: Dict[str, Any] = None) -> pd.DataFrame:
    if author_history is None:
        author_history = {}

    lines_added = commit.lines_added
    lines_deleted = commit.lines_deleted
    files_changed = max(commit.files_changed, 1)
    total_churn = lines_added + lines_deleted
    churn_ratio = lines_deleted / lines_added if lines_added > 0 else 0.0

    complexity_score = min(total_churn / 500.0, 1.0)

    total_commits = author_history.get("total_commits", 1)
    buggy_commits = author_history.get("buggy_commits", 0)
    bug_rate = buggy_commits / total_commits if total_commits > 0 else 0.0
    recent_frequency = author_history.get("recent_frequency", 0)
    avg_lines_added = author_history.get("avg_lines_added", float(lines_added))
    avg_lines_deleted = author_history.get("avg_lines_deleted", float(lines_deleted))
    avg_files_changed = author_history.get("avg_files_changed", float(files_changed))

    ts = commit.timestamp
    hour_of_day = ts.hour
    day_of_week = ts.weekday()
    is_weekend = int(day_of_week >= 5)
    month = ts.month

    features = pd.DataFrame([{
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "files_changed": files_changed,
        "total_churn": total_churn,
        "churn_ratio": churn_ratio,
        "touches_core": commit.touches_core,
        "touches_tests": commit.touches_tests,
        "complexity_score": complexity_score,
        "total_commits": total_commits,
        "buggy_commits": buggy_commits,
        "bug_rate": bug_rate,
        "recent_frequency": recent_frequency,
        "avg_lines_added": avg_lines_added,
        "avg_lines_deleted": avg_lines_deleted,
        "avg_files_changed": avg_files_changed,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": month,
    }])

    features = features[FEATURE_COLUMNS]

    logger.debug(f"Built feature vector for {commit.commit_hash[:8]}: {features.to_dict(orient='records')[0]}")
    return features
