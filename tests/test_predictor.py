import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from src.inference.predictor import CommitPredictor


@pytest.fixture
def mock_predictor():
    with patch("src.inference.predictor.ModelLoader") as MockLoader:
        instance = MockLoader.return_value
        instance.predict_proba.return_value = 0.75
        instance.predict.return_value = 1
        instance.get_model_info.return_value = {
            "model_type": "XGBoost",
            "num_features": 19,
            "accuracy": 0.80
        }
        predictor = CommitPredictor.__new__(CommitPredictor)
        predictor.model_loader = instance
        yield predictor


def base_commit(**kwargs):
    data = {
        "commit_hash": "abc123",
        "lines_added": 100,
        "lines_deleted": 40,
        "files_changed": 5,
        "touches_core": 1,
        "touches_tests": 0,
        "total_commits": 80,
        "buggy_commits": 20,
        "recent_frequency": 6,
    }
    data.update(kwargs)
    return data


class TestRiskLevel:
    def test_high_risk(self, mock_predictor):
        mock_predictor.model_loader.predict_proba.return_value = 0.75
        assert mock_predictor._get_risk_level(0.75) == "HIGH"

    def test_medium_risk(self, mock_predictor):
        assert mock_predictor._get_risk_level(0.55) == "MEDIUM"

    def test_low_risk(self, mock_predictor):
        assert mock_predictor._get_risk_level(0.2) == "LOW"

    def test_boundary_high(self, mock_predictor):
        assert mock_predictor._get_risk_level(0.70) == "HIGH"

    def test_boundary_medium(self, mock_predictor):
        assert mock_predictor._get_risk_level(0.40) == "MEDIUM"


class TestFeatureExtraction:
    def test_churn_ratio(self, mock_predictor):
        features = mock_predictor._extract_features(base_commit(lines_added=100, lines_deleted=50))
        assert features["churn_ratio"].iloc[0] == pytest.approx(0.5)

    def test_churn_ratio_zero_division(self, mock_predictor):
        features = mock_predictor._extract_features(base_commit(lines_added=0, lines_deleted=10))
        assert features["churn_ratio"].iloc[0] == 0.0

    def test_bug_rate(self, mock_predictor):
        features = mock_predictor._extract_features(base_commit(total_commits=100, buggy_commits=25))
        assert features["bug_rate"].iloc[0] == pytest.approx(0.25)

    def test_none_optional_fields_cast_to_float(self, mock_predictor):
        commit = base_commit()
        commit["complexity_score"] = None
        commit["avg_lines_added"] = None
        features = mock_predictor._extract_features(commit)
        assert isinstance(features["complexity_score"].iloc[0], float)
        assert isinstance(features["avg_lines_added"].iloc[0], float)

    def test_weekend_flag(self, mock_predictor):
        features = mock_predictor._extract_features(
            base_commit(timestamp="2024-03-02 10:00:00")
        )
        assert features["is_weekend"].iloc[0] == 1

    def test_weekday_flag(self, mock_predictor):
        features = mock_predictor._extract_features(
            base_commit(timestamp="2024-03-04 10:00:00")
        )
        assert features["is_weekend"].iloc[0] == 0


class TestPredictCommit:
    def test_returns_required_keys(self, mock_predictor):
        result = mock_predictor.predict_commit(base_commit())
        for key in ["commit_hash", "risk_score", "risk_level", "recommendation", "prediction_time"]:
            assert key in result

    def test_risk_score_is_float(self, mock_predictor):
        result = mock_predictor.predict_commit(base_commit())
        assert isinstance(result["risk_score"], float)

    def test_commit_hash_passed_through(self, mock_predictor):
        result = mock_predictor.predict_commit(base_commit(commit_hash="xyz999"))
        assert result["commit_hash"] == "xyz999"

    def test_recommendation_matches_risk_level(self, mock_predictor):
        mock_predictor.model_loader.predict_proba.return_value = 0.8
        result = mock_predictor.predict_commit(base_commit())
        assert "full test suite" in result["recommendation"].lower()


class TestPredictBatch:
    def test_batch_returns_all_results(self, mock_predictor):
        commits = [base_commit(commit_hash=f"hash{i}") for i in range(3)]
        results = mock_predictor.predict_batch(commits)
        assert len(results) == 3

    def test_batch_error_handled_gracefully(self, mock_predictor):
        mock_predictor.model_loader.predict_proba.side_effect = Exception("model error")
        results = mock_predictor.predict_batch([base_commit()])
        assert "error" in results[0]
