"""
tests/test_pipeline.py
========================
Unit and integration tests for the AccessGuard MLOps pipeline.
Run with: pytest tests/ -v
"""

import json
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_LOGINS = {
    "Username":    ["alice", "alice", "bob",   "bob",   "carol", "carol",
                    "alice", "bob",   "carol",  "alice", "bob"],
    "IP":          ["1.1.1.1","1.1.1.1","2.2.2.2","3.3.3.3","4.4.4.4","4.4.4.4",
                    "1.1.1.1","3.3.3.3","5.5.5.5","9.9.9.9","2.2.2.2"],
    "Device":      ["Windows 10","Windows 10","macOS","macOS","Ubuntu","Ubuntu",
                    "Windows 10","macOS","Ubuntu","Windows 11","macOS"],
    "Browser":     ["Chrome","Chrome","Firefox","Firefox","Edge","Edge",
                    "Chrome","Firefox","Edge","Chrome","Safari"],
    "Timestamp":   pd.date_range("2025-01-01", periods=11, freq="h").astype(str).tolist(),
    "MFA Enabled": [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    "Outcome":     [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
}

SAMPLE_USERS = {
    "User ID":    [1000, 1001, 1002],
    "Username":   ["alice", "bob", "carol"],
    "Password":   ["hash1", "hash2", "hash3"],
    "MFA Enabled": [1, 0, 1],
    "IP":         ["1.1.1.1", "2.2.2.2", "4.4.4.4"],
    "Device":     ["Windows 10", "macOS", "Ubuntu"],
    "Browser":    ["Chrome", "Firefox", "Edge"],
    "Registration Timestamp": [
        "2025-01-01T00:00:00",
        "2025-01-02T00:00:00",
        "2025-01-03T00:00:00",
    ],
}


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with sample CSVs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(SAMPLE_LOGINS).to_csv(data_dir / "logins.csv", index=False)
    pd.DataFrame(SAMPLE_USERS).to_csv(data_dir / "users.csv", index=False)
    return tmp_path


# ── Tests: Data Loading & Validation ──────────────────────────────────────────

class TestDataLoading:
    def test_load_valid_csv(self, tmp_data_dir, monkeypatch):
        from mlops.pipeline import train_pipeline as tp
        monkeypatch.setattr(tp, "LOGINS_CSV", tmp_data_dir / "data" / "logins.csv")
        monkeypatch.setattr(tp, "MODELS_DIR", tmp_data_dir / "models")
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_data_dir / "mlops" / "monitoring")
        (tmp_data_dir / "models").mkdir(parents=True, exist_ok=True)
        (tmp_data_dir / "mlops" / "monitoring").mkdir(parents=True, exist_ok=True)

        df = tp.load_and_validate()
        assert len(df) == 11
        assert "Outcome" in df.columns

    def test_missing_csv_raises(self, tmp_path, monkeypatch):
        from mlops.pipeline import train_pipeline as tp
        monkeypatch.setattr(tp, "LOGINS_CSV", tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            tp.load_and_validate()

    def test_missing_column_raises(self, tmp_data_dir, monkeypatch):
        from mlops.pipeline import train_pipeline as tp
        # Write a CSV missing the Outcome column
        bad_csv = tmp_data_dir / "data" / "logins_bad.csv"
        df = pd.read_csv(tmp_data_dir / "data" / "logins.csv").drop(columns=["Outcome"])
        df.to_csv(bad_csv, index=False)
        monkeypatch.setattr(tp, "LOGINS_CSV", bad_csv)
        with pytest.raises(ValueError, match="Missing required columns"):
            tp.load_and_validate()

    def test_insufficient_data_raises(self, tmp_data_dir, monkeypatch):
        from mlops.pipeline import train_pipeline as tp
        tiny_csv = tmp_data_dir / "data" / "tiny.csv"
        pd.DataFrame({k: v[:2] for k, v in SAMPLE_LOGINS.items()}).to_csv(tiny_csv, index=False)
        monkeypatch.setattr(tp, "LOGINS_CSV", tiny_csv)
        with pytest.raises(ValueError, match="Insufficient training data"):
            tp.load_and_validate()


# ── Tests: Preprocessing ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_login_hour_created(self):
        from mlops.pipeline.train_pipeline import preprocess
        df = pd.DataFrame(SAMPLE_LOGINS)
        result = preprocess(df)
        assert "LoginHour" in result.columns
        assert result["LoginHour"].between(0, 23).all()

    def test_user_new_ip_created(self):
        from mlops.pipeline.train_pipeline import preprocess
        df = pd.DataFrame(SAMPLE_LOGINS)
        result = preprocess(df)
        assert "User_New_IP" in result.columns
        assert set(result["User_New_IP"].unique()).issubset({0, 1})

    def test_new_ip_flagged_correctly(self):
        from mlops.pipeline.train_pipeline import preprocess
        df = pd.DataFrame({
            "Username":    ["alice", "alice"],
            "IP":          ["1.1.1.1", "9.9.9.9"],    # second IP is new
            "Device":      ["Win", "Win"],
            "Browser":     ["Chrome", "Chrome"],
            "Timestamp":   ["2025-01-01T00:00:00", "2025-01-01T01:00:00"],
            "MFA Enabled": [1, 1],
            "Outcome":     [0, 1],
        })
        result = preprocess(df)
        result = result.sort_values("Timestamp").reset_index(drop=True)
        assert result.loc[0, "User_New_IP"] == 1   # first login always new
        assert result.loc[1, "User_New_IP"] == 1   # second IP is different

    def test_mfa_cleaned_to_int(self):
        from mlops.pipeline.train_pipeline import preprocess
        df = pd.DataFrame(SAMPLE_LOGINS)
        result = preprocess(df)
        assert result["MFA Enabled"].dtype in [int, np.int64, np.int32]


# ── Tests: Model Training ─────────────────────────────────────────────────────

class TestModelTraining:
    def test_train_returns_model_and_encoders(self):
        from mlops.pipeline.train_pipeline import preprocess, train
        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)
        model, encoders, X_test, y_test, oenc = train(df)
        assert model is not None
        assert "IP" in encoders
        assert "Device" in encoders
        assert "Browser" in encoders
        assert "Outcome" in encoders

    def test_model_can_predict(self):
        from mlops.pipeline.train_pipeline import preprocess, train
        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)
        model, encoders, X_test, y_test, oenc = train(df)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)


# ── Tests: Metrics & Evaluation ───────────────────────────────────────────────

class TestEvaluation:
    def test_metrics_written_to_file(self, tmp_data_dir):
        from mlops.pipeline.train_pipeline import preprocess, train, evaluate
        metrics_dir = tmp_data_dir / "mlops" / "monitoring"
        metrics_dir.mkdir(parents=True)
        metrics_file = metrics_dir / "metrics.json"

        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)
        model, encoders, X_test, y_test, oenc = train(df)

        import mlops.pipeline.train_pipeline as tp
        orig = tp.METRICS_FILE
        tp.METRICS_FILE = metrics_file
        try:
            metrics = evaluate(model, X_test, y_test, oenc, encoders)
        finally:
            tp.METRICS_FILE = orig

        assert metrics_file.exists()
        with open(metrics_file) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert "accuracy" in data[-1]
        assert "f1_weighted" in data[-1]

    def test_metrics_values_in_range(self):
        from mlops.pipeline.train_pipeline import preprocess, train, evaluate
        import mlops.pipeline.train_pipeline as tp
        df = pd.DataFrame(SAMPLE_LOGINS)
        df = preprocess(df)
        model, encoders, X_test, y_test, oenc = train(df)

        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            tp.METRICS_FILE = pathlib.Path(td) / "metrics.json"
            metrics = evaluate(model, X_test, y_test, oenc, encoders)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_weighted"] <= 1.0


# ── Tests: Drift Monitor ───────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_no_drift_returns_empty_violations(self):
        from mlops.monitoring.monitor_drift import check_drift
        metrics = {"accuracy": 0.95, "f1_weighted": 0.92}
        violations = check_drift(metrics)
        assert violations == []

    def test_low_accuracy_triggers_violation(self):
        from mlops.monitoring.monitor_drift import check_drift
        metrics = {"accuracy": 0.50, "f1_weighted": 0.90}
        violations = check_drift(metrics)
        assert any("accuracy" in v for v in violations)

    def test_low_f1_triggers_violation(self):
        from mlops.monitoring.monitor_drift import check_drift
        metrics = {"accuracy": 0.90, "f1_weighted": 0.40}
        violations = check_drift(metrics)
        assert any("f1_weighted" in v for v in violations)

    def test_missing_metrics_file_raises(self, tmp_path, monkeypatch):
        from mlops.monitoring import monitor_drift as md
        monkeypatch.setattr(md, "METRICS_FILE", tmp_path / "none.json")
        with pytest.raises(FileNotFoundError):
            md.load_latest_metrics()


# ── Tests: Full Pipeline End-to-End ───────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_runs(self, tmp_data_dir, monkeypatch):
        """Integration test: run_pipeline() should complete with exit code 0."""
        import mlops.pipeline.train_pipeline as tp

        monkeypatch.setattr(tp, "LOGINS_CSV",  tmp_data_dir / "data" / "logins.csv")
        monkeypatch.setattr(tp, "USERS_CSV",   tmp_data_dir / "data" / "users.csv")
        monkeypatch.setattr(tp, "MODEL_FILE",  tmp_data_dir / "models" / "model.pkl")
        monkeypatch.setattr(tp, "METRICS_FILE",tmp_data_dir / "mlops" / "monitoring" / "metrics.json")
        monkeypatch.setattr(tp, "MODELS_DIR",  tmp_data_dir / "models")
        monkeypatch.setattr(tp, "METRICS_DIR", tmp_data_dir / "mlops" / "monitoring")

        (tmp_data_dir / "models").mkdir(parents=True, exist_ok=True)
        (tmp_data_dir / "mlops" / "monitoring").mkdir(parents=True, exist_ok=True)

        exit_code = tp.run_pipeline()
        assert exit_code == 0

        # Model file should exist
        assert (tmp_data_dir / "models" / "model.pkl").exists()
