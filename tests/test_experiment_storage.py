"""
Unit tests for experiment storage and integrity verification.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from src.services.experiment_service import (
    ExperimentIntegrityError,
    ExperimentStorageService,
)


class TestExperimentStorage(unittest.TestCase):
    def setUp(self):
        os.environ["EXPERIMENT_INTEGRITY_KEY"] = "test-secret-key"
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_experiments.db"
        self.storage = ExperimentStorageService(db_path=self.db_path)
        self.username = "test_user"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_retrieve_classification_experiment(self):
        config = {
            "model_name": "Logistic Regression",
            "task_type": "classification",
            "target_column": "diagnosis",
            "feature_count": 30,
            "split_ratio": 0.20,
            "training_time_seconds": 0.125,
            "parameters": {"C": 1.0, "max_iter": 1000},
        }
        metrics = {
            "accuracy": 0.965,
            "precision": 0.952,
            "recall": 0.970,
            "f1_score": 0.961,
            "labels": ["benign", "malignant"],
            "confusion_matrix": [[70, 2], [2, 40]],
        }

        exp_id = self.storage.save_experiment(
            config=config, metrics=metrics, username=self.username
        )
        self.assertTrue(exp_id.startswith("EXP-"))

        record = self.storage.get_experiment(exp_id, username=self.username)
        self.assertIsNotNone(record)
        self.assertEqual(record["experiment_id"], exp_id)
        self.assertEqual(record["model_name"], "Logistic Regression")
        self.assertEqual(record["task_type"], "classification")
        self.assertEqual(record["config"]["target_column"], "diagnosis")
        self.assertAlmostEqual(record["metrics"]["accuracy"], 0.965)
        self.assertEqual(record["status"], "COMPLETED")
        self.assertEqual(record["integrity_status"], "VERIFIED")

    def test_save_and_retrieve_regression_experiment(self):
        config = {
            "model_name": "Linear Regression",
            "task_type": "regression",
            "target_column": "price",
            "feature_count": 8,
            "split_ratio": 0.25,
        }
        metrics = {
            "mean_absolute_error": 12.5,
            "mean_squared_error": 250.0,
            "root_mean_squared_error": 15.81,
            "r2_score": 0.88,
        }

        exp_id = self.storage.save_experiment(
            config=config, metrics=metrics, username=self.username
        )
        record = self.storage.get_experiment(exp_id, username=self.username)
        self.assertIsNotNone(record)
        self.assertEqual(record["task_type"], "regression")
        self.assertAlmostEqual(record["metrics"]["r2_score"], 0.88)

    def test_list_experiments_chronological_order(self):
        id1 = self.storage.save_experiment(
            {"model_name": "Model 1", "task_type": "classification"},
            {"accuracy": 0.80}, username=self.username,
        )
        id2 = self.storage.save_experiment(
            {"model_name": "Model 2", "task_type": "classification"},
            {"accuracy": 0.85}, username=self.username,
        )
        id3 = self.storage.save_experiment(
            {"model_name": "Model 3", "task_type": "regression"},
            {"r2_score": 0.90}, username=self.username,
        )

        history = self.storage.list_experiments(
            limit=10, username=self.username
        )
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["experiment_id"], id3)
        self.assertEqual(history[1]["experiment_id"], id2)
        self.assertEqual(history[2]["experiment_id"], id1)

    def test_retrieval_nonexistent_experiment_returns_none(self):
        record = self.storage.get_experiment(
            "NONEXISTENT-ID-999", username=self.username
        )
        self.assertIsNone(record)

    def test_delete_experiment(self):
        exp_id = self.storage.save_experiment(
            config={"model_name": "Model to delete"},
            metrics={"accuracy": 0.5},
            username=self.username,
        )
        self.assertEqual(self.storage.count_experiments(), 1)
        self.assertTrue(
            self.storage.delete_experiment(exp_id, username=self.username)
        )
        self.assertEqual(self.storage.count_experiments(), 0)
        self.assertIsNone(
            self.storage.get_experiment(exp_id, username=self.username)
        )

    def test_data_integrity_stored_results_correspond_to_correct_experiment(self):
        exp_a = self.storage.save_experiment(
            {"model_name": "Alpha", "target_column": "targetA"},
            {"accuracy": 0.91}, username=self.username,
        )
        exp_b = self.storage.save_experiment(
            {"model_name": "Beta", "target_column": "targetB"},
            {"accuracy": 0.72}, username=self.username,
        )

        res_a = self.storage.get_experiment(exp_a, username=self.username)
        res_b = self.storage.get_experiment(exp_b, username=self.username)
        self.assertEqual(res_a["model_name"], "Alpha")
        self.assertEqual(res_a["config"]["target_column"], "targetA")
        self.assertEqual(res_a["metrics"]["accuracy"], 0.91)
        self.assertEqual(res_b["model_name"], "Beta")
        self.assertEqual(res_b["config"]["target_column"], "targetB")
        self.assertEqual(res_b["metrics"]["accuracy"], 0.72)


def test_configuration_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    experiment_id = service.save_experiment(
        {"model_name": "RandomForest", "n_estimators": 100},
        {"accuracy": 0.95}, username="tester",
    )
    with service._get_connection() as conn:
        conn.execute(
            "UPDATE experiments SET config_json = ? WHERE experiment_id = ?",
            ('{"model_name":"ModifiedModel"}', experiment_id),
        )
        conn.commit()
    with pytest.raises(ExperimentIntegrityError):
        service.get_experiment(experiment_id, username="tester")


def test_metrics_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    experiment_id = service.save_experiment(
        {"model_name": "RandomForest"}, {"accuracy": 0.95},
        username="tester",
    )
    with service._get_connection() as conn:
        conn.execute(
            "UPDATE experiments SET metrics_json = ? WHERE experiment_id = ?",
            ('{"accuracy":0.10}', experiment_id),
        )
        conn.commit()
    with pytest.raises(ExperimentIntegrityError):
        service.get_experiment(experiment_id, username="tester")


def test_stored_configuration_and_metrics_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    config = {"model_name": "RandomForest", "n_estimators": 100,
              "features": ["age", "income"]}
    metrics = {"accuracy": 0.95, "f1_score": 0.92}
    experiment_id = service.save_experiment(config, metrics, username="tester")
    retrieved = service.get_experiment(experiment_id, username="tester")
    assert retrieved["config"] == config
    assert retrieved["metrics"] == metrics


def test_config_hash_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    experiment_id = service.save_experiment(
        {"model_name": "TestModel"}, {"accuracy": 0.95}, username="tester"
    )
    with service._get_connection() as conn:
        conn.execute(
            "UPDATE experiments SET config_hash = ? WHERE experiment_id = ?",
            ("invalid-hash", experiment_id),
        )
        conn.commit()
    with pytest.raises(ExperimentIntegrityError):
        service.get_experiment(experiment_id, username="tester")


def test_metrics_hash_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    experiment_id = service.save_experiment(
        {"model_name": "TestModel"}, {"accuracy": 0.95}, username="tester"
    )
    with service._get_connection() as conn:
        conn.execute(
            "UPDATE experiments SET metrics_hash = ? WHERE experiment_id = ?",
            ("invalid-hash", experiment_id),
        )
        conn.commit()
    with pytest.raises(ExperimentIntegrityError):
        service.get_experiment(experiment_id, username="tester")


def test_legacy_experiment_is_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    experiment_id = service.save_experiment(
        {"model_name": "TestModel"}, {"accuracy": 0.95}, username="tester"
    )
    with service._get_connection() as conn:
        conn.execute(
            """UPDATE experiments
               SET config_hash = NULL, metrics_hash = NULL
               WHERE experiment_id = ?""",
            (experiment_id,),
        )
        conn.commit()
    experiment = service.get_experiment(experiment_id, username="tester")
    assert experiment is not None
    assert experiment["integrity_status"] == "UNVERIFIED"


def test_legacy_experiment_data_remains_retrievable(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_INTEGRITY_KEY", "test-secret")
    service = ExperimentStorageService(tmp_path / "test.db")
    config = {"model_name": "TestModel"}
    metrics = {"accuracy": 0.95}
    experiment_id = service.save_experiment(config, metrics, username="tester")
    with service._get_connection() as conn:
        conn.execute(
            """UPDATE experiments
               SET config_hash = NULL, metrics_hash = NULL
               WHERE experiment_id = ?""",
            (experiment_id,),
        )
        conn.commit()
    experiment = service.get_experiment(experiment_id, username="tester")
    assert experiment["config"] == config
    assert experiment["metrics"] == metrics
    assert experiment["integrity_status"] == "UNVERIFIED"


if __name__ == "__main__":
    unittest.main()
