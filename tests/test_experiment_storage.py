"""
Unit test for Issue #16 (ML-F-016 Experiment Storage, TC-HIST-01).
Tests persistent storage, retrieval, chronological listing, and data integrity
for machine learning experiment configurations and evaluation results.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.services.experiment_service import ExperimentStorageService


class TestExperimentStorage(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for isolated database testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_experiments.db"
        self.storage = ExperimentStorageService(db_path=self.db_path)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_retrieve_classification_experiment(self):
        """TC-HIST-01 Step 1 & 2: Store and retrieve classification experiment with full metrics."""
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

        exp_id = self.storage.save_experiment(config=config, metrics=metrics)
        self.assertTrue(exp_id.startswith("EXP-"))

        # Retrieve experiment
        record = self.storage.get_experiment(exp_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["experiment_id"], exp_id)
        self.assertEqual(record["model_name"], "Logistic Regression")
        self.assertEqual(record["task_type"], "classification")
        self.assertEqual(record["config"]["target_column"], "diagnosis")
        self.assertAlmostEqual(record["metrics"]["accuracy"], 0.965)
        self.assertEqual(record["status"], "COMPLETED")

    def test_save_and_retrieve_regression_experiment(self):
        """TC-HIST-01 Step 3: Store and retrieve regression experiment metrics (MAE, MSE, R2)."""
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

        exp_id = self.storage.save_experiment(config=config, metrics=metrics)
        record = self.storage.get_experiment(exp_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["task_type"], "regression")
        self.assertAlmostEqual(record["metrics"]["r2_score"], 0.88)

    def test_list_experiments_chronological_order(self):
        """TC-HIST-01 Step 4: Stored experiments appear in history in descending order."""
        id1 = self.storage.save_experiment(
            config={"model_name": "Model 1", "task_type": "classification"},
            metrics={"accuracy": 0.80},
        )
        id2 = self.storage.save_experiment(
            config={"model_name": "Model 2", "task_type": "classification"},
            metrics={"accuracy": 0.85},
        )
        id3 = self.storage.save_experiment(
            config={"model_name": "Model 3", "task_type": "regression"},
            metrics={"r2_score": 0.90},
        )

        history = self.storage.list_experiments(limit=10)
        self.assertEqual(len(history), 3)
        # Most recent first
        self.assertEqual(history[0]["experiment_id"], id3)
        self.assertEqual(history[1]["experiment_id"], id2)
        self.assertEqual(history[2]["experiment_id"], id1)

    def test_retrieval_nonexistent_experiment_returns_none(self):
        """TC-HIST-01 Step 5: Querying a nonexistent experiment ID returns None safely."""
        record = self.storage.get_experiment("NONEXISTENT-ID-999")
        self.assertIsNone(record)

    def test_delete_experiment(self):
        """TC-HIST-01 Step 6: Stored experiment can be deleted."""
        exp_id = self.storage.save_experiment(
            config={"model_name": "Model to delete"},
            metrics={"accuracy": 0.5},
        )
        self.assertEqual(self.storage.count_experiments(), 1)

        deleted = self.storage.delete_experiment(exp_id)
        self.assertTrue(deleted)
        self.assertEqual(self.storage.count_experiments(), 0)
        self.assertIsNone(self.storage.get_experiment(exp_id))

    def test_data_integrity_stored_results_correspond_to_correct_experiment(self):
        """TC-HIST-01 Step 7: Multi-experiment results do not cross-contaminate."""
        exp_a = self.storage.save_experiment(
            config={"model_name": "Alpha", "target_column": "targetA"},
            metrics={"accuracy": 0.91},
        )
        exp_b = self.storage.save_experiment(
            config={"model_name": "Beta", "target_column": "targetB"},
            metrics={"accuracy": 0.72},
        )

        res_a = self.storage.get_experiment(exp_a)
        res_b = self.storage.get_experiment(exp_b)

        self.assertEqual(res_a["model_name"], "Alpha")
        self.assertEqual(res_a["config"]["target_column"], "targetA")
        self.assertEqual(res_a["metrics"]["accuracy"], 0.91)

        self.assertEqual(res_b["model_name"], "Beta")
        self.assertEqual(res_b["config"]["target_column"], "targetB")
        self.assertEqual(res_b["metrics"]["accuracy"], 0.72)


if __name__ == "__main__":
    unittest.main()
