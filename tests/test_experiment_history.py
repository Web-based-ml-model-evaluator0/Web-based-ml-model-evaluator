"""Tests for experiment history retrieval (ML-F-017 / issue #17)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.services.experiment_service import ExperimentStorageService


def config(model_name="Logistic Regression", task_type="classification"):
    return {
        "model_name": model_name,
        "task_type": task_type,
        "target_column": "diagnosis",
        "feature_count": 30,
        "split_ratio": 0.2,
    }


def metrics(task_type="classification"):
    if task_type == "classification":
        return {"task_type": task_type, "accuracy": 0.93, "f1_score": 0.92}
    return {"task_type": task_type, "r2_score": 0.88, "mean_absolute_error": 0.3}


class TestExperimentHistory(unittest.TestCase):
    def setUp(self):
        os.environ["EXPERIMENT_INTEGRITY_KEY"] = "test-secret-key"
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ExperimentStorageService(
            db_path=Path(self.temp_dir) / "history.db"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_history_is_empty_for_new_user(self):
        history = self.storage.list_experiments(username="new_user", role="user")
        self.assertEqual(history, [])

    def test_history_lists_own_experiments(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-USER1", username="alice"
        )
        self.storage.save_experiment(
            config("Random Forest"), metrics(), experiment_id="EXP-USER2", username="alice"
        )

        history = self.storage.list_experiments(username="alice", role="user")
        ids = [row["experiment_id"] for row in history]

        self.assertEqual(ids, ["EXP-USER2", "EXP-USER1"])

    def test_history_summary_contains_expected_fields(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-1", username="alice"
        )
        row = self.storage.list_experiments(username="alice", role="user")[0]

        for field in (
            "experiment_id",
            "created_at",
            "model_name",
            "task_type",
            "target_column",
            "status",
            "integrity_status",
            "accuracy",
            "f1_score",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["integrity_status"], "VERIFIED")

    def test_history_is_isolated_between_users(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-A", username="alice"
        )
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-B", username="bob"
        )

        alice_history = self.storage.list_experiments(username="alice", role="user")
        bob_history = self.storage.list_experiments(username="bob", role="user")

        self.assertEqual([row["experiment_id"] for row in alice_history], ["EXP-A"])
        self.assertEqual([row["experiment_id"] for row in bob_history], ["EXP-B"])

    def test_admin_and_evaluator_see_all_experiments(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-A", username="alice"
        )
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-B", username="bob"
        )

        admin_history = self.storage.list_experiments(username="admin", role="admin")
        evaluator_history = self.storage.list_experiments(
            username="eval", role="evaluator"
        )

        self.assertEqual(len(admin_history), 2)
        self.assertEqual(len(evaluator_history), 2)

    def test_history_respects_limit(self):
        for index in range(5):
            self.storage.save_experiment(
                config(),
                metrics(),
                experiment_id=f"EXP-{index}",
                username="alice",
            )

        history = self.storage.list_experiments(
            limit=3, username="alice", role="user"
        )
        self.assertEqual(len(history), 3)

    def test_history_detail_returns_full_record(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-DETAIL", username="alice"
        )
        detail = self.storage.get_experiment(
            "EXP-DETAIL", username="alice", role="user"
        )

        self.assertIsNotNone(detail)
        self.assertIn("config", detail)
        self.assertIn("metrics", detail)
        self.assertAlmostEqual(detail["metrics"]["accuracy"], 0.93)
        self.assertEqual(detail["integrity_status"], "VERIFIED")

    def test_history_detail_hidden_from_other_users(self):
        self.storage.save_experiment(
            config(), metrics(), experiment_id="EXP-PRIVATE", username="alice"
        )
        detail = self.storage.get_experiment(
            "EXP-PRIVATE", username="bob", role="user"
        )
        self.assertIsNone(detail)

    def test_unknown_experiment_returns_none(self):
        detail = self.storage.get_experiment(
            "EXP-DOES-NOT-EXIST", username="alice", role="user"
        )
        self.assertIsNone(detail)


if __name__ == "__main__":
    unittest.main()
