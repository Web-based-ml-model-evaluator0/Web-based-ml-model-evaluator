
"""Tests for Issue #33: Experiment History Access Control."""

import unittest
from pathlib import Path
import tempfile
import shutil
import os

from src.services.experiment_service import ExperimentStorageService


class TestExperimentAccessControl(unittest.TestCase):

    def setUp(self):
        os.environ["EXPERIMENT_INTEGRITY_KEY"] = "test-secret-key"
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_access.db"
        self.storage = ExperimentStorageService(db_path=self.db_path)

        self.exp_a = self.storage.save_experiment(
            config={"model_name": "User A Model"},
            metrics={"accuracy": 0.91},
            username="user_a",
        )

        self.exp_b = self.storage.save_experiment(
            config={"model_name": "User B Model"},
            metrics={"accuracy": 0.75},
            username="user_b",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_can_access_own_experiment(self):
        record = self.storage.get_experiment(
            self.exp_a,
            username="user_a",
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["model_name"], "User A Model")

    def test_user_cannot_access_another_users_experiment(self):
        record = self.storage.get_experiment(
            self.exp_b,
            username="user_a",
        )
        self.assertIsNone(record)

    def test_user_history_contains_only_own_experiments(self):
        history = self.storage.list_experiments(
            username="user_a",
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["experiment_id"], self.exp_a)

    def test_admin_can_access_other_users_experiment(self):
        record = self.storage.get_experiment(
            self.exp_b,
            username="admin_user",
            role="admin",
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["model_name"], "User B Model")

    def test_admin_can_view_all_experiment_history(self):
        history = self.storage.list_experiments(
            username="admin_user",
            role="admin",
        )
        self.assertEqual(len(history), 2)

    def test_user_cannot_delete_another_users_experiment(self):
        deleted = self.storage.delete_experiment(
            self.exp_b,
            username="user_a",
        )
        self.assertFalse(deleted)

        # Owner can still access the experiment
        record = self.storage.get_experiment(
            self.exp_b,
            username="user_b",
        )
        self.assertIsNotNone(record)

    def test_user_can_delete_own_experiment(self):
        deleted = self.storage.delete_experiment(
            self.exp_a,
            username="user_a",
        )
        self.assertTrue(deleted)

    def test_evaluator_can_access_other_users_experiment(self):
        record = self.storage.get_experiment(
            self.exp_a,
            username="evaluator_user",
            role="evaluator",
        )
        self.assertIsNotNone(record)


if __name__ == "__main__":
    unittest.main()
