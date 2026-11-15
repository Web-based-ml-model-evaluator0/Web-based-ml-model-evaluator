"""Tests for model run comparison (ML-F-015)."""

import unittest

from src.services.comparison_service import ComparisonError, ModelComparisonService


def experiment(exp_id, model, task_type="classification", metrics=None, status="COMPLETED"):
    return {
        "experiment_id": exp_id,
        "model_name": model,
        "task_type": task_type,
        "status": status,
        "metrics": metrics or {},
    }


class TestClassificationComparison(unittest.TestCase):
    def setUp(self):
        self.runs = [
            experiment("EXP-1", "Logistic Regression", metrics={"accuracy": 0.90, "precision": 0.88, "recall": 0.85, "f1_score": 0.86}),
            experiment("EXP-2", "Random Forest", metrics={"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1_score": 0.935}),
        ]

    def test_compare_returns_all_runs(self):
        result = ModelComparisonService.compare(self.runs)
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(len(result["runs"]), 2)

    def test_metric_rows_include_every_run(self):
        result = ModelComparisonService.compare(self.runs)
        row = next(r for r in result["metric_rows"] if r["Metric"] == "accuracy")
        self.assertAlmostEqual(row["EXP-1"], 0.90)
        self.assertAlmostEqual(row["EXP-2"], 0.95)

    def test_best_run_identified_for_higher_is_better(self):
        result = ModelComparisonService.compare(self.runs)
        self.assertEqual(result["best_per_metric"]["accuracy"]["experiment_id"], "EXP-2")
        self.assertEqual(result["best_per_metric"]["f1_score"]["experiment_id"], "EXP-2")

    def test_metric_table_helper(self):
        rows = ModelComparisonService.metric_table(self.runs)
        self.assertEqual(len(rows), 4)


class TestRegressionComparison(unittest.TestCase):
    def test_lower_is_better_for_error_metrics(self):
        runs = [
            experiment("EXP-1", "Linear Regression", task_type="regression", metrics={"mean_absolute_error": 0.8, "mean_squared_error": 1.2, "root_mean_squared_error": 1.1, "r2_score": 0.80}),
            experiment("EXP-2", "Ridge Regression", task_type="regression", metrics={"mean_absolute_error": 0.4, "mean_squared_error": 0.5, "root_mean_squared_error": 0.7, "r2_score": 0.90}),
        ]
        result = ModelComparisonService.compare(runs)

        self.assertEqual(result["best_per_metric"]["mean_absolute_error"]["experiment_id"], "EXP-2")
        self.assertEqual(result["best_per_metric"]["r2_score"]["experiment_id"], "EXP-2")


class TestComparisonValidation(unittest.TestCase):
    def test_fewer_than_two_runs_rejected(self):
        with self.assertRaises(ComparisonError):
            ModelComparisonService.compare([])
        with self.assertRaises(ComparisonError):
            ModelComparisonService.compare([experiment("EXP-1", "A")])

    def test_incomplete_run_rejected(self):
        with self.assertRaises(ComparisonError):
            ModelComparisonService.compare([
                experiment("EXP-1", "A"),
                experiment("EXP-2", "B", status="RUNNING"),
            ])

    def test_mixed_task_types_rejected(self):
        with self.assertRaises(ComparisonError):
            ModelComparisonService.compare([
                experiment("EXP-1", "A", task_type="classification"),
                experiment("EXP-2", "B", task_type="regression"),
            ])

    def test_missing_run_rejected(self):
        with self.assertRaises(ComparisonError):
            ModelComparisonService.compare([experiment("EXP-1", "A"), None])


if __name__ == "__main__":
    unittest.main()
