"""Tests for evaluation metrics (ML-F-012 classification, ML-F-013 regression)."""

import unittest

from src.services.evaluation_service import EvaluationService


class TestClassificationEvaluation(unittest.TestCase):
    def test_accuracy_precision_recall_f1_are_computed(self):
        result = EvaluationService.evaluate_classification(
            y_true=[1, 0, 1, 1, 0, 0],
            y_pred=[1, 0, 1, 0, 0, 1],
        )

        self.assertEqual(result["task_type"], "classification")
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1_score"], 2 / 3)

    def test_confusion_matrix_and_labels(self):
        result = EvaluationService.evaluate_classification(
            y_true=[1, 0, 1, 1, 0, 0],
            y_pred=[1, 0, 1, 0, 0, 1],
        )

        self.assertEqual(result["labels"], ["0", "1"])
        # rows = actual 0, 1 ; columns = predicted 0, 1
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])

    def test_perfect_predictions(self):
        result = EvaluationService.evaluate_classification(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1],
        )

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1_score"], 1.0)

    def test_multiclass_uses_weighted_average(self):
        result = EvaluationService.evaluate_classification(
            y_true=[0, 1, 2, 0, 1, 2],
            y_pred=[0, 2, 2, 0, 0, 1],
        )

        self.assertEqual(result["labels"], ["0", "1", "2"])
        self.assertEqual(len(result["confusion_matrix"]), 3)
        self.assertTrue(0.0 <= result["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= result["f1_score"] <= 1.0)

    def test_explicit_labels_are_preserved(self):
        result = EvaluationService.evaluate_classification(
            y_true=["cat", "dog", "cat"],
            y_pred=["cat", "dog", "dog"],
            labels=["cat", "dog"],
        )

        self.assertEqual(result["labels"], ["cat", "dog"])
        self.assertIn("classification_report", result)


class TestRegressionEvaluation(unittest.TestCase):
    def test_mae_mse_rmse_and_r2_are_computed(self):
        result = EvaluationService.evaluate_regression(
            y_true=[3.0, -0.5, 2.0, 7.0],
            y_pred=[2.5, 0.0, 2.0, 8.0],
        )

        self.assertEqual(result["task_type"], "regression")
        self.assertAlmostEqual(result["mean_absolute_error"], 0.5)
        self.assertAlmostEqual(result["mean_squared_error"], 0.375)
        self.assertAlmostEqual(result["root_mean_squared_error"], 0.375 ** 0.5)
        self.assertAlmostEqual(result["r2_score"], 0.9486081370449679, places=6)

    def test_rmse_is_square_root_of_mse(self):
        result = EvaluationService.evaluate_regression(
            y_true=[1.0, 2.0, 3.0, 4.0],
            y_pred=[1.5, 2.5, 2.5, 4.5],
        )
        self.assertAlmostEqual(
            result["root_mean_squared_error"],
            result["mean_squared_error"] ** 0.5,
        )

    def test_perfect_predictions(self):
        result = EvaluationService.evaluate_regression(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[1.0, 2.0, 3.0],
        )

        self.assertAlmostEqual(result["mean_absolute_error"], 0.0)
        self.assertAlmostEqual(result["mean_squared_error"], 0.0)
        self.assertAlmostEqual(result["root_mean_squared_error"], 0.0)
        self.assertAlmostEqual(result["r2_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
