"""Tests for evaluation result visualization helpers (ML-F-014)."""

import unittest

import pandas as pd

from src.services.visualization_service import EvaluationVisualizationService


CLASSIFICATION = {
    "task_type": "classification",
    "accuracy": 0.9,
    "precision": 0.8,
    "recall": 0.7,
    "f1_score": 0.75,
    "labels": ["0", "1"],
    "confusion_matrix": [[4, 1], [1, 4]],
}

REGRESSION = {
    "task_type": "regression",
    "mean_absolute_error": 0.5,
    "mean_squared_error": 0.375,
    "root_mean_squared_error": 0.612,
    "r2_score": 0.95,
}


class TestMetricTable(unittest.TestCase):
    def test_classification_table_contains_all_metrics(self):
        table = EvaluationVisualizationService.metric_table(CLASSIFICATION)

        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(list(table.columns), ["Metric", "Value"])
        self.assertEqual(
            table["Metric"].tolist(),
            ["Accuracy", "Precision", "Recall", "F1-Score"],
        )
        self.assertAlmostEqual(table.loc[0, "Value"], 0.9)

    def test_regression_table_contains_all_metrics(self):
        table = EvaluationVisualizationService.metric_table(REGRESSION)

        self.assertEqual(
            table["Metric"].tolist(),
            ["MAE", "MSE", "RMSE", "R2 Score"],
        )

    def test_missing_metrics_are_skipped(self):
        table = EvaluationVisualizationService.metric_table(
            {"task_type": "classification", "accuracy": 0.5}
        )
        self.assertEqual(table["Metric"].tolist(), ["Accuracy"])

    def test_empty_metrics_return_empty_table(self):
        table = EvaluationVisualizationService.metric_table({"task_type": "classification"})
        self.assertTrue(table.empty)

    def test_chart_data_is_indexed_by_metric(self):
        chart = EvaluationVisualizationService.metric_chart_data(CLASSIFICATION)
        self.assertEqual(chart.index.tolist()[0], "Accuracy")
        self.assertIn("Value", chart.columns)


class TestConfusionMatrixFrame(unittest.TestCase):
    def test_matrix_is_labelled(self):
        frame = EvaluationVisualizationService.confusion_matrix_frame(CLASSIFICATION)

        self.assertEqual(list(frame.columns), ["Predicted 0", "Predicted 1"])
        self.assertEqual(list(frame.index), ["Actual 0", "Actual 1"])
        self.assertEqual(frame.iloc[0, 0], 4)

    def test_missing_matrix_returns_empty_frame(self):
        frame = EvaluationVisualizationService.confusion_matrix_frame({})
        self.assertTrue(frame.empty)

    def test_default_labels_when_not_provided(self):
        frame = EvaluationVisualizationService.confusion_matrix_frame(
            {"confusion_matrix": [[2, 0], [0, 3]]}
        )
        self.assertEqual(list(frame.columns), ["Predicted 0", "Predicted 1"])


class TestRegressionFrames(unittest.TestCase):
    def test_scatter_frame_computes_residuals(self):
        frame = EvaluationVisualizationService.regression_scatter_frame(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[1.5, 2.0, 2.0],
        )

        self.assertEqual(list(frame.columns), ["Sample", "Actual", "Predicted", "Residual"])
        self.assertAlmostEqual(frame.loc[0, "Residual"], -0.5)
        self.assertAlmostEqual(frame.loc[2, "Residual"], 1.0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            EvaluationVisualizationService.regression_scatter_frame([1.0], [1.0, 2.0])

    def test_residual_chart_data_indexed_by_sample(self):
        chart = EvaluationVisualizationService.residual_chart_data(
            y_true=[1.0, 2.0],
            y_pred=[1.0, 3.0],
        )
        self.assertEqual(chart.index.tolist(), [0, 1])
        self.assertEqual(chart.columns.tolist(), ["Residual"])


if __name__ == "__main__":
    unittest.main()
