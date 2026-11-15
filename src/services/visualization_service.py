"""Evaluation visualization helpers (ML-F-014).

Transforms evaluation metric dictionaries into chart/table friendly
structures. Keeping this logic out of the presentation layer makes the
visualization behaviour independently testable.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


CLASSIFICATION_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-Score"),
]

REGRESSION_METRICS = [
    ("mean_absolute_error", "MAE"),
    ("mean_squared_error", "MSE"),
    ("root_mean_squared_error", "RMSE"),
    ("r2_score", "R2 Score"),
]


class EvaluationVisualizationService:
    """Builds tables and chart-ready data frames for evaluation results."""

    @staticmethod
    def metric_table(metrics: Dict[str, Any]) -> pd.DataFrame:
        """Return a readable Metric/Value table for a metric dictionary."""
        task_type = metrics.get("task_type", "classification")
        definitions = (
            REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
        )

        rows: List[Dict[str, Any]] = []
        for key, label in definitions:
            value = metrics.get(key)
            if value is None:
                continue
            rows.append({"Metric": label, "Value": float(value)})

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    @staticmethod
    def metric_chart_data(metrics: Dict[str, Any]) -> pd.DataFrame:
        """Return a DataFrame indexed by metric name for charting."""
        table = EvaluationVisualizationService.metric_table(metrics)
        if table.empty:
            return pd.DataFrame(columns=["Value"])
        return table.set_index("Metric")[["Value"]]

    @staticmethod
    def confusion_matrix_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
        """Return the confusion matrix as a labelled DataFrame."""
        matrix = metrics.get("confusion_matrix")
        labels = metrics.get("labels")

        if not matrix:
            return pd.DataFrame()

        if not labels:
            size = len(matrix)
            labels = [str(i) for i in range(size)]

        labels = [str(label) for label in labels]
        return pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        )

    @staticmethod
    def regression_scatter_frame(
        y_true: Any,
        y_pred: Any,
    ) -> pd.DataFrame:
        """Return actual/predicted/residual rows for a regression scatter plot."""
        actual = list(y_true)
        predicted = list(y_pred)
        if len(actual) != len(predicted):
            raise ValueError("y_true and y_pred must have the same length.")

        rows = []
        for index, (a, p) in enumerate(zip(actual, predicted)):
            rows.append(
                {
                    "Sample": index,
                    "Actual": float(a),
                    "Predicted": float(p),
                    "Residual": float(a) - float(p),
                }
            )
        return pd.DataFrame(rows, columns=["Sample", "Actual", "Predicted", "Residual"])

    @staticmethod
    def residual_chart_data(y_true: Any, y_pred: Any) -> pd.DataFrame:
        """Return residual values indexed by sample for a line chart."""
        frame = EvaluationVisualizationService.regression_scatter_frame(y_true, y_pred)
        if frame.empty:
            return pd.DataFrame(columns=["Residual"])
        return frame.set_index("Sample")[["Residual"]]
