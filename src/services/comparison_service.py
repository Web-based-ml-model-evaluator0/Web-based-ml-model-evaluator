"""Model run comparison (ML-F-015).

Compares two or more completed experiment records and builds side-by-side
metric tables. Invalid comparisons (fewer than two runs, mixed task types,
incomplete runs) are rejected with a clear error.
"""

from typing import Any, Dict, List

# Metrics where a larger value is better. All other metrics are treated as
# lower-is-better.
HIGHER_IS_BETTER = {"accuracy", "precision", "recall", "f1_score", "r2_score"}

CLASSIFICATION_KEYS = ["accuracy", "precision", "recall", "f1_score"]
REGRESSION_KEYS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
]


class ComparisonError(ValueError):
    """Raised when experiment runs cannot be compared."""


class ModelComparisonService:
    """Builds comparison tables from completed experiment records."""

    @staticmethod
    def _validate(experiments: List[Dict[str, Any]]) -> None:
        if not experiments or len(experiments) < 2:
            raise ComparisonError("Select at least two completed runs to compare.")

        for experiment in experiments:
            if experiment is None:
                raise ComparisonError("One of the selected runs could not be found.")
            if experiment.get("status") != "COMPLETED":
                raise ComparisonError(
                    f"Run {experiment.get('experiment_id', '?')} is not completed."
                )

        task_types = {experiment.get("task_type") for experiment in experiments}
        if len(task_types) > 1:
            raise ComparisonError(
                "Runs with different task types (classification vs regression) "
                "cannot be compared."
            )

    @staticmethod
    def compare(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a side-by-side comparison for the provided experiment records."""
        ModelComparisonService._validate(experiments)

        task_type = experiments[0].get("task_type", "classification")
        metric_keys = CLASSIFICATION_KEYS if task_type == "classification" else REGRESSION_KEYS

        runs: List[Dict[str, Any]] = []
        for experiment in experiments:
            metrics = experiment.get("metrics", {}) or {}
            runs.append(
                {
                    "experiment_id": experiment.get("experiment_id"),
                    "model_name": experiment.get("model_name"),
                    "task_type": task_type,
                    "metrics": metrics,
                }
            )

        metric_rows: List[Dict[str, Any]] = []
        for key in metric_keys:
            row: Dict[str, Any] = {"Metric": key}
            for run in runs:
                value = run["metrics"].get(key)
                row[run["experiment_id"]] = value
            metric_rows.append(row)

        best_per_metric: Dict[str, Any] = {}
        for key in metric_keys:
            values = [
                (run["experiment_id"], run["metrics"].get(key))
                for run in runs
                if run["metrics"].get(key) is not None
            ]
            if not values:
                continue
            reverse = key in HIGHER_IS_BETTER
            best_id, best_value = sorted(
                values,
                key=lambda item: item[1],
                reverse=reverse,
            )[0]
            best_per_metric[key] = {"experiment_id": best_id, "value": best_value}

        return {
            "task_type": task_type,
            "runs": runs,
            "metric_rows": metric_rows,
            "best_per_metric": best_per_metric,
        }

    @staticmethod
    def metric_table(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convenience wrapper returning only the side-by-side metric rows."""
        return ModelComparisonService.compare(experiments)["metric_rows"]
