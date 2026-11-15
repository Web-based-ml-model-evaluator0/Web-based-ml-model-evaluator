"""Evaluation report generation (ML-F-018).

Builds a self-contained, downloadable HTML report for a completed
experiment. The report contains dataset information, model information,
configuration and evaluation results.
"""

import csv
import html
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ReportGenerationError(ValueError):
    """Raised when a report cannot be generated for an experiment."""


class EvaluationReportService:
    """Generates downloadable reports for completed experiment records."""

    @staticmethod
    def _validate(experiment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not experiment:
            raise ReportGenerationError("A completed experiment is required to build a report.")
        if experiment.get("status") != "COMPLETED":
            raise ReportGenerationError(
                f"Experiment {experiment.get('experiment_id', '?')} is not completed."
            )
        return experiment

    @staticmethod
    def build_report_data(
        experiment: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect report sections into a single serialisable structure."""
        experiment = EvaluationReportService._validate(experiment)
        config = experiment.get("config", {}) or {}
        metrics = experiment.get("metrics", {}) or {}

        return {
            "report_title": "ML Model Evaluation Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": {
                "experiment_id": experiment.get("experiment_id"),
                "created_at": experiment.get("created_at"),
                "status": experiment.get("status"),
                "integrity_status": experiment.get("integrity_status"),
            },
            "dataset_information": dataset_info or {
                "target_column": experiment.get("target_column"),
                "feature_count": experiment.get("feature_count"),
                "split_ratio": experiment.get("split_ratio"),
            },
            "model_information": {
                "model_name": experiment.get("model_name"),
                "task_type": experiment.get("task_type"),
                "training_time_seconds": experiment.get("training_time_seconds"),
            },
            "configuration": config,
            "evaluation_results": metrics,
        }

    @staticmethod
    def _table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
        cells = "".join(
            "<th>{}</th>".format(html.escape(str(header))) for header in headers
        )
        body = ""
        for row in rows:
            body += "<tr>" + "".join(
                "<td>{}</td>".format(html.escape(str(row.get(header, ""))))
                for header in headers
            ) + "</tr>"
        return f"<table><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table>"

    @staticmethod
    def to_html(
        experiment: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the report as a standalone HTML document."""
        data = EvaluationReportService.build_report_data(experiment, dataset_info)

        experiment_rows = [
            {"Field": key, "Value": value}
            for key, value in data["experiment"].items()
        ]
        dataset_rows = [
            {"Field": key, "Value": value}
            for key, value in data["dataset_information"].items()
        ]
        model_rows = [
            {"Field": key, "Value": value}
            for key, value in data["model_information"].items()
        ]
        config_rows = [
            {"Field": key, "Value": value}
            for key, value in data["configuration"].items()
        ]
        metric_rows = [
            {"Metric": key, "Value": value}
            for key, value in data["evaluation_results"].items()
            if not isinstance(value, (list, dict))
        ]

        section = "<h2>{}</h2>{}"

        parts = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            "<title>{}</title>".format(html.escape(data["report_title"])),
            "<style>",
            "body{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#1f2933;}",
            "h1{color:#0f766e;}h2{color:#0f766e;border-bottom:1px solid #cbd5e1;padding-bottom:4px;}",
            "table{border-collapse:collapse;width:100%;margin:8px 0 20px;}",
            "th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:left;font-size:14px;}",
            "th{background:#f1f5f9;}",
            "</style></head><body>",
            "<h1>{}</h1>".format(html.escape(data["report_title"])),
            "<p>Generated at: {}</p>".format(html.escape(data["generated_at"])),
            section.format("Experiment", EvaluationReportService._table(experiment_rows, ["Field", "Value"])),
            section.format("Dataset Information", EvaluationReportService._table(dataset_rows, ["Field", "Value"])),
            section.format("Model Information", EvaluationReportService._table(model_rows, ["Field", "Value"])),
            section.format("Configuration", EvaluationReportService._table(config_rows, ["Field", "Value"])),
            section.format("Evaluation Results", EvaluationReportService._table(metric_rows, ["Metric", "Value"])),
        ]

        confusion = data["evaluation_results"].get("confusion_matrix")
        labels = data["evaluation_results"].get("labels")
        if confusion and labels:
            cm_rows = []
            for index, row in enumerate(confusion):
                cm_row = {"Actual": labels[index] if index < len(labels) else index}
                for col_index, value in enumerate(row):
                    cm_row[f"Pred {labels[col_index] if col_index < len(labels) else col_index}"] = value
                cm_rows.append(cm_row)
            headers = list(cm_rows[0].keys()) if cm_rows else []
            parts.append(section.format("Confusion Matrix", EvaluationReportService._table(cm_rows, headers)))

        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def to_csv(experiment: Dict[str, Any]) -> str:
        """Render the evaluation results as CSV text."""
        experiment = EvaluationReportService._validate(experiment)
        metrics = experiment.get("metrics", {}) or {}

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                writer.writerow([key, json.dumps(value, default=str)])
            else:
                writer.writerow([key, value])
        return buffer.getvalue()

    @staticmethod
    def to_json(experiment: Dict[str, Any]) -> str:
        """Render the full report payload as JSON text."""
        data = EvaluationReportService.build_report_data(experiment)
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def save_html(
        experiment: Dict[str, Any],
        path: str,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write an HTML report to disk and return the path."""
        content = EvaluationReportService.to_html(experiment, dataset_info)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path
