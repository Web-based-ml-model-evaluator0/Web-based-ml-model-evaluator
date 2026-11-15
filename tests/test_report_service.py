"""Tests for evaluation report generation (ML-F-018)."""

import json
import os
import unittest

from src.services.report_service import EvaluationReportService, ReportGenerationError


def completed_experiment():
    return {
        "experiment_id": "EXP-TEST01",
        "created_at": "2026-11-15T10:00:00+00:00",
        "status": "COMPLETED",
        "integrity_status": "VERIFIED",
        "model_name": "Logistic Regression",
        "task_type": "classification",
        "target_column": "diagnosis",
        "feature_count": 30,
        "split_ratio": 0.2,
        "training_time_seconds": 0.42,
        "config": {
            "model_name": "Logistic Regression",
            "task_type": "classification",
            "target_column": "diagnosis",
            "feature_count": 30,
        },
        "metrics": {
            "task_type": "classification",
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
            "labels": ["0", "1"],
            "confusion_matrix": [[40, 3], [2, 45]],
        },
    }


class TestReportData(unittest.TestCase):
    def test_build_report_data_contains_all_sections(self):
        data = EvaluationReportService.build_report_data(completed_experiment())

        self.assertEqual(data["report_title"], "ML Model Evaluation Report")
        self.assertIn("generated_at", data)
        for key in (
            "experiment",
            "dataset_information",
            "model_information",
            "configuration",
            "evaluation_results",
        ):
            self.assertIn(key, data)

    def test_dataset_information_can_be_supplied(self):
        data = EvaluationReportService.build_report_data(
            completed_experiment(),
            dataset_info={"target_column": "diagnosis", "rows": 569},
        )
        self.assertEqual(data["dataset_information"]["rows"], 569)

    def test_incomplete_experiment_rejected(self):
        exp = completed_experiment()
        exp["status"] = "RUNNING"
        with self.assertRaises(ReportGenerationError):
            EvaluationReportService.build_report_data(exp)

    def test_missing_experiment_rejected(self):
        with self.assertRaises(ReportGenerationError):
            EvaluationReportService.build_report_data(None)


class TestReportFormats(unittest.TestCase):
    def test_html_report_contains_key_sections(self):
        html = EvaluationReportService.to_html(completed_experiment())

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("ML Model Evaluation Report", html)
        self.assertIn("Dataset Information", html)
        self.assertIn("Model Information", html)
        self.assertIn("Configuration", html)
        self.assertIn("Evaluation Results", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn("Confusion Matrix", html)

    def test_html_escapes_untrusted_values(self):
        exp = completed_experiment()
        exp["model_name"] = "<script>alert('x')</script>"
        html = EvaluationReportService.to_html(exp)
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)

    def test_csv_contains_metrics(self):
        csv_text = EvaluationReportService.to_csv(completed_experiment())
        self.assertIn("metric,value", csv_text)
        self.assertIn("accuracy", csv_text)

    def test_json_report_is_valid_json(self):
        payload = EvaluationReportService.to_json(completed_experiment())
        data = json.loads(payload)
        self.assertEqual(data["experiment"]["experiment_id"], "EXP-TEST01")

    def test_save_html_writes_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "report.html")
            returned = EvaluationReportService.save_html(completed_experiment(), path)

            self.assertEqual(returned, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                self.assertIn("ML Model Evaluation Report", handle.read())


if __name__ == "__main__":
    unittest.main()
