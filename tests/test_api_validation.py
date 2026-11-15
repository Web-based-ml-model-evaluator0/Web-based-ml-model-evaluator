"""
Unit test for Issue #35 (ML-SR-007 Server-side API Input Validation, TC-SEC-07).
Tests server-side API validation, payload verification, boundary checking,
safe handling of unexpected parameters, and application crash prevention.
"""

import unittest

from src.services.validation_service import APIInputValidator, APIValidationError


class TestAPIInputValidation(unittest.TestCase):
    def test_valid_training_payload_accepted(self):
        """TC-SEC-07 Step 1: Valid training payloads pass validation and return sanitized dict."""
        payload = {
            "model_name": "Logistic Regression",
            "task_type": "classification",
            "split_ratio": 0.20,
            "target_column": "target",
            "feature_columns": ["f1", "f2"],
            "parameters": {"C": 0.5},
        }
        sanitized = APIInputValidator.validate_training_payload(payload)
        self.assertEqual(sanitized["model_name"], "Logistic Regression")
        self.assertEqual(sanitized["task_type"], "classification")
        self.assertEqual(sanitized["split_ratio"], 0.20)
        self.assertEqual(sanitized["parameters"]["C"], 0.5)

    def test_missing_required_fields_rejected(self):
        """TC-SEC-07 Step 2: Missing required payload fields raise controlled APIValidationError."""
        payload = {"task_type": "classification", "split_ratio": 0.20}  # missing model_name
        with self.assertRaises(APIValidationError) as ctx:
            APIInputValidator.validate_training_payload(payload)
        self.assertIn("model_name", str(ctx.exception))

    def test_invalid_task_type_rejected(self):
        """TC-SEC-07 Step 3: Unsupported task types are rejected with helpful message."""
        payload = {
            "model_name": "Logistic Regression",
            "task_type": "reinforcement_learning",
            "split_ratio": 0.20,
        }
        with self.assertRaises(APIValidationError) as ctx:
            APIInputValidator.validate_training_payload(payload)
        self.assertIn("task_type", str(ctx.exception))

    def test_split_ratio_boundaries_enforced(self):
        """TC-SEC-07 Step 4: Split ratio boundaries (0.05 <= r <= 0.50) are enforced."""
        invalid_ratios = [-0.1, 0.0, 0.01, 0.51, 0.99, 2.0]
        for r in invalid_ratios:
            with self.subTest(ratio=r):
                payload = {
                    "model_name": "Linear Regression",
                    "task_type": "regression",
                    "split_ratio": r,
                }
                with self.assertRaises(APIValidationError):
                    APIInputValidator.validate_training_payload(payload)

    def test_unexpected_parameters_handled_safely(self):
        """TC-SEC-07 Step 5: Unexpected/malicious payload fields are filtered out safely without crashing."""
        payload = {
            "model_name": "Logistic Regression",
            "task_type": "classification",
            "split_ratio": 0.20,
            "__malicious_injected_code__": "eval('import os; os.system()')",
            "extra_key": 12345,
        }
        sanitized = APIInputValidator.validate_training_payload(payload)
        self.assertNotIn("__malicious_injected_code__", sanitized)
        self.assertNotIn("extra_key", sanitized)
        self.assertEqual(sanitized["model_name"], "Logistic Regression")

    def test_malformed_non_dict_payload_rejected(self):
        """TC-SEC-07 Step 6: Non-dict / malformed payloads are safely rejected."""
        malformed_inputs = [None, "string_payload", [1, 2, 3], 42, False]
        for inp in malformed_inputs:
            with self.subTest(inp=inp):
                with self.assertRaises(APIValidationError):
                    APIInputValidator.validate_training_payload(inp)

    def test_experiment_query_limit_validation(self):
        """TC-SEC-07 Step 7: Query limits are constrained to valid integer range."""
        # Valid limit
        q1 = APIInputValidator.validate_experiment_query({"limit": "25"})
        self.assertEqual(q1["limit"], 25)

        # Negative limit
        with self.assertRaises(APIValidationError):
            APIInputValidator.validate_experiment_query({"limit": -5})

        # Excessive limit
        with self.assertRaises(APIValidationError):
            APIInputValidator.validate_experiment_query({"limit": 10000})

    def test_experiment_query_malformed_id_rejected(self):
        """TC-SEC-07 Step 8: Malformed experiment IDs (e.g. SQL injection attempts) are rejected."""
        malicious_ids = [
            "EXP-'; DROP TABLE experiments; --",
            "123",
            "EXP-NONHEX!@#",
            "../../../etc/passwd",
        ]
        for mid in malicious_ids:
            with self.subTest(mid=mid):
                with self.assertRaises(APIValidationError):
                    APIInputValidator.validate_experiment_query({"experiment_id": mid})


if __name__ == "__main__":
    unittest.main()
