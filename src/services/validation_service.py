"""
API Input Validation module (ML-SR-007, TC-SEC-07).
Provides server-side payload validation, type checking, boundary enforcement,
and safe handling of malformed or unexpected parameters without crashing the application.
"""

from typing import Any, Dict, List, Optional, Set
import re


class APIValidationError(ValueError):
    """Raised when server-side API input validation fails (ML-SR-007 / TC-SEC-07)."""
    pass


class APIInputValidator:
    """
    Validates server-side API inputs, payloads, and parameters.
    Designed according to SAD 4.3 and OWASP guidelines.
    """

    ALLOWED_TASK_TYPES: Set[str] = {"classification", "regression"}
    VALID_EXPERIMENT_ID_REGEX = re.compile(r"^EXP-[A-F0-9]{4,16}$")

    @classmethod
    def validate_training_payload(cls, payload: Any) -> Dict[str, Any]:
        """
        Validates training request payload (TC-SEC-07).
        Rejects malformed structures, missing fields, or out-of-range values.
        """
        if not isinstance(payload, dict):
            raise APIValidationError("Payload must be a valid JSON dictionary.")

        # Required fields check
        required_fields = ["model_name", "task_type", "split_ratio"]
        for field in required_fields:
            if field not in payload or payload[field] is None:
                raise APIValidationError(f"Missing required payload field: '{field}'.")

        # 1. model_name validation
        model_name = payload["model_name"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise APIValidationError("Field 'model_name' must be a non-empty string.")
        if len(model_name) > 100:
            raise APIValidationError("Field 'model_name' exceeds maximum length of 100 characters.")

        # 2. task_type validation
        task_type = str(payload["task_type"]).strip().lower()
        if task_type not in cls.ALLOWED_TASK_TYPES:
            raise APIValidationError(
                f"Invalid 'task_type': '{task_type}'. Must be one of {cls.ALLOWED_TASK_TYPES}."
            )

        # 3. split_ratio validation
        try:
            split_ratio = float(payload["split_ratio"])
        except (ValueError, TypeError):
            raise APIValidationError("Field 'split_ratio' must be a valid numeric float.")

        if not (0.05 <= split_ratio <= 0.50):
            raise APIValidationError(
                f"Field 'split_ratio' must be between 0.05 and 0.50 (got {split_ratio})."
            )

        # 4. Optional parameters validation
        parameters = payload.get("parameters", {})
        if parameters is not None and not isinstance(parameters, dict):
            raise APIValidationError("Field 'parameters' must be a dictionary.")

        # 5. Sanitize and ignore unexpected parameters safely
        sanitized = {
            "model_name": model_name.strip(),
            "task_type": task_type,
            "split_ratio": split_ratio,
            "parameters": parameters or {},
        }

        # Preserve known target and feature definitions if present
        if "target_column" in payload:
            if not isinstance(payload["target_column"], str) or not payload["target_column"].strip():
                raise APIValidationError("Field 'target_column' must be a non-empty string.")
            sanitized["target_column"] = payload["target_column"].strip()

        if "feature_columns" in payload:
            features = payload["feature_columns"]
            if not isinstance(features, list) or len(features) == 0:
                raise APIValidationError("Field 'feature_columns' must be a non-empty list of strings.")
            if not all(isinstance(c, str) and c.strip() for c in features):
                raise APIValidationError("All elements in 'feature_columns' must be valid non-empty strings.")
            sanitized["feature_columns"] = [c.strip() for c in features]

        return sanitized

    @classmethod
    def validate_experiment_query(cls, query_params: Any) -> Dict[str, Any]:
        """
        Validates experiment querying and pagination parameters (TC-SEC-07).
        Protects against invalid types, negative limits, and malicious query inputs.
        """
        if not isinstance(query_params, dict):
            raise APIValidationError("Query parameters must be a dictionary.")

        sanitized: Dict[str, Any] = {}

        # 1. limit parameter validation
        if "limit" in query_params:
            try:
                limit = int(query_params["limit"])
            except (ValueError, TypeError):
                raise APIValidationError("Query parameter 'limit' must be an integer.")

            if limit < 1:
                raise APIValidationError("Query parameter 'limit' must be at least 1.")
            if limit > 500:
                raise APIValidationError("Query parameter 'limit' cannot exceed 500.")
            sanitized["limit"] = limit
        else:
            sanitized["limit"] = 50

        # 2. experiment_id parameter validation
        if "experiment_id" in query_params:
            exp_id = str(query_params["experiment_id"]).strip().upper()
            if not cls.VALID_EXPERIMENT_ID_REGEX.match(exp_id):
                raise APIValidationError(
                    f"Malformed 'experiment_id': '{exp_id}'. Expected format 'EXP-XXXX'."
                )
            sanitized["experiment_id"] = exp_id

        return sanitized

    @classmethod
    def validate_prediction_payload(cls, payload: Any) -> Dict[str, Any]:
        """
        Validates payload for real-time model predictions.
        """
        if not isinstance(payload, dict):
            raise APIValidationError("Prediction payload must be a valid dictionary.")

        if "inputs" not in payload or not isinstance(payload["inputs"], list):
            raise APIValidationError("Payload must contain an 'inputs' list.")

        inputs = payload["inputs"]
        if len(inputs) == 0:
            raise APIValidationError("Prediction 'inputs' list cannot be empty.")

        # Ensure all rows are lists or numeric records
        for i, row in enumerate(inputs):
            if not isinstance(row, (list, tuple)):
                raise APIValidationError(f"Input record at index {i} must be a list/vector of features.")

        return payload
