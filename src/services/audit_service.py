
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class AuditService:
    """Record timestamped application and experiment events."""

    def __init__(self, log_file="logs/audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use a separate logger for each log file.
        logger_name = (
            f"ml_model_evaluator.audit.{self.log_file.resolve()}"
        )

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.FileHandler(
                self.log_file,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log_event(self, event_type, username=None, details=None):
        """Record an event without storing sensitive credentials."""

        details = details or {}

        sensitive_fields = {
            "password",
            "token",
            "secret",
            "api_key",
            "authorization",
            "credential",
        }

        safe_details = {
            key: value
            for key, value in details.items()
            if not any(
                field in key.lower()
                for field in sensitive_fields
            )
        }

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "username": username,
            "details": safe_details,
        }

        self.logger.info(json.dumps(event, default=str))
