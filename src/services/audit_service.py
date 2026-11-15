
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path


class AuditService:
    """Record timestamped application and experiment events."""

    SENSITIVE_FIELDS = {
        "password",
        "token",
        "secret",
        "api_key",
        "authorization",
        "credential",
    }

    SENSITIVE_VALUE_PATTERN = re.compile(
        r"""(?i)\b(password|token|secret|api[_-]?key|
        authorization|credential)\b\s*[:=]\s*
        (\"[^\"]*\"|'[^']*'|[^\s,;]+)""",
        re.VERBOSE,
    )

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

    @classmethod
    def _sanitize(cls, value):
        """Recursively redact sensitive credentials."""

        if isinstance(value, dict):
            safe = {}

            for key, item in value.items():
                key_lower = str(key).lower()

                if any(
                    field in key_lower
                    for field in cls.SENSITIVE_FIELDS
                ):
                    safe[key] = "[REDACTED]"
                else:
                    safe[key] = cls._sanitize(item)

            return safe

        if isinstance(value, list):
            return [
                cls._sanitize(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls._sanitize(item)
                for item in value
            )

        if isinstance(value, str):
            return cls.SENSITIVE_VALUE_PATTERN.sub(
                lambda match: (
                    f"{match.group(1)}=[REDACTED]"
                ),
                value,
            )

        return value

    def log_event(self, event_type, username=None, details=None):
        """Record an event without storing sensitive credentials."""

        safe_details = self._sanitize(details or {})

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "username": username,
            "details": safe_details,
        }

        self.logger.info(json.dumps(event, default=str))
