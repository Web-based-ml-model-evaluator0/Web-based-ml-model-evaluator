
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
        r"""(?i)\b(password|token|secret|api[_-]?key|authorization|credential)\b\s*[:=]\s*("[^"]*"|'[^']*'|[^\s,;]+)""",
        re.VERBOSE,
    )

    BEARER_TOKEN_PATTERN = re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
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
            # Redact complete Bearer authorization tokens.
            value = cls.BEARER_TOKEN_PATTERN.sub(
                "Bearer [REDACTED]",
                value,
            )

            # Redact other credential assignments.
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

    def get_events(self):
        """Read recorded audit events from the log file."""

        events = []

        if not self.log_file.exists():
            return events

        with self.log_file.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return events
