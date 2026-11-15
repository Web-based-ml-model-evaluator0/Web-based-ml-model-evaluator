
import json
import logging

from src.services.audit_service import AuditService


def test_audit_event_contains_timestamp_and_details(tmp_path):
    log_file = tmp_path / "audit.log"
    service = AuditService(log_file=str(log_file))

    service.log_event(
        event_type="LOGIN_SUCCESS",
        username="student",
        details={"role": "user"},
    )

    log_contents = log_file.read_text(encoding="utf-8")
    event = json.loads(log_contents.strip().splitlines()[-1])

    assert event["event_type"] == "LOGIN_SUCCESS"
    assert event["username"] == "student"
    assert event["details"]["role"] == "user"
    assert "timestamp" in event


def test_audit_event_excludes_sensitive_fields(tmp_path):
    log_file = tmp_path / "audit.log"
    service = AuditService(log_file=str(log_file))

    service.log_event(
        event_type="LOGIN_SUCCESS",
        username="student",
        details={
            "role": "user",
            "password": "secret123",
            "api_key": "test-api-key",
            "token": "test-token",
        },
    )

    log_contents = log_file.read_text(encoding="utf-8")

    assert "secret123" not in log_contents
    assert "test-api-key" not in log_contents
    assert "test-token" not in log_contents
    assert '"role": "user"' in log_contents


def test_audit_event_without_details(tmp_path):
    log_file = tmp_path / "audit.log"
    service = AuditService(log_file=str(log_file))

    service.log_event(
        event_type="LOGOUT",
        username="student",
    )

    log_contents = log_file.read_text(encoding="utf-8")
    event = json.loads(log_contents.strip().splitlines()[-1])

    assert event["event_type"] == "LOGOUT"
    assert event["username"] == "student"
    assert event["details"] == {}


def test_audit_event_timestamp_is_valid(tmp_path):
    log_file = tmp_path / "audit.log"
    service = AuditService(log_file=str(log_file))

    service.log_event(
        event_type="MODEL_TRAINING_SUCCESS",
        username="student",
    )

    log_contents = log_file.read_text(encoding="utf-8")
    event = json.loads(log_contents.strip().splitlines()[-1])

    from datetime import datetime

    timestamp = datetime.fromisoformat(event["timestamp"])

    assert timestamp.tzinfo is not None
