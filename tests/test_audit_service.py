
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


def test_audit_service_redacts_nested_credentials(tmp_path):
    from src.services.audit_service import AuditService

    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "test",
        details={
            "user": {
                "username": "alice",
                "password": "mysecret123",
            }
        },
    )

    content = log_file.read_text(encoding="utf-8")

    assert "alice" in content
    assert "mysecret123" not in content
    assert "[REDACTED]" in content


def test_audit_service_redacts_credentials_inside_lists(tmp_path):
    from src.services.audit_service import AuditService

    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "test",
        details={
            "users": [
                {"username": "alice", "api_key": "secret-key-123"}
            ]
        },
    )

    content = log_file.read_text(encoding="utf-8")

    assert "alice" in content
    assert "secret-key-123" not in content
    assert "[REDACTED]" in content


def test_audit_service_redacts_credentials_in_text(tmp_path):
    from src.services.audit_service import AuditService

    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "test",
        details={
            "message": "Login failed password=mysecret123"
        },
    )

    content = log_file.read_text(encoding="utf-8")

    assert "Login failed" in content
    assert "mysecret123" not in content
    assert "[REDACTED]" in content


def test_audit_service_preserves_non_sensitive_details(tmp_path):
    from src.services.audit_service import AuditService

    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "test",
        details={
            "username": "alice",
            "action": "login",
            "status": "success",
        },
    )

    content = log_file.read_text(encoding="utf-8")

    assert "alice" in content
    assert "login" in content
    assert "success" in content


def test_audit_service_redacts_bearer_tokens(tmp_path):
    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "test",
        details={
            "message": "Authorization: Bearer abc123secret",
        },
    )

    content = log_file.read_text(encoding="utf-8")

    assert "abc123secret" not in content
    assert "[REDACTED]" in content


def test_audit_service_get_events(tmp_path):
    log_file = tmp_path / "audit.log"
    audit = AuditService(log_file)

    audit.log_event(
        "LOGIN_SUCCESS",
        username="alice",
        details={"role": "student"},
    )

    events = audit.get_events()

    assert len(events) == 1
    assert events[0]["event_type"] == "LOGIN_SUCCESS"
    assert events[0]["username"] == "alice"
    assert events[0]["details"]["role"] == "student"


