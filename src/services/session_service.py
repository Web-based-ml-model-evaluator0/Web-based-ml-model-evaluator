"""User session lifecycle management (ML-F-001).

Provides explicit session creation and termination helpers so login and
logout behaviour can be exercised independently of the Streamlit runtime.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class SessionError(ValueError):
    """Raised when a session cannot be created or used."""


class SessionService:
    """Creates and terminates authenticated user sessions."""

    VALID_ROLES = ("user", "admin")

    def __init__(self) -> None:
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_session(self, user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create and register a new session for an authenticated user."""
        if not isinstance(user, dict):
            raise SessionError("A valid user is required to create a session.")

        username = user.get("username")
        role = user.get("role")

        if not isinstance(username, str) or not username.strip():
            raise SessionError("A valid username is required to create a session.")
        if role not in self.VALID_ROLES:
            raise SessionError("A valid role is required to create a session.")

        session = {
            "session_id": uuid.uuid4().hex,
            "username": username.strip(),
            "role": role,
            "created_at": self._now(),
            "active": True,
        }
        self._active_sessions[session["session_id"]] = session
        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return an active session by id, or None when it is not active."""
        session = self._active_sessions.get(session_id)
        if session is None or not session.get("active"):
            return None
        return session

    def is_active(self, session_id: str) -> bool:
        """Return True when the session exists and has not been terminated."""
        return self.get_session(session_id) is not None

    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session and return True when one was active."""
        session = self._active_sessions.get(session_id)
        if session is None or not session.get("active"):
            return False
        session["active"] = False
        session["terminated_at"] = self._now()
        return True

    def terminate_all(self, username: str) -> int:
        """Terminate every active session belonging to a username."""
        terminated = 0
        for session in self._active_sessions.values():
            if session.get("username") == username and session.get("active"):
                session["active"] = False
                session["terminated_at"] = self._now()
                terminated += 1
        return terminated

    def active_session_count(self) -> int:
        """Return the number of currently active sessions."""
        return sum(1 for session in self._active_sessions.values() if session.get("active"))
