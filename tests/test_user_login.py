"""Tests for user login and session lifecycle (ML-F-001 / issue #37).

Covers valid credentials, invalid password, invalid username, empty fields,
session creation and logout/session termination.
"""

import unittest

from src.services.auth_service import AuthService
from src.services.authentication_service import AuthenticationService
from src.services.authorization_service import AuthorizationService
from src.services.session_service import SessionError, SessionService


class TestUserLogin(unittest.TestCase):
    def setUp(self):
        self.auth = AuthenticationService()
        self.auth.add_user("student", "Student@123", role="user")
        self.auth.add_user("admin", "Admin@123", role="admin")
        self.sessions = SessionService()

    def test_valid_credentials_return_user(self):
        user = self.auth.authenticate("student", "Student@123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "student")
        self.assertEqual(user["role"], "user")

    def test_valid_admin_credentials_return_admin_role(self):
        user = self.auth.authenticate("admin", "Admin@123")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "admin")

    def test_invalid_password_rejected(self):
        self.assertIsNone(self.auth.authenticate("student", "WrongPassword"))

    def test_invalid_username_rejected(self):
        self.assertIsNone(self.auth.authenticate("unknown", "Student@123"))

    def test_empty_fields_rejected(self):
        self.assertIsNone(self.auth.authenticate("", "Student@123"))
        self.assertIsNone(self.auth.authenticate("student", ""))
        self.assertIsNone(self.auth.authenticate("", ""))

    def test_whitespace_only_fields_rejected(self):
        self.assertIsNone(self.auth.authenticate("   ", "Student@123"))
        self.assertIsNone(self.auth.authenticate("student", "   "))

    def test_non_string_fields_rejected(self):
        self.assertIsNone(self.auth.authenticate(None, "Student@123"))
        self.assertIsNone(self.auth.authenticate("student", None))

    def test_username_is_trimmed_before_authentication(self):
        user = self.auth.authenticate("  student  ", "Student@123")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "student")

    def test_password_hash_never_returned(self):
        user = self.auth.authenticate("student", "Student@123")
        self.assertNotIn("password", user)
        self.assertNotIn("password_hash", user)

    def test_session_creation_for_valid_user(self):
        user = self.auth.authenticate("student", "Student@123")
        session = self.sessions.create_session(user)

        self.assertTrue(session["active"])
        self.assertEqual(session["username"], "student")
        self.assertEqual(session["role"], "user")
        self.assertIn("session_id", session)
        self.assertTrue(self.sessions.is_active(session["session_id"]))

    def test_session_creation_rejects_empty_user(self):
        with self.assertRaises(SessionError):
            self.sessions.create_session({})
        with self.assertRaises(SessionError):
            self.sessions.create_session(None)

    def test_session_creation_rejects_invalid_role(self):
        with self.assertRaises(SessionError):
            self.sessions.create_session({"username": "ghost", "role": "superuser"})

    def test_session_grants_authenticated_access(self):
        user = self.auth.authenticate("student", "Student@123")
        session = self.sessions.create_session(user)
        AuthorizationService.require_authenticated(
            {"username": session["username"], "role": session["role"]}
        )

    def test_logout_terminates_session(self):
        user = self.auth.authenticate("student", "Student@123")
        session = self.sessions.create_session(user)

        self.assertTrue(self.sessions.terminate_session(session["session_id"]))
        self.assertFalse(self.sessions.is_active(session["session_id"]))
        self.assertEqual(self.sessions.active_session_count(), 0)

    def test_terminating_unknown_session_returns_false(self):
        self.assertFalse(self.sessions.terminate_session("does-not-exist"))

    def test_terminating_twice_returns_false(self):
        user = self.auth.authenticate("student", "Student@123")
        session = self.sessions.create_session(user)
        self.assertTrue(self.sessions.terminate_session(session["session_id"]))
        self.assertFalse(self.sessions.terminate_session(session["session_id"]))

    def test_terminate_all_sessions_for_user(self):
        user = self.auth.authenticate("student", "Student@123")
        first = self.sessions.create_session(user)
        second = self.sessions.create_session(user)

        self.assertEqual(self.sessions.terminate_all("student"), 2)
        self.assertFalse(self.sessions.is_active(first["session_id"]))
        self.assertFalse(self.sessions.is_active(second["session_id"]))

    def test_sessions_are_unique_per_login(self):
        user = self.auth.authenticate("student", "Student@123")
        first = self.sessions.create_session(user)
        second = self.sessions.create_session(user)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(self.sessions.active_session_count(), 2)


if __name__ == "__main__":
    unittest.main()
