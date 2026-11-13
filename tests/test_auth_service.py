
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.services import auth_service


class TestAuthService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_users.db"

        self.db_patch = patch.object(
            auth_service,
            "DATABASE_PATH",
            self.db_path
        )
        self.db_patch.start()

        # Reduce hashing iterations during tests only.
        self.hash_patch = patch.object(
            auth_service,
            "HASH_ITERATIONS",
            1000
        )
        self.hash_patch.start()

        auth_service.initialize_database()

        auth_service.create_user(
            "admin",
            "admin@example.com",
            "SecurePassword123",
            role="admin"
        )

    def tearDown(self):
        self.hash_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_valid_username_login(self):
        user = auth_service.authenticate_user(
            "admin",
            "SecurePassword123"
        )

        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "admin")
        self.assertEqual(user["role"], "admin")

    def test_valid_email_login(self):
        user = auth_service.authenticate_user(
            "admin@example.com",
            "SecurePassword123"
        )

        self.assertIsNotNone(user)
        self.assertEqual(user["email"], "admin@example.com")

    def test_invalid_password(self):
        user = auth_service.authenticate_user(
            "admin",
            "WrongPassword123"
        )

        self.assertIsNone(user)

    def test_invalid_username(self):
        user = auth_service.authenticate_user(
            "unknown",
            "SecurePassword123"
        )

        self.assertIsNone(user)

    def test_empty_credentials(self):
        self.assertIsNone(
            auth_service.authenticate_user("", "")
        )

    def test_password_is_not_stored_as_plaintext(self):
        with auth_service._get_connection() as connection:
            row = connection.execute(
                """
                SELECT password_hash, salt
                FROM users
                WHERE username = ?
                """,
                ("admin",)
            ).fetchone()

        self.assertNotEqual(
            row[0],
            b"SecurePassword123"
        )

        self.assertTrue(row[1])
        self.assertEqual(len(row[1]), 16)

    def test_duplicate_user_rejected(self):
        with self.assertRaises(ValueError):
            auth_service.create_user(
                "admin",
                "another@example.com",
                "AnotherPassword123"
            )

    def test_short_password_rejected(self):
        with self.assertRaises(ValueError):
            auth_service.create_user(
                "newuser",
                "new@example.com",
                "short"
            )

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            auth_service.create_user(
                "newuser",
                "new@example.com",
                "Password123",
                role="superuser"
            )


if __name__ == "__main__":
    unittest.main()
