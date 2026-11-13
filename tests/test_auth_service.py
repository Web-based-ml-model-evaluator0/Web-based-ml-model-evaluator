import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.services import auth_service
from src.services.auth_service import AuthService


class TestBcryptPasswordHashing(unittest.TestCase):

    def test_password_is_hashed(self):
        password = "MySecurePassword123!"
        hashed = AuthService.hash_password(password)

        self.assertNotEqual(hashed, password)
        self.assertTrue(hashed.startswith("$2b$"))

    def test_correct_password_verifies(self):
        password = "MySecurePassword123!"
        hashed = AuthService.hash_password(password)

        self.assertTrue(AuthService.verify_password(password, hashed))

    def test_wrong_password_fails(self):
        hashed = AuthService.hash_password("CorrectPassword123!")

        self.assertFalse(
            AuthService.verify_password("WrongPassword123!", hashed)
        )

    def test_hashes_are_unique(self):
        password = "MySecurePassword123!"

        hash1 = AuthService.hash_password(password)
        hash2 = AuthService.hash_password(password)

        self.assertNotEqual(hash1, hash2)

    def test_empty_password_rejected(self):
        with self.assertRaises(ValueError):
            AuthService.hash_password("")

    def test_non_string_password_rejected(self):
        with self.assertRaises(TypeError):
            AuthService.hash_password(12345)

    def test_long_password_rejected(self):
        with self.assertRaises(ValueError):
            AuthService.hash_password("a" * 73)


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

    def test_new_user_uses_bcrypt(self):
        with auth_service._get_connection() as connection:
            row = connection.execute(
                """
                SELECT password_hash, salt
                FROM users
                WHERE username = ?
                """,
                ("admin",)
            ).fetchone()

        stored_hash = row[0].decode("utf-8") if isinstance(row[0], bytes) else row[0]

        self.assertTrue(stored_hash.startswith("$2b$"))
        self.assertEqual(row[1], b"")
        self.assertNotEqual(stored_hash, "SecurePassword123")

    def test_legacy_pbkdf2_login(self):
        password = "LegacyPassword123"
        salt = b"0123456789abcdef"

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            auth_service.HASH_ITERATIONS
        )

        with auth_service._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                (username, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "legacy@example.com",
                    password_hash,
                    salt,
                    "user"
                )
            )

        user = auth_service.authenticate_user(
            "legacy",
            password
        )

        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "legacy")

    def test_legacy_pbkdf2_rejects_wrong_password(self):
        password = "LegacyPassword123"
        salt = b"0123456789abcdef"

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            auth_service.HASH_ITERATIONS
        )

        with auth_service._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                (username, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "legacy",
                    "legacy@example.com",
                    password_hash,
                    salt,
                    "user"
                )
            )

        user = auth_service.authenticate_user(
            "legacy",
            "WrongPassword123"
        )

        self.assertIsNone(user)

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