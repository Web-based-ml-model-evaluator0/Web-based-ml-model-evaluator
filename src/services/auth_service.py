"""
Authentication service for secure password hashing and verification.
"""

import hashlib
import hmac
import os
import sqlite3
import bcrypt

from contextlib import contextmanager
from pathlib import Path


DATABASE_PATH = Path(
    os.getenv("AUTH_DATABASE_PATH", "instance/users.db")
)

HASH_ITERATIONS = 600_000
HASH_ALGORITHM = "sha256"


class AuthService:
    """Provides secure password hashing and verification using bcrypt."""

    @staticmethod
    def hash_password(password: str) -> str:
        if not isinstance(password, str):
            raise TypeError("Password must be a string.")

        if not password:
            raise ValueError("Password cannot be empty.")

        password_bytes = password.encode("utf-8")

        if len(password_bytes) > 72:
            raise ValueError("Password cannot exceed 72 bytes when encoded.")

        hashed = bcrypt.hashpw(
            password_bytes,
            bcrypt.gensalt(rounds=12)
        )

        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        if not isinstance(password, str):
            return False

        if not isinstance(hashed_password, str):
            return False

        password_bytes = password.encode("utf-8")

        if len(password_bytes) > 72:
            return False

        try:
            return bcrypt.checkpw(
                password_bytes,
                hashed_password.encode("utf-8")
            )
        except (ValueError, TypeError):
            return False


@contextmanager
def _get_connection():
    """Create and safely close a SQLite database connection."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _hash_password(password, salt):
    """Verify legacy passwords using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS
    )


def initialize_database():
    """Create the users table if it does not exist."""
    with _get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_user(username, email, password, role="user"):
    """Register a user with a bcrypt-hashed password."""

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        raise ValueError("All fields are required.")

    if len(password) < 8:
        raise ValueError(
            "Password must contain at least 8 characters."
        )

    if role not in ("user", "admin"):
        raise ValueError("Invalid user role.")

    password_hash = AuthService.hash_password(password)

    try:
        with _get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                (username, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    password_hash,
                    b"",
                    role
                )
            )

    except sqlite3.IntegrityError:
        raise ValueError(
            "Username or email is already registered."
        )


def authenticate_user(identifier, password):
    """Authenticate a user using username/email and password."""

    if not identifier or not password:
        return None

    identifier = identifier.strip()

    with _get_connection() as connection:
        connection.row_factory = sqlite3.Row

        user = connection.execute(
            """
            SELECT
                id,
                username,
                email,
                password_hash,
                salt,
                role
            FROM users
            WHERE username = ? COLLATE NOCASE
               OR email = ? COLLATE NOCASE
            """,
            (identifier, identifier)
        ).fetchone()

    if user is None:
        return None

    stored_hash = user["password_hash"]
    salt = user["salt"]

    # Existing accounts use PBKDF2.
    if salt:
        attempted_hash = _hash_password(password, salt)

        if not hmac.compare_digest(
            attempted_hash,
            stored_hash
        ):
            return None

    # New accounts use bcrypt.
    else:
        if isinstance(stored_hash, bytes):
            stored_hash = stored_hash.decode("utf-8")

        if not AuthService.verify_password(password, stored_hash):
            return None

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }