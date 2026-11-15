
"""
Authentication service for secure password hashing and verification.
"""

import bcrypt


class AuthService:
    """Provides secure password hashing and verification using bcrypt."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Returns:
            A bcrypt hash suitable for database storage.
        """
        if not isinstance(password, str):
            raise TypeError("Password must be a string.")

        if not password:
            raise ValueError("Password cannot be empty.")

        password_bytes = password.encode("utf-8")

        # bcrypt only supports passwords up to 72 bytes.
        if len(password_bytes) > 72:
            raise ValueError("Password cannot exceed 72 bytes when encoded.")

        hashed = bcrypt.hashpw(
            password_bytes,
            bcrypt.gensalt(rounds=12)
        )

        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """
        Verify a password against its stored bcrypt hash.

        Returns:
            True if the password matches, otherwise False.
        """
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
