
from src.services.auth_service import AuthService


class AuthenticationService:
    """Authenticate users using stored password hashes."""

    def __init__(self, users=None):
        self.users = users or {}

    def authenticate(self, username, password):
        """Return user information if credentials are valid."""
        if not isinstance(username, str) or not isinstance(password, str):
            return None

        username = username.strip()

        if not username or not password:
            return None

        user = self.users.get(username)

        if not user:
            return None

        if not AuthService.verify_password(
            password,
            user["password_hash"]
        ):
            return None

        return {
            "username": username,
            "role": user["role"],
        }

    def add_user(self, username, password, role="user"):
        """Register a user with a securely hashed password."""
        if not isinstance(username, str) or not username.strip():
            raise ValueError("Username cannot be empty.")

        if role not in ("user", "admin"):
            raise ValueError("Role must be user or admin.")

        username = username.strip()

        if username in self.users:
            raise ValueError("Username already exists.")

        self.users[username] = {
            "password_hash": AuthService.hash_password(password),
            "role": role,
        }
