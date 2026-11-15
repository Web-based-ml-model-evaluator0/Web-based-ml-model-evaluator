
"""
Authorization service for checking user permissions and roles.
"""


class AuthorizationService:
    """Handles authentication checks and role-based authorization."""

    VALID_ROLES = ("user", "admin")

    @staticmethod
    def is_authenticated(user):
        """Check whether a valid user session exists."""
        return (
            isinstance(user, dict)
            and bool(user.get("username"))
            and user.get("role") in AuthorizationService.VALID_ROLES
        )

    @staticmethod
    def has_role(user, required_role):
        """Check whether a user has the required role."""
        if not AuthorizationService.is_authenticated(user):
            return False

        if required_role == "user":
            return user["role"] in ("user", "admin")

        if required_role == "admin":
            return user["role"] == "admin"

        return False

    @staticmethod
    def require_authenticated(user):
        """Reject requests from unauthenticated users."""
        if not AuthorizationService.is_authenticated(user):
            raise PermissionError(
                "Authentication required. Please log in."
            )

    @staticmethod
    def require_admin(user):
        """Reject users without administrator privileges."""
        AuthorizationService.require_authenticated(user)

        if user["role"] != "admin":
            raise PermissionError(
                "Access denied. Administrator privileges required."
            )
