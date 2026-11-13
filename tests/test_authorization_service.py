
import pytest

from src.services.authorization_service import AuthorizationService


@pytest.fixture
def admin_user():
    return {"username": "admin", "role": "admin"}


@pytest.fixture
def normal_user():
    return {"username": "student", "role": "user"}


def test_authenticated_user(admin_user):
    assert AuthorizationService.is_authenticated(admin_user)


def test_unauthenticated_user():
    assert not AuthorizationService.is_authenticated(None)


def test_admin_has_admin_access(admin_user):
    assert AuthorizationService.has_role(admin_user, "admin")


def test_user_cannot_access_admin(normal_user):
    assert not AuthorizationService.has_role(normal_user, "admin")


def test_user_has_user_access(normal_user):
    assert AuthorizationService.has_role(normal_user, "user")


def test_admin_has_user_access(admin_user):
    assert AuthorizationService.has_role(admin_user, "user")


def test_require_authenticated_rejects_unauthenticated():
    with pytest.raises(PermissionError):
        AuthorizationService.require_authenticated(None)


def test_require_admin_rejects_normal_user(normal_user):
    with pytest.raises(PermissionError):
        AuthorizationService.require_admin(normal_user)


def test_require_admin_accepts_admin(admin_user):
    AuthorizationService.require_admin(admin_user)
