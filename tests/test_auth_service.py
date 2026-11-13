
import pytest

from src.services.auth_service import AuthService


def test_password_is_hashed():
    password = "MySecurePassword123!"
    hashed = AuthService.hash_password(password)

    assert hashed != password
    assert hashed.startswith("$2b$")


def test_correct_password_verifies():
    password = "MySecurePassword123!"
    hashed = AuthService.hash_password(password)

    assert AuthService.verify_password(password, hashed) is True


def test_wrong_password_fails():
    hashed = AuthService.hash_password("CorrectPassword123!")

    assert AuthService.verify_password("WrongPassword123!", hashed) is False


def test_hashes_are_unique():
    password = "MySecurePassword123!"

    hash1 = AuthService.hash_password(password)
    hash2 = AuthService.hash_password(password)

    assert hash1 != hash2


def test_empty_password_rejected():
    with pytest.raises(ValueError):
        AuthService.hash_password("")


def test_non_string_password_rejected():
    with pytest.raises(TypeError):
        AuthService.hash_password(12345)


def test_long_password_rejected():
    with pytest.raises(ValueError):
        AuthService.hash_password("a" * 73)
