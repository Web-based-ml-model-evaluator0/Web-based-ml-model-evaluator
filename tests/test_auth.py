import unittest
from auth import authenticate_credentials, hash_password, verify_password

class PasswordHashTests(unittest.TestCase):
    def test_password_hash_verifies_correct_password(self):
        stored = hash_password("correct horse battery staple", iterations=100_000, salt="test-salt")
        self.assertTrue(verify_password("correct horse battery staple", stored))
    def test_wrong_password_is_rejected(self):
        stored = hash_password("correct-password", iterations=100_000, salt="test-salt")
        self.assertFalse(verify_password("wrong-password", stored))
    def test_malformed_hash_is_rejected(self):
        self.assertFalse(verify_password("anything", "not-a-valid-hash"))
    def test_low_iteration_hash_is_rejected(self):
        self.assertFalse(verify_password("anything", "pbkdf2_sha256$1$salt$digest"))
    def test_valid_registered_user_is_accepted(self):
        users = {"student": hash_password("student-password", iterations=100_000, salt="student-salt")}
        self.assertTrue(authenticate_credentials("student", "student-password", users))
    def test_invalid_username_or_password_is_rejected(self):
        users = {"student": hash_password("student-password", iterations=100_000, salt="student-salt")}
        self.assertFalse(authenticate_credentials("unknown", "student-password", users))
        self.assertFalse(authenticate_credentials("student", "wrong", users))
        self.assertFalse(authenticate_credentials(" ", "student-password", users))

if __name__ == "__main__":
    unittest.main()
