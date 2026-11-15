"""
Unit test for Issue #29 (ML-SR-001 HTTPS/TLS Communication, TC-SEC-01).
Tests TLS/HTTPS enforcement, rejection of HTTP-only access in production,
hardened SSL context protocols (TLS 1.2+ minimum), and Streamlit TLS arguments.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import ssl

from src.services.security_service import SecurityConfigService, TLSSecurityError


class TestTLSSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cert_path = Path(self.temp_dir) / "cert.pem"
        self.key_path = Path(self.temp_dir) / "key.pem"

        # Create dummy cert and key files for testing existence validation
        self.cert_path.write_text("-----BEGIN CERTIFICATE-----\nDUMMY\n-----END CERTIFICATE-----")
        self.key_path.write_text("-----BEGIN PRIVATE KEY-----\nDUMMY\n-----END PRIVATE KEY-----")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_https_enabled_with_valid_files(self):
        """TC-SEC-01 Step 1: Valid certificate and key paths enable HTTPS successfully."""
        config = SecurityConfigService.validate_tls_config(
            cert_file=self.cert_path,
            key_file=self.key_path,
            require_https=True,
        )
        self.assertTrue(config["https_enabled"])
        self.assertEqual(config["min_tls_version"], "TLSv1.2")
        self.assertTrue(config["enforce_https"])

    def test_http_only_rejected_when_https_required(self):
        """TC-SEC-01 Step 2: HTTP-only access is strictly rejected in production mode."""
        with self.assertRaises(TLSSecurityError) as ctx:
            SecurityConfigService.validate_tls_config(
                cert_file=None,
                key_file=None,
                require_https=True,
            )
        self.assertIn("not permitted", str(ctx.exception).lower())

    def test_missing_cert_file_raises_error(self):
        """TC-SEC-01 Step 3: Nonexistent certificate file raises TLSSecurityError."""
        with self.assertRaises(TLSSecurityError) as ctx:
            SecurityConfigService.validate_tls_config(
                cert_file="nonexistent_cert.pem",
                key_file=self.key_path,
                require_https=True,
            )
        self.assertIn("not found", str(ctx.exception).lower())

    def test_ssl_context_enforces_tls_1_2_minimum(self):
        """TC-SEC-01 Step 4: SSL context enforces TLSv1.2 minimum and disables insecure protocols."""
        ctx = SecurityConfigService.create_ssl_context()
        self.assertEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertGreaterEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)

    def test_streamlit_tls_cli_args_generation(self):
        """TC-SEC-01 Step 5: Generates compliant CLI arguments for Streamlit TLS server configuration."""
        args = SecurityConfigService.get_streamlit_server_tls_args(
            cert_file=str(self.cert_path),
            key_file=str(self.key_path),
        )
        self.assertEqual(len(args), 2)
        self.assertIn(f"--server.sslCertFile={self.cert_path}", args)
        self.assertIn(f"--server.sslKeyFile={self.key_path}", args)


if __name__ == "__main__":
    unittest.main()
