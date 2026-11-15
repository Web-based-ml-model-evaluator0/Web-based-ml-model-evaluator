"""
Security Configuration module (ML-SR-001, TC-SEC-01).
Provides TLS/HTTPS enforcement, certificate inspection, and secure transport validation.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import ssl


class TLSSecurityError(ValueError):
    """Raised when TLS/HTTPS security requirements are violated (ML-SR-001)."""
    pass


class SecurityConfigService:
    """
    Manages HTTPS/TLS transport security, configuration validation,
    and cipher/protocol enforcement according to SAD 3.1 & 3.4 (ML-SR-001 / TC-SEC-01).
    """

    MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2

    @classmethod
    def validate_tls_config(
        cls,
        cert_file: Optional[Path | str] = None,
        key_file: Optional[Path | str] = None,
        require_https: bool = True,
    ) -> Dict[str, Any]:
        """
        Validates SSL certificate and private key files for HTTPS/TLS communication.
        Ensures HTTP-only access is rejected when require_https=True.
        """
        if require_https and (not cert_file or not key_file):
            raise TLSSecurityError(
                "Production HTTPS/TLS requires valid SSL certificate and key files. "
                "HTTP-only access is not permitted."
            )

        config: Dict[str, Any] = {
            "https_enabled": bool(cert_file and key_file),
            "cert_file": str(cert_file) if cert_file else None,
            "key_file": str(key_file) if key_file else None,
            "min_tls_version": "TLSv1.2",
            "enforce_https": require_https,
        }

        if cert_file:
            cert_path = Path(cert_file)
            if not cert_path.exists():
                raise TLSSecurityError(f"SSL certificate file not found: '{cert_path}'")

        if key_file:
            key_path = Path(key_file)
            if not key_path.exists():
                raise TLSSecurityError(f"SSL private key file not found: '{key_path}'")

        return config

    @classmethod
    def create_ssl_context(
        cls,
        cert_file: Optional[Path | str] = None,
        key_file: Optional[Path | str] = None,
    ) -> ssl.SSLContext:
        """
        Builds a hardened SSLContext enforcing modern TLS (TLS 1.2+).
        Enforces minimum protocol version TLSv1.2 natively according to modern SSL standards.
        """
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.minimum_version = cls.MIN_TLS_VERSION

        if cert_file and key_file:
            ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))

        return ctx

    @classmethod
    def get_streamlit_server_tls_args(cls, cert_file: str, key_file: str) -> List[str]:
        """
        Returns CLI parameters for launching Streamlit with strict TLS enforcement.
        """
        cls.validate_tls_config(cert_file=cert_file, key_file=key_file, require_https=True)
        return [
            f"--server.sslCertFile={cert_file}",
            f"--server.sslKeyFile={key_file}",
        ]
