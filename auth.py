"""Authentication helpers for the Streamlit ML evaluator."""
from __future__ import annotations
import hashlib, hmac, secrets
from typing import Mapping

ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 310_000

def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS, salt: str | None = None) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    if iterations < 100_000:
        raise ValueError("Use at least 100,000 PBKDF2 iterations.")
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_value.encode("utf-8"), iterations).hex()
    return f"{ALGORITHM}${iterations}${salt_value}${digest}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iteration_text, salt, expected = stored_hash.split("$", 3)
        iterations = int(iteration_text)
        if algorithm != ALGORITHM or iterations < 100_000 or not salt:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
        return hmac.compare_digest(actual, expected)
    except (AttributeError, TypeError, ValueError):
        return False

def authenticate_credentials(username: str, password: str, users: Mapping[str, str]) -> bool:
    if not isinstance(username, str) or not isinstance(password, str):
        return False
    normalized = username.strip()
    if not normalized:
        return False
    stored_hash = users.get(normalized)
    if not stored_hash:
        hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), b"unknown-user-timing-salt", DEFAULT_ITERATIONS)
        return False
    return verify_password(password, stored_hash)

def show_login_gate(st) -> None:
    try:
        users = st.secrets.get("auth_users", {})
    except Exception:
        # Streamlit raises when no secrets file/provider configuration exists.
        users = {}
    if not st.session_state.get("authenticated", False):
        st.title("Sign in")
        st.caption("Use the account provided by your project administrator.")
        if not users:
            st.warning("Login is not configured yet. An administrator must add accounts to Streamlit secrets.")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            if authenticate_credentials(username, password, users):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.stop()

def show_logout_button(st) -> None:
    st.sidebar.caption(f"Signed in as: {st.session_state.get('username', 'user')}")
    if st.sidebar.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()
