"""Print a PBKDF2 password hash for adding an account to Streamlit secrets."""
from getpass import getpass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth import hash_password

if __name__ == "__main__":
    first = getpass("New password: ")
    second = getpass("Repeat password: ")
    if first != second:
        raise SystemExit("Passwords do not match.")
    print("\nCopy this complete hash into .streamlit/secrets.toml under [auth_users]:")
    print(hash_password(first))
