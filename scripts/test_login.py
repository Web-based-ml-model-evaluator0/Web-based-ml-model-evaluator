
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import getpass

from src.services.auth_service import authenticate_user


def main():
    """Manually test the user login workflow."""

    username = input("Enter username or email: ").strip()
    password = getpass.getpass("Enter password: ")

    user = authenticate_user(username, password)

    if user:
        print("\nLogin successful!")
        print("Authenticated user:", user["username"])
        print("Role:", user["role"])
    else:
        print("\nLogin failed. Invalid credentials.")


if __name__ == "__main__":
    main()
