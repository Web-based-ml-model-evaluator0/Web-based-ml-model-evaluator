import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import getpass

from src.services.auth_service import (
    initialize_database,
    create_user
)


def main():
    initialize_database()

    print("Create a registered user account")

    username = input("Username: ").strip()
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Confirm password: ")

    if password != confirm_password:
        print("Passwords do not match.")
        return

    try:
        create_user(username, email, password, role="admin")
        print("User account created successfully.")
    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
