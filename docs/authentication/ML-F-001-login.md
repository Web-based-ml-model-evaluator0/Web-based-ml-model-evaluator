# User login setup (ML-F-001)

## Included
- Login form before the evaluator.
- Registered users only; unknown usernames and incorrect passwords receive the same generic error.
- PBKDF2-HMAC-SHA256 password-hash verification (no plaintext passwords in source).
- Session-scoped authentication and a logout button.
- Unit tests for valid/invalid credentials and malformed hashes.

## Configure accounts
1. Run `python scripts/hash_password.py` and enter the password twice.
2. Create `.streamlit/secrets.toml` locally (do not commit it):
   ```toml
   [auth_users]
   student = "PASTE_THE_COMPLETE_HASH_PRINTED_BY_THE_SCRIPT"
   ```
3. Replace the example with the complete generated hash and add one entry per registered user.
4. For deployment, configure the same `auth_users` mapping in the hosting provider's Streamlit secrets settings.

The repository does not include self-service registration, password reset, email verification, or a central account database. An administrator provisions accounts. Never commit plaintext passwords or `.streamlit/secrets.toml`.

## Run tests
```bash
python -m unittest discover -s tests -p "test_auth.py" -v
```

## Acceptance checks (TC-AUTH-01)
- Valid configured username/password: evaluator opens.
- Wrong password and unknown username: login rejected with generic error.
- Empty username: login rejected.
- Logout: evaluator is hidden again until successful login.
- Missing `auth_users` configuration: no login succeeds; configure accounts before deploying.

## Limitations / review
This is a minimal project-level login gate, not a complete identity-management system. Users are provisioned via secrets. For production/public deployment, review an established OIDC identity provider and deployment-level access controls. Issue #1 / #36 was previously assigned to another teammate, so this branch is separate for team review before merging.
