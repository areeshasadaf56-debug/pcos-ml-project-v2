"""
auth.py

Account endpoints + real session tokens for the Flutter app
(see lib/providers/cycle_provider.dart: signUp / signIn / resetPassword).

Design notes for your viva:
  - Passwords are hashed with bcrypt -- a slow, salted hash designed
    specifically for passwords, unlike a fast general-purpose hash
    like SHA-256.
  - Sign up/sign in/reset now issue an opaque session token (a random
    32-byte value, not a JWT -- there's nothing that needs to be
    *decoded* client-side, so a random token looked up in the
    `sessions` table is simpler and just as secure). The app stores
    this token and sends it back as `Authorization: Bearer <token>`
    on every request that touches personal data (/profile, /chat).
    main.py's `get_current_user_id` dependency verifies it.
  - Rate limiting on sign-in/sign-up/reset is a small in-memory
    sliding-window counter (see `_check_rate_limit` below). It resets
    if the server restarts and doesn't share state across multiple
    worker processes -- both fine for a single-process PythonAnywhere
    deployment, but call this out as a known limitation if you ever
    move to multiple workers/instances (the real fix there is Redis-
    backed rate limiting).

SECURITY NOTES still worth being upfront about:
  1. /reset_password has no email-based verification step (no email
     service is wired up), so it can't confirm the person resetting
     the password is actually the account owner -- it can only rate
     limit attempts and avoid confirming *whether* an email exists.
     A real fix needs an emailed one-time code/link; flag this as
     known future work, not silently ignored.
  2. Sessions don't rotate/refresh -- a token is valid for its full
     lifetime (30 days) or until logout/password reset. Fine for an
     FYP; a production app would add refresh tokens.
"""

import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import bcrypt

import database

# Using the `bcrypt` package directly rather than passlib's CryptContext
# wrapper. passlib 1.7.4 (its latest release) has a known incompatibility
# with bcrypt>=4.0 -- its internal self-test misreads bcrypt's version
# and raises a spurious "password cannot be longer than 72 bytes" error
# even for short passwords. Calling bcrypt directly sidesteps that
# entirely and is one fewer dependency.

SESSION_TTL_DAYS = 30


class AuthError(Exception):
    """Raised for any expected auth failure; main.py turns this into
    the appropriate HTTP status code + `detail` message the Flutter
    app already expects (see cycle_provider.dart: `body['detail']`)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ---------------------------------------------------------------
# Rate limiting -- simple in-memory sliding window per key
# ---------------------------------------------------------------

_attempts: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    now = time.time()
    dq = _attempts[key]
    while dq and now - dq[0] > window_seconds:
        dq.popleft()
    if len(dq) >= max_attempts:
        raise AuthError(
            429, "Too many attempts. Please wait a few minutes and try again."
        )
    dq.append(now)


def _clear_rate_limit(key: str) -> None:
    _attempts.pop(key, None)


# ---------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------

def _issue_session(user_id: int) -> tuple[str, str]:
    """Creates and stores a new session token for user_id. Returns
    (token, expires_at_iso)."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
    expires_at_iso = expires_at.isoformat()
    database.create_session(token, user_id, expires_at_iso)
    return token, expires_at_iso


def verify_token(token: str) -> int:
    """Returns the user_id for a valid, unexpired token. Raises
    AuthError(401) otherwise."""
    session = database.get_session(token)
    if session is None:
        raise AuthError(401, "Your session has expired. Please sign in again.")
    return session["user_id"]


def invalidate_session(token: str) -> None:
    database.delete_session(token)


# ---------------------------------------------------------------
# Account endpoints
# ---------------------------------------------------------------

def sign_up(name: str, email: str, password: str, client_key: str) -> dict:
    """Creates a new account and an initial session. Returns
    {user_id, name, token, expires_at} on success."""
    _check_rate_limit(f"signup:{client_key}", max_attempts=5, window_seconds=600)

    if len(password) < 6:
        raise AuthError(422, "Password must be at least 6 characters.")

    if database.get_user_by_email(email) is not None:
        raise AuthError(409, "An account with this email already exists.")

    user = database.create_user(name.strip(), email, hash_password(password))
    token, expires_at = _issue_session(user["id"])
    return {
        "user_id": user["id"],
        "name": user["name"],
        "token": token,
        "expires_at": expires_at,
    }


def sign_in(email: str, password: str, client_key: str) -> dict:
    """Validates credentials and issues a session. Returns
    {user_id, name, token, expires_at} on success."""
    rate_key = f"signin:{client_key}:{email}"
    _check_rate_limit(rate_key, max_attempts=8, window_seconds=300)

    user = database.get_user_by_email(email)
    if user is None or not verify_password(password, user["password_hash"]):
        # Deliberately the SAME message for "no such email" and "wrong
        # password" -- distinguishing them lets an attacker enumerate
        # which emails have accounts.
        raise AuthError(401, "Incorrect email or password.")

    _clear_rate_limit(rate_key)
    token, expires_at = _issue_session(user["id"])
    return {
        "user_id": user["id"],
        "name": user["name"],
        "token": token,
        "expires_at": expires_at,
    }


def reset_password(email: str, new_password: str, client_key: str) -> None:
    """Resets the password if the email exists. Always returns
    normally (never reveals whether the email was found) -- the
    caller in main.py always responds with the same generic
    {"status": "ok"}, so this can't be used to enumerate accounts.

    Known limitation: with no email-verification step, this endpoint
    can't confirm the caller owns the account -- it can only be rate
    limited. See module docstring."""
    _check_rate_limit(f"reset:{client_key}:{email}", max_attempts=5, window_seconds=900)

    if len(new_password) < 6:
        raise AuthError(422, "Password must be at least 6 characters.")

    user = database.get_user_by_email(email)
    if user is None:
        return  # Silently no-op -- see docstring above.

    database.update_password(email, hash_password(new_password))
    # Force every existing session for this account to re-authenticate
    # with the new password.
    database.delete_all_sessions_for_user(user["id"])