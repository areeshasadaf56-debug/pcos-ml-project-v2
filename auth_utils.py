"""
auth_utils.py

Shared session-token verification and rate limiting, used by
main_flask.py, health_profile_api.py, and chat_api.py.

Split into its own module rather than living in main_flask.py because
health_profile_api.py and chat_api.py are imported BY main_flask.py --
putting shared auth code there would create a circular import.

Tokens are opaque random strings (not JWTs) looked up in the
`sessions` table below -- there's nothing that needs to be *decoded*
client-side, so a random token looked up in SQLite is simpler and
just as secure. This table lives in the same accounts.db file that
main_flask.py's `accounts` table uses, so a session's account_user_id
can be looked up in the same place it was issued.

SECURITY NOTE for your viva: rate limiting below is a small in-memory
sliding-window counter. It resets if the server restarts and doesn't
share state across multiple worker processes -- both fine for a
single-process PythonAnywhere deployment, but call this out as a
known limitation if you ever move to multiple workers/instances (the
real fix there is Redis-backed rate limiting).
"""

import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import g, jsonify, request

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.db")
SESSION_TTL_DAYS = 30


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_sessions_table():
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            account_user_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# Run at import time, same pattern as health_profile_api.py's init_db().
_init_sessions_table()


# ---------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------

def issue_session(account_user_id: str) -> tuple[str, str]:
    """Creates and stores a new session token. Returns (token, expires_at_iso)."""
    token = secrets.token_urlsafe(32)
    expires_at_iso = (
        datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)
    ).isoformat()

    conn = _get_connection()
    conn.execute(
        "INSERT INTO sessions (token, account_user_id, expires_at) VALUES (?, ?, ?)",
        (token, account_user_id, expires_at_iso),
    )
    # Opportunistic cleanup -- piggybacks on every login instead of
    # needing a separate cron job for an app this size.
    conn.execute("DELETE FROM sessions WHERE expires_at < datetime('now')")
    conn.commit()
    conn.close()
    return token, expires_at_iso


def verify_token(token: str) -> str | None:
    """Returns the account_user_id for a valid, unexpired token, or
    None if the token is missing/expired."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT account_user_id FROM sessions WHERE token = ? AND expires_at > datetime('now')",
        (token,),
    ).fetchone()
    conn.close()
    return row["account_user_id"] if row else None


def invalidate_session(token: str) -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def invalidate_all_sessions_for_account(account_user_id: str) -> None:
    """Used on password reset -- forces every other device/session for
    this account to sign in again with the new password."""
    conn = _get_connection()
    conn.execute("DELETE FROM sessions WHERE account_user_id = ?", (account_user_id,))
    conn.commit()
    conn.close()


def require_auth(f):
    """Route decorator: verifies `Authorization: Bearer <token>` and
    stashes the resulting account_user_id on flask.g.current_user_id
    for the route to use. Returns 401 if missing/invalid/expired.
    Lets CORS preflight (OPTIONS) requests through untouched."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({
                "detail": "Missing or invalid Authorization header. Please sign in again."
            }), 401

        token = auth_header[len("Bearer "):].strip()
        account_user_id = verify_token(token)
        if account_user_id is None:
            return jsonify({"detail": "Your session has expired. Please sign in again."}), 401

        g.current_user_id = account_user_id
        return f(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------
# Rate limiting -- simple in-memory sliding window per key
# ---------------------------------------------------------------

class RateLimitError(Exception):
    def __init__(self, detail: str):
        self.detail = detail


_attempts: dict[str, deque] = defaultdict(deque)


def check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    now = time.time()
    dq = _attempts[key]
    while dq and now - dq[0] > window_seconds:
        dq.popleft()
    if len(dq) >= max_attempts:
        raise RateLimitError("Too many attempts. Please wait a few minutes and try again.")
    dq.append(now)


def clear_rate_limit(key: str) -> None:
    _attempts.pop(key, None)