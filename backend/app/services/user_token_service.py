"""Short-lived tokens that carry which customer is asking.

Until now ``user_id`` was a field in the request body. The site key proved the
caller was the website; nothing proved *which* of its customers the request was
for, so a bug in the website — a stale variable, a mixed-up loop — spent one
customer's credits on another's account and read their data back. With money
attached to a plan, that is no longer a theoretical shape of bug.

A token changes where the answer comes from: the customer is read from a value
this service signed, not from a field the caller filled in. The site still asks
for the token on its customer's behalf, so it is still trusted to know who is
logged in — that cannot be removed without the site handing over its own login.
What it does remove is the ability to *keep* getting it wrong: the token is
bound to one user, expires, and any mismatch between it and the body is refused
rather than reconciled.

Deliberately not JWT. A JWT library brings an algorithm field the verifier is
supposed to police, and "alg: none" is the most repeated authentication bug in
the industry. This format has one algorithm, no negotiation, and no header a
caller can influence.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600
# Long enough that a customer is not signed out mid-carousel, short enough that a
# token found in a log or a proxy cache is worth little by the time it is read.
MAX_TTL_SECONDS = 24 * 3600


class UserTokenError(RuntimeError):
    """The token is absent, malformed, expired, or not ours."""


class UserTokenNotConfigured(RuntimeError):
    """No signing secret. Retrying will not help until an operator sets one."""


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class UserTokenService:
    def __init__(self, *, secret: str | None = None, ttl_seconds: int | None = None):
        self._secret = (secret if secret is not None else os.getenv("USER_TOKEN_SECRET", "")).strip()
        configured_ttl = int(ttl_seconds or os.getenv("USER_TOKEN_TTL_SECONDS", "") or DEFAULT_TTL_SECONDS)
        self.ttl_seconds = max(60, min(configured_ttl, MAX_TTL_SECONDS))

    def is_configured(self) -> bool:
        return bool(self._secret)

    def _sign(self, payload: bytes) -> str:
        if not self._secret:
            raise UserTokenNotConfigured(
                "USER_TOKEN_SECRET is not set. User tokens cannot be issued or verified."
            )
        return _b64(hmac.new(self._secret.encode(), payload, hashlib.sha256).digest())

    def issue(self, user_id: str, *, now: float | None = None) -> dict:
        user_id = (user_id or "").strip()
        if not user_id:
            raise UserTokenError("user_id is required to issue a token")

        issued_at = int(now if now is not None else time.time())
        expires_at = issued_at + self.ttl_seconds
        body = _b64(json.dumps({"sub": user_id, "iat": issued_at, "exp": expires_at}, separators=(",", ":")).encode())
        return {
            "user_token": f"{body}.{self._sign(body.encode())}",
            "user_id": user_id,
            "expires_at": expires_at,
            "expires_in": self.ttl_seconds,
        }

    def verify(self, token: str, *, now: float | None = None) -> str:
        """Return the user id this token was issued for, or raise.

        Every failure raises the same exception type with a generic message. A
        verifier that distinguished "bad signature" from "expired" from
        "malformed" would tell an attacker which part of a forgery to fix next.
        """
        token = (token or "").strip()
        if not token or token.count(".") != 1:
            raise UserTokenError("Invalid user token")

        body, signature = token.split(".", 1)
        # compare_digest, not ==: a plain comparison returns at the first
        # differing byte, which leaks a valid signature one character at a time
        # to anyone able to time the responses.
        if not hmac.compare_digest(signature, self._sign(body.encode())):
            raise UserTokenError("Invalid user token")

        try:
            payload = json.loads(_unb64(body))
            user_id = str(payload["sub"]).strip()
            expires_at = int(payload["exp"])
        except Exception as exc:  # noqa: BLE001 - any shape problem is one failure
            raise UserTokenError("Invalid user token") from exc

        # Checked after the signature: an expiry read from an unverified body is
        # a value the caller chose.
        if expires_at <= int(now if now is not None else time.time()):
            raise UserTokenError("Invalid user token")
        if not user_id:
            raise UserTokenError("Invalid user token")

        return user_id
