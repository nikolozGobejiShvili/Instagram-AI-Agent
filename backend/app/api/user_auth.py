"""Resolving which customer a request is for.

The rule every money- or data-touching route follows: **the token decides the
user, the body does not.** When both are present and disagree, the request is
refused rather than reconciled — silently preferring one would turn a website
bug into a cross-account read that still returns 200.

Enforcement is opt-in per deployment via ``USER_TOKEN_SECRET``. With no secret
set, routes fall back to the body's ``user_id`` and behave exactly as before,
which is what keeps the currently-deployed service working while the website
that will issue tokens does not exist yet. Startup logs which mode is active,
because "we thought tokens were on" is the failure this note exists to prevent.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from app.services.user_token_service import UserTokenError, UserTokenService

logger = logging.getLogger(__name__)

USER_TOKEN_HEADER = "X-User-Token"

user_token_service = UserTokenService()


def enforcement_enabled() -> bool:
    # Re-read per call rather than cached at import: tests and operators change
    # the secret at runtime, and a cached answer would report the wrong mode.
    return UserTokenService().is_configured()


def resolve_user_id(request: Request, body_user_id: str | None) -> str:
    """The customer this request is for.

    Raises 401 when enforcement is on and the token is missing or invalid, and
    403 when a valid token names a different customer than the body.
    """
    service = UserTokenService()

    if not service.is_configured():
        if not body_user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        return body_user_id

    provided = request.headers.get(USER_TOKEN_HEADER, "").strip()
    if not provided:
        raise HTTPException(
            status_code=401,
            detail=f"A valid {USER_TOKEN_HEADER} header is required.",
        )

    try:
        token_user_id = service.verify(provided)
    except UserTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired user token") from exc

    if body_user_id and body_user_id != token_user_id:
        # Refused, not reconciled. A website that sends the wrong id is broken,
        # and answering for whichever id we preferred would hide that while
        # charging somebody.
        logger.warning(
            "Rejected request whose user token and body disagree token_user=%s body_user=%s",
            token_user_id,
            body_user_id,
        )
        raise HTTPException(
            status_code=403,
            detail="The user token does not match the user_id in this request.",
        )

    return token_user_id
