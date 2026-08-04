"""Shared guard for privileged, operator-only endpoints.

This lived inside ``app/api/routes/knowledge_packs.py``, which meant the billing
and maintenance routers could only reuse it by importing from a sibling route
module. It is the same check in all three places, so it belongs here.

Scope note: this authenticates the *operator*, not the end user. It closes
privilege escalation (granting yourself a plan, resetting everyone's usage). It
does not establish who the calling end user is -- ``user_id`` is still a trusted
request parameter everywhere, which remains an open decision.
"""
import os

from fastapi import HTTPException, Request

INTERNAL_ADMIN_HEADER = "X-Internal-Admin-Key"


def require_internal_admin_access(request: Request) -> None:
    """Reject the request unless it carries the configured internal admin key.

    Fails closed: when ``INTERNAL_ADMIN_KEY`` is unset or blank every request is
    denied, so a missing configuration cannot silently leave the endpoint open.
    """
    expected_key = os.getenv("INTERNAL_ADMIN_KEY", "").strip()
    provided_key = request.headers.get(INTERNAL_ADMIN_HEADER, "").strip()
    if not expected_key or provided_key != expected_key:
        raise HTTPException(status_code=403, detail="Internal admin access is required")
