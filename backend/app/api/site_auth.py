"""Authenticates the *website* that fronts this API.

The API is about to be reachable on a public domain, and ``user_id`` is still a
plain request parameter: without a guard, anyone who found the hostname could
read and spend any customer's account by naming it. This closes that.

It is middleware rather than a dependency on each router deliberately. A
dependency has to be remembered -- a router added later ships open, and the
mistake is invisible because everything still works. A path-prefixed middleware
is covered by default and has to be *explicitly* exempted instead.

Two things this is not:

* It is not end-user authentication. It proves the caller is the website, not
  which customer is asking. The website is responsible for establishing that and
  for sending the right ``user_id``.
* It is not a browser credential. A key shipped to browser JavaScript is
  readable by anyone who opens devtools, so this belongs on the website's
  *server*: browser -> website backend -> this API. That is also why no CORS
  middleware is configured -- allowing cross-origin calls would invite exactly
  the pattern that leaks the key.

Privileged operator routes keep their separate ``X-Internal-Admin-Key`` check on
top of this (see ``internal_auth.py``), so a leaked site key still cannot grant
itself a plan.
"""
from __future__ import annotations

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

SITE_KEY_HEADER = "X-Site-Key"
GUARDED_PREFIX = "/api/v1"

# Rendered carousel images are loaded by the browser in <img src="...">, which
# cannot carry a custom header. They are capability URLs instead: the asset id is
# 128 bits of opaque randomness minted per render, so the URL is the credential.
# Guarding this path would not add security, it would just stop the images from
# appearing.
EXEMPT_PREFIXES = (
    "/api/v1/carousel-media/",
    # Stripe cannot send a site key. Its request is authenticated by the
    # signature over the raw body instead, which is stronger -- it proves the
    # payload came from Stripe unmodified, where a shared header only proves the
    # caller had it. See payment_service.verify_signature, which refuses every
    # event when no signing secret is set precisely because this path is open.
    "/api/v1/payments/stripe/webhook",
)


def site_key() -> str:
    return os.getenv("SITE_API_KEY", "").strip()


def is_guarded(path: str) -> bool:
    if not path.startswith(GUARDED_PREFIX):
        return False
    return not path.startswith(EXEMPT_PREFIXES)


def _refuse(status_code: int, code: str, message: str) -> JSONResponse:
    # Matches app.error_handling's envelope by hand: middleware runs outside the
    # exception handlers, so raising HTTPException here would escape them and
    # return a shape no client is written against.
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": None,
                "task_type": None,
                "account_id": None,
            }
        },
    )


class SiteKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not is_guarded(request.url.path):
            return await call_next(request)

        expected = site_key()
        if not expected:
            # Fails closed. An unset key denying every request is a loud,
            # immediate outage; an unset key waving every request through is a
            # silent one, and the whole point of this module is that the service
            # is publicly reachable.
            logger.error("SITE_API_KEY is not configured; refusing %s", request.url.path)
            return _refuse(503, "site_auth_not_configured", "This API is not accepting requests.")

        provided = request.headers.get(SITE_KEY_HEADER, "").strip()
        # compare_digest, not ==: a plain comparison returns as soon as two bytes
        # differ, which leaks the key one character at a time to anyone timing
        # the responses.
        if not hmac.compare_digest(provided, expected):
            return _refuse(401, "site_auth_required", f"A valid {SITE_KEY_HEADER} header is required.")

        return await call_next(request)
