"""Anthropic credential handling — setup tokens and API keys.

Ported from the sityvis-akademia-ai-agent project so both agents configure
Claude the same way. ``claude setup-token`` mints an OAuth token
(``sk-ant-oat01-…``) that bills against a Claude Pro/Max **subscription**
rather than pay-per-token API credits, so which credential shape is in use is
a billing decision, not a code one.

One variable accepts either shape and the prefix decides the transport:

  * ``sk-ant-oat*``  → ``Authorization: Bearer`` + the OAuth beta header
  * ``sk-ant-api*``  → the classic ``x-api-key`` header

Three details are load-bearing. Each one fails in a way that does not point at
its own cause:

1. **Billing attribution goes in the SYSTEM PROMPT, not a header.** Despite
   its ``x-anthropic-…`` name, :data:`CC_BILLING_HEADER` is prepended as the
   first line of the system string. Without it an OAuth token only reaches
   Haiku — Sonnet and Opus answer 400. It has to be applied on every call
   path; miss one and exactly that path breaks. API keys skip it.

2. **Tokens pasted from a clipboard carry invisible characters.** Zero-width
   marks, non-breaking spaces and stray newlines corrupt the auth header and
   the API replies 401 without mentioning whitespace. :func:`sanitize_token`
   strips everything outside printable ASCII. (This is not hypothetical: a
   BOM smuggled into an env var through a shell pipe cost us an afternoon on
   this project's own admin key.)

3. **The token is never logged** — not the value, not its length. Log lines
   carry the credential *kind* only.
"""
from __future__ import annotations

import os

#: Setup tokens (``claude setup-token``) start with this prefix. Anything else
#: is treated as a classic API key.
OAUTH_TOKEN_PREFIX = "sk-ant-oat"

#: Beta features requested for OAuth tokens. ``oauth-2025-04-20`` is what makes
#: the Bearer credential acceptable at all; prompt caching rides along because
#: it is free to ask for and cheap when it hits.
OAUTH_BETA = "prompt-caching-2024-07-31,oauth-2025-04-20"

#: Prepended to the system prompt for OAuth tokens ONLY. Not an HTTP header,
#: despite the name — see point (1) in the module docstring.
CC_BILLING_HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.63.0a5; "
    "cc_entrypoint=cli; cch=00000;"
)


def sanitize_token(raw: str | None) -> str:
    """Strip everything outside printable ASCII, then trim.

    Clipboard round-trips (terminal → browser → dashboard) routinely inject
    zero-width spaces, non-breaking spaces and trailing newlines. Any of them
    makes the Authorization header invalid and the API replies 401 without
    hinting at the cause.
    """
    if not raw:
        return ""
    return "".join(ch for ch in raw if 32 <= ord(ch) <= 126).strip()


def auth_token() -> str:
    """The configured credential, sanitized. Empty string when unset.

    ``ANTHROPIC_AUTH_TOKEN`` is the documented name; ``ANTHROPIC_API_KEY`` is
    accepted as an alias so an existing deployment keeps working.
    """
    return sanitize_token(
        os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    )


def is_setup_token(token: str | None) -> bool:
    """True when *token* is a ``claude setup-token`` OAuth credential."""
    return bool(token) and token.startswith(OAUTH_TOKEN_PREFIX)


def auth_kind(token: str | None = None) -> str:
    """Non-secret description of the active credential, safe to log."""
    resolved = auth_token() if token is None else token
    if not resolved:
        return "none"
    return "setup-token" if is_setup_token(resolved) else "api-key"


def is_configured() -> bool:
    return bool(auth_token())


def client_kwargs(token: str) -> dict:
    """Constructor arguments for ``anthropic.Anthropic`` for *token*.

    A setup token authenticates as a Bearer credential and needs the OAuth
    beta; an API key uses the standard header. The SDK exposes both, so this
    stays on the official client rather than hand-rolling HTTP.
    """
    if is_setup_token(token):
        return {
            "auth_token": token,
            "default_headers": {"anthropic-beta": OAUTH_BETA},
        }
    return {"api_key": token}


def apply_billing_attribution(system_prompt: str, token: str) -> str:
    """Prepend the Claude Code billing attribution for OAuth tokens.

    Required server-side for a setup token to reach Sonnet/Opus — without it
    every model except Haiku answers 400. API keys pass through untouched.
    """
    if not is_setup_token(token):
        return system_prompt
    if not system_prompt:
        return CC_BILLING_HEADER
    return f"{CC_BILLING_HEADER}\n{system_prompt}"
