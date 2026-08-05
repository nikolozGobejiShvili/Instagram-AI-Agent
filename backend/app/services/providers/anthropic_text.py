"""Anthropic text provider — Claude Sonnet 4.6.

This is the product's analysis and text model. Model id, thinking configuration
and effort placement follow the current Anthropic API contract:

* ``claude-sonnet-4-6`` is the complete model id -- never append a date suffix.
* Sonnet 4.6 uses **adaptive thinking** (``{"type": "adaptive"}``). The older
  ``{"type": "enabled", "budget_tokens": N}`` form is deprecated on this model;
  depth is controlled with effort instead.
* ``effort`` is nested inside ``output_config``, not a top-level parameter.

Only Anthropic SDK calls live in this module.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.services.providers.anthropic_auth import (
    apply_billing_attribution,
    auth_kind,
    auth_token,
    client_kwargs,
)
from app.services.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    TextProvider,
    sections_to_system_and_messages,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Cheap tasks don't need deep reasoning; audits and plans do. Effort is the
# cost/quality dial on 4.6 -- `medium` is the balanced default.
EFFORT_BY_TASK: dict[str, str] = {
    "chat": "low",
    "caption": "low",
    "reel_idea": "medium",
    "reel_script": "medium",
    "reel_feedback": "medium",
    "carousel": "medium",
    "performance_summary": "medium",
    "link_analysis": "high",
    "profile_audit": "high",
    "content_plan": "high",
}
DEFAULT_EFFORT = "medium"


class AnthropicTextProvider(TextProvider):
    name = PROVIDER_NAME

    def __init__(self, client: Any | None = None, model_name: str | None = None):
        # The client is injectable so tests can drive a stub transport without
        # credentials or network access.
        self._client = client
        self.model_name = model_name or os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        token = auth_token()
        if not token:
            raise ProviderNotConfigured(PROVIDER_NAME, missing="ANTHROPIC_AUTH_TOKEN")

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise ProviderError(
                "Anthropic support is not installed.",
                provider=PROVIDER_NAME,
                status_code=503,
                code="provider_not_configured",
            ) from exc

        # A setup token authenticates as Bearer + OAuth beta; an API key uses
        # x-api-key. The prefix decides, so switching the credential shape is
        # a configuration change, never a code change.
        logger.info("[anthropic] model=%s auth=%s", self.model_name, auth_kind(token))
        self._client = anthropic.Anthropic(**client_kwargs(token))
        return self._client

    def effort_for(self, task_type: str) -> str:
        return EFFORT_BY_TASK.get(task_type, DEFAULT_EFFORT)

    def generate(
        self,
        *,
        task_type: str,
        response_input: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        system_prompt, messages = sections_to_system_and_messages(response_input)
        client = self._get_client()

        # Billing attribution must ride on EVERY call path. A setup token that
        # reaches this line without it answers 400 for Sonnet and Opus while
        # still working for Haiku — a failure that looks like a model problem.
        system_prompt = apply_billing_attribution(system_prompt, auth_token())

        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_output_tokens,
            "messages": messages,
            # Adaptive thinking: Claude decides depth per request. `budget_tokens`
            # is deprecated on 4.6 and removed on later models.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort_for(task_type)},
        }
        if system_prompt:
            request["system"] = system_prompt

        try:
            message = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise self._as_provider_error(exc) from exc

        return {
            "reply": self._extract_text(message),
            "model_provider": PROVIDER_NAME,
            "model_name": getattr(message, "model", self.model_name),
            "stop_reason": getattr(message, "stop_reason", None),
            "usage": self._extract_usage(message),
        }

    # ------------------------------------------------------------------
    # response handling
    # ------------------------------------------------------------------
    def _extract_text(self, message: Any) -> str:
        """Concatenate text blocks, skipping thinking blocks.

        ``content`` is a list of typed blocks, not a string. Thinking blocks come
        first when adaptive thinking runs, so indexing ``content[0]`` would return
        reasoning instead of the answer.
        """
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)

        reply = "\n".join(parts).strip()
        if not reply:
            stop_reason = getattr(message, "stop_reason", None)
            if stop_reason == "refusal":
                raise ProviderError(
                    "The model declined this request.",
                    provider=PROVIDER_NAME,
                    model_name=self.model_name,
                    status_code=422,
                    code="generation_refused",
                )
            raise ProviderError(
                "AI generation returned no content.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
            )
        return reply

    def _extract_usage(self, message: Any) -> dict[str, int]:
        usage = getattr(message, "usage", None)
        if usage is None:
            return {}
        return {
            key: int(value)
            for key in ("input_tokens", "output_tokens")
            if isinstance(value := getattr(usage, key, None), int)
        }

    def _as_provider_error(self, exc: Exception) -> ProviderError:
        """Map SDK exceptions to a caller-safe error.

        Typed SDK exception classes are used rather than string matching, so this
        keeps working when Anthropic edits a message.
        """
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return ProviderError(
                "AI generation failed. Please try again shortly.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
            )

        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(
                "AI generation is rate limited. Please try again shortly.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
                status_code=503,
                code="llm_rate_limited",
            )
        if isinstance(exc, anthropic.AuthenticationError):
            return ProviderError(
                "AI generation is not configured correctly.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
                status_code=503,
                code="provider_not_configured",
            )
        if isinstance(exc, anthropic.APIStatusError):
            logger.warning("Anthropic API error status=%s", getattr(exc, "status_code", None))
            return ProviderError(
                "AI generation failed. Please try again shortly.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(
                "AI generation could not reach the model provider.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
                status_code=504,
                code="llm_unreachable",
            )

        logger.exception("Unexpected Anthropic failure")
        return ProviderError(
            "AI generation failed. Please try again shortly.",
            provider=PROVIDER_NAME,
            model_name=self.model_name,
        )
