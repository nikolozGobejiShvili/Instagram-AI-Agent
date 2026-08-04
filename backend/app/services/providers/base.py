"""Provider interfaces for text and image generation.

Before this package, `LLMService.run_agent` raised outright for any provider that
was not OpenAI, so there was nowhere for a second model to attach (audit,
2026-08-04, finding 7). The product needs two: Anthropic Sonnet 4.6 for analysis
and text, Gemini for carousel imagery.

Each provider lives in its own module. That is deliberate rather than tidy-minded:
mixing SDKs in one file is how a "provider-neutral" module quietly becomes a
single-vendor one.

The prompt-building layer stays where it is. A text provider receives the
already-assembled `response_input` -- the section list `LLMService` builds -- and
is responsible only for translating it to its own wire format and returning a
normalised result.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class ProviderError(RuntimeError):
    """A provider could not fulfil a request.

    Carries a caller-safe message plus the provider and model that failed, so the
    route layer can report which model broke without leaking vendor internals.
    """

    def __init__(
        self,
        safe_message: str,
        *,
        provider: str,
        model_name: str | None = None,
        status_code: int = 502,
        code: str = "generation_failed",
    ):
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.provider = provider
        self.model_name = model_name
        self.status_code = status_code
        self.code = code


class ProviderNotConfigured(ProviderError):
    """Required configuration (usually an API key) is absent.

    Distinct from a generation failure: nothing was attempted, and retrying will
    not help until the operator sets the missing value. Surfaced as 503 so it is
    not mistaken for a model error.
    """

    def __init__(self, provider: str, missing: str):
        super().__init__(
            f"{provider} is not configured yet.",
            provider=provider,
            status_code=503,
            code="provider_not_configured",
        )
        self.missing = missing


def require_env(name: str, *, provider: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ProviderNotConfigured(provider, missing=name)
    return value


def sections_to_system_and_messages(response_input: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split the internal section list into a system prompt and chat messages.

    ``LLMService`` builds sections shaped like
    ``{"type": "message", "role": ..., "content": [{"type": "input_text", "text": ...}]}``.
    Providers that separate the system prompt from the conversation need the
    system-role sections concatenated and the rest kept in order.
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for section in response_input:
        text = "\n".join(
            str(part.get("text") or "")
            for part in section.get("content", [])
            if str(part.get("text") or "").strip()
        ).strip()
        if not text:
            continue

        if section.get("role") == "system":
            system_parts.append(text)
        else:
            messages.append({"role": section.get("role", "user"), "content": text})

    if not messages:
        # Every provider requires at least one non-system turn. This only happens
        # if the caller passed nothing but system sections.
        messages.append({"role": "user", "content": "Continue."})

    return "\n\n".join(system_parts), messages


class TextProvider(ABC):
    """Generates text for a task type."""

    name: str

    @abstractmethod
    def generate(
        self,
        *,
        task_type: str,
        response_input: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """Return at least ``reply``, ``model_provider`` and ``model_name``."""


class ImageProvider(ABC):
    """Generates raster images."""

    name: str

    @abstractmethod
    def generate_image(self, *, prompt: str, aspect_ratio: str | None = None) -> bytes:
        """Return encoded image bytes (PNG or JPEG) for ``prompt``."""
