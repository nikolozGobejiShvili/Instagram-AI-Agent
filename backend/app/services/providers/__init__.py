"""Provider registry.

Resolution is by name so a provider can be swapped with configuration rather than
a code change, and so `LLMService` never imports a vendor SDK directly.

Providers are constructed lazily and cached. Constructing eagerly would read
credentials at import time, which is what made the previous OpenAI-only client
fail at startup rather than at first use.
"""
from __future__ import annotations

from typing import Callable

from app.services.providers.base import (
    ImageProvider,
    ProviderError,
    ProviderNotConfigured,
    TextProvider,
)

_TEXT_BUILDERS: dict[str, Callable[[], TextProvider]] = {}
_IMAGE_BUILDERS: dict[str, Callable[[], ImageProvider]] = {}
_TEXT_CACHE: dict[str, TextProvider] = {}
_IMAGE_CACHE: dict[str, ImageProvider] = {}


def _build_anthropic() -> TextProvider:
    from app.services.providers.anthropic_text import AnthropicTextProvider

    return AnthropicTextProvider()


def _build_openai_image() -> ImageProvider:
    from app.services.providers.openai_image import OpenAIImageProvider

    return OpenAIImageProvider()


def _build_gemini() -> ImageProvider:
    from app.services.providers.gemini_image import GeminiImageProvider

    return GeminiImageProvider()


_TEXT_BUILDERS["anthropic"] = _build_anthropic
_IMAGE_BUILDERS["openai"] = _build_openai_image
# Kept registered but no longer the default. Removing it would turn a
# configuration rollback into a code change, and the module costs nothing until
# something asks for it by name.
_IMAGE_BUILDERS["gemini"] = _build_gemini


def supported_text_providers() -> list[str]:
    return sorted(_TEXT_BUILDERS)


def supported_image_providers() -> list[str]:
    return sorted(_IMAGE_BUILDERS)


def get_text_provider(name: str) -> TextProvider:
    key = (name or "").strip().lower()
    if key not in _TEXT_BUILDERS:
        raise ProviderError(
            f"Unsupported text provider '{name}'.",
            provider=key or "unknown",
            status_code=503,
            code="provider_not_configured",
        )
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = _TEXT_BUILDERS[key]()
    return _TEXT_CACHE[key]


def get_image_provider(name: str) -> ImageProvider:
    key = (name or "").strip().lower()
    if key not in _IMAGE_BUILDERS:
        raise ProviderError(
            f"Unsupported image provider '{name}'.",
            provider=key or "unknown",
            status_code=503,
            code="provider_not_configured",
        )
    if key not in _IMAGE_CACHE:
        _IMAGE_CACHE[key] = _IMAGE_BUILDERS[key]()
    return _IMAGE_CACHE[key]


def register_text_provider(name: str, builder: Callable[[], TextProvider]) -> None:
    _TEXT_BUILDERS[name.strip().lower()] = builder
    _TEXT_CACHE.pop(name.strip().lower(), None)


def register_image_provider(name: str, builder: Callable[[], ImageProvider]) -> None:
    _IMAGE_BUILDERS[name.strip().lower()] = builder
    _IMAGE_CACHE.pop(name.strip().lower(), None)


def reset_cache() -> None:
    """Drop cached provider instances (used by tests)."""
    _TEXT_CACHE.clear()
    _IMAGE_CACHE.clear()


__all__ = [
    "ImageProvider",
    "ProviderError",
    "ProviderNotConfigured",
    "TextProvider",
    "get_image_provider",
    "get_text_provider",
    "register_image_provider",
    "register_text_provider",
    "reset_cache",
    "supported_image_providers",
    "supported_text_providers",
]
