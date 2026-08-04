"""Gemini image provider — carousel backgrounds.

Carousel slides are composed in two stages: this provider produces a background
image containing **no text**, and the slide's headline and body are drawn on top
in code. Image models render text unreliably, and Georgian in particular; keeping
the words out of the generated pixels is what makes the output legible.

The model id is **not defaulted**. It is read from ``GEMINI_IMAGE_MODEL`` and the
provider refuses to run without it, rather than shipping a guessed identifier
that would fail at request time with a confusing error. Set it from Google's
current model list.

Only Google GenAI SDK calls live in this module.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.services.providers.base import (
    ImageProvider,
    ProviderError,
    ProviderNotConfigured,
    require_env,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "gemini"

# Instagram portrait. The renderer composites text onto this canvas.
DEFAULT_ASPECT_RATIO = "4:5"

NO_TEXT_DIRECTIVE = (
    "Produce a background image only. Do not render any text, letters, words, "
    "numbers, logos, watermarks or UI elements anywhere in the image. Leave the "
    "composition uncluttered so text can be overlaid afterwards."
)


class GeminiImageProvider(ImageProvider):
    name = PROVIDER_NAME

    def __init__(self, client: Any | None = None, model_name: str | None = None):
        self._client = client
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        name = self._model_name or os.getenv("GEMINI_IMAGE_MODEL", "").strip()
        if not name:
            # Deliberately not defaulted -- see module docstring.
            raise ProviderNotConfigured(PROVIDER_NAME, missing="GEMINI_IMAGE_MODEL")
        return name

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = require_env("GEMINI_API_KEY", provider=PROVIDER_NAME)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise ProviderError(
                "Gemini support is not installed.",
                provider=PROVIDER_NAME,
                status_code=503,
                code="provider_not_configured",
            ) from exc

        self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_image(self, *, prompt: str, aspect_ratio: str | None = None) -> bytes:
        model_name = self.model_name
        client = self._get_client()
        full_prompt = f"{prompt.strip()}\n\n{NO_TEXT_DIRECTIVE}"

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config={
                    "response_modalities": ["IMAGE"],
                    "image_config": {"aspect_ratio": aspect_ratio or DEFAULT_ASPECT_RATIO},
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            logger.exception("Gemini image generation failed")
            raise ProviderError(
                "Image generation failed. Please try again shortly.",
                provider=PROVIDER_NAME,
                model_name=model_name,
            ) from exc

        image_bytes = self._extract_image_bytes(response)
        if not image_bytes:
            raise ProviderError(
                "Image generation returned no image.",
                provider=PROVIDER_NAME,
                model_name=model_name,
            )
        return image_bytes

    def _extract_image_bytes(self, response: Any) -> bytes | None:
        """Pull the first inline image out of the response.

        A response can interleave text and image parts, so the parts are scanned
        rather than indexed -- taking ``parts[0]`` returns commentary whenever the
        model narrates before emitting the image.
        """
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None)
                if data:
                    return data
        return None
