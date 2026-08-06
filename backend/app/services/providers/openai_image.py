"""Carousel backgrounds from OpenAI's image model.

Replaces the Gemini provider. Same contract, same rule: the model produces a
*background* and never text. Slide copy is composited in code because image
models render text unreliably, and Georgian script especially -- letters baked
into the pixels would sit behind the real ones.

One size mismatch has to be handled rather than hoped away. Slides are 1080x1350
(4:5, the tallest Instagram allows); `gpt-image-1` offers only 1024x1024,
1536x1024 and 1024x1536. The nearest portrait is 1024x1536 (2:3), which is taller
than the slide, so the render step crops rather than stretches -- a stretched
background is visibly wrong in a way a cropped one is not.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

from app.services.providers.base import ImageProvider, ProviderError, ProviderNotConfigured

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-image-1"

# Taller than the 4:5 slide on purpose: the compositor can crop height away, but
# it cannot invent it.
GENERATION_SIZE = "1024x1536"

NO_TEXT_RULE = (
    "Do not render any text, letters, numbers, logos, watermarks or UI in the image. "
    "Produce a clean background composition only, with an uncluttered area where text "
    "will later be placed on top."
)


class OpenAIImageProvider(ImageProvider):
    name = PROVIDER_NAME

    def __init__(self, *, api_key: str | None = None, model_name: str | None = None, timeout: float | None = None):
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name = model_name or os.getenv("OPENAI_IMAGE_MODEL", "").strip() or DEFAULT_MODEL
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        # Image generation is slow; the default httpx timeout would abandon a
        # request the API is still working on and the customer would be charged
        # for a slide that never arrives.
        self.timeout = float(timeout or os.getenv("IMAGE_TIMEOUT_SECONDS", "120") or 120)
        self.quality = os.getenv("OPENAI_IMAGE_QUALITY", "").strip() or "medium"

    def generate_image(self, *, prompt: str, aspect_ratio: str | None = None) -> bytes:
        if not self._api_key:
            raise ProviderNotConfigured(PROVIDER_NAME, missing="OPENAI_API_KEY")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/images/generations",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self.model_name,
                        "prompt": f"{prompt}\n\n{NO_TEXT_RULE}",
                        "size": GENERATION_SIZE,
                        "quality": self.quality,
                        "n": 1,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("OpenAI image error status=%s", status)
            if status == 401:
                raise ProviderNotConfigured(PROVIDER_NAME, missing="OPENAI_API_KEY") from exc
            raise ProviderError(
                "Image generation is rate limited. Please try again shortly."
                if status == 429
                else "Image generation failed. Please try again shortly.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
                status_code=503 if status == 429 else 502,
                code="image_rate_limited" if status == 429 else "image_generation_failed",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Image generation could not reach the model provider.",
                provider=PROVIDER_NAME,
                model_name=self.model_name,
                status_code=504,
                code="image_unreachable",
            ) from exc

        return self._extract_image(payload)

    def _extract_image(self, payload: dict[str, Any]) -> bytes:
        """Pull the image bytes out of the response.

        gpt-image models always return base64 in ``data[].b64_json`` and ignore
        ``response_format``, so there is no url branch to handle -- but a
        response carrying no image at all still has to fail loudly rather than
        return empty bytes the compositor would happily draw onto.
        """
        entries = payload.get("data")
        if isinstance(entries, list):
            for entry in entries:
                encoded = (entry or {}).get("b64_json")
                if encoded:
                    return base64.b64decode(encoded)

        raise ProviderError(
            "Image generation returned no image.",
            provider=PROVIDER_NAME,
            model_name=self.model_name,
            code="no_image_returned",
        )
