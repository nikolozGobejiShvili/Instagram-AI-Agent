"""End-to-end carousel generation.

Three stages, deliberately separated because only the middle one is expensive
and only the last one is deterministic:

1. **Copy + art direction** (Sonnet 4.6) -- slide text, plus a per-slide prompt
   describing a background containing *no* text.
2. **Backgrounds** (Gemini) -- one image per slide.
3. **Composition** (Pillow) -- the copy is drawn onto the background here, in
   code, because image models render text unreliably and Georgian especially.

A slide whose background fails still ships: it is composited on the brand colour
instead. Losing decoration is not a reason to lose the copy the customer paid
for, and a partially-illustrated carousel is far more useful than an error.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES
from app.services.carousel_media_service import CarouselMediaService
from app.services.carousel_render_service import CarouselRenderError, CarouselRenderService

logger = logging.getLogger(__name__)

DEFAULT_SLIDE_COUNT = 5
DEFAULT_TEMPLATE_ID = "default"


class CarouselPipelineService:
    def __init__(
        self,
        *,
        render_service: CarouselRenderService | None = None,
        media_service: CarouselMediaService | None = None,
        image_provider: Any | None = None,
    ):
        self.render_service = render_service or CarouselRenderService()
        self.media_service = media_service or CarouselMediaService()
        self._image_provider = image_provider

    def _get_image_provider(self) -> Any | None:
        """Resolve the image provider, or None if it is not configured.

        Returning None rather than raising keeps a text-only carousel possible
        when no image credentials are set -- the slides still render on the brand
        colour.
        """
        if self._image_provider is not None:
            return self._image_provider
        try:
            from app.services.providers import get_image_provider

            return get_image_provider("gemini")
        except Exception as exc:  # noqa: BLE001 - absence is a valid state here
            logger.warning("Carousel image provider unavailable: %s", exc)
            return None

    def resolve_slide_count(self, requested: int | None, *, tier_maximum: int | None = None) -> int:
        """Clamp the requested slide count to what the product and plan allow."""
        count = int(requested or DEFAULT_SLIDE_COUNT)
        ceiling = MAX_CAROUSEL_SLIDES if tier_maximum is None else min(int(tier_maximum), MAX_CAROUSEL_SLIDES)
        return max(2, min(count, max(2, ceiling)))

    def render_carousel(
        self,
        *,
        structured_output: dict[str, Any],
        template_id: str = DEFAULT_TEMPLATE_ID,
        generate_images: bool = True,
    ) -> dict[str, Any]:
        """Render every slide in ``structured_output`` and attach asset urls.

        Returns the structured output with each slide's ``image_url`` and
        ``template_id`` populated, plus a per-slide record of what failed.
        """
        slides = list(structured_output.get("slides") or [])
        if not slides:
            return {**structured_output, "slides": [], "image_failures": []}

        provider = self._get_image_provider() if generate_images else None
        total = len(slides)
        rendered: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for index, slide in enumerate(slides, start=1):
            slide = dict(slide)
            slide_number = slide.get("slide_number") or index

            background = None
            prompt = (slide.get("image_prompt") or "").strip()
            if provider is not None and prompt:
                try:
                    background = provider.generate_image(prompt=prompt)
                except Exception as exc:  # noqa: BLE001 - degrade, don't abort
                    logger.warning("Background generation failed for slide %s: %s", slide_number, exc)
                    failures.append({"slide_number": slide_number, "reason": str(exc)})

            try:
                png = self.render_service.render_slide(
                    headline=slide.get("headline") or "",
                    body=slide.get("body") or "",
                    slide_number=slide_number,
                    total_slides=total,
                    background=background,
                )
            except CarouselRenderError:
                # Rendering is the one stage that must not fail silently: without
                # it there is no slide at all, only copy in a JSON blob.
                raise

            asset_id = self.media_service.store(png)
            slide["image_url"] = self.media_service.url_for(asset_id)
            slide["template_id"] = template_id
            slide["slide_number"] = slide_number
            rendered.append(slide)

        return {**structured_output, "slides": rendered, "image_failures": failures}
