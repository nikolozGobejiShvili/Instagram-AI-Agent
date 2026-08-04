"""Serves rendered carousel images by asset id.

Assets are addressed by an opaque id generated when the image was stored, never
by a caller-supplied path, so this endpoint cannot be walked out of the media
directory. An unknown or malformed id is a plain 404 -- the two are deliberately
indistinguishable, so the endpoint does not confirm which ids exist.
"""
from fastapi import APIRouter, HTTPException, Response

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.services.carousel_media_service import CONTENT_TYPE, CarouselMediaService

router = APIRouter(prefix="/api/v1/carousel-media", tags=["carousel-media"])

carousel_media_service = CarouselMediaService()


@router.get(
    "/{asset_id}",
    responses={**STANDARD_ERROR_RESPONSES, 200: {"content": {CONTENT_TYPE: {}}}},
    response_class=Response,
)
def get_carousel_asset(asset_id: str):
    image_bytes = carousel_media_service.load(asset_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Carousel image was not found")

    return Response(
        content=image_bytes,
        media_type=CONTENT_TYPE,
        # Assets are immutable: the id is derived per render, so a changed image
        # is always a new id. Safe to cache hard.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
