from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.instagram_media import InstagramMediaListResponse
from app.services.instagram_media_service import InstagramMediaService

router = APIRouter(prefix="/api/v1/instagram-media", tags=["instagram-media"])

instagram_media_service = InstagramMediaService()


@router.get("/{user_id}", response_model=InstagramMediaListResponse, responses=STANDARD_ERROR_RESPONSES)
def get_instagram_media(user_id: str, account_id: str | None = None, limit: int = 10):
    return instagram_media_service.get_media(user_id, account_id, limit)
