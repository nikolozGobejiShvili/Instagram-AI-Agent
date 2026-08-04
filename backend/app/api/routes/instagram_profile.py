from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.instagram_profile import InstagramProfileResponse
from app.services.instagram_profile_service import InstagramProfileService

router = APIRouter(prefix="/api/v1/instagram-profile", tags=["instagram-profile"])

instagram_profile_service = InstagramProfileService()


@router.get("/{user_id}", response_model=InstagramProfileResponse, responses=STANDARD_ERROR_RESPONSES)
def get_instagram_profile(user_id: str, account_id: str | None = None):
    return instagram_profile_service.get_profile(user_id, account_id)
