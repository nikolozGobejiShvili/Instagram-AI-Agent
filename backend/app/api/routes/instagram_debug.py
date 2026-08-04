from fastapi import APIRouter

from app.services.instagram_profile_service import InstagramProfileService

router = APIRouter(prefix="/api/v1/instagram-debug", tags=["instagram-debug"])

instagram_profile_service = InstagramProfileService()


@router.get("/pages/{user_id}")
def get_instagram_page_diagnostics(user_id: str, account_id: str | None = None):
    """Return safe Meta page-discovery diagnostics without exposing tokens."""
    return instagram_profile_service.get_page_diagnostics(user_id, account_id)
