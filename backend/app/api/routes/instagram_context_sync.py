from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.services.instagram_context_sync_service import InstagramContextSyncService

router = APIRouter(prefix="/api/v1/instagram-context-sync", tags=["instagram-context-sync"])

instagram_context_sync_service = InstagramContextSyncService()


@router.post("/{user_id}", responses=STANDARD_ERROR_RESPONSES)
def sync_instagram_context(user_id: str, account_id: str | None = None):
    return instagram_context_sync_service.sync(user_id, account_id)
