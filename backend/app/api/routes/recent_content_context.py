from fastapi import APIRouter
from app.schemas.recent_content_context import RecentContentContextResponse
from app.services.recent_content_context_service import RecentContentContextService

router = APIRouter(prefix="/api/v1/recent-content-context", tags=["recent-content-context"])

recent_content_context_service = RecentContentContextService()


@router.get("/{account_id}", response_model=RecentContentContextResponse)
def get_recent_content_context(account_id: str):
    return recent_content_context_service.get_context(account_id)


@router.post("")
def save_recent_content_context(payload: RecentContentContextResponse):
    return recent_content_context_service.save_context(payload.model_dump())