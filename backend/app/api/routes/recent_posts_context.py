from fastapi import APIRouter
from app.schemas.recent_posts_context import RecentPostsContextResponse
from app.services.recent_posts_context_service import RecentPostsContextService

router = APIRouter(prefix="/api/v1/recent-posts-context", tags=["recent-posts-context"])

recent_posts_context_service = RecentPostsContextService()


@router.get("/{account_id}", response_model=RecentPostsContextResponse)
def get_recent_posts_context(account_id: str):
    return recent_posts_context_service.get_context(account_id)


@router.post("", response_model=RecentPostsContextResponse)
def save_recent_posts_context(payload: RecentPostsContextResponse):
    return recent_posts_context_service.save_context(payload.model_dump())
