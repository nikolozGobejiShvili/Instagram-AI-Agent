from fastapi import APIRouter

from app.schemas.generation_history import GenerationHistoryItem
from app.services.generation_history_service import GenerationHistoryService

router = APIRouter(prefix="/api/v1/generation-history", tags=["generation-history"])

generation_history_service = GenerationHistoryService()


@router.get("/user/{user_id}", response_model=list[GenerationHistoryItem])
def get_generation_history_by_user(user_id: str):
    return generation_history_service.get_history_by_user(user_id)
