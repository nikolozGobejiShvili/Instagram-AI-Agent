from fastapi import APIRouter
from app.services.profile_context_service import ProfileContextService
from app.schemas.profile_context import ProfileContextPayload

router = APIRouter(prefix="/api/v1/profile-context", tags=["profile-context"])

profile_context_service = ProfileContextService()


@router.get("/{account_id}")
def get_profile_context(account_id: str):
    return profile_context_service.get_context(account_id)


@router.post("")
def save_profile_context(payload: ProfileContextPayload):
    return profile_context_service.save_context(payload.model_dump())