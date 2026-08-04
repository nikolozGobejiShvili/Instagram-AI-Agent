from fastapi import APIRouter
from app.schemas.link_context import LinkContextRequest, LinkContextResponse
from app.services.link_context_service import LinkContextService

router = APIRouter(prefix="/api/v1/link-context", tags=["link-context"])

link_context_service = LinkContextService()


@router.post("", response_model=LinkContextResponse)
def get_link_context(payload: LinkContextRequest):
    result = link_context_service.extract_context(str(payload.link))
    return LinkContextResponse(**result)