from fastapi import APIRouter, HTTPException, Request

from app.api.routes.knowledge_packs import _require_internal_admin_access
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.internal_generation_debug import InternalGenerationDebugResponse
from app.services.generation_history_service import GenerationHistoryService

router = APIRouter(prefix="/api/v1/internal/generation-debug", tags=["internal-generation-debug"])

generation_history_service = GenerationHistoryService()


@router.get("/latest", response_model=InternalGenerationDebugResponse, responses=STANDARD_ERROR_RESPONSES, include_in_schema=False)
def get_latest_generation_debug(
    request: Request,
    user_id: str | None = None,
    task_type: str | None = None,
    account_id: str | None = None,
):
    _require_internal_admin_access(request)
    item = generation_history_service.get_latest_item(
        user_id=user_id,
        task_type=task_type,
        account_id=account_id,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Generation metadata was not found")

    return {
        "user_id": item.get("user_id"),
        "task_type": item.get("task_type"),
        "account_id": item.get("account_id"),
        "used_system_knowledge": item.get("used_system_knowledge"),
        "matched_knowledge_domain": item.get("matched_knowledge_domain"),
        "matched_knowledge_pack_ids": list(item.get("matched_knowledge_pack_ids") or []),
        "retrieved_chunk_count": int(item.get("retrieved_chunk_count") or 0),
        "retrieved_chunk_titles": list(item.get("retrieved_chunk_titles") or []),
        "knowledge_retrieval_used": item.get("knowledge_retrieval_used"),
        "knowledge_retrieval_top_k": item.get("knowledge_retrieval_top_k"),
        "knowledge_retrieved_count": int(item.get("knowledge_retrieved_count") or 0),
        "knowledge_collection_name": item.get("knowledge_collection_name"),
        "used_langflow": item.get("used_langflow"),
        "model_provider": item.get("model_provider"),
        "model_name": item.get("model_name"),
        "prompt_section_names": list(item.get("prompt_section_names") or []),
        "prompt_token_estimate": item.get("prompt_token_estimate"),
        "retry_count": int(item.get("retry_count") or 0),
        "rate_limited": bool(item.get("rate_limited")),
        "parse_status": item.get("parse_status"),
        "created_at": item.get("timestamp"),
    }
