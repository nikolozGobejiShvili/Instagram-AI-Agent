import json
import importlib.util

from fastapi import APIRouter, HTTPException, Request

from app.api.internal_auth import INTERNAL_ADMIN_HEADER, require_internal_admin_access
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.knowledge_pack import (
    KnowledgePackDeleteResponse,
    KnowledgePackListResponse,
    KnowledgePackResponse,
)
from app.services.knowledge_pack_service import KnowledgePackService
from app.services.langflow_service import LangflowService, LangflowServiceError

router = APIRouter(prefix="/api/v1/internal/knowledge-packs", tags=["internal-knowledge-packs"])

knowledge_pack_service = KnowledgePackService()
langflow_service = LangflowService()
MULTIPART_AVAILABLE = bool(importlib.util.find_spec("multipart"))

# The guard now lives in app.api.internal_auth so billing and maintenance share
# one implementation. Kept under the original private name because the handlers
# below call it directly.
_require_internal_admin_access = require_internal_admin_access


def _parse_supported_task_types(value: str | None) -> list[str]:
    normalized_value = (value or "").strip()
    if not normalized_value:
        return []

    if normalized_value.startswith("["):
        try:
            parsed_value = json.loads(normalized_value)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="supported_task_types must be a JSON array or a comma-separated string",
            ) from exc
        if not isinstance(parsed_value, list):
            raise HTTPException(
                status_code=400,
                detail="supported_task_types must be a JSON array or a comma-separated string",
            )
        return [str(item).strip() for item in parsed_value if str(item).strip()]

    return [item.strip() for item in normalized_value.split(",") if item.strip()]


UPLOAD_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["title", "files"],
                    "properties": {
                        "title": {
                            "type": "string",
                            "title": "Title",
                            "description": "Human-readable internal knowledge pack name.",
                            "example": "Mariami Reels Playbook",
                        },
                        "description": {
                            "type": "string",
                            "title": "Description",
                            "description": "Optional short summary of the internal reels methodology.",
                            "example": "Internal reels strategy guidance for hook logic, retention, and CTA structure.",
                        },
                        "domain": {
                            "type": "string",
                            "title": "Domain",
                            "description": "Phase 1 currently supports reels only.",
                            "example": "reels",
                            "default": "reels",
                        },
                        "supported_task_types": {
                            "type": "string",
                            "title": "Supported Task Types",
                            "description": "Optional comma-separated list or JSON array string. Phase 1 supports reel_idea,reel_script,reel_feedback",
                            "example": "reel_idea,reel_script,reel_feedback",
                        },
                        "scope": {
                            "type": "string",
                            "title": "Scope",
                            "description": "Phase 1 internal knowledge must use system scope.",
                            "example": "system",
                            "default": "system",
                        },
                        "visibility": {
                            "type": "string",
                            "title": "Visibility",
                            "description": "Phase 1 internal knowledge must remain internal.",
                            "example": "internal",
                            "default": "internal",
                        },
                        "status": {
                            "type": "string",
                            "title": "Status",
                            "description": "Use active to enable the pack immediately.",
                            "example": "active",
                            "default": "active",
                        },
                        "files": {
                            "type": "array",
                            "title": "Files",
                            "items": {
                                "type": "string",
                                "format": "binary",
                            },
                            "description": "One or more .txt, .md, .pdf, or .docx files.",
                        },
                    },
                }
            }
        },
    }
}


@router.post(
    "/upload",
    response_model=KnowledgePackResponse,
    responses=STANDARD_ERROR_RESPONSES,
    openapi_extra=UPLOAD_OPENAPI_EXTRA,
    include_in_schema=False,
)
async def upload_knowledge_pack(request: Request):
    _require_internal_admin_access(request)

    if not MULTIPART_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="python-multipart is required for knowledge pack uploads",
        )

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Knowledge pack upload failed: multipart form data could not be parsed",
        ) from exc

    title = " ".join(str(form.get("title") or "").split()).strip()
    description_raw = form.get("description")
    description = " ".join(str(description_raw or "").split()).strip() or None
    domain = " ".join(str(form.get("domain") or "reels").split()).strip() or "reels"
    scope = " ".join(str(form.get("scope") or "system").split()).strip() or "system"
    visibility = " ".join(str(form.get("visibility") or "internal").split()).strip() or "internal"
    status = " ".join(str(form.get("status") or "active").split()).strip() or "active"
    supported_task_types = form.get("supported_task_types")

    if not title:
        raise HTTPException(status_code=400, detail="title is required for knowledge pack upload")

    uploaded_files = []
    raw_files = form.getlist("files")
    for file in raw_files:
        file_name = str(getattr(file, "filename", "") or "").strip()
        if not file_name:
            continue

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge pack upload failed: file '{file_name}' is empty",
            )

        uploaded_files.append({
            "file_name": file_name,
            "content_type": getattr(file, "content_type", None),
            "content": content,
        })

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Knowledge pack upload requires at least one file")

    parsed_supported_task_types = _parse_supported_task_types(
        str(supported_task_types) if supported_task_types is not None else None,
    )

    uploaded_pack = knowledge_pack_service.upload_pack(
        owner_user_id="system",
        title=title,
        description=description,
        supported_task_types=parsed_supported_task_types,
        uploaded_files=uploaded_files,
        scope=scope,
        domain=domain,
        visibility=visibility,
        status=status,
    )
    # Vectorised for every domain, not just reels. The gate used to require
    # domain == "reels", so material uploaded to teach the agent about
    # carousels, audits or planning was stored and then never ingested -- it sat
    # on disk looking uploaded while retrieval found nothing. The retrieval side
    # already covers those task types (see RAG_TASK_TYPES); only ingestion was
    # narrower than the feature it fed.
    if uploaded_pack.get("scope") == "system" and uploaded_pack.get("visibility") == "internal":
        try:
            langflow_service.ingest_system_reels_knowledge(
                knowledge_pack_id=uploaded_pack["knowledge_pack_id"],
                title=uploaded_pack["title"],
                description=uploaded_pack.get("description"),
                file_paths=knowledge_pack_service.get_pack_file_paths(uploaded_pack["knowledge_pack_id"]),
                supported_task_types=list(uploaded_pack.get("supported_task_types") or []),
            )
        except LangflowServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.safe_message) from exc

    return uploaded_pack


@router.get("", response_model=KnowledgePackListResponse, responses=STANDARD_ERROR_RESPONSES, include_in_schema=False)
def list_knowledge_packs(request: Request, domain: str | None = None):
    _require_internal_admin_access(request)
    return knowledge_pack_service.list_packs(scope="system", domain=domain, visibility="internal")


@router.post(
    "/{knowledge_pack_id}/activate",
    response_model=KnowledgePackResponse,
    responses=STANDARD_ERROR_RESPONSES,
    include_in_schema=False,
)
def activate_knowledge_pack(knowledge_pack_id: str, request: Request):
    _require_internal_admin_access(request)
    return knowledge_pack_service.activate_pack(knowledge_pack_id)


@router.post(
    "/{knowledge_pack_id}/deactivate",
    response_model=KnowledgePackResponse,
    responses=STANDARD_ERROR_RESPONSES,
    include_in_schema=False,
)
def deactivate_knowledge_pack(knowledge_pack_id: str, request: Request):
    _require_internal_admin_access(request)
    return knowledge_pack_service.deactivate_pack(knowledge_pack_id)


@router.delete(
    "/{knowledge_pack_id}",
    response_model=KnowledgePackDeleteResponse,
    responses=STANDARD_ERROR_RESPONSES,
    include_in_schema=False,
)
def delete_knowledge_pack(knowledge_pack_id: str, request: Request):
    _require_internal_admin_access(request)
    return knowledge_pack_service.delete_pack(knowledge_pack_id)
