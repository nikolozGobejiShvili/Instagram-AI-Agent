import json
import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


def _parse_body_payload(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value

    if value is None:
        return None

    raw_value = value
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")

    if not isinstance(raw_value, str) or not raw_value.strip():
        return None

    try:
        parsed_value = json.loads(raw_value)
    except ValueError:
        return None

    return parsed_value if isinstance(parsed_value, dict) else None


async def _extract_request_context(request: Request, body_override: Any = None) -> tuple[str | None, str | None]:
    task_type = request.query_params.get("task_type")
    account_id = request.query_params.get("account_id") or request.path_params.get("account_id")

    payload = _parse_body_payload(body_override)
    if payload is None:
        try:
            payload = _parse_body_payload(await request.body())
        except Exception:
            payload = None

    if payload:
        task_type = payload.get("task_type") or task_type
        account_id = payload.get("account_id") or account_id

    return task_type, account_id


def _normalize_detail(detail: Any) -> tuple[str, str | None]:
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or "Request failed")
        extra_payload = {key: value for key, value in detail.items() if key not in {"message", "detail"}}
        details = json.dumps(extra_payload, ensure_ascii=False) if extra_payload else None
        return message, details

    if isinstance(detail, list):
        return "Request validation failed", json.dumps(detail, ensure_ascii=False)

    if detail is None:
        return "Request failed", None

    return str(detail), None


def _combine_details(original_message: str, original_details: str | None, mapped_message: str) -> str | None:
    detail_parts = []
    if original_message and original_message != mapped_message:
        detail_parts.append(original_message)
    if original_details:
        detail_parts.append(original_details)
    return " | ".join(detail_parts) if detail_parts else None


def _map_error_code(status_code: int, original_message: str, path: str) -> tuple[str, str]:
    normalized_message = (original_message or "").lower()
    normalized_path = (path or "").lower()

    if "unsupported task type" in normalized_message:
        return "unsupported_task_type", "Unsupported task type"

    if "reel feedback requires media_id or an instagram reel link" in normalized_message:
        return "reel_feedback_input_required", "Reel feedback needs a target Reel"

    if "reel feedback requires a connected instagram account" in normalized_message:
        return "account_not_connected", "Instagram account is not connected"

    if "invalid instagram reel link" in normalized_message:
        return "invalid_instagram_reel_link", "Invalid Instagram Reel link"

    if "requested reel was not found" in normalized_message:
        return "reel_not_found", "Requested Reel was not found"

    if "internal admin access is required" in normalized_message:
        return "internal_access_required", "Internal admin access is required"

    if "must be reconnected" in normalized_message or "saved meta connection is no longer valid" in normalized_message:
        return "reconnect_required", "Instagram account must be reconnected."

    if "knowledge pack was not found" in normalized_message:
        return "knowledge_pack_not_found", "Knowledge pack was not found"

    if "no active knowledge pack" in normalized_message:
        return "no_active_knowledge_pack", "No active knowledge pack is available"

    if "unsupported knowledge pack file type" in normalized_message:
        return "knowledge_pack_file_unsupported", "Knowledge pack file type is not supported"

    if "python-multipart is required for knowledge pack uploads" in normalized_message:
        return "knowledge_pack_upload_unavailable", "Knowledge pack upload is unavailable"

    if (
        "knowledge pack document parsing failed" in normalized_message
        or "knowledge pack upload failed" in normalized_message
        or "pdf parsing dependency is unavailable" in normalized_message
        or "docx parsing dependency is unavailable" in normalized_message
    ):
        return "knowledge_pack_parse_failed", "Knowledge pack document parsing failed"

    if "no active plan" in normalized_message:
        return "no_active_plan", "No active plan is available"

    if "plan expired" in normalized_message:
        return "plan_expired", "Plan has expired"

    if "ai generation is temporarily rate limited" in normalized_message:
        return "llm_rate_limited", "AI generation is temporarily rate limited. Please try again shortly."

    if "not available on the current plan" in normalized_message:
        return "task_not_in_plan", "Task is not available on the current plan"

    if "monthly generation limit reached" in normalized_message or status_code == 429:
        return "generation_limit_reached", "Monthly generation limit reached"

    if "connected account limit reached" in normalized_message:
        return "connected_account_limit_reached", "Connected account limit reached"

    if "instagram context auto-sync failed" in normalized_message:
        return "instagram_context_sync_failed", "Instagram context sync failed"

    if "parse failed" in normalized_message:
        return "parse_failed", "Agent response parsing failed"

    if "instagram account is not connected" in normalized_message:
        return "account_not_connected", "Instagram account is not connected"

    if "multiple active connected accounts" in normalized_message or "multiple connected accounts found" in normalized_message:
        return "multiple_accounts_require_selection", "Multiple accounts require explicit selection"

    if "no connected accounts found" in normalized_message or "no active connected account" in normalized_message:
        return "no_active_account", "No active Instagram account is available"

    if "connected account '" in normalized_message and "was not found" in normalized_message:
        return "account_not_found", "Requested account was not found"

    if normalized_path.startswith("/api/v1/agent/chat") and status_code in {502, 504}:
        return "generation_failed", "Agent generation failed"

    if normalized_path.startswith("/api/v1/instagram-context-sync") and status_code >= 500:
        return "instagram_context_sync_failed", "Instagram context sync failed"

    if (
        "meta " in normalized_message
        or normalized_message.startswith("meta")
        or normalized_path.startswith("/api/v1/instagram-") and status_code in {502, 504}
    ):
        return "meta_api_failed", "Meta API request failed"

    if status_code == 422:
        return "validation_failed", "Request validation failed"

    if status_code >= 500:
        return "internal_error", "Internal server error"

    if status_code == 404:
        return "not_found", "Requested resource was not found"

    return "validation_failed", "Request validation failed"


def _build_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: str | None,
    task_type: str | None,
    account_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "task_type": task_type,
                "account_id": account_id,
            }
        },
    )


def _format_validation_details(errors: list[dict]) -> str | None:
    detail_parts = []
    for error in errors[:8]:
        location = ".".join(str(part) for part in error.get("loc", []))
        message = error.get("msg", "Invalid value")
        detail_parts.append(f"{location}: {message}")
    return " | ".join(detail_parts) if detail_parts else None


def _validation_error_code(errors: list[dict]) -> tuple[str, str]:
    for error in errors:
        location = error.get("loc", [])
        if location and location[-1] == "task_type":
            return "unsupported_task_type", "Unsupported task type"
    return "validation_failed", "Request validation failed"


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    task_type, account_id = await _extract_request_context(request)
    original_message, original_details = _normalize_detail(exc.detail)
    code, message = _map_error_code(exc.status_code, original_message, request.url.path)
    details = _combine_details(original_message, original_details, message)

    logger.warning(
        "Standardized API error path=%s status_code=%s code=%s task_type=%s account_id=%s",
        request.url.path,
        exc.status_code,
        code,
        task_type,
        account_id,
    )

    return _build_error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        details=details,
        task_type=task_type,
        account_id=account_id,
    )


async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    task_type, account_id = await _extract_request_context(request, exc.body)
    code, message = _validation_error_code(exc.errors())
    details = _format_validation_details(exc.errors())

    logger.warning(
        "Standardized validation error path=%s code=%s task_type=%s account_id=%s",
        request.url.path,
        code,
        task_type,
        account_id,
    )

    return _build_error_response(
        status_code=422,
        code=code,
        message=message,
        details=details,
        task_type=task_type,
        account_id=account_id,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    task_type, account_id = await _extract_request_context(request)
    logger.exception(
        "Unhandled API error path=%s task_type=%s account_id=%s",
        request.url.path,
        task_type,
        account_id,
        exc_info=exc,
    )

    return _build_error_response(
        status_code=500,
        code="internal_error",
        message="Internal server error",
        details=None,
        task_type=task_type,
        account_id=account_id,
    )
