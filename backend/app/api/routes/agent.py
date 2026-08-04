import logging
import os
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from app.schemas.agent import AgentCapabilitiesResponse, AgentChatRequest, AgentChatResponse, validate_structured_output_payload
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.services.billing_service import BillingService
from app.services.langflow_service import LangflowService, LangflowServiceError
from app.services.llm_service import LLMService, LLMServiceError
from app.services.profile_context_service import ProfileContextService
from app.services.link_context_service import LinkContextService
from app.services.recent_content_context_service import RecentContentContextService
from app.services.recent_posts_context_service import RecentPostsContextService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.generation_history_service import GenerationHistoryService
from app.services.knowledge_pack_service import KnowledgePackService
from app.services.deterministic_knowledge_retrieval_service import (
    DeterministicKnowledgeRetrievalService,
    should_use_knowledge_retrieval,
)
from app.services.agent_response_formatter_service import AgentResponseFormatterService
from app.services.instagram_context_sync_service import InstagramContextSyncService
from app.services.instagram_media_service import InstagramMediaService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

logger = logging.getLogger(__name__)

langflow_service = LangflowService()
profile_context_service = ProfileContextService()
link_context_service = LinkContextService()
recent_content_context_service = RecentContentContextService()
recent_posts_context_service = RecentPostsContextService()
connected_accounts_service = ConnectedAccountsService()
generation_history_service = GenerationHistoryService()
knowledge_pack_service = KnowledgePackService()
knowledge_retrieval_service = DeterministicKnowledgeRetrievalService()
agent_response_formatter_service = AgentResponseFormatterService()
instagram_context_sync_service = InstagramContextSyncService()
billing_service = BillingService()
instagram_media_service = InstagramMediaService()
llm_service = LLMService()
REELS_SYSTEM_KNOWLEDGE_TASK_TYPES = {"reel_idea", "reel_script", "reel_feedback"}

AGENT_CAPABILITIES = AgentCapabilitiesResponse(
    supported_task_types=[
        {
            "task_type": "chat",
            "structured_output_supported": False,
            "fields": [],
        },
        {
            "task_type": "reel_idea",
            "structured_output_supported": True,
            "fields": ["ideas"],
        },
        {
            "task_type": "reel_script",
            "structured_output_supported": True,
            "fields": ["script"],
        },
        {
            "task_type": "reel_feedback",
            "structured_output_supported": True,
            "fields": ["feedback"],
        },
        {
            "task_type": "caption",
            "structured_output_supported": True,
            "fields": ["hook", "body", "cta", "full_caption"],
        },
        {
            "task_type": "carousel",
            "structured_output_supported": True,
            "fields": ["title", "slides", "cta"],
        },
        {
            "task_type": "profile_audit",
            "structured_output_supported": True,
            "fields": ["strengths", "weak_points", "quick_fixes", "priority_actions", "summary"],
        },
        {
            "task_type": "content_plan",
            "structured_output_supported": True,
            "fields": ["plan_title", "content_items", "summary"],
        },
        {
            "task_type": "link_analysis",
            "structured_output_supported": True,
            "fields": ["current_state", "issues", "recommended_changes", "best_cta_direction", "summary"],
        },
        {
            "task_type": "performance_summary",
            "structured_output_supported": True,
            "fields": ["what_worked", "what_did_not_work", "content_patterns", "recommended_next_moves", "summary"],
        },
    ],
    features={
        "auto_sync": True,
        "active_account_fallback": True,
        "generation_history": True,
        "knowledge_packs": False,
        "plans_and_limits": True,
    },
)


def _build_preview(value: str | None, limit: int = 160) -> str | None:
    if value is None:
        return None

    normalized_value = " ".join(value.split()).strip()
    if len(normalized_value) <= limit:
        return normalized_value

    return f"{normalized_value[: limit - 3]}..."


def _use_langflow_for_agent_chat() -> bool:
    return (os.getenv("USE_LANGFLOW_FOR_AGENT_CHAT", "false").strip().lower() == "true")


def _is_reconnect_required_error(detail: object) -> bool:
    normalized_detail = str(detail or "").lower()
    return (
        "must be reconnected" in normalized_detail
        or "saved meta connection is no longer valid" in normalized_detail
    )


def _is_reel_media_item(media_item: dict | None) -> bool:
    if not media_item:
        return False

    media_type = str(media_item.get("media_type") or "").upper()
    media_product_type = str(media_item.get("media_product_type") or "").upper()
    permalink = str(media_item.get("permalink") or "").lower()
    return bool(
        media_item.get("is_reel")
        or media_type == "REEL"
        or media_product_type == "REELS"
        or "/reel/" in permalink
        or "/reels/" in permalink
    )


def _is_valid_reel_link_context(link_context: dict | None) -> bool:
    return bool(
        link_context
        and link_context.get("detected_platform") == "instagram"
        and link_context.get("content_type") == "reel"
    )


def _normalize_source_url(value: str | None) -> str | None:
    normalized_value = " ".join((value or "").split()).strip()
    return normalized_value or None


def _instagram_media_shortcode(value: str | None) -> str | None:
    normalized_value = _normalize_source_url(value)
    if not normalized_value:
        return None

    parsed = urlparse(normalized_value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0].lower() in {"reel", "reels", "p", "tv"}:
        return path_parts[1].strip().lower() or None
    return None


def _normalized_instagram_permalink(value: str | None) -> str | None:
    normalized_value = _normalize_source_url(value)
    if not normalized_value:
        return None

    parsed = urlparse(normalized_value)
    host = parsed.netloc.lower().removeprefix("www.")
    path = "/".join(part.lower() for part in parsed.path.split("/") if part)
    if not host and not path:
        return None
    return f"{host}/{path}".strip("/")


def _instagram_media_urls_match(source_url: str | None, media_permalink: str | None) -> bool:
    source_permalink = _normalized_instagram_permalink(source_url)
    target_permalink = _normalized_instagram_permalink(media_permalink)
    if source_permalink and target_permalink and source_permalink == target_permalink:
        return True

    source_shortcode = _instagram_media_shortcode(source_url)
    target_shortcode = _instagram_media_shortcode(media_permalink)
    return bool(source_shortcode and target_shortcode and source_shortcode == target_shortcode)


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_caption_summary(value: str | None) -> str | None:
    return _build_preview(value, limit=220)


def _content_type_from_media_item(media_item: dict) -> str:
    media_type = str(media_item.get("media_type") or media_item.get("media_product_type") or "").upper()
    permalink = media_item.get("permalink")
    if _is_reel_media_item(media_item):
        return "reel"
    if media_type == "CAROUSEL_ALBUM":
        return "carousel"
    if media_type == "IMAGE":
        return "post"
    if media_type == "VIDEO":
        return "video"
    if permalink and "/p/" in str(permalink).lower():
        return "post"
    return media_type.lower() or "unknown"


def _build_safe_media_context(media_item: dict, *, account_id: str, source_url: str) -> dict:
    return {
        "source": "connected_account_permalink",
        "account_id": account_id,
        "media_id": media_item.get("media_id"),
        "media_type": media_item.get("media_type"),
        "content_type": _content_type_from_media_item(media_item),
        "caption_summary": _safe_caption_summary(media_item.get("caption")),
        "timestamp": media_item.get("timestamp"),
        "permalink": media_item.get("permalink"),
        "source_url": source_url,
        "metrics": {
            "likes": _safe_int(media_item.get("like_count")),
            "comments": _safe_int(media_item.get("comments_count")),
        },
    }


def _build_reel_context_from_safe_media(media_context: dict) -> dict:
    metrics = media_context.get("metrics") if isinstance(media_context.get("metrics"), dict) else {}
    performance_signals = []
    if metrics.get("likes") is not None:
        performance_signals.append(f"likes={metrics.get('likes')}")
    if metrics.get("comments") is not None:
        performance_signals.append(f"comments={metrics.get('comments')}")

    analysis_parts = ["Review this connected-account media using safe metadata and account context only."]
    if media_context.get("caption_summary"):
        analysis_parts.append(f"Caption/theme summary: {media_context.get('caption_summary')}")
    if performance_signals:
        analysis_parts.append(f"Available performance signals: {', '.join(performance_signals)}")

    return {
        "source": media_context.get("source"),
        "account_id": media_context.get("account_id"),
        "media_id": media_context.get("media_id"),
        "permalink": media_context.get("permalink") or media_context.get("source_url"),
        "source_url": media_context.get("source_url"),
        "media_type": media_context.get("media_type"),
        "content_type": media_context.get("content_type"),
        "caption_summary": media_context.get("caption_summary"),
        "caption": media_context.get("caption_summary"),
        "timestamp": media_context.get("timestamp"),
        "like_count": metrics.get("likes"),
        "comments_count": metrics.get("comments"),
        "analysis_brief": " ".join(part for part in analysis_parts if part),
    }


def _resolve_connected_media_context_for_link(
    *,
    user_id: str | None,
    account_id: str | None,
    source_url: str | None,
) -> dict | None:
    if not user_id or not account_id or not source_url:
        return None

    parsed = urlparse(source_url)
    if "instagram.com" not in parsed.netloc.lower():
        return None

    if not _instagram_media_shortcode(source_url):
        return None

    try:
        media_payload = instagram_media_service.get_media(user_id, account_id, limit=25)
    except Exception as exc:
        logger.info(
            "Skipped connected-account media permalink matching user_id=%s account_id=%s error_type=%s",
            user_id,
            account_id,
            type(exc).__name__,
        )
        return None

    for media_item in media_payload.get("items", []):
        if not isinstance(media_item, dict):
            continue
        if _instagram_media_urls_match(source_url, media_item.get("permalink")):
            return _build_safe_media_context(
                media_item,
                account_id=account_id,
                source_url=source_url,
            )

    return None


def _build_media_reel_context(media_item: dict, *, source: str) -> dict:
    performance_signals = []
    if media_item.get("like_count") is not None:
        performance_signals.append(f"likes={media_item.get('like_count')}")
    if media_item.get("comments_count") is not None:
        performance_signals.append(f"comments={media_item.get('comments_count')}")

    analysis_parts = ["Review this specific Reel using the provided metadata and account context only."]
    if media_item.get("caption"):
        analysis_parts.append(f"Current caption/theme: {media_item.get('caption')}")
    if performance_signals:
        analysis_parts.append(f"Available performance signals: {', '.join(performance_signals)}")

    return {
        "source": source,
        "media_id": media_item.get("media_id"),
        "permalink": media_item.get("permalink"),
        "media_type": media_item.get("media_type"),
        "caption": media_item.get("caption"),
        "timestamp": media_item.get("timestamp"),
        "like_count": media_item.get("like_count"),
        "comments_count": media_item.get("comments_count"),
        "analysis_brief": " ".join(part for part in analysis_parts if part),
    }


def _build_link_reel_context(link_context: dict) -> dict:
    analysis_parts = ["Review this Reel link using safe public metadata and account context."]
    if link_context.get("summary"):
        analysis_parts.append(f"Public summary: {link_context.get('summary')}")
    if link_context.get("hook_style"):
        analysis_parts.append(f"Inferred hook style: {link_context.get('hook_style')}")
    if link_context.get("source_patterns"):
        analysis_parts.append(f"Transferable patterns: {', '.join(link_context.get('source_patterns', []))}")

    return {
        "source": "link",
        "media_id": None,
        "permalink": link_context.get("link"),
        "media_type": "REEL",
        "caption": link_context.get("summary"),
        "timestamp": None,
        "like_count": None,
        "comments_count": None,
        "analysis_brief": " ".join(part for part in analysis_parts if part),
    }


def _resolve_reel_feedback_context(
    *,
    user_id: str,
    account_id: str | None,
    media_id: str | None,
    link_context: dict | None,
) -> dict:
    normalized_media_id = " ".join((media_id or "").split()).strip()
    if normalized_media_id:
        if not account_id:
            raise HTTPException(
                status_code=400,
                detail="Reel feedback requires a connected Instagram account when media_id is provided",
            )
        media_item = instagram_media_service.get_media_item(user_id, normalized_media_id, account_id)
        if not _is_reel_media_item(media_item):
            raise HTTPException(
                status_code=404,
                detail="Requested Reel was not found in the connected Instagram account",
            )
        return _build_media_reel_context(media_item, source="media_id")

    if link_context:
        matched_media_context = link_context.get("matched_connected_media")
        if (
            isinstance(matched_media_context, dict)
            and str(matched_media_context.get("content_type") or "").lower() == "reel"
        ):
            return _build_reel_context_from_safe_media(matched_media_context)

        if not _is_valid_reel_link_context(link_context):
            raise HTTPException(
                status_code=400,
                detail="Invalid Instagram Reel link. Provide an instagram.com/reel/... URL.",
            )
        return _build_link_reel_context(link_context)

    if not account_id:
        raise HTTPException(
            status_code=400,
            detail="Reel feedback requires media_id or an Instagram Reel link when no recent Reel is available for fallback",
        )

    recent_media_payload = instagram_media_service.get_media(user_id, account_id, limit=20)
    recent_reel = next(
        (item for item in recent_media_payload.get("items", []) if _is_reel_media_item(item)),
        None,
    )
    if not recent_reel:
        raise HTTPException(
            status_code=400,
            detail="Reel feedback requires media_id or an Instagram Reel link when no recent Reel is available for fallback",
        )

    return _build_media_reel_context(recent_reel, source="recent_reel_fallback")


def _try_save_generation_history(
    *,
    user_id: str | None,
    account_id: str | None,
    task_type: str,
    status: str,
    message_preview: str,
    response_preview: str | None = None,
    error_message: str | None = None,
    niche: str | None = None,
    target_audience: str | None = None,
    goal: str | None = None,
    auto_sync: bool = False,
    sync_attempted: bool = False,
    sync_succeeded: bool | None = None,
    context_was_fresh: bool | None = None,
    sync_skipped: bool | None = None,
    last_synced_at: str | None = None,
    used_real_instagram_context: bool = False,
    used_system_knowledge: bool | None = None,
    matched_knowledge_domain: str | None = None,
    matched_knowledge_pack_ids: list[str] | None = None,
    retrieved_chunk_count: int | None = None,
    retrieved_chunk_titles: list[str] | None = None,
    knowledge_retrieval_used: bool | None = None,
    knowledge_retrieval_top_k: int | None = None,
    knowledge_retrieved_count: int | None = None,
    knowledge_collection_name: str | None = None,
    parse_status: str | None = None,
    used_langflow: bool | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    prompt_section_names: list[str] | None = None,
    prompt_token_estimate: int | None = None,
    retry_count: int | None = None,
    rate_limited: bool | None = None,
) -> None:
    try:
        generation_history_service.save_item({
            "user_id": user_id,
            "account_id": account_id,
            "task_type": task_type,
            "status": status,
            "message_preview": message_preview,
            "response_preview": response_preview,
            "error_message": error_message,
            "niche": niche,
            "target_audience": target_audience,
            "goal": goal,
            "auto_sync": auto_sync,
            "sync_attempted": sync_attempted,
            "sync_succeeded": sync_succeeded,
            "context_was_fresh": context_was_fresh,
            "sync_skipped": sync_skipped,
            "last_synced_at": last_synced_at,
            "used_real_instagram_context": used_real_instagram_context,
            "used_system_knowledge": used_system_knowledge,
            "matched_knowledge_domain": matched_knowledge_domain,
            "matched_knowledge_pack_ids": list(matched_knowledge_pack_ids or []),
            "retrieved_chunk_count": retrieved_chunk_count,
            "retrieved_chunk_titles": list(retrieved_chunk_titles or []),
            "knowledge_retrieval_used": knowledge_retrieval_used,
            "knowledge_retrieval_top_k": knowledge_retrieval_top_k,
            "knowledge_retrieved_count": knowledge_retrieved_count,
            "knowledge_collection_name": knowledge_collection_name,
            "parse_status": parse_status,
            "used_langflow": used_langflow,
            "model_provider": model_provider,
            "model_name": model_name,
            "prompt_section_names": list(prompt_section_names or []),
            "prompt_token_estimate": prompt_token_estimate,
            "retry_count": retry_count,
            "rate_limited": rate_limited,
        })
    except Exception:
        pass


def _clean_context_value(value: str | None) -> str | None:
    normalized_value = " ".join((value or "").split()).strip()
    if not normalized_value:
        return None
    if normalized_value.lower() in {"unknown", "not clearly defined yet", "unknown brand"}:
        return None
    return normalized_value


def _resolve_history_niche(payload: AgentChatRequest, profile_context: dict | None) -> str | None:
    return _clean_context_value(payload.niche) or _clean_context_value((profile_context or {}).get("niche"))


def _resolve_history_target_audience(payload: AgentChatRequest, profile_context: dict | None) -> str | None:
    return _clean_context_value(payload.target_audience) or _clean_context_value((profile_context or {}).get("target_audience"))


def _context_has_real_signal(
    profile_context: dict | None,
    recent_content_context: dict | None,
    recent_posts_context: dict | None,
) -> bool:
    if profile_context:
        if any([
            _clean_context_value(profile_context.get("brand_name")),
            _clean_context_value(profile_context.get("bio")),
            profile_context.get("content_focus"),
            profile_context.get("strengths"),
            profile_context.get("weak_points"),
        ]):
            return True

    if recent_content_context and any([
        recent_content_context.get("top_formats"),
        recent_content_context.get("best_topics"),
        recent_content_context.get("notes"),
    ]):
        return True

    if recent_posts_context and recent_posts_context.get("posts"):
        return True

    return False


def _run_auto_sync(user_id: str, account_id: str) -> None:
    sync_callable = getattr(
        instagram_context_sync_service,
        "sync_user_context",
        instagram_context_sync_service.sync,
    )

    try:
        sync_callable(user_id, account_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Instagram context auto-sync failed: {exc.detail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Instagram context auto-sync failed: {exc}",
        ) from exc


def _context_age_seconds_from_freshness(freshness: dict, *, minimum_one: bool = False) -> int | None:
    raw_age = freshness.get("context_age_seconds")
    if raw_age is None:
        return None

    try:
        age_seconds = int(raw_age)
    except (TypeError, ValueError):
        return None

    age_seconds = max(0, age_seconds)
    if minimum_one:
        return max(1, age_seconds)
    return age_seconds


def _evaluate_auto_sync(user_id: str, account_id: str) -> dict:
    freshness = instagram_context_sync_service.get_context_freshness(account_id)
    context_was_fresh = bool(freshness.get("context_was_fresh"))
    sync_required = bool(freshness.get("sync_required", not context_was_fresh))
    logger.info(
        "Evaluated Instagram context freshness user_id=%s account_id=%s context_was_fresh=%s last_synced_at=%s stale_reasons=%s",
        user_id,
        account_id,
        context_was_fresh,
        freshness.get("last_synced_at"),
        ",".join(freshness.get("stale_reasons", [])) if freshness.get("stale_reasons") else "none",
    )

    if context_was_fresh or not sync_required:
        logger.info(
            "Skipping Instagram context auto-sync because context is fresh user_id=%s account_id=%s",
            user_id,
            account_id,
        )
        return {
            "sync_attempted": False,
            "sync_succeeded": None,
            "context_was_fresh": True,
            "context_fresh": True,
            "sync_skipped": True,
            "last_synced_at": freshness.get("last_synced_at"),
            "context_age_seconds": _context_age_seconds_from_freshness(freshness, minimum_one=True),
        }

    try:
        _run_auto_sync(user_id, account_id)
    except HTTPException as exc:
        if _is_reconnect_required_error(exc.detail):
            raise
        if freshness.get("has_complete_context"):
            logger.warning(
                "Instagram context auto-sync failed but existing context will be reused user_id=%s account_id=%s error=%s",
                user_id,
                account_id,
                exc.detail,
            )
            return {
                "sync_attempted": True,
                "sync_succeeded": False,
                "context_was_fresh": False,
                "context_fresh": False,
                "sync_skipped": False,
                "last_synced_at": freshness.get("last_synced_at"),
                "context_age_seconds": _context_age_seconds_from_freshness(freshness),
            }
        raise

    refreshed_state = instagram_context_sync_service.get_context_freshness(account_id)
    logger.info(
        "Completed Instagram context auto-sync user_id=%s account_id=%s last_synced_at=%s",
        user_id,
        account_id,
        refreshed_state.get("last_synced_at"),
    )
    return {
        "sync_attempted": True,
        "sync_succeeded": True,
        "context_was_fresh": False,
        "context_fresh": bool(refreshed_state.get("context_was_fresh")),
        "sync_skipped": False,
        "last_synced_at": refreshed_state.get("last_synced_at"),
        "context_age_seconds": _context_age_seconds_from_freshness(refreshed_state),
    }


@router.get("/capabilities", response_model=AgentCapabilitiesResponse, responses=STANDARD_ERROR_RESPONSES)
def get_agent_capabilities():
    return AGENT_CAPABILITIES


@router.post("/chat", response_model=AgentChatResponse, responses=STANDARD_ERROR_RESPONSES)
def agent_chat(payload: AgentChatRequest):
    profile_context = None
    link_context = None
    reel_context = None
    recent_content_context = None
    recent_posts_context = None
    playbook_context = None
    effective_user_id = payload.user_id
    effective_account_id = payload.account_id
    message_preview = _build_preview(payload.message) or ""
    sync_attempted = False
    sync_succeeded: bool | None = None
    context_was_fresh: bool | None = None
    sync_skipped: bool | None = None
    last_synced_at: str | None = None
    context_fresh: bool | None = None
    context_age_seconds: int | None = None
    used_real_instagram_context = False
    used_system_knowledge = False
    matched_knowledge_domain: str | None = None
    matched_knowledge_pack_ids: list[str] = []
    retrieved_chunk_count = 0
    retrieved_chunk_titles: list[str] = []
    knowledge_context: str | None = None
    knowledge_retrieval_used = False
    knowledge_retrieval_top_k: int | None = None
    knowledge_retrieved_count = 0
    knowledge_collection_name: str | None = knowledge_retrieval_service.collection_name
    used_langflow = _use_langflow_for_agent_chat()
    use_safe_langflow_reels = langflow_service.should_use_safe_reels_rag(payload.task_type)
    model_provider = "langflow" if used_langflow else llm_service.provider_name()
    model_name: str | None = langflow_service.flow_id if used_langflow else None
    prompt_section_names: list[str] = []
    prompt_token_estimate: int | None = None
    retry_count = 0
    rate_limited = False
    source_url = _normalize_source_url(payload.source_url or payload.link)

    try:
        if payload.user_id:
            billing_service.enforce_agent_access(effective_user_id, payload.task_type)
            if payload.task_type == "reel_feedback" and source_url and not payload.media_id:
                try:
                    effective_account_id = connected_accounts_service.resolve_account_id(
                        payload.user_id,
                        payload.account_id,
                    )
                except HTTPException as exc:
                    if payload.account_id:
                        raise
                    if exc.status_code in {404, 409}:
                        effective_account_id = None
                    else:
                        raise
            else:
                effective_account_id = connected_accounts_service.resolve_account_id(
                    payload.user_id,
                    payload.account_id,
                )
        elif effective_account_id:
            effective_user_id = connected_accounts_service.find_user_id_by_account_id(effective_account_id)
            if not effective_user_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Connected account '{effective_account_id}' was not found for any user",
                )
            billing_service.enforce_agent_access(effective_user_id, payload.task_type)
        else:
            raise HTTPException(
                status_code=400,
                detail="account_id is required when user_id is not provided",
            )

        if payload.auto_sync and payload.user_id and effective_account_id:
            auto_sync_result = _evaluate_auto_sync(payload.user_id, effective_account_id)
            sync_attempted = auto_sync_result.get("sync_attempted", False)
            sync_succeeded = auto_sync_result.get("sync_succeeded")
            context_was_fresh = auto_sync_result.get("context_was_fresh")
            context_fresh = auto_sync_result.get("context_fresh", context_was_fresh)
            sync_skipped = auto_sync_result.get("sync_skipped")
            last_synced_at = auto_sync_result.get("last_synced_at")
            context_age_seconds = auto_sync_result.get("context_age_seconds")

        if effective_account_id:
            profile_context = profile_context_service.get_context(effective_account_id)

        if source_url:
            link_context = link_context_service.extract_context(source_url)
            link_context["link"] = link_context.get("link") or source_url
            link_context["source_url"] = link_context.get("source_url") or source_url

            matched_media_context = _resolve_connected_media_context_for_link(
                user_id=effective_user_id,
                account_id=effective_account_id,
                source_url=source_url,
            )
            if matched_media_context:
                link_context["matched_connected_media"] = matched_media_context
                if payload.task_type == "link_analysis":
                    reel_context = _build_reel_context_from_safe_media(matched_media_context)

            if payload.task_type == "reel_feedback" and not _is_valid_reel_link_context(link_context):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Instagram Reel link. Provide an instagram.com/reel/... URL.",
                )

        if effective_account_id and (
            payload.task_type in {"reel_idea", "reel_script", "reel_feedback", "caption", "carousel", "profile_audit", "content_plan", "performance_summary"}
            or source_url
        ):
            recent_posts_context = recent_posts_context_service.get_context(effective_account_id)

        if payload.task_type in {"content_plan", "profile_audit", "performance_summary", "reel_feedback", "link_analysis"} and effective_account_id:
            recent_content_context = recent_content_context_service.get_context(effective_account_id)

        if payload.task_type == "reel_feedback":
            reel_context = _resolve_reel_feedback_context(
                user_id=effective_user_id,
                account_id=effective_account_id,
                media_id=payload.media_id,
                link_context=link_context,
            )

        used_real_instagram_context = bool(effective_account_id) and _context_has_real_signal(
            profile_context,
            recent_content_context,
            recent_posts_context,
        )

        if (
            used_langflow
            and not use_safe_langflow_reels
            and should_use_knowledge_retrieval(payload.task_type, payload.message)
        ):
            knowledge_retrieval_result = knowledge_retrieval_service.retrieve(
                task_type=payload.task_type,
                message=payload.message,
                goal=payload.goal,
                profile_context=profile_context,
                recent_posts_context=recent_posts_context,
                recent_content_context=recent_content_context,
            )
            knowledge_retrieval_used = bool(knowledge_retrieval_result.used)
            knowledge_retrieval_top_k = knowledge_retrieval_result.top_k
            knowledge_retrieved_count = int(knowledge_retrieval_result.retrieved_count or 0)
            knowledge_collection_name = knowledge_retrieval_result.collection_name
            knowledge_context = knowledge_retrieval_result.knowledge_context

        if (
            effective_user_id
            and payload.task_type in REELS_SYSTEM_KNOWLEDGE_TASK_TYPES
            and not use_safe_langflow_reels
            and not used_langflow
        ):
            try:
                playbook_context = knowledge_pack_service.retrieve_system_context(
                    domain="reels",
                    task_type=payload.task_type,
                    message=payload.message,
                    goal=payload.goal,
                    profile_context=profile_context,
                    recent_content_context=recent_content_context,
                    recent_posts_context=recent_posts_context,
                )
                used_system_knowledge = bool(playbook_context.get("used_system_knowledge"))
                matched_knowledge_domain = playbook_context.get("matched_knowledge_domain")
                matched_knowledge_pack_ids = list(playbook_context.get("matched_knowledge_pack_ids") or [])
                retrieved_chunk_count = int(playbook_context.get("retrieved_chunk_count") or 0)
                retrieved_chunk_titles = list(playbook_context.get("retrieved_chunk_titles") or [])
            except Exception as exc:
                logger.warning(
                    "System knowledge retrieval failed user_id=%s account_id=%s task_type=%s error=%s",
                    effective_user_id,
                    effective_account_id,
                    payload.task_type,
                    exc,
                )
                playbook_context = None

        if use_safe_langflow_reels:
            used_langflow = True
            model_provider = "langflow"
            model_name = langflow_service.reels_generation_flow_id or model_name
            result = langflow_service.run_reels_rag_agent(
                message=payload.message,
                task_type=payload.task_type,
                account_id=effective_account_id,
                goal=payload.goal,
                profile_context=profile_context,
                recent_posts_context=recent_posts_context,
                recent_content_context=recent_content_context,
                reel_context=reel_context,
                link_context=link_context,
            )
            prompt_section_names = list(result.get("prompt_section_names") or ["langflow_runtime_payload"])
            used_system_knowledge = bool(result.get("used_system_knowledge", used_system_knowledge))
            matched_knowledge_domain = result.get("matched_knowledge_domain") or matched_knowledge_domain
            matched_knowledge_pack_ids = list(result.get("matched_knowledge_pack_ids") or matched_knowledge_pack_ids)
            retrieved_chunk_count = int(result.get("retrieved_chunk_count") or retrieved_chunk_count)
            retrieved_chunk_titles = list(result.get("retrieved_chunk_titles") or retrieved_chunk_titles)
        elif used_langflow:
            logger.warning(
                "USE_LANGFLOW_FOR_AGENT_CHAT=true is enabled. Agent chat will send sanitized runtime variables to the configured Langflow flow."
            )
            langflow_agent_kwargs = {
                "message": payload.message,
                "task_type": payload.task_type,
                "account_id": effective_account_id,
                "niche": payload.niche,
                "target_audience": payload.target_audience,
                "goal": payload.goal,
                "link": source_url,
                "profile_context": profile_context,
                "link_context": link_context,
                "recent_content_context": recent_content_context,
                "recent_posts_context": recent_posts_context,
                "playbook_context": playbook_context,
                "reel_context": reel_context,
            }
            if knowledge_context:
                langflow_agent_kwargs["knowledge_context"] = knowledge_context
            result = langflow_service.run_agent(**langflow_agent_kwargs)
            prompt_section_names = list(result.get("prompt_section_names") or ["main_agent_runtime_payload"])
        else:
            result = llm_service.run_agent(
                message=payload.message,
                task_type=payload.task_type,
                account_id=effective_account_id,
                niche=payload.niche,
                target_audience=payload.target_audience,
                goal=payload.goal,
                link=source_url,
                profile_context=profile_context,
                link_context=link_context,
                recent_content_context=recent_content_context,
                recent_posts_context=recent_posts_context,
                playbook_context=playbook_context,
                reel_context=reel_context,
            )
            prompt_section_names = list(result.get("prompt_section_names") or [])

        used_langflow = bool(result.get("used_langflow", used_langflow))
        model_provider = result.get("model_provider") or model_provider
        model_name = result.get("model_name") or model_name
        prompt_token_estimate = result.get("prompt_token_estimate")
        retry_count = int(result.get("retry_count") or 0)
        rate_limited = bool(result.get("rate_limited"))
        result["knowledge_retrieval_used"] = knowledge_retrieval_used
        result["knowledge_retrieval_top_k"] = knowledge_retrieval_top_k
        result["knowledge_retrieved_count"] = knowledge_retrieved_count
        result["knowledge_collection_name"] = knowledge_collection_name

        if result.get("structured_output") is not None:
            formatted_reply, _ = agent_response_formatter_service.format_reply(
                payload.task_type,
                result.get("reply"),
            )
            if formatted_reply is not None:
                result["reply"] = formatted_reply
            structured_output, parse_status, _ = validate_structured_output_payload(
                payload.task_type,
                result.get("parse_status") or "parsed",
                result.get("structured_output"),
            )
            if structured_output is None:
                normalized_reply = agent_response_formatter_service.normalize_reply(
                    payload.task_type,
                    result.get("reply"),
                )
                if normalized_reply.get("reply") is not None:
                    result["reply"] = normalized_reply["reply"]
                result["parse_status"] = normalized_reply.get("parse_status")
                result["structured_output"] = normalized_reply.get("structured_output")
            else:
                result["parse_status"] = parse_status
                result["structured_output"] = structured_output
        else:
            normalized_reply = agent_response_formatter_service.normalize_reply(
                payload.task_type,
                result.get("reply"),
            )
            if normalized_reply.get("reply") is not None:
                result["reply"] = normalized_reply["reply"]
            result["parse_status"] = normalized_reply.get("parse_status")
            result["structured_output"] = normalized_reply.get("structured_output")

        result["task_type"] = payload.task_type
        billing_service.increment_generation_usage(effective_user_id)
    except HTTPException as exc:
        if sync_attempted and sync_succeeded is None:
            sync_succeeded = False
        _try_save_generation_history(
            user_id=effective_user_id,
            account_id=effective_account_id,
            task_type=payload.task_type,
            status="failed",
            message_preview=message_preview,
            error_message=str(exc.detail),
            niche=_resolve_history_niche(payload, profile_context),
            target_audience=_resolve_history_target_audience(payload, profile_context),
            goal=_clean_context_value(payload.goal),
            auto_sync=payload.auto_sync,
            sync_attempted=sync_attempted,
            sync_succeeded=sync_succeeded,
            context_was_fresh=context_was_fresh,
            sync_skipped=sync_skipped,
            last_synced_at=last_synced_at,
            used_real_instagram_context=used_real_instagram_context,
            used_system_knowledge=used_system_knowledge,
            matched_knowledge_domain=matched_knowledge_domain,
            matched_knowledge_pack_ids=matched_knowledge_pack_ids,
            retrieved_chunk_count=retrieved_chunk_count,
            retrieved_chunk_titles=retrieved_chunk_titles,
            knowledge_retrieval_used=knowledge_retrieval_used,
            knowledge_retrieval_top_k=knowledge_retrieval_top_k,
            knowledge_retrieved_count=knowledge_retrieved_count,
            knowledge_collection_name=knowledge_collection_name,
            used_langflow=used_langflow,
            model_provider=model_provider,
            model_name=model_name,
            prompt_section_names=prompt_section_names,
            prompt_token_estimate=prompt_token_estimate,
            retry_count=retry_count,
            rate_limited=rate_limited,
        )
        raise
    except LangflowServiceError as exc:
        if sync_attempted and sync_succeeded is None:
            sync_succeeded = False
        used_langflow = bool(exc.used_langflow)
        model_provider = exc.model_provider or model_provider
        model_name = exc.model_name or model_name
        prompt_section_names = list(exc.prompt_section_names or prompt_section_names)
        retry_count = int(exc.retry_count or 0)
        rate_limited = bool(exc.rate_limited)
        _try_save_generation_history(
            user_id=effective_user_id,
            account_id=effective_account_id,
            task_type=payload.task_type,
            status="failed",
            message_preview=message_preview,
            error_message=exc.safe_message,
            niche=_resolve_history_niche(payload, profile_context),
            target_audience=_resolve_history_target_audience(payload, profile_context),
            goal=_clean_context_value(payload.goal),
            auto_sync=payload.auto_sync,
            sync_attempted=sync_attempted,
            sync_succeeded=sync_succeeded,
            context_was_fresh=context_was_fresh,
            sync_skipped=sync_skipped,
            last_synced_at=last_synced_at,
            used_real_instagram_context=used_real_instagram_context,
            used_system_knowledge=used_system_knowledge,
            matched_knowledge_domain=matched_knowledge_domain,
            matched_knowledge_pack_ids=matched_knowledge_pack_ids,
            retrieved_chunk_count=retrieved_chunk_count,
            retrieved_chunk_titles=retrieved_chunk_titles,
            knowledge_retrieval_used=knowledge_retrieval_used,
            knowledge_retrieval_top_k=knowledge_retrieval_top_k,
            knowledge_retrieved_count=knowledge_retrieved_count,
            knowledge_collection_name=knowledge_collection_name,
            used_langflow=used_langflow,
            model_provider=model_provider,
            model_name=model_name,
            prompt_section_names=prompt_section_names,
            prompt_token_estimate=prompt_token_estimate,
            retry_count=retry_count,
            rate_limited=rate_limited,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.safe_message) from exc
    except LLMServiceError as exc:
        if sync_attempted and sync_succeeded is None:
            sync_succeeded = False
        prompt_token_estimate = exc.prompt_token_estimate
        retry_count = int(exc.retry_count or 0)
        rate_limited = bool(exc.rate_limited)
        model_provider = exc.model_provider or model_provider
        model_name = exc.model_name or model_name
        prompt_section_names = list(exc.prompt_section_names or prompt_section_names)
        _try_save_generation_history(
            user_id=effective_user_id,
            account_id=effective_account_id,
            task_type=payload.task_type,
            status="failed",
            message_preview=message_preview,
            error_message=exc.safe_message,
            niche=_resolve_history_niche(payload, profile_context),
            target_audience=_resolve_history_target_audience(payload, profile_context),
            goal=_clean_context_value(payload.goal),
            auto_sync=payload.auto_sync,
            sync_attempted=sync_attempted,
            sync_succeeded=sync_succeeded,
            context_was_fresh=context_was_fresh,
            sync_skipped=sync_skipped,
            last_synced_at=last_synced_at,
            used_real_instagram_context=used_real_instagram_context,
            used_system_knowledge=used_system_knowledge,
            matched_knowledge_domain=matched_knowledge_domain,
            matched_knowledge_pack_ids=matched_knowledge_pack_ids,
            retrieved_chunk_count=retrieved_chunk_count,
            retrieved_chunk_titles=retrieved_chunk_titles,
            knowledge_retrieval_used=knowledge_retrieval_used,
            knowledge_retrieval_top_k=knowledge_retrieval_top_k,
            knowledge_retrieved_count=knowledge_retrieved_count,
            knowledge_collection_name=knowledge_collection_name,
            used_langflow=used_langflow,
            model_provider=model_provider,
            model_name=model_name,
            prompt_section_names=prompt_section_names,
            prompt_token_estimate=prompt_token_estimate,
            retry_count=retry_count,
            rate_limited=rate_limited,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.safe_message) from exc
    except RuntimeError as exc:
        if sync_attempted and sync_succeeded is None:
            sync_succeeded = False
        _try_save_generation_history(
            user_id=effective_user_id,
            account_id=effective_account_id,
            task_type=payload.task_type,
            status="failed",
            message_preview=message_preview,
            error_message=str(exc),
            niche=_resolve_history_niche(payload, profile_context),
            target_audience=_resolve_history_target_audience(payload, profile_context),
            goal=_clean_context_value(payload.goal),
            auto_sync=payload.auto_sync,
            sync_attempted=sync_attempted,
            sync_succeeded=sync_succeeded,
            context_was_fresh=context_was_fresh,
            sync_skipped=sync_skipped,
            last_synced_at=last_synced_at,
            used_real_instagram_context=used_real_instagram_context,
            used_system_knowledge=used_system_knowledge,
            matched_knowledge_domain=matched_knowledge_domain,
            matched_knowledge_pack_ids=matched_knowledge_pack_ids,
            retrieved_chunk_count=retrieved_chunk_count,
            retrieved_chunk_titles=retrieved_chunk_titles,
            knowledge_retrieval_used=knowledge_retrieval_used,
            knowledge_retrieval_top_k=knowledge_retrieval_top_k,
            knowledge_retrieved_count=knowledge_retrieved_count,
            knowledge_collection_name=knowledge_collection_name,
            used_langflow=used_langflow,
            model_provider=model_provider,
            model_name=model_name,
            prompt_section_names=prompt_section_names,
            prompt_token_estimate=prompt_token_estimate,
            retry_count=retry_count,
            rate_limited=rate_limited,
        )
        if str(exc) == "Langflow request timed out":
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _try_save_generation_history(
        user_id=effective_user_id,
        account_id=result.get("account_id"),
        task_type=payload.task_type,
        status="success",
        message_preview=message_preview,
        response_preview=_build_preview(result.get("reply")),
        niche=_resolve_history_niche(payload, profile_context),
        target_audience=_resolve_history_target_audience(payload, profile_context),
        goal=_clean_context_value(payload.goal),
        auto_sync=payload.auto_sync,
        sync_attempted=sync_attempted,
        sync_succeeded=sync_succeeded,
        context_was_fresh=context_was_fresh,
        sync_skipped=sync_skipped,
        last_synced_at=last_synced_at,
        used_real_instagram_context=used_real_instagram_context,
        used_system_knowledge=used_system_knowledge,
        matched_knowledge_domain=matched_knowledge_domain,
        matched_knowledge_pack_ids=matched_knowledge_pack_ids,
        retrieved_chunk_count=retrieved_chunk_count,
        retrieved_chunk_titles=retrieved_chunk_titles,
        knowledge_retrieval_used=knowledge_retrieval_used,
        knowledge_retrieval_top_k=knowledge_retrieval_top_k,
        knowledge_retrieved_count=knowledge_retrieved_count,
        knowledge_collection_name=knowledge_collection_name,
        parse_status=result.get("parse_status"),
        used_langflow=used_langflow,
        model_provider=model_provider,
        model_name=model_name,
        prompt_section_names=prompt_section_names,
        prompt_token_estimate=prompt_token_estimate,
        retry_count=retry_count,
        rate_limited=rate_limited,
    )

    return AgentChatResponse(
        reply=result["reply"],
        account_id=result.get("account_id"),
        task_type=result.get("task_type"),
        parse_status=result.get("parse_status"),
        structured_output=result.get("structured_output"),
        knowledge_retrieval_used=result.get("knowledge_retrieval_used"),
        knowledge_retrieval_top_k=result.get("knowledge_retrieval_top_k"),
        knowledge_retrieved_count=int(result.get("knowledge_retrieved_count") or 0),
        knowledge_collection_name=result.get("knowledge_collection_name"),
        auto_sync_requested=bool(payload.auto_sync),
        auto_sync_performed=bool(sync_attempted),
        context_fresh=context_fresh,
        context_age_seconds=context_age_seconds,
    )
