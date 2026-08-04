from fastapi import APIRouter

from app.api.routes.agent import AGENT_CAPABILITIES
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.app_bootstrap import AppBootstrapResponse, BootstrapNextAction
from app.services.billing_service import BillingService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService
from app.services.instagram_context_sync_service import InstagramContextSyncService
from app.services.profile_context_service import ProfileContextService
from app.services.recent_content_context_service import RecentContentContextService
from app.services.recent_posts_context_service import RecentPostsContextService

router = APIRouter(prefix="/api/v1/app", tags=["app"])

billing_service = BillingService()
connected_accounts_service = ConnectedAccountsService()
instagram_connection_service = InstagramConnectionService()
instagram_context_sync_service = InstagramContextSyncService()
profile_context_service = ProfileContextService()
recent_posts_context_service = RecentPostsContextService()
recent_content_context_service = RecentContentContextService()


def _normalize_text(value: str | None) -> str | None:
    normalized_value = " ".join((value or "").split()).strip()
    return normalized_value or None


def _resolve_primary_account(accounts: list[dict]) -> dict | None:
    valid_accounts = [
        account for account in accounts
        if _normalize_text(str(account.get("account_id") or ""))
    ]
    active_accounts = [account for account in valid_accounts if account.get("is_active")]

    if len(active_accounts) == 1:
        return active_accounts[0]
    if len(valid_accounts) == 1:
        return valid_accounts[0]
    return None


def _resolve_primary_connection(connections: list[dict], active_account_id: str | None) -> dict | None:
    if active_account_id:
        matching_connection = next(
            (
                connection for connection in connections
                if str(connection.get("account_id") or "") == active_account_id
            ),
            None,
        )
        if matching_connection:
            return matching_connection

    if len(connections) == 1:
        return connections[0]

    reconnect_connection = next(
        (
            connection for connection in connections
            if connection.get("requires_reconnect") or connection.get("connection_status") == "reconnect_required"
        ),
        None,
    )
    if reconnect_connection:
        return reconnect_connection

    return connections[0] if connections else None


def _derive_next_actions(
    *,
    has_active_plan: bool,
    current_plan: str | None,
    instagram_connected: bool,
    requires_reconnect: bool,
    active_account_id: str | None,
    connected_accounts_count: int,
    has_complete_context: bool,
    context_is_fresh: bool,
) -> list[BootstrapNextAction]:
    blocking_actions: list[BootstrapNextAction] = []

    if not has_active_plan:
        blocking_actions.append("start_trial" if current_plan is None else "upgrade_plan")

    if not instagram_connected:
        blocking_actions.append("connect_instagram")
    elif requires_reconnect:
        blocking_actions.append("reconnect_instagram")
    elif not active_account_id and connected_accounts_count > 0:
        blocking_actions.append("run_first_sync")
    elif not has_complete_context or not context_is_fresh:
        blocking_actions.append("run_first_sync")

    if blocking_actions:
        return blocking_actions

    return ["ready_to_generate"]


@router.get("/bootstrap/{user_id}", response_model=AppBootstrapResponse, responses=STANDARD_ERROR_RESPONSES)
def get_app_bootstrap(user_id: str):
    plan = billing_service.peek_plan(user_id)
    accounts_payload = connected_accounts_service.get_accounts(user_id)
    accounts = accounts_payload.get("accounts", [])
    valid_accounts = [
        account for account in accounts
        if _normalize_text(str(account.get("account_id") or ""))
    ]
    primary_account = _resolve_primary_account(valid_accounts)
    active_account_id = _normalize_text(str((primary_account or {}).get("account_id") or ""))
    active_account_label = _normalize_text((primary_account or {}).get("label"))

    connections_payload = instagram_connection_service.get_connections_by_user(user_id)
    connections = connections_payload.get("connections", [])
    reconnect_status = instagram_connection_service.get_reconnect_status(user_id)
    primary_connection = _resolve_primary_connection(connections, active_account_id)

    instagram_connected = any(
        str(connection.get("connection_status") or "") in {"connected", "stale", "failed", "reconnect_required"}
        for connection in connections
    )
    connection_status = _normalize_text(str((primary_connection or {}).get("connection_status") or ""))
    requires_reconnect = bool(reconnect_status.get("requires_reconnect"))
    if primary_connection:
        requires_reconnect = bool(primary_connection.get("requires_reconnect")) or requires_reconnect
    reconnect_message = reconnect_status.get("message") if requires_reconnect else None

    has_profile_context = False
    has_recent_posts_context = False
    has_recent_content_context = False
    last_synced_at = None
    context_is_fresh = False
    has_complete_context = False

    if active_account_id:
        has_profile_context = profile_context_service.get_stored_context(active_account_id) is not None
        has_recent_posts_context = recent_posts_context_service.get_stored_context(active_account_id) is not None
        has_recent_content_context = recent_content_context_service.get_stored_context(active_account_id) is not None
        freshness = instagram_context_sync_service.get_context_freshness(active_account_id)
        last_synced_at = freshness.get("last_synced_at")
        context_is_fresh = bool(freshness.get("context_was_fresh"))
        has_complete_context = bool(freshness.get("has_complete_context"))

    supported_task_types = [
        capability.task_type
        for capability in AGENT_CAPABILITIES.supported_task_types
    ]
    structured_output_task_types = [
        capability.task_type
        for capability in AGENT_CAPABILITIES.supported_task_types
        if capability.structured_output_supported
    ]
    has_active_plan = bool(plan and plan.get("plan_status") in {"active", "trial"})
    next_actions = _derive_next_actions(
        has_active_plan=has_active_plan,
        current_plan=(plan or {}).get("current_plan"),
        instagram_connected=instagram_connected,
        requires_reconnect=requires_reconnect,
        active_account_id=active_account_id,
        connected_accounts_count=len(valid_accounts),
        has_complete_context=has_complete_context,
        context_is_fresh=context_is_fresh,
    )

    return AppBootstrapResponse(
        user_id=user_id,
        has_active_plan=has_active_plan,
        current_plan=(plan or {}).get("current_plan"),
        plan_status=(plan or {}).get("plan_status"),
        monthly_generation_limit=(plan or {}).get("monthly_generation_limit"),
        monthly_generation_used=(plan or {}).get("monthly_generation_used"),
        connected_accounts_count=len(valid_accounts),
        active_account_id=active_account_id,
        active_account_label=active_account_label,
        instagram_connected=instagram_connected,
        connection_status=connection_status,
        requires_reconnect=requires_reconnect,
        reconnect_message=reconnect_message,
        has_profile_context=has_profile_context,
        has_recent_posts_context=has_recent_posts_context,
        has_recent_content_context=has_recent_content_context,
        last_synced_at=last_synced_at,
        context_is_fresh=context_is_fresh,
        supported_task_types=supported_task_types,
        structured_output_task_types=structured_output_task_types,
        auto_sync_supported=AGENT_CAPABILITIES.features.auto_sync,
        active_account_fallback_supported=AGENT_CAPABILITIES.features.active_account_fallback,
        next_actions=next_actions,
    )
