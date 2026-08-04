from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent import TaskType


BootstrapNextAction = Literal[
    "start_trial",
    "upgrade_plan",
    "connect_instagram",
    "reconnect_instagram",
    "run_first_sync",
    "ready_to_generate",
]

PlanId = Literal["trial", "creator", "pro", "agency"]
PlanStatus = Literal["trial", "active", "expired", "cancelled"]


class AppBootstrapResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "user_id": "user-1",
                    "has_active_plan": True,
                    "current_plan": "pro",
                    "plan_status": "active",
                    "monthly_generation_limit": 400,
                    "monthly_generation_used": 28,
                    "connected_accounts_count": 1,
                    "active_account_id": "test-account-3",
                    "active_account_label": "Main Instagram",
                    "instagram_connected": True,
                    "connection_status": "connected",
                    "requires_reconnect": False,
                    "reconnect_message": None,
                    "has_profile_context": True,
                    "has_recent_posts_context": True,
                    "has_recent_content_context": True,
                    "last_synced_at": "2026-04-22T08:30:00+00:00",
                    "context_is_fresh": True,
                    "supported_task_types": [
                        "reel_idea",
                        "reel_script",
                        "reel_feedback",
                        "caption",
                        "carousel",
                        "profile_audit",
                        "content_plan",
                        "link_analysis",
                        "performance_summary",
                    ],
                    "structured_output_task_types": [
                        "reel_idea",
                        "reel_script",
                        "reel_feedback",
                        "caption",
                        "carousel",
                        "profile_audit",
                        "content_plan",
                        "link_analysis",
                        "performance_summary",
                    ],
                    "auto_sync_supported": True,
                    "active_account_fallback_supported": True,
                    "next_actions": [
                        "ready_to_generate",
                    ],
                },
                {
                    "user_id": "new-user-1",
                    "has_active_plan": False,
                    "current_plan": None,
                    "plan_status": None,
                    "monthly_generation_limit": None,
                    "monthly_generation_used": None,
                    "connected_accounts_count": 0,
                    "active_account_id": None,
                    "active_account_label": None,
                    "instagram_connected": False,
                    "connection_status": None,
                    "requires_reconnect": False,
                    "reconnect_message": None,
                    "has_profile_context": False,
                    "has_recent_posts_context": False,
                    "has_recent_content_context": False,
                    "last_synced_at": None,
                    "context_is_fresh": False,
                    "supported_task_types": [
                        "reel_idea",
                        "reel_script",
                        "reel_feedback",
                        "caption",
                        "carousel",
                        "profile_audit",
                        "content_plan",
                        "link_analysis",
                        "performance_summary",
                    ],
                    "structured_output_task_types": [
                        "reel_idea",
                        "reel_script",
                        "reel_feedback",
                        "caption",
                        "carousel",
                        "profile_audit",
                        "content_plan",
                        "link_analysis",
                        "performance_summary",
                    ],
                    "auto_sync_supported": True,
                    "active_account_fallback_supported": True,
                    "next_actions": [
                        "start_trial",
                        "connect_instagram",
                    ],
                },
            ]
        },
    )

    user_id: str
    has_active_plan: bool
    current_plan: PlanId | None = None
    plan_status: PlanStatus | None = None
    monthly_generation_limit: int | None = None
    monthly_generation_used: int | None = None
    connected_accounts_count: int
    active_account_id: str | None = None
    active_account_label: str | None = None
    instagram_connected: bool
    connection_status: str | None = None
    requires_reconnect: bool
    reconnect_message: str | None = None
    has_profile_context: bool
    has_recent_posts_context: bool
    has_recent_content_context: bool
    last_synced_at: str | None = None
    context_is_fresh: bool
    supported_task_types: list[TaskType] = Field(default_factory=list)
    structured_output_task_types: list[TaskType] = Field(default_factory=list)
    auto_sync_supported: bool
    active_account_fallback_supported: bool
    next_actions: list[BootstrapNextAction] = Field(default_factory=list)
