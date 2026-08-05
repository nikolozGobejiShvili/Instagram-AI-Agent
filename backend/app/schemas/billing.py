from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


BillingPlanId = Literal["trial", "creator", "pro", "agency"]
BillingPlanStatus = Literal["active", "expired", "cancelled", "trial"]


class BillingPlanResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "user_id": "user-1",
                    "current_plan": "trial",
                    "plan_status": "trial",
                    "plan_started_at": "2026-04-21T09:00:00+00:00",
                    "plan_expires_at": "2026-05-05T09:00:00+00:00",
                    "monthly_generation_limit": 15,
                    "monthly_generation_used": 3,
                    "monthly_generation_remaining": 12,
                    "connected_account_limit": 1,
                    "tracked_accounts_limit": 0,
                    "carousel_slide_limit": 5,
                    "allowed_task_types": [
                        "reel_idea",
                        "caption",
                        "performance_summary",
                        "profile_audit",
                        "carousel",
                    ],
                    "usage_month": "2026-04",
                },
                {
                    "user_id": "user-2",
                    "current_plan": "pro",
                    "plan_status": "active",
                    "plan_started_at": "2026-04-01T00:00:00+00:00",
                    "plan_expires_at": None,
                    "monthly_generation_limit": 400,
                    "monthly_generation_used": 42,
                    "monthly_generation_remaining": 358,
                    "connected_account_limit": 2,
                    "tracked_accounts_limit": 10,
                    "carousel_slide_limit": 10,
                    "allowed_task_types": [
                        "reel_idea",
                        "caption",
                        "performance_summary",
                        "profile_audit",
                        "reel_script",
                        "carousel",
                        "content_plan",
                        "link_analysis",
                    ],
                    "usage_month": "2026-04",
                },
            ]
        },
    )

    user_id: str
    current_plan: BillingPlanId
    plan_status: BillingPlanStatus
    plan_started_at: str
    plan_expires_at: str | None = None
    monthly_generation_limit: int
    monthly_generation_used: int
    monthly_generation_remaining: int
    connected_account_limit: int
    tracked_accounts_limit: int
    carousel_slide_limit: int
    allowed_task_types: list[str] = Field(default_factory=list)
    usage_month: str


class BillingPlanCatalogueEntry(BaseModel):
    """One purchasable tier, described without reference to any customer."""

    model_config = ConfigDict(extra="forbid")

    plan_id: BillingPlanId
    connected_account_limit: int
    tracked_accounts_limit: int
    monthly_generation_limit: int
    carousel_slide_limit: int
    allowed_task_types: list[str] = Field(default_factory=list)
    # Only the trial expires on its own, so this is null for every paid tier
    # rather than a shared default that would imply paid plans lapse too.
    trial_duration_days: int | None = None


class BillingPlanCatalogueResponse(BaseModel):
    """What the storefront needs to render pricing before anyone signs up.

    Published because the tier table previously existed only inside the service:
    a site selling these subscriptions had no way to ask what it was selling, and
    would have had to keep a second hard-coded copy that drifts the first time a
    limit changes here.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "plans": [
                    {
                        "plan_id": "creator",
                        "connected_account_limit": 1,
                        "tracked_accounts_limit": 3,
                        "monthly_generation_limit": 120,
                        "carousel_slide_limit": 7,
                        "allowed_task_types": ["chat", "caption", "carousel"],
                        "trial_duration_days": None,
                    }
                ],
                "generation_costs": {"chat": 1, "profile_audit": 2, "carousel": 5},
                "default_generation_cost": 1,
            }
        },
    )

    plans: list[BillingPlanCatalogueEntry]
    # Tasks are not equally expensive to serve, so a credit does not buy a fixed
    # amount of work. Publishing the weights lets a client show what an action
    # will cost before the customer commits to it.
    generation_costs: dict[str, int]
    default_generation_cost: int


class BillingPlanSetPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "current_plan": "trial",
                    "plan_status": "trial",
                },
                {
                    "current_plan": "pro",
                    "plan_status": "active",
                    "plan_started_at": "2026-04-01T00:00:00+00:00",
                },
            ]
        },
    )

    current_plan: BillingPlanId
    plan_status: BillingPlanStatus | None = None
    plan_started_at: str | None = None
    plan_expires_at: str | None = None


class BillingUsageResetResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "user-1",
                "current_plan": "trial",
                "usage_month": "2026-04",
                "monthly_generation_used": 0,
                "monthly_generation_limit": 15,
            }
        },
    )

    user_id: str
    current_plan: BillingPlanId
    usage_month: str
    monthly_generation_used: int
    monthly_generation_limit: int
