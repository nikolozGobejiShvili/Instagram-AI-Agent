from fastapi import APIRouter, Depends

from app.api.internal_auth import require_internal_admin_access
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.billing import (
    BillingPlanCatalogueResponse,
    BillingPlanResponse,
    BillingPlanSetPayload,
    BillingUsageResetResponse,
)
from app.services.billing_service import BillingService

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])

billing_service = BillingService()

# Changing a plan or clearing a usage counter is an operator action: without a
# guard, any caller able to reach the API could put itself on the top plan or
# wipe the usage it had been billed for. Reads stay open -- reading a plan is not
# privilege escalation, and end-user identity is a separate, still-open decision.
ADMIN_ONLY = [Depends(require_internal_admin_access)]


@router.get(
    "/plans",
    response_model=BillingPlanCatalogueResponse,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_plan_catalogue():
    """The tier table, for a pricing page. Deliberately requires no customer.

    Declared before ``/plan/{user_id}``-style paths in this router so the literal
    segment is never shadowed by a path parameter.
    """
    return BillingService.plan_catalogue()


@router.get("/plan/{user_id}", response_model=BillingPlanResponse, responses=STANDARD_ERROR_RESPONSES)
def get_billing_plan(user_id: str):
    return billing_service.get_plan(user_id)


@router.post(
    "/plan/{user_id}/set",
    response_model=BillingPlanResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def set_billing_plan(user_id: str, payload: BillingPlanSetPayload):
    return billing_service.set_plan(user_id, payload.model_dump(exclude_none=True))


@router.post(
    "/usage/{user_id}/reset-month",
    response_model=BillingUsageResetResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def reset_billing_usage(user_id: str):
    return billing_service.reset_monthly_usage(user_id)
