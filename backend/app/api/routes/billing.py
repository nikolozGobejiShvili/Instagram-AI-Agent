from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.billing import BillingPlanResponse, BillingPlanSetPayload, BillingUsageResetResponse
from app.services.billing_service import BillingService

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])

billing_service = BillingService()


@router.get("/plan/{user_id}", response_model=BillingPlanResponse, responses=STANDARD_ERROR_RESPONSES)
def get_billing_plan(user_id: str):
    return billing_service.get_plan(user_id)


@router.post("/plan/{user_id}/set", response_model=BillingPlanResponse, responses=STANDARD_ERROR_RESPONSES)
def set_billing_plan(user_id: str, payload: BillingPlanSetPayload):
    return billing_service.set_plan(user_id, payload.model_dump(exclude_none=True))


@router.post("/usage/{user_id}/reset-month", response_model=BillingUsageResetResponse, responses=STANDARD_ERROR_RESPONSES)
def reset_billing_usage(user_id: str):
    return billing_service.reset_monthly_usage(user_id)
