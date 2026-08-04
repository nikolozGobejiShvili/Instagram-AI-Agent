from fastapi import APIRouter, HTTPException

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.connected_accounts import ConnectedAccountsResponse, ConnectedAccountsSavePayload, SetActiveAccountPayload
from app.services.billing_service import BillingService
from app.services.connected_accounts_service import ConnectedAccountsService

router = APIRouter(prefix="/api/v1/connected-accounts", tags=["connected-accounts"])

connected_accounts_service = ConnectedAccountsService()
billing_service = BillingService()


@router.get("/{user_id}", response_model=ConnectedAccountsResponse)
def get_connected_accounts(user_id: str):
    return connected_accounts_service.get_accounts(user_id)


@router.post("", response_model=ConnectedAccountsResponse, responses=STANDARD_ERROR_RESPONSES)
def save_connected_accounts(payload: ConnectedAccountsSavePayload):
    billing_service.assert_connected_account_limit(payload.user_id, len(payload.accounts))
    return connected_accounts_service.save_accounts(payload.model_dump())


@router.post("/{user_id}/set-active", response_model=ConnectedAccountsResponse)
def set_active_connected_account(user_id: str, payload: SetActiveAccountPayload):
    try:
        return connected_accounts_service.set_active_account(user_id, payload.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
