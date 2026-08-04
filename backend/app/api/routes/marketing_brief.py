"""The customer's marketing brief: state the goal once, steer every generation.

Reads return an empty brief rather than 404 when none has been set — the website
renders the same form either way, and "not filled in yet" is a normal state, not
an error.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.marketing_brief import MarketingBriefPayload, MarketingBriefResponse
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.marketing_brief_service import MarketingBriefService

router = APIRouter(prefix="/api/v1/marketing-brief", tags=["marketing-brief"])

marketing_brief_service = MarketingBriefService()
connected_accounts_service = ConnectedAccountsService()


def _resolve_account(user_id: str, account_id: str | None) -> str:
    """Reuse the agent's account-resolution so the brief attaches to the same account."""
    return connected_accounts_service.resolve_account_id(user_id, account_id)


@router.get("/{user_id}", response_model=MarketingBriefResponse, responses=STANDARD_ERROR_RESPONSES)
def get_marketing_brief(user_id: str, account_id: str | None = None):
    return marketing_brief_service.get_brief(user_id, _resolve_account(user_id, account_id))


@router.post("/{user_id}", response_model=MarketingBriefResponse, responses=STANDARD_ERROR_RESPONSES)
def save_marketing_brief(user_id: str, payload: MarketingBriefPayload, account_id: str | None = None):
    # exclude_unset so an omitted field keeps its stored value: the brief can be
    # filled in progressively instead of resent whole on every edit.
    return marketing_brief_service.save_brief(
        user_id,
        _resolve_account(user_id, account_id),
        payload.model_dump(exclude_unset=True),
    )


@router.delete("/{user_id}", response_model=MarketingBriefResponse, responses=STANDARD_ERROR_RESPONSES)
def delete_marketing_brief(user_id: str, account_id: str | None = None):
    resolved = _resolve_account(user_id, account_id)
    if not marketing_brief_service.delete_brief(user_id, resolved):
        raise HTTPException(status_code=404, detail="No marketing brief exists for this account")
    return marketing_brief_service.get_brief(user_id, resolved)
