"""The competitor watch list.

Adding and listing need no Instagram connection — they are the customer's own
list. Refreshing does, because business_discovery is a field on the caller's own
Instagram Business account and there is no anonymous form of that call.

Refreshing is not charged as a generation. It is one Meta read with no model
behind it, and the thing that bounds it is the plan's `tracked_accounts_limit` —
a field that already existed in every tier and was published in the catalogue
while nothing enforced it.
"""
import logging

from fastapi import APIRouter, Request

from app.api.user_auth import resolve_user_id
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.tracked_account import (
    TrackedAccountAddRequest,
    TrackedAccountChangeResponse,
    TrackedAccountListResponse,
)
from app.services.billing_service import BillingService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.public_profile_service import PublicProfileService
from app.services.tracked_account_service import TrackedAccountService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tracked-accounts", tags=["tracked-accounts"])

tracked_account_service = TrackedAccountService()
public_profile_service = PublicProfileService()
billing_service = BillingService()
connected_accounts_service = ConnectedAccountsService()


@router.post("", response_model=TrackedAccountListResponse, responses=STANDARD_ERROR_RESPONSES)
def add_tracked_account(payload: TrackedAccountAddRequest, request: Request):
    user_id = resolve_user_id(request, payload.user_id)
    plan = billing_service.get_plan(user_id)

    # Normalised before storage so the same account added as "nike", "@nike" and
    # a profile URL is one entry rather than three, each with its own history.
    handle = PublicProfileService.normalize_handle(payload.handle)

    tracked_account_service.add(
        user_id=user_id,
        handle=handle,
        label=payload.label,
        limit=int(plan["tracked_accounts_limit"]),
    )
    return _listing(user_id, plan)


@router.get("", response_model=TrackedAccountListResponse, responses=STANDARD_ERROR_RESPONSES)
def list_tracked_accounts(request: Request, user_id: str | None = None):
    resolved = resolve_user_id(request, user_id)
    return _listing(resolved, billing_service.get_plan(resolved))


@router.delete("/{handle}", response_model=TrackedAccountListResponse, responses=STANDARD_ERROR_RESPONSES)
def remove_tracked_account(handle: str, request: Request, user_id: str | None = None):
    resolved = resolve_user_id(request, user_id)
    tracked_account_service.remove(user_id=resolved, handle=PublicProfileService.normalize_handle(handle))
    return _listing(resolved, billing_service.get_plan(resolved))


@router.post(
    "/{handle}/refresh",
    response_model=TrackedAccountChangeResponse,
    responses=STANDARD_ERROR_RESPONSES,
)
def refresh_tracked_account(handle: str, request: Request, user_id: str | None = None, account_id: str | None = None):
    """Read the account again and report what changed since the last check."""
    resolved = resolve_user_id(request, user_id)
    normalized = PublicProfileService.normalize_handle(handle)

    profile = public_profile_service.fetch(
        user_id=resolved,
        account_id=connected_accounts_service.resolve_account_id(resolved, account_id),
        handle=normalized,
    )
    return tracked_account_service.compare_and_store(
        user_id=resolved, handle=normalized, profile=profile
    )


def _listing(user_id: str, plan: dict) -> dict:
    return {
        "user_id": user_id,
        "tracked_accounts": tracked_account_service.list_tracked(user_id),
        # Returned with the list so a client can show "3 of 10" without a second
        # request against the catalogue.
        "tracked_accounts_limit": int(plan["tracked_accounts_limit"]),
    }
