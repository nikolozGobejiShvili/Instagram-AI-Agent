import os
from datetime import datetime, timezone
from secrets import token_urlsafe
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.instagram_connection import (
    InstagramConnectionCallbackRequest,
    InstagramConnectionRecord,
    InstagramConnectionHealthResponse,
    InstagramReconnectStatusResponse,
    InstagramConnectionStartRequest,
    InstagramConnectionStartResponse,
    InstagramConnectionStatusResponse,
)
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService

router = APIRouter(prefix="/api/v1/instagram-connections", tags=["instagram-connections"])

instagram_connection_service = InstagramConnectionService()
connected_accounts_service = ConnectedAccountsService()


def _build_oauth_start_url(state: str) -> str:
    auth_base_url = os.getenv("META_AUTH_BASE_URL", "").strip()
    app_id = os.getenv("META_APP_ID", "").strip()
    redirect_uri = os.getenv("META_REDIRECT_URI", "").strip()
    scopes = os.getenv("META_SCOPES", "").strip()

    if not auth_base_url or not app_id or not redirect_uri or not scopes:
        raise HTTPException(
            status_code=500,
            detail="Instagram OAuth config is incomplete",
        )

    query = urlencode({
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "response_type": "code",
        "state": state,
    })

    return f"{auth_base_url}?{query}"


def _exchange_connection_by_state(state: str, code: str) -> dict:
    record = instagram_connection_service.get_connection_by_state(state)
    if not record:
        raise HTTPException(status_code=404, detail="Instagram connection state was not found")

    if record.get("status") != "pending":
        raise HTTPException(status_code=400, detail="Instagram connection is not pending")

    if os.getenv("META_MOCK_MODE", "").lower() == "true":
        updated_record = instagram_connection_service.update_connection_by_state(
            state,
            status="connected",
            connection_status="connected",
            access_token="mock-token",
            refresh_token=None,
            connected_at=datetime.now(timezone.utc).isoformat(),
            error_message=None,
            last_error_code=None,
            last_error_message=None,
            last_failed_meta_call_at=None,
            last_successful_meta_call_at=None,
            token_last_checked_at=None,
            requires_reconnect=False,
            health_checked_at=None,
        )
        _mirror_connected_account_selector(updated_record)
        return updated_record

    try:
        updated_record = instagram_connection_service.exchange_code_for_connection(
            state,
            code,
        )
        _mirror_connected_account_selector(updated_record)
        return updated_record
    except RuntimeError as exc:
        if str(exc) == "Meta token exchange timed out":
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        if str(exc) == "Meta OAuth token config is incomplete":
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _mirror_connected_account_selector(record: dict) -> None:
    if record.get("status") != "connected" or record.get("connection_status") != "connected":
        return

    user_id = str(record.get("user_id") or "").strip()
    account_id = str(record.get("account_id") or "").strip()
    if not user_id or not account_id:
        return

    connected_accounts_service.upsert_account(
        user_id=user_id,
        account_id=account_id,
        label=account_id,
        niche="",
        is_active=False,
    )


@router.post("/start", response_model=InstagramConnectionStartResponse, responses=STANDARD_ERROR_RESPONSES)
def start_instagram_connection(payload: InstagramConnectionStartRequest):
    state = token_urlsafe(24)
    record_payload = {
        "user_id": payload.user_id,
        "account_id": payload.account_id,
        "state": state,
        "status": "pending",
        "connection_status": "disconnected",
        "platform": "instagram",
        "access_token": None,
        "refresh_token": None,
        "connected_at": None,
        "error_message": None,
        "last_successful_meta_call_at": None,
        "last_failed_meta_call_at": None,
        "last_error_code": None,
        "last_error_message": None,
        "token_last_checked_at": None,
        "requires_reconnect": False,
        "health_checked_at": None,
    }

    saved_record = instagram_connection_service.save_or_update_record(record_payload)
    oauth_start_url = _build_oauth_start_url(state)

    return InstagramConnectionStartResponse(
        oauth_start_url=oauth_start_url,
        connection=InstagramConnectionRecord(**saved_record),
    )


@router.post("/callback", response_model=InstagramConnectionRecord, responses=STANDARD_ERROR_RESPONSES)
def instagram_connection_callback(payload: InstagramConnectionCallbackRequest):
    updated_record = _exchange_connection_by_state(payload.state, payload.code)
    return InstagramConnectionRecord(**updated_record)


@router.get("/callback", response_class=HTMLResponse, responses=STANDARD_ERROR_RESPONSES)
def instagram_connection_callback_redirect(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_reason: str | None = None,
    error_description: str | None = None,
):
    if error:
        error_parts = [error]
        if error_reason:
            error_parts.append(error_reason)
        if error_description:
            error_parts.append(error_description)
        raise HTTPException(status_code=400, detail="Meta OAuth callback failed: " + " | ".join(error_parts))

    if not code or not state:
        raise HTTPException(status_code=400, detail="Meta OAuth callback is missing code or state")

    updated_record = _exchange_connection_by_state(state, code)

    return HTMLResponse(
        content=(
            "<html><body>"
            "<h2>Instagram connection completed</h2>"
            f"<p>Account ID: {updated_record.get('account_id', '')}</p>"
            "<p>You can close this window and return to the app.</p>"
            "</body></html>"
        ),
        status_code=200,
    )


@router.get("/{user_id}/health", response_model=InstagramConnectionHealthResponse, responses=STANDARD_ERROR_RESPONSES)
def get_instagram_connection_health(user_id: str, account_id: str | None = None):
    return instagram_connection_service.get_connection_health(user_id, account_id)


@router.get(
    "/{user_id}/reconnect-status",
    response_model=InstagramReconnectStatusResponse,
    responses=STANDARD_ERROR_RESPONSES,
)
def get_instagram_connection_reconnect_status(user_id: str):
    return instagram_connection_service.get_reconnect_status(user_id)


@router.get("/{user_id}", response_model=InstagramConnectionStatusResponse, responses=STANDARD_ERROR_RESPONSES)
def get_instagram_connections(user_id: str):
    return instagram_connection_service.get_connections_by_user(user_id)
