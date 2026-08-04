from pydantic import BaseModel, ConfigDict


class InstagramConnectionRecord(BaseModel):
    user_id: str
    account_id: str
    state: str
    status: str
    connection_status: str
    platform: str
    connected_at: str | None = None
    error_message: str | None = None
    last_successful_meta_call_at: str | None = None
    last_failed_meta_call_at: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    token_last_checked_at: str | None = None
    requires_reconnect: bool = False
    health_checked_at: str | None = None


class InstagramConnectionStartRequest(BaseModel):
    user_id: str
    account_id: str


class InstagramConnectionCallbackRequest(BaseModel):
    code: str
    state: str


class InstagramConnectionStatusResponse(BaseModel):
    user_id: str
    connections: list[InstagramConnectionRecord]


class InstagramConnectionStartResponse(BaseModel):
    oauth_start_url: str
    connection: InstagramConnectionRecord


class InstagramConnectionHealthRecord(BaseModel):
    user_id: str
    account_id: str
    connection_status: str
    requires_reconnect: bool
    token_last_checked_at: str | None = None
    last_successful_meta_call_at: str | None = None
    last_failed_meta_call_at: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    health_checked_at: str | None = None


class InstagramConnectionHealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user-1",
                "connections": [
                    {
                        "user_id": "user-1",
                        "account_id": "test-account-3",
                        "connection_status": "connected",
                        "requires_reconnect": False,
                        "token_last_checked_at": "2026-04-21T12:10:00+00:00",
                        "last_successful_meta_call_at": "2026-04-21T12:10:00+00:00",
                        "last_failed_meta_call_at": None,
                        "last_error_code": None,
                        "last_error_message": None,
                        "health_checked_at": "2026-04-21T12:10:00+00:00",
                    }
                ],
            }
        }
    )

    user_id: str
    connections: list[InstagramConnectionHealthRecord]


class InstagramReconnectStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user-1",
                "requires_reconnect": True,
                "account_ids": ["test-account-3"],
                "message": "One or more Instagram accounts must be reconnected.",
            }
        }
    )

    user_id: str
    requires_reconnect: bool
    account_ids: list[str]
    message: str
