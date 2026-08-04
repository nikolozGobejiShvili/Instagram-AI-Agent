from pydantic import BaseModel


class GenerationHistoryItem(BaseModel):
    log_id: str
    timestamp: str
    user_id: str | None
    account_id: str | None
    task_type: str
    status: str
    message_preview: str
    response_preview: str | None
    error_message: str | None
    niche: str | None = None
    target_audience: str | None = None
    goal: str | None = None
    auto_sync: bool | None = None
    sync_attempted: bool | None = None
    sync_succeeded: bool | None = None
    context_was_fresh: bool | None = None
    sync_skipped: bool | None = None
    last_synced_at: str | None = None
    used_real_instagram_context: bool | None = None
