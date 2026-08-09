from pydantic import BaseModel, ConfigDict, Field


class TrackedAccountAddRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"user_id": "user-1", "handle": "@nike", "label": "format reference"}},
    )

    user_id: str = Field(..., min_length=1)
    # A handle, an @handle or the profile URL — normalised server-side, because a
    # customer pastes whichever their browser gave them.
    handle: str = Field(..., min_length=1)
    # Why this account is on the list. A watch list of five handles with no note
    # is unreadable a month later.
    label: str | None = None


class TrackedAccountItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str
    label: str | None = None
    added_at: str
    # Null until the first refresh. Shown so a stale entry is visible as stale
    # rather than looking like an account that never changes.
    last_checked_at: str | None = None
    followers_count: int | None = None
    media_count: int | None = None


class TrackedAccountListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    tracked_accounts: list[TrackedAccountItem] = Field(default_factory=list)
    tracked_accounts_limit: int


class TrackedPostItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_id: str | None = None
    caption: str | None = None
    media_type: str | None = None
    product_type: str | None = None
    like_count: int | None = None
    comment_count: int | None = None
    permalink: str | None = None
    posted_at: str | None = None


class TrackedAccountChangeResponse(BaseModel):
    """What changed since the previous check — the reason this feature exists.

    A snapshot restates facts nobody needs. The comparison is what changes what
    the customer publishes next, so the deltas are the response and the raw
    totals are supporting detail.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "handle": "nike",
                "is_first_check": False,
                "since": "2026-08-02T09:00:00+00:00",
                "new_posts": [],
                "followers_change": 120000,
                "median_likes_change": -400,
                "followers_count": 302000000,
                "media_count": 1204,
                "median_likes": 24000,
                "post_ids": ["18001", "18002"],
            }
        },
    )

    handle: str
    # True on the first reading. Deltas are null rather than measured against
    # zero — "+302,000,000 followers" is derivable and useless.
    is_first_check: bool
    since: str | None = None
    new_posts: list[TrackedPostItem] = Field(default_factory=list)
    followers_change: int | None = None
    median_likes_change: int | None = None
    followers_count: int | None = None
    media_count: int | None = None
    median_likes: int | None = None
    post_ids: list[str] = Field(default_factory=list)
