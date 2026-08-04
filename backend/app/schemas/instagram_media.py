from pydantic import BaseModel


class InstagramMediaItem(BaseModel):
    media_id: str
    account_id: str
    media_type: str
    caption: str
    permalink: str
    thumbnail_url: str | None = None
    media_url: str | None = None
    timestamp: str
    like_count: int | None = None
    comments_count: int | None = None


class InstagramMediaListResponse(BaseModel):
    user_id: str
    account_id: str
    platform: str
    items: list[InstagramMediaItem]
