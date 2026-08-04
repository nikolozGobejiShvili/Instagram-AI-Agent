from pydantic import BaseModel


class InstagramInsightsResponse(BaseModel):
    user_id: str
    account_id: str
    platform: str
    period: str
    followers_count: int | None = None
    reach: int | None = None
    impressions: int | None = None
    profile_views: int | None = None
    website_clicks: int | None = None
    total_likes: int | None = None
    total_comments: int | None = None
    total_saves: int | None = None
    total_shares: int | None = None
    reels_views: int | None = None
    top_content_type: str | None = None
