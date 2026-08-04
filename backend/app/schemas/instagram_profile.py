from pydantic import BaseModel


class InstagramProfileResponse(BaseModel):
    user_id: str
    account_id: str
    platform: str
    instagram_username: str
    display_name: str
    biography: str
    followers_count: int
    following_count: int
    media_count: int
    is_connected: bool
