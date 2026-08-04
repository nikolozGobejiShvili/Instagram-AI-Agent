from pydantic import BaseModel


class RecentPostItem(BaseModel):
    post_id: str
    content_type: str
    topic: str
    caption: str
    views: int
    likes: int
    comments: int
    saves: int


class RecentPostsContextResponse(BaseModel):
    account_id: str
    posts: list[RecentPostItem]