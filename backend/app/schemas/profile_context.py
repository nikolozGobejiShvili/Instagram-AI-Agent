from pydantic import BaseModel


class ProfileContextPayload(BaseModel):
    account_id: str
    brand_name: str
    niche: str
    target_audience: str
    brand_voice: str
    bio: str
    content_focus: list[str]
    strengths: list[str]
    weak_points: list[str]