from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FunnelStage = Literal["awareness", "consideration", "conversion", "retention"]

BRIEF_EXAMPLE = {
    "user_id": "user-1",
    "account_id": "acct-1",
    "objective": "Book 10 discovery calls per month from Instagram",
    "offer": "1:1 Instagram strategy intensive, 400 GEL",
    "ideal_customer": "Georgian service businesses with 2k-20k followers",
    "funnel_stage": "conversion",
    "tone_of_voice": "direct, warm, no hype",
    "topics_to_avoid": ["competitor comparisons", "discount promises"],
    "primary_kpi": "qualified DM conversations per week",
}


class MarketingBriefPayload(BaseModel):
    """What the customer is actually trying to achieve.

    Stored once per (user, account) and fed into every generation, rather than
    re-supplied per request and forgotten. Every field is optional so a brief can
    be filled in progressively — a partial brief is more useful than none.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": BRIEF_EXAMPLE})

    objective: str | None = Field(default=None, max_length=500)
    offer: str | None = Field(default=None, max_length=500)
    ideal_customer: str | None = Field(default=None, max_length=500)
    funnel_stage: FunnelStage | None = None
    tone_of_voice: str | None = Field(default=None, max_length=300)
    topics_to_avoid: list[str] = Field(default_factory=list, max_length=20)
    primary_kpi: str | None = Field(default=None, max_length=200)


class MarketingBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": BRIEF_EXAMPLE})

    user_id: str
    account_id: str
    objective: str | None = None
    offer: str | None = None
    ideal_customer: str | None = None
    funnel_stage: FunnelStage | None = None
    tone_of_voice: str | None = None
    topics_to_avoid: list[str] = Field(default_factory=list)
    primary_kpi: str | None = None
    is_empty: bool = False
    updated_at: str | None = None
