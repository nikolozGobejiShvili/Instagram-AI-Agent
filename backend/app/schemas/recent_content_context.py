from pydantic import BaseModel


class RecentContentContextResponse(BaseModel):
    account_id: str
    top_formats: list[str]
    best_topics: list[str]
    weak_topics: list[str]
    best_ctas: list[str]
    weak_ctas: list[str]
    notes: list[str]