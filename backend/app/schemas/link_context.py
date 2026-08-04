from pydantic import BaseModel, HttpUrl


class LinkContextRequest(BaseModel):
    link: HttpUrl


class LinkContextResponse(BaseModel):
    link: str
    source_type: str
    detected_platform: str
    summary: str
    content_type: str
    hook_style: str
    probable_strengths: list[str]
    adaptation_notes: list[str]