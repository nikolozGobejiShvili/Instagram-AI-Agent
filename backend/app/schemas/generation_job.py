from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent import TaskType
from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class GenerationJobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {
            "user_id": "user-1",
            "task_type": "carousel",
            "message": "მომეცი კარუსელი ჩემი ახალი პროდუქტისთვის",
            "goal": "increase qualified inbound leads from Instagram",
            "slide_count": 8,
        }
    })

    user_id: str = Field(..., min_length=1)
    task_type: TaskType
    message: str = Field(..., min_length=1)
    account_id: str | None = None
    niche: str | None = None
    target_audience: str | None = None
    goal: str | None = None
    link: str | None = None

    # Carousel only. The upper bound here is the parser's ceiling, not the
    # customer's: a request above the plan's slide limit is clamped rather than
    # rejected, so asking for more than the tier covers returns the carousel the
    # tier does cover instead of an error.
    slide_count: int | None = Field(default=None, ge=2, le=MAX_CAROUSEL_SLIDES)
    # A text-only carousel skips the per-slide image generation entirely, which
    # is what makes a cheap preview possible. Default stays True so the normal
    # request produces the finished article.
    generate_images: bool = True


class GenerationJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {
            "job_id": "9f2c1a4b7e8d4f0a",
            "user_id": "user-1",
            "kind": "agent_generation",
            "status": "queued",
            "result": None,
            "error": None,
            "attempts": 0,
            "created_at": "2026-08-04T12:00:00+00:00",
            "updated_at": "2026-08-04T12:00:00+00:00",
        }
    })

    job_id: str
    user_id: str
    kind: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    created_at: str
    updated_at: str


class GenerationJobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[GenerationJobResponse]
