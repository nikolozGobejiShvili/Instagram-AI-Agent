"""Asynchronous generation, for work too slow to answer inside a request.

``POST /api/v1/agent/chat`` stays as it is: it is synchronous, and existing
callers depend on that. This router is the path for generation that runs for
tens of seconds -- carousels with per-slide image generation above all -- which
cannot be served synchronously without timing out the caller and exhausting the
threadpool for every other request.

Contract for the website: POST returns **202 Accepted** with a job id, then poll
``GET /api/v1/generation-jobs/{job_id}`` until ``status`` is ``succeeded`` or
``failed``.
"""
import logging

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.generation_job import (
    GenerationJobCreateRequest,
    GenerationJobListResponse,
    GenerationJobResponse,
)
from app.services.billing_service import BillingService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.job_service import JobService, JobWorker
from app.services.llm_service import LLMService
from app.services.marketing_brief_service import MarketingBriefService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generation-jobs", tags=["generation-jobs"])

job_service = JobService()
billing_service = BillingService()
llm_service = LLMService()
marketing_brief_service = MarketingBriefService()
connected_accounts_service = ConnectedAccountsService()

AGENT_GENERATION = "agent_generation"
CAROUSEL_GENERATION = "carousel_generation"

_GENERATION_FIELDS = (
    "message",
    "task_type",
    "account_id",
    "niche",
    "target_audience",
    "goal",
    "link",
)


def _generation_kwargs(payload: dict) -> dict:
    """Assemble generation inputs, including the customer's stored brief.

    The brief is loaded here rather than passed by the caller: a goal the
    customer stated once should shape every task without the website having to
    remember to resend it.
    """
    kwargs = {k: payload.get(k) for k in _GENERATION_FIELDS}
    account_id = payload.get("account_id")
    try:
        brief = marketing_brief_service.get_brief(
            payload["user_id"],
            connected_accounts_service.resolve_account_id(payload["user_id"], account_id),
        )
        kwargs["brief_context"] = marketing_brief_service.as_prompt_context(brief)
    except Exception as exc:  # noqa: BLE001 - a missing brief must not fail generation
        logger.warning("Could not load marketing brief: %s", exc)
        kwargs["brief_context"] = None
    return kwargs


def run_agent_generation(payload: dict) -> dict:
    """Execute one queued generation.

    Usage is charged **here**, on success, rather than at enqueue time: a job
    that fails before producing anything must not consume the customer's monthly
    allowance. Entitlement was already checked when the job was accepted, so a
    caller cannot use the queue to bypass their plan.
    """
    result = llm_service.run_agent(**_generation_kwargs(payload))
    billing_service.increment_generation_usage(payload["user_id"])
    return result


def run_carousel_generation(payload: dict) -> dict:
    """Generate a carousel: copy and art direction, then backgrounds, then slides.

    This is the reason the job layer exists. One text call plus one image
    generation per slide runs for tens of seconds -- far past what a synchronous
    request can hold without timing out the caller and occupying a threadpool
    worker the whole time.
    """
    from app.services.carousel_pipeline_service import CarouselPipelineService

    generation = llm_service.run_agent(**_generation_kwargs(payload))

    structured = generation.get("structured_output")
    if not isinstance(structured, dict) or not structured.get("slides"):
        # No slides parsed means there is nothing to render. Fail the job rather
        # than storing an empty carousel that looks successful.
        raise ValueError("Carousel generation did not produce any slides")

    rendered = CarouselPipelineService().render_carousel(
        structured_output=structured,
        generate_images=bool(payload.get("generate_images", True)),
    )

    billing_service.increment_generation_usage(payload["user_id"])
    return {**generation, "structured_output": rendered}


job_worker = JobWorker(
    job_service,
    handlers={
        AGENT_GENERATION: run_agent_generation,
        CAROUSEL_GENERATION: run_carousel_generation,
    },
)


@router.post(
    "",
    response_model=GenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=STANDARD_ERROR_RESPONSES,
)
def create_generation_job(payload: GenerationJobCreateRequest, response: Response):
    # Entitlement is enforced before the job is accepted, so an over-quota or
    # unentitled caller gets an immediate, honest refusal instead of a queued job
    # that fails later.
    billing_service.enforce_agent_access(payload.user_id, payload.task_type)

    # A carousel needs the render pipeline; everything else is text only.
    kind = CAROUSEL_GENERATION if payload.task_type == "carousel" else AGENT_GENERATION

    job = job_service.enqueue(
        user_id=payload.user_id,
        kind=kind,
        payload=payload.model_dump(),
    )
    response.headers["Location"] = f"{router.prefix}/{job['job_id']}"
    return _serialize(job)


@router.get("/{job_id}", response_model=GenerationJobResponse, responses=STANDARD_ERROR_RESPONSES)
def get_generation_job(job_id: str):
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Generation job '{job_id}' was not found")
    return _serialize(job)


@router.get("", response_model=GenerationJobListResponse, responses=STANDARD_ERROR_RESPONSES)
def list_generation_jobs(user_id: str = Query(..., min_length=1), limit: int = Query(50, ge=1, le=200)):
    return {"jobs": [_serialize(job) for job in job_service.list_for_user(user_id, limit=limit)]}


def _serialize(job: dict) -> dict:
    """Strip the stored payload from the response.

    The payload echoes the caller's own request and can carry account context;
    the job record is a status resource, not a mirror of the submission.
    """
    return {key: value for key, value in job.items() if key != "payload"}
