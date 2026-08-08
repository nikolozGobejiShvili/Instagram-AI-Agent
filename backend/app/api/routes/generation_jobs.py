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

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.api.user_auth import enforcement_enabled, resolve_user_id
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.generation_job import (
    GenerationJobCreateRequest,
    GenerationJobListResponse,
    GenerationJobResponse,
)
from app.services.billing_service import BillingService
from app.services.carousel_pipeline_service import CarouselPipelineService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.job_service import JobService, JobWorker
from app.services.llm_service import LLMService
from app.services.marketing_brief_service import MarketingBriefService
from app.services.public_profile_service import PublicProfileService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generation-jobs", tags=["generation-jobs"])

job_service = JobService()
billing_service = BillingService()
llm_service = LLMService()
marketing_brief_service = MarketingBriefService()
connected_accounts_service = ConnectedAccountsService()
carousel_pipeline_service = CarouselPipelineService()
public_profile_service = PublicProfileService()

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
    # Carousel only, and already clamped to the plan when the job was accepted.
    # It reaches the prompt so the model is asked for the right number of slides
    # rather than a fixed five that later has to be trimmed.
    "slide_count",
)

# Not in _GENERATION_FIELDS: it is an input to the Meta lookup above, not
# something run_agent takes. Passing it through would be an unexpected keyword.
_REFERENCE_HANDLE_FIELD = "reference_handle"


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

    # Unlike the brief, this one is not optional. The whole task is "read that
    # account and adapt it"; without the data the model would answer from what it
    # remembers about the brand, which is exactly the generic reply the customer
    # could have got for free.
    if payload.get("task_type") == "public_profile_analysis":
        profile = public_profile_service.fetch(
            user_id=payload["user_id"],
            account_id=connected_accounts_service.resolve_account_id(
                payload["user_id"], payload.get("account_id")
            ),
            handle=payload.get("reference_handle") or payload.get("link") or "",
        )
        kwargs["public_profile_context"] = public_profile_service.as_prompt_context(profile)

    return kwargs


def run_agent_generation(payload: dict) -> dict:
    """Execute one queued generation.

    Usage is charged **here**, on success, rather than at enqueue time: a job
    that fails before producing anything must not consume the customer's monthly
    allowance. Entitlement was already checked when the job was accepted, so a
    caller cannot use the queue to bypass their plan.
    """
    result = llm_service.run_agent(**_generation_kwargs(payload))
    billing_service.increment_generation_usage(payload["user_id"], payload.get("task_type"))
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

    # The model is *asked* for a slide count but is not bound by it, and every
    # slide that reaches the renderer costs one image generation. Clamping here
    # -- after the copy exists, before anything is rendered -- is what makes a
    # plan's slide limit cost anything. Trimming beats failing: the customer gets
    # the carousel their tier covers instead of an error about a number the model
    # chose.
    plan = billing_service.get_plan(payload["user_id"])
    slide_limit = int(plan["carousel_slide_limit"])
    slides = structured["slides"]
    if len(slides) > slide_limit:
        logger.info(
            "Trimming carousel to plan limit user_id=%s plan=%s produced=%s limit=%s",
            payload["user_id"],
            plan["current_plan"],
            len(slides),
            slide_limit,
        )
        structured = {**structured, "slides": slides[:slide_limit]}

    rendered = CarouselPipelineService().render_carousel(
        structured_output=structured,
        generate_images=bool(payload.get("generate_images", True)),
    )

    billing_service.increment_generation_usage(payload["user_id"], payload.get("task_type"))
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
def create_generation_job(payload: GenerationJobCreateRequest, response: Response, request: Request):
    # The token decides whose credits are spent, not the body. This is the route
    # that costs money, so it is the one where a website sending a stale user_id
    # must be refused rather than obeyed.
    user_id = resolve_user_id(request, payload.user_id)

    # Entitlement is enforced before the job is accepted, so an over-quota or
    # unentitled caller gets an immediate, honest refusal instead of a queued job
    # that fails later.
    plan = billing_service.enforce_agent_access(user_id, payload.task_type)

    # A carousel needs the render pipeline; everything else is text only.
    kind = CAROUSEL_GENERATION if payload.task_type == "carousel" else AGENT_GENERATION

    job_payload = {**payload.model_dump(), "user_id": user_id}
    if kind == CAROUSEL_GENERATION:
        # Clamped at accept time so the model is *asked* for a count the plan
        # allows. Trimming after generation alone would still pay Sonnet to write
        # slides that are then thrown away, and would leave the carousel's own
        # "1 of N" numbering describing a length it no longer has.
        job_payload["slide_count"] = carousel_pipeline_service.resolve_slide_count(
            payload.slide_count,
            tier_maximum=plan["carousel_slide_limit"],
        )

    job = job_service.enqueue(
        user_id=user_id,
        kind=kind,
        payload=job_payload,
    )
    response.headers["Location"] = f"{router.prefix}/{job['job_id']}"
    return _serialize(job)


@router.get("/{job_id}", response_model=GenerationJobResponse, responses=STANDARD_ERROR_RESPONSES)
def get_generation_job(job_id: str, request: Request):
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Generation job '{job_id}' was not found")

    # Ownership is checked, and a job belonging to someone else answers 404
    # rather than 403: 403 would confirm the id exists, which turns a failed
    # guess into a positive result.
    if enforcement_enabled():
        owner = resolve_user_id(request, None)
        if job.get("user_id") != owner:
            raise HTTPException(status_code=404, detail=f"Generation job '{job_id}' was not found")

    return _serialize(job)


@router.get("", response_model=GenerationJobListResponse, responses=STANDARD_ERROR_RESPONSES)
def list_generation_jobs(
    request: Request,
    user_id: str | None = Query(default=None, min_length=1),
    limit: int = Query(50, ge=1, le=200),
):
    # With enforcement on, the query parameter cannot select whose jobs are
    # listed -- otherwise the whole token is a formality on the one endpoint
    # that returns another customer's finished work.
    resolved = resolve_user_id(request, user_id)
    return {"jobs": [_serialize(job) for job in job_service.list_for_user(resolved, limit=limit)]}


def _serialize(job: dict) -> dict:
    """Strip the stored payload from the response.

    The payload echoes the caller's own request and can carry account context;
    the job record is a status resource, not a mirror of the submission.
    """
    return {key: value for key, value in job.items() if key != "payload"}
