"""Tiers as a sellable product: a public catalogue, priced tasks, per-plan caps.

Three properties are covered here, each of which the tier table failed before:

1. The tier table is reachable from outside the process, so a storefront can
   render pricing without keeping a second hard-coded copy.
2. A task is charged what it costs to serve. Charging a carousel -- one Sonnet
   call plus an image generation per slide -- the same single credit as a `chat`
   made the heaviest users the least profitable.
3. A plan's slide limit actually binds, including when the model returns more
   slides than it was asked for.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import generation_jobs as jobs_route  # noqa: E402
from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES  # noqa: E402
from app.services.billing_service import BillingService  # noqa: E402
from app.services.carousel_pipeline_service import DEFAULT_SLIDE_COUNT  # noqa: E402
from app.services.job_service import JobService, JobWorker  # noqa: E402

USER = "tier-user"


@pytest.fixture()
def billing(tmp_path):
    return BillingService(
        db_path=tmp_path / "billing.sqlite3",
        legacy_json_file=tmp_path / "absent.json",
    )


# ------------------------------------------------------------------ catalogue


def test_catalogue_is_served_without_a_customer():
    """A pricing page renders before anyone has signed up."""
    response = TestClient(app).get("/api/v1/billing/plans")

    assert response.status_code == 200
    body = response.json()
    assert {p["plan_id"] for p in body["plans"]} == set(BillingService.PLAN_DEFAULTS)


def test_catalogue_reports_what_each_tier_actually_enforces():
    """The published numbers must be the ones the service checks against.

    A catalogue assembled by hand is the failure this guards: it agrees with the
    code on the day it is written and silently drifts afterwards.
    """
    body = TestClient(app).get("/api/v1/billing/plans").json()

    for entry in body["plans"]:
        defaults = BillingService.PLAN_DEFAULTS[entry["plan_id"]]
        assert entry["monthly_generation_limit"] == defaults["monthly_generation_limit"]
        assert entry["carousel_slide_limit"] == defaults["carousel_slide_limit"]
        assert entry["allowed_task_types"] == list(defaults["allowed_task_types"])


def test_catalogue_prices_every_supported_task():
    """A client cannot show the cost of an action it has no price for."""
    body = TestClient(app).get("/api/v1/billing/plans").json()

    assert set(body["generation_costs"]) == set(BillingService.SUPPORTED_TASK_TYPES)
    assert all(cost >= 1 for cost in body["generation_costs"].values())


def test_only_the_trial_advertises_an_expiry():
    """Paid tiers do not lapse on their own; saying otherwise would be a lie."""
    plans = {p["plan_id"]: p for p in TestClient(app).get("/api/v1/billing/plans").json()["plans"]}

    assert plans["trial"]["trial_duration_days"] == BillingService.TRIAL_DURATION_DAYS
    assert [p["trial_duration_days"] for p in plans.values() if p["plan_id"] != "trial"] == [None, None, None]


def test_every_tier_can_run_a_carousel():
    """The subscription is sold on carousels.

    A trial that cannot run the headline feature demonstrates nothing, and a
    first paid tier without it would mean paying to lose access.
    """
    for entry in TestClient(app).get("/api/v1/billing/plans").json()["plans"]:
        assert "carousel" in entry["allowed_task_types"], entry["plan_id"]
        assert entry["carousel_slide_limit"] >= 2, entry["plan_id"]


def test_the_ladder_never_steps_backwards():
    """Every rung must be at least as generous as the one below it."""
    order = ["trial", "creator", "pro", "agency"]
    entries = {p["plan_id"]: p for p in TestClient(app).get("/api/v1/billing/plans").json()["plans"]}

    for lower, higher in zip(order, order[1:]):
        low, high = entries[lower], entries[higher]
        for field in (
            "monthly_generation_limit",
            "carousel_slide_limit",
            "connected_account_limit",
            "tracked_accounts_limit",
        ):
            assert high[field] >= low[field], f"{higher}.{field} < {lower}.{field}"
        assert set(low["allowed_task_types"]) <= set(high["allowed_task_types"]), (
            f"{higher} drops a task {lower} has"
        )


# ------------------------------------------------------------- weighted cost


def test_a_carousel_costs_more_than_a_chat():
    """Equal pricing made the customers using the headline feature the ones the
    plan loses money on."""
    assert BillingService.generation_cost("carousel") > BillingService.generation_cost("chat")
    assert BillingService.generation_cost("chat") == BillingService.DEFAULT_GENERATION_COST


def test_an_unpriced_task_still_charges_the_base_rate():
    """An unknown task must not be free -- the cheapest way to get free work
    would otherwise be to send a task type nobody priced."""
    assert BillingService.generation_cost("something_new") == BillingService.DEFAULT_GENERATION_COST
    assert BillingService.generation_cost(None) == BillingService.DEFAULT_GENERATION_COST


def test_usage_is_charged_at_the_task_rate(billing):
    billing.set_plan(USER, {"current_plan": "pro"})

    billing.increment_generation_usage(USER, "chat")
    assert billing.get_plan(USER)["monthly_generation_used"] == 1

    billing.increment_generation_usage(USER, "carousel")
    assert billing.get_plan(USER)["monthly_generation_used"] == 1 + BillingService.generation_cost("carousel")


def test_a_task_the_remaining_credits_cannot_pay_for_is_refused(billing):
    """The charge lands after the work succeeds, so an unaffordable task has to
    be refused up front or the overspend is only discovered once it happened."""
    billing.set_plan(USER, {"current_plan": "pro"})
    limit = billing.get_plan(USER)["monthly_generation_limit"]

    # Leave exactly two credits: enough for a chat, not for a carousel.
    for _ in range(limit - 2):
        billing.increment_generation_usage(USER, "chat")

    billing.enforce_agent_access(USER, "chat")  # affordable, must not raise

    with pytest.raises(Exception) as exc:
        billing.enforce_agent_access(USER, "carousel")
    assert getattr(exc.value, "status_code", None) == 429


def test_a_full_allowance_still_refuses_the_cheapest_task(billing):
    billing.set_plan(USER, {"current_plan": "trial"})
    limit = billing.get_plan(USER)["monthly_generation_limit"]
    for _ in range(limit):
        billing.increment_generation_usage(USER, "chat")

    with pytest.raises(Exception) as exc:
        billing.enforce_agent_access(USER, "chat")
    assert getattr(exc.value, "status_code", None) == 429


def test_every_plan_can_afford_its_own_most_expensive_task():
    """A tier that allows a task it can never pay for sells a dead feature."""
    for plan_id, defaults in BillingService.PLAN_DEFAULTS.items():
        dearest = max(BillingService.generation_cost(t) for t in defaults["allowed_task_types"])
        assert defaults["monthly_generation_limit"] >= dearest, plan_id


# ---------------------------------------------------------------- slide cap


@pytest.fixture()
def carousel_client(monkeypatch, tmp_path):
    """A client whose carousel job renders no images and bills in isolation."""
    job_service = JobService(db_path=tmp_path / "jobs.sqlite3")
    billing_service = BillingService(
        db_path=tmp_path / "billing.sqlite3",
        legacy_json_file=tmp_path / "absent.json",
    )
    monkeypatch.setattr(jobs_route, "job_service", job_service)
    monkeypatch.setattr(jobs_route, "billing_service", billing_service)

    # Deliberately returns more slides than any plan allows, and more than it was
    # asked for: the model is prompted for a count but is not bound by one, so
    # the cap has to hold even when the prompt is ignored.
    calls = []

    def _run_agent(**kwargs):
        calls.append(kwargs)
        return {
            "reply": "ok",
            "model_provider": "stub",
            "used_langflow": False,
            "structured_output": {
                "title": "კარუსელი",
                "slides": [
                    {"slide_number": i, "headline": f"სათაური {i}", "body": "ტექსტი"}
                    for i in range(1, MAX_CAROUSEL_SLIDES + 1)
                ],
            },
        }

    monkeypatch.setattr(jobs_route.llm_service, "run_agent", _run_agent)

    client = TestClient(app)
    client.job_service = job_service
    client.billing_service = billing_service
    client.generation_calls = calls
    client.worker = JobWorker(job_service, handlers=dict(jobs_route.job_worker.handlers))
    return client


def _run_carousel(client, plan_id, **overrides):
    client.billing_service.set_plan(USER, {"current_plan": plan_id})
    body = {
        "user_id": USER,
        "task_type": "carousel",
        "message": "გააკეთე კარუსელი",
        "generate_images": False,
    }
    body.update(overrides)
    response = client.post("/api/v1/generation-jobs", json=body)
    assert response.status_code == 202, response.text
    client.worker.run_once()
    job = client.get(f"/api/v1/generation-jobs/{response.json()['job_id']}").json()
    assert job["status"] == "succeeded", job
    return job["result"]["structured_output"]["slides"]


@pytest.mark.parametrize("plan_id", ["trial", "creator", "pro", "agency"])
def test_a_carousel_is_trimmed_to_the_plans_slide_limit(carousel_client, plan_id):
    """Each rendered slide costs an image generation, so an unbound slide count
    is an unbound bill."""
    slides = _run_carousel(carousel_client, plan_id)

    assert len(slides) == BillingService.PLAN_DEFAULTS[plan_id]["carousel_slide_limit"]


def test_trimming_keeps_the_opening_slides(carousel_client):
    """The hook has to survive the trim -- dropping from the front would leave a
    carousel that starts in the middle of its own argument."""
    slides = _run_carousel(carousel_client, "trial")

    assert [s["slide_number"] for s in slides] == list(range(1, len(slides) + 1))


def test_a_request_over_the_tier_limit_is_clamped_not_refused(carousel_client):
    """Asking for more slides than the plan covers returns the carousel the plan
    does cover. Refusing would punish the customer for a number they had no way
    to know."""
    _run_carousel(carousel_client, "trial", slide_count=MAX_CAROUSEL_SLIDES)

    asked_for = carousel_client.generation_calls[-1]["slide_count"]
    assert asked_for == BillingService.PLAN_DEFAULTS["trial"]["carousel_slide_limit"]


def test_the_model_is_asked_for_the_count_the_customer_requested(carousel_client):
    """Clamping at accept time is what stops Sonnet being paid to write slides
    that are then thrown away."""
    _run_carousel(carousel_client, "agency", slide_count=8)

    assert carousel_client.generation_calls[-1]["slide_count"] == 8


def test_the_requested_count_reaches_the_prompt():
    """Threading the count only as far as the service would leave the prompt
    still asking for its fixed default."""
    from app.services.langflow_service import LangflowService

    instruction = LangflowService()._output_format_instruction("carousel", 8)

    assert instruction.count("Slide ") == 8
    assert "Slide 8:" in instruction
    assert "Slide 9:" not in instruction


def test_a_carousel_without_a_requested_count_still_gets_one(carousel_client):
    """The pipeline must never see None -- an absent count would fall back to a
    default that ignores the plan."""
    _run_carousel(carousel_client, "creator")

    asked_for = carousel_client.generation_calls[-1]["slide_count"]
    assert asked_for == DEFAULT_SLIDE_COUNT


def test_text_only_carousels_are_requestable(carousel_client):
    """`generate_images` was read by the handler but forbidden by the request
    schema, so no caller could ever set it and every carousel paid for images."""
    response = TestClient(app).post(
        "/api/v1/generation-jobs",
        json={
            "user_id": USER,
            "task_type": "carousel",
            "message": "გააკეთე კარუსელი",
            "generate_images": False,
        },
    )
    assert response.status_code != 422, response.text
