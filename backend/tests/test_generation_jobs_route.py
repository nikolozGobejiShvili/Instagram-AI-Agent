"""The async generation contract the subscription website will consume.

POST accepts and returns 202 + a job id; the caller polls GET until the job
reaches a terminal state. Entitlement is checked at accept time; the monthly
allowance is only consumed when a job actually produces something.
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
from app.services.billing_service import BillingService  # noqa: E402
from app.services.job_service import JobService, JobWorker  # noqa: E402


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """A client whose jobs and billing are isolated, with the worker driven manually."""
    job_service = JobService(db_path=tmp_path / "jobs.sqlite3")
    billing_service = BillingService(
        db_path=tmp_path / "billing.sqlite3",
        legacy_json_file=tmp_path / "absent.json",
    )
    monkeypatch.setattr(jobs_route, "job_service", job_service)
    monkeypatch.setattr(jobs_route, "billing_service", billing_service)

    generated = {"reply": "carousel ready", "model_provider": "stub", "used_langflow": False}
    monkeypatch.setattr(jobs_route.llm_service, "run_agent", lambda **kwargs: dict(generated))

    test_client = TestClient(app)
    test_client.job_service = job_service
    test_client.billing_service = billing_service
    # Copied from the app's own registry rather than rebuilt, so a job kind added
    # to production can't silently go untested here. Stepped by hand so
    # assertions never race a background thread.
    test_client.worker = JobWorker(job_service, handlers=dict(jobs_route.job_worker.handlers))
    return test_client


# content_plan, not carousel: these tests cover the generic job contract, and a
# carousel now routes to the render pipeline (see test_carousel_pipeline). It is
# still a pro-only task, which keeps the entitlement case below honest.
DEFAULT_TASK_TYPE = "content_plan"


def _create(client, **overrides):
    body = {"user_id": "jobs-user", "task_type": DEFAULT_TASK_TYPE, "message": "მომეცი გეგმა"}
    body.update(overrides)
    return client.post("/api/v1/generation-jobs", json=body)


def test_post_accepts_immediately_with_202_and_a_job_id(client):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})

    response = _create(client)

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"]
    assert response.headers["Location"].endswith(body["job_id"])


def test_response_does_not_echo_the_submitted_payload(client):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})
    body = _create(client).json()
    assert "payload" not in body


def test_polling_reaches_a_succeeded_job_with_its_result(client):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})
    job_id = _create(client).json()["job_id"]

    assert client.get(f"/api/v1/generation-jobs/{job_id}").json()["status"] == "queued"
    client.worker.run_once()

    done = client.get(f"/api/v1/generation-jobs/{job_id}").json()
    assert done["status"] == "succeeded"
    assert done["result"]["reply"] == "carousel ready"
    assert done["error"] is None


def test_a_failing_generation_surfaces_as_a_failed_job(client, monkeypatch):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})

    def boom(**kwargs):
        raise RuntimeError("image provider unavailable")

    monkeypatch.setattr(jobs_route.llm_service, "run_agent", boom)
    job_id = _create(client).json()["job_id"]
    client.worker.run_once()

    failed = client.get(f"/api/v1/generation-jobs/{job_id}").json()
    assert failed["status"] == "failed"
    assert "image provider unavailable" in failed["error"]


def test_entitlement_is_refused_before_a_job_is_created(client):
    """A trial plan does not include content_plan, so nothing should be queued."""
    client.billing_service.set_plan("jobs-user", {"current_plan": "trial"})

    response = _create(client)

    assert response.status_code == 403
    assert client.job_service.list_for_user("jobs-user") == []


def test_usage_is_charged_only_when_the_job_succeeds(client, monkeypatch):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})
    before = client.billing_service.get_plan("jobs-user")["monthly_generation_used"]
    # Tasks are priced individually, so the expected charge is looked up rather
    # than assumed to be one -- hard-coding it would make this test fail on any
    # future price change while saying nothing about charge-on-success.
    cost = client.billing_service.generation_cost(DEFAULT_TASK_TYPE)
    assert cost > 0, "a successful generation must cost something"

    # Accepting a job must not consume the allowance on its own.
    _create(client)
    assert client.billing_service.get_plan("jobs-user")["monthly_generation_used"] == before

    client.worker.run_once()
    assert client.billing_service.get_plan("jobs-user")["monthly_generation_used"] == before + cost

    # A failed job must not be charged.
    monkeypatch.setattr(
        jobs_route.llm_service,
        "run_agent",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    _create(client)
    client.worker.run_once()
    assert client.billing_service.get_plan("jobs-user")["monthly_generation_used"] == before + cost


def test_unknown_job_id_is_404(client):
    assert client.get("/api/v1/generation-jobs/nope").status_code == 404


def test_the_app_lifespan_actually_runs_queued_jobs(monkeypatch):
    """End-to-end on the production path: no hand-stepped worker.

    Everything above drives the worker manually. This exercises the wiring the
    deployed service relies on -- lifespan starts the background thread, and a
    job posted over HTTP reaches 'succeeded' without anyone stepping it.
    """
    import time

    monkeypatch.setenv("JOB_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        jobs_route.llm_service,
        "run_agent",
        lambda **kwargs: {"reply": "produced by the background worker"},
    )
    jobs_route.billing_service.set_plan("lifespan-user", {"current_plan": "pro"})

    # Entering the context manager is what triggers FastAPI's lifespan.
    with TestClient(app) as live_client:
        created = live_client.post(
            "/api/v1/generation-jobs",
            json={"user_id": "lifespan-user", "task_type": "content_plan", "message": "hi"},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            body = live_client.get(f"/api/v1/generation-jobs/{job_id}").json()
            if body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)

    assert body["status"] == "succeeded", f"job did not complete: {body}"
    assert body["result"]["reply"] == "produced by the background worker"


def test_listing_is_scoped_to_the_requested_user(client):
    client.billing_service.set_plan("jobs-user", {"current_plan": "pro"})
    client.billing_service.set_plan("other-user", {"current_plan": "pro"})
    mine = _create(client).json()["job_id"]
    _create(client, user_id="other-user")

    listed = client.get("/api/v1/generation-jobs", params={"user_id": "jobs-user"}).json()["jobs"]
    assert [j["job_id"] for j in listed] == [mine]
