"""Asynchronous generation jobs.

Audit of 2026-08-04, finding 5: all 73 route handlers are ``sync def``, so each
request holds a threadpool worker for its whole duration. Carousel generation is
one text call plus one image generation per slide -- plausibly 40-90 seconds. As
a synchronous endpoint that guarantees gateway timeouts for the caller and
threadpool exhaustion for everyone else.

Long work therefore becomes a job: accepted immediately, executed in the
background, polled for completion. These tests pin the contract the website will
depend on.
"""
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.job_service import JobService, JobWorker  # noqa: E402


@pytest.fixture()
def jobs(tmp_path):
    return JobService(db_path=tmp_path / "jobs.sqlite3")


def test_enqueue_returns_immediately_with_a_queued_job(jobs):
    job = jobs.enqueue(user_id="u1", kind="carousel", payload={"message": "hi"})

    assert job["job_id"]
    assert job["status"] == "queued"
    assert job["user_id"] == "u1"
    assert job["kind"] == "carousel"
    assert job["result"] is None
    assert job["error"] is None


def test_job_can_be_fetched_by_id(jobs):
    job = jobs.enqueue(user_id="u1", kind="carousel", payload={})
    fetched = jobs.get(job["job_id"])

    assert fetched is not None
    assert fetched["job_id"] == job["job_id"]
    assert jobs.get("does-not-exist") is None


def test_worker_runs_a_job_to_success(jobs):
    job = jobs.enqueue(user_id="u1", kind="carousel", payload={"slides": 3})
    worker = JobWorker(jobs, handlers={"carousel": lambda payload: {"slides": payload["slides"] * 2}})

    assert worker.run_once() is True

    done = jobs.get(job["job_id"])
    assert done["status"] == "succeeded"
    assert done["result"] == {"slides": 6}
    assert done["error"] is None


def test_worker_records_failure_without_crashing(jobs):
    job = jobs.enqueue(user_id="u1", kind="carousel", payload={})

    def explode(payload):
        raise ValueError("gemini refused the prompt")

    worker = JobWorker(jobs, handlers={"carousel": explode})
    assert worker.run_once() is True

    failed = jobs.get(job["job_id"])
    assert failed["status"] == "failed"
    assert "gemini refused the prompt" in failed["error"]
    assert failed["result"] is None


def test_unknown_kind_fails_the_job_rather_than_hanging_it(jobs):
    job = jobs.enqueue(user_id="u1", kind="not_a_real_kind", payload={})
    worker = JobWorker(jobs, handlers={})

    assert worker.run_once() is True
    assert jobs.get(job["job_id"])["status"] == "failed"


def test_run_once_reports_when_there_is_nothing_to_do(jobs):
    worker = JobWorker(jobs, handlers={})
    assert worker.run_once() is False


def test_a_job_is_claimed_by_exactly_one_worker(jobs):
    """Two workers racing must not both execute the same job."""
    jobs.enqueue(user_id="u1", kind="carousel", payload={})

    executions = []
    lock = threading.Lock()

    def handler(payload):
        with lock:
            executions.append(1)
        return {}

    workers = [JobWorker(jobs, handlers={"carousel": handler}) for _ in range(8)]
    threads = [threading.Thread(target=w.run_once) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(executions) == 1, f"job executed {len(executions)} times; it must run exactly once"


def test_jobs_are_processed_in_the_order_they_were_accepted(jobs):
    for i in range(3):
        jobs.enqueue(user_id="u1", kind="carousel", payload={"n": i})

    seen = []
    worker = JobWorker(jobs, handlers={"carousel": lambda p: seen.append(p["n"]) or {}})
    while worker.run_once():
        pass

    assert seen == [0, 1, 2]


def test_running_jobs_are_recovered_after_a_restart(jobs, tmp_path):
    """A job interrupted by a crash must not be stranded in 'running' forever."""
    job = jobs.enqueue(user_id="u1", kind="carousel", payload={})
    claimed = jobs.claim_next()
    assert claimed["status"] == "running"

    # Simulate a process restart against the same database.
    restarted = JobService(db_path=tmp_path / "jobs.sqlite3")
    recovered = restarted.requeue_stale_running()

    assert recovered == 1
    assert restarted.get(job["job_id"])["status"] == "queued"


def test_listing_is_scoped_to_the_requesting_user(jobs):
    mine = jobs.enqueue(user_id="u1", kind="carousel", payload={})
    jobs.enqueue(user_id="u2", kind="carousel", payload={})

    listed = jobs.list_for_user("u1")
    assert [j["job_id"] for j in listed] == [mine["job_id"]]
