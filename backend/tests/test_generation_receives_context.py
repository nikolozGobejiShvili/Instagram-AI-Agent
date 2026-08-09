"""What the model actually receives when a job runs.

Written because 365 passing tests did not notice that the asynchronous
generation path — the one the website uses — passed no Instagram context at all.
Every existing test asserted a *response*: the job succeeded, the payload
validated, the credits were charged. All of that stayed true while the agent
generated with no idea whose account it was for, because a missing context is
silently omitted from the prompt rather than raised.

So these assert the input, not the output. A generic answer is indistinguishable
from a good one at a glance; an absent profile in the arguments is not.
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
from app.services.account_context_service import (  # noqa: E402
    TASKS_NEEDING_RECENT_PERFORMANCE,
    TASKS_NEEDING_RECENT_POSTS,
    AccountContextService,
)
from app.services.billing_service import BillingService  # noqa: E402
from app.services.job_service import JobService, JobWorker  # noqa: E402

PROFILE = {"username": "ceramics_tbilisi", "followers": 1200, "biography": "handmade ceramics"}
POSTS = {"posts": [{"topic": "studio process", "content_type": "REELS"}]}
PERFORMANCE = {"best_topics": ["studio process"]}


class StubContexts:
    def __init__(self, payload):
        self.payload = payload
        self.asked_for = []

    def get_context(self, account_id):
        self.asked_for.append(account_id)
        return self.payload


class StubRetrieval:
    """Stands in for the whole recall+rerank stack."""

    def __init__(self, knowledge_context="Open with tension, never a greeting.", used=True):
        self.calls = []
        self._context = knowledge_context
        self._used = used

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        from app.services.deterministic_knowledge_retrieval_service import KnowledgeRetrievalResult

        return KnowledgeRetrievalResult(
            used=self._used,
            top_k=3,
            retrieved_count=1 if self._context else 0,
            collection_name="pgvector:knowledge_chunks",
            knowledge_context=self._context,
        )


@pytest.fixture()
def client(monkeypatch, tmp_path):
    job_service = JobService(db_path=tmp_path / "jobs.sqlite3")
    billing_service = BillingService(
        db_path=tmp_path / "billing.sqlite3", legacy_json_file=tmp_path / "absent.json"
    )
    monkeypatch.setattr(jobs_route, "job_service", job_service)
    monkeypatch.setattr(jobs_route, "billing_service", billing_service)
    monkeypatch.setattr(
        jobs_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id: "acct-1"
    )

    retrieval = StubRetrieval()
    monkeypatch.setattr(
        jobs_route,
        "account_context_service",
        AccountContextService(
            profile_context_service=StubContexts(PROFILE),
            recent_posts_context_service=StubContexts(POSTS),
            recent_content_context_service=StubContexts(PERFORMANCE),
            knowledge_retrieval_service=retrieval,
        ),
    )

    captured = []

    def _run_agent(**kwargs):
        captured.append(kwargs)
        return {"reply": "ok", "model_provider": "stub", "used_langflow": False}

    monkeypatch.setattr(jobs_route.llm_service, "run_agent", _run_agent)

    test_client = TestClient(app)
    test_client.captured = captured
    test_client.retrieval = retrieval
    test_client.billing_service = billing_service
    test_client.worker = JobWorker(job_service, handlers=dict(jobs_route.job_worker.handlers))
    return test_client


def _run(client, task_type="content_plan", plan="pro"):
    client.billing_service.set_plan("ctx-user", {"current_plan": plan})
    response = client.post(
        "/api/v1/generation-jobs",
        json={"user_id": "ctx-user", "task_type": task_type, "message": "გეგმა მჭირდება"},
    )
    assert response.status_code == 202, response.text
    client.worker.run_once()
    assert client.captured, "run_agent was never called"
    return client.captured[-1]


# ------------------------------------------------- the account's own signal


def test_the_generation_knows_whose_account_it_is_for(client):
    """The defect this file exists for: the async path sent no profile at all."""
    kwargs = _run(client)

    assert kwargs["profile_context"] == PROFILE


def test_the_generation_sees_what_the_account_recently_posted(client):
    kwargs = _run(client)

    assert kwargs["recent_posts_context"] == POSTS


def test_a_planning_task_sees_recent_performance(client):
    kwargs = _run(client, task_type="content_plan")

    assert kwargs["recent_content_context"] == PERFORMANCE


def test_a_caption_is_not_charged_a_month_of_performance_history(client):
    """Context is not free: it competes with the instructions for attention and
    costs tokens. A caption needs to know what the account posts, not how every
    post performed."""
    kwargs = _run(client, task_type="caption")

    assert kwargs["recent_posts_context"] == POSTS
    assert kwargs["recent_content_context"] is None


@pytest.mark.parametrize("task_type", sorted(TASKS_NEEDING_RECENT_POSTS))
def test_every_content_task_receives_the_accounts_posts(client, task_type):
    # _offered, not the raw list: a withheld task is still in PLAN_DEFAULTS and
    # answers 503, so checking the raw list would try to run one.
    if task_type not in BillingService._offered(BillingService.PLAN_DEFAULTS["agency"]["allowed_task_types"]):
        pytest.skip(f"{task_type} is not sold")

    assert _run(client, task_type=task_type, plan="agency")["recent_posts_context"] == POSTS


@pytest.mark.parametrize("task_type", sorted(TASKS_NEEDING_RECENT_PERFORMANCE))
def test_every_analysis_task_receives_performance_history(client, task_type):
    # _offered, not the raw list: a withheld task is still in PLAN_DEFAULTS and
    # answers 503, so checking the raw list would try to run one.
    if task_type not in BillingService._offered(BillingService.PLAN_DEFAULTS["agency"]["allowed_task_types"]):
        pytest.skip(f"{task_type} is not sold")

    assert _run(client, task_type=task_type, plan="agency")["recent_content_context"] == PERFORMANCE


# ------------------------------------------------------- uploaded material


def test_uploaded_material_reaches_the_generation(client):
    """Indexing worked all along; reading it back did not. The material was
    written to pgvector on upload and never once retrieved."""
    kwargs = _run(client)

    playbook = kwargs["playbook_context"]
    assert playbook is not None
    assert "Open with tension" in playbook["chunks"][0]["text"]


def test_retrieval_is_given_the_account_signal_to_search_with(client):
    """Searching on the message alone throws away the niche and the history,
    which is what makes a retrieved passage relevant to *this* account."""
    _run(client)

    call = client.retrieval.calls[-1]
    assert call["profile_context"] == PROFILE
    assert call["recent_posts_context"] == POSTS


def test_nothing_relevant_means_no_material_block(client, monkeypatch):
    """The retriever is allowed to decline. An empty block would read to the
    model as material that exists and is blank."""
    monkeypatch.setattr(
        jobs_route,
        "account_context_service",
        AccountContextService(
            profile_context_service=StubContexts(PROFILE),
            recent_posts_context_service=StubContexts(POSTS),
            recent_content_context_service=StubContexts(PERFORMANCE),
            knowledge_retrieval_service=StubRetrieval(knowledge_context=None),
        ),
    )

    assert _run(client)["playbook_context"] is None


# --------------------------------------------------------------- resilience


def test_one_broken_context_does_not_cost_the_others(monkeypatch, client):
    """A stale profile cache must not also take away the recent posts."""
    class Exploding:
        def get_context(self, account_id):
            raise RuntimeError("cache corrupt")

    monkeypatch.setattr(
        jobs_route,
        "account_context_service",
        AccountContextService(
            profile_context_service=Exploding(),
            recent_posts_context_service=StubContexts(POSTS),
            recent_content_context_service=StubContexts(PERFORMANCE),
            knowledge_retrieval_service=StubRetrieval(),
        ),
    )

    kwargs = _run(client)

    assert kwargs["profile_context"] is None
    assert kwargs["recent_posts_context"] == POSTS


def test_a_failing_retriever_does_not_lose_the_generation(monkeypatch, client):
    class Exploding:
        def retrieve(self, **kwargs):
            raise RuntimeError("pgvector down")

    monkeypatch.setattr(
        jobs_route,
        "account_context_service",
        AccountContextService(
            profile_context_service=StubContexts(PROFILE),
            recent_posts_context_service=StubContexts(POSTS),
            recent_content_context_service=StubContexts(PERFORMANCE),
            knowledge_retrieval_service=Exploding(),
        ),
    )

    kwargs = _run(client)

    assert kwargs["playbook_context"] is None
    assert kwargs["profile_context"] == PROFILE


def test_no_connected_account_still_generates(monkeypatch, client):
    """A customer who has not connected Instagram gets a generic answer, not an
    error — the product has to be usable on day one."""
    monkeypatch.setattr(
        jobs_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id: None
    )

    kwargs = _run(client)

    assert kwargs["profile_context"] is None
