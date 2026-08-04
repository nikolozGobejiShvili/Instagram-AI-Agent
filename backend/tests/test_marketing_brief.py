"""Goal intake: state the objective once, steer every generation.

Before this, goal/niche/target_audience arrived per request and were discarded
with it, so nothing the customer said once shaped anything afterwards.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import marketing_brief as brief_route  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.marketing_brief_service import MarketingBriefService  # noqa: E402

BRIEF = {
    "objective": "Book 10 discovery calls per month from Instagram",
    "offer": "1:1 Instagram strategy intensive",
    "ideal_customer": "Georgian service businesses",
    "funnel_stage": "conversion",
    "tone_of_voice": "direct, warm, no hype",
    "topics_to_avoid": ["discount promises"],
    "primary_kpi": "qualified DM conversations per week",
}


@pytest.fixture()
def service(tmp_path):
    return MarketingBriefService(db_path=tmp_path / "briefs.sqlite3")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    service = MarketingBriefService(db_path=tmp_path / "briefs.sqlite3")
    monkeypatch.setattr(brief_route, "marketing_brief_service", service)
    monkeypatch.setattr(
        brief_route.connected_accounts_service, "resolve_account_id",
        lambda user_id, account_id=None: account_id or "acct-1",
    )
    test_client = TestClient(app)
    test_client.brief_service = service
    return test_client


# ------------------------------------------------------------------- storage


def test_absent_brief_reads_as_empty_not_an_error(service):
    """'Not filled in yet' is a normal state — the site renders the same form."""
    brief = service.get_brief("u1", "acct-1")
    assert brief["is_empty"] is True
    assert brief["objective"] is None
    assert brief["topics_to_avoid"] == []


def test_brief_round_trips(service):
    service.save_brief("u1", "acct-1", BRIEF)
    stored = service.get_brief("u1", "acct-1")

    assert stored["objective"] == BRIEF["objective"]
    assert stored["topics_to_avoid"] == ["discount promises"]
    assert stored["is_empty"] is False
    assert stored["updated_at"]


def test_partial_update_preserves_untouched_fields(service):
    """A brief is filled in progressively; a save must not blank the rest."""
    service.save_brief("u1", "acct-1", BRIEF)
    service.save_brief("u1", "acct-1", {"tone_of_voice": "playful"})

    stored = service.get_brief("u1", "acct-1")
    assert stored["tone_of_voice"] == "playful"
    assert stored["objective"] == BRIEF["objective"], "unrelated field was cleared"
    assert stored["topics_to_avoid"] == ["discount promises"]


def test_briefs_are_isolated_per_account(service):
    service.save_brief("u1", "acct-1", {"objective": "calls"})
    service.save_brief("u1", "acct-2", {"objective": "product sales"})

    assert service.get_brief("u1", "acct-1")["objective"] == "calls"
    assert service.get_brief("u1", "acct-2")["objective"] == "product sales"


def test_blank_strings_do_not_count_as_content(service):
    service.save_brief("u1", "acct-1", {"objective": "   ", "topics_to_avoid": ["  ", ""]})
    stored = service.get_brief("u1", "acct-1")
    assert stored["objective"] is None
    assert stored["topics_to_avoid"] == []
    assert stored["is_empty"] is True


# ------------------------------------------------------------ prompt context


def test_empty_brief_yields_no_prompt_context(service):
    """A block of empty labels reads as 'unknown' and invites invention."""
    assert service.as_prompt_context(service.get_brief("u1", "acct-1")) is None
    assert service.as_prompt_context(None) is None


def test_prompt_context_carries_only_populated_fields(service):
    service.save_brief("u1", "acct-1", {"objective": "Book calls", "primary_kpi": "DMs"})
    context = service.as_prompt_context(service.get_brief("u1", "acct-1"))

    assert context == {"Business objective": "Book calls", "Primary KPI": "DMs"}
    assert "Offer being sold" not in context


# --------------------------------------------------------- generation wiring


def test_brief_reaches_the_prompt(service):
    """The whole point: a goal stated once shapes later generations."""
    service.save_brief("u1", "acct-1", BRIEF)
    context = service.as_prompt_context(service.get_brief("u1", "acct-1"))

    prepared = LLMService()._prepare_generation(
        message="მომეცი იდეები", task_type="reel_idea", brief_context=context
    )

    assert "marketing_brief" in prepared["prompt_section_names"]
    rendered = "\n".join(
        part["text"]
        for section in prepared["response_input"]
        for part in section["content"]
    )
    assert "Book 10 discovery calls" in rendered
    assert "discount promises" in rendered


def test_no_brief_means_no_brief_section(service):
    prepared = LLMService()._prepare_generation(
        message="hi", task_type="chat", brief_context=None
    )
    assert "marketing_brief" not in prepared["prompt_section_names"]


def test_audit_tasks_are_told_to_judge_against_the_brief():
    """2.3: audits measure against the stated goal, not generic best practice."""
    from app.services.langflow_service import LangflowService

    service = LangflowService()
    audit = service._task_instruction("profile_audit").lower()
    summary = service._task_instruction("performance_summary").lower()

    assert "brief" in audit and "objective" in audit
    assert "brief" in summary and "kpi" in summary


# ----------------------------------------------------------------- endpoints


def test_get_returns_an_empty_brief_rather_than_404(client):
    response = client.get("/api/v1/marketing-brief/u1")
    assert response.status_code == 200
    assert response.json()["is_empty"] is True


def test_save_and_read_back_over_http(client):
    saved = client.post("/api/v1/marketing-brief/u1", json=BRIEF)
    assert saved.status_code == 200, saved.text
    assert saved.json()["objective"] == BRIEF["objective"]

    assert client.get("/api/v1/marketing-brief/u1").json()["primary_kpi"] == BRIEF["primary_kpi"]


def test_partial_save_over_http_keeps_other_fields(client):
    client.post("/api/v1/marketing-brief/u1", json=BRIEF)
    client.post("/api/v1/marketing-brief/u1", json={"tone_of_voice": "playful"})

    body = client.get("/api/v1/marketing-brief/u1").json()
    assert body["tone_of_voice"] == "playful"
    assert body["objective"] == BRIEF["objective"]


def test_unknown_fields_are_rejected(client):
    assert client.post("/api/v1/marketing-brief/u1", json={"nope": "x"}).status_code == 422


def test_invalid_funnel_stage_is_rejected(client):
    assert client.post("/api/v1/marketing-brief/u1", json={"funnel_stage": "wat"}).status_code == 422


def test_delete_removes_the_brief(client):
    client.post("/api/v1/marketing-brief/u1", json=BRIEF)
    assert client.delete("/api/v1/marketing-brief/u1").status_code == 200
    assert client.get("/api/v1/marketing-brief/u1").json()["is_empty"] is True
    assert client.delete("/api/v1/marketing-brief/u1").status_code == 404
