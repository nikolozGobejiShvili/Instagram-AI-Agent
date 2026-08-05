import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import agent as agent_route  # noqa: E402
from app.api.routes import knowledge_packs as knowledge_pack_route  # noqa: E402
from app.services.knowledge_pack_service import KnowledgePackService  # noqa: E402


INTERNAL_HEADERS = {"X-Internal-Admin-Key": "test-internal-key"}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERNAL_ADMIN_KEY", INTERNAL_HEADERS["X-Internal-Admin-Key"])
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "false")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    knowledge_service = KnowledgePackService(
        data_file=tmp_path / "knowledge_packs.json",
        chunks_file=tmp_path / "knowledge_pack_chunks.json",
        storage_dir=tmp_path / "knowledge_packs_files",
    )
    monkeypatch.setattr(knowledge_pack_route, "knowledge_pack_service", knowledge_service)
    monkeypatch.setattr(agent_route, "knowledge_pack_service", knowledge_service)

    captured_contexts: list[dict] = []

    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id, task_type=None: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.connected_accounts_service, "find_user_id_by_account_id", lambda account_id: "user-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {
        "brand_name": "Brand",
        "niche": "beauty",
        "target_audience": "women 24-35",
        "brand_voice": "direct",
        "bio": "Helping creators convert views into DMs",
        "content_focus": ["reels"],
        "strengths": ["clear offer"],
        "weak_points": ["slow hooks"],
    })
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {
        "posts": [
            {"post_id": "p1", "content_type": "REEL", "topic": "hooks", "caption": "Better hooks", "views": 100, "likes": 10, "comments": 1, "saves": 2},
        ]
    })
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {
        "top_formats": ["Reels"],
        "best_topics": ["hook logic"],
        "weak_topics": ["generic intros"],
        "best_ctas": ["DM CTA"],
        "weak_ctas": ["follow for more"],
        "notes": ["retention drops after the first second"],
    })
    monkeypatch.setattr(agent_route.link_context_service, "extract_context", lambda link: {
        "link": link,
        "detected_platform": "instagram",
        "content_type": "reel",
        "summary": "Public Reel summary",
        "hook_style": "problem-first",
        "source_patterns": ["direct hook", "proof moment"],
    })
    monkeypatch.setattr(agent_route.instagram_media_service, "get_media", lambda user_id, account_id, limit=20: {
        "items": [
            {
                "media_id": "m1",
                "media_type": "REEL",
                "caption": "Current reel",
                "permalink": "https://www.instagram.com/reel/example/",
                "timestamp": "2026-04-27T10:00:00+00:00",
                "like_count": 10,
                "comments_count": 2,
                "is_reel": True,
            }
        ]
    })
    monkeypatch.setattr(agent_route.instagram_media_service, "get_media_item", lambda user_id, media_id, account_id=None: {
        "media_id": media_id,
        "media_type": "REEL",
        "caption": "Current reel",
        "permalink": "https://www.instagram.com/reel/example/",
        "timestamp": "2026-04-27T10:00:00+00:00",
        "like_count": 10,
        "comments_count": 2,
        "is_reel": True,
    })
    monkeypatch.setattr(agent_route.agent_response_formatter_service, "normalize_reply", lambda task_type, reply: {
        "reply": reply,
        "parse_status": "raw_only",
        "structured_output": None,
    })
    monkeypatch.setattr(agent_route.langflow_service, "run_agent", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Langflow should not be called when USE_LANGFLOW_FOR_AGENT_CHAT=false")))

    def fake_run_agent(**kwargs):
        captured_contexts.append({
            "task_type": kwargs.get("task_type"),
            "playbook_context": kwargs.get("playbook_context"),
        })
        return {
            "reply": f"{kwargs.get('task_type')} ok",
            "account_id": kwargs.get("account_id"),
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "used_langflow": False,
            "prompt_section_names": ["base_system_instruction", "user_request"],
        }

    monkeypatch.setattr(agent_route.llm_service, "run_agent", fake_run_agent)

    test_client = TestClient(app)
    test_client.captured_contexts = captured_contexts
    test_client.knowledge_service = knowledge_service
    return test_client


def _upload_internal_reels_pack(client: TestClient, *, status: str = "active") -> dict:
    response = client.post(
        "/api/v1/internal/knowledge-packs/upload",
        headers=INTERNAL_HEADERS,
        data={
            "title": "Mariami Reels Playbook",
            "description": "Internal reels methodology",
            "domain": "reels",
            "supported_task_types": "reel_idea,reel_script,reel_feedback",
            "scope": "system",
            "visibility": "internal",
            "status": status,
        },
        files=[
            ("files", ("mariami-reels.md", b"# Hooks\nStart with tension.\n\n# CTA\nUse one CTA.", "text/markdown")),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_system_reels_knowledge_can_be_uploaded_and_activated(client: TestClient):
    uploaded = _upload_internal_reels_pack(client, status="inactive")
    assert uploaded["scope"] == "system"
    assert uploaded["domain"] == "reels"
    assert uploaded["visibility"] == "internal"
    assert uploaded["status"] == "inactive"

    activate_response = client.post(
        f"/api/v1/internal/knowledge-packs/{uploaded['knowledge_pack_id']}/activate",
        headers=INTERNAL_HEADERS,
    )
    assert activate_response.status_code == 200, activate_response.text
    activated = activate_response.json()
    assert activated["status"] == "active"


@pytest.mark.parametrize(
    ("task_type", "payload_overrides"),
    [
        ("reel_idea", {}),
        ("reel_script", {}),
        ("reel_feedback", {"link": "https://www.instagram.com/reel/example/"}),
    ],
)
def test_reels_tasks_use_active_system_knowledge(client: TestClient, task_type: str, payload_overrides: dict):
    _upload_internal_reels_pack(client)

    payload = {
        "message": f"Help me with {task_type}",
        "task_type": task_type,
        "user_id": "user-1",
        "auto_sync": False,
    }
    payload.update(payload_overrides)

    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["reply"] == f"{task_type} ok"

    captured = client.captured_contexts[-1]
    playbook_context = captured["playbook_context"]
    assert playbook_context is not None
    assert playbook_context["used_system_knowledge"] is True
    assert playbook_context["matched_knowledge_domain"] == "reels"
    assert playbook_context["matched_knowledge_pack_ids"]
    assert playbook_context["retrieved_chunk_count"] > 0


@pytest.mark.parametrize("task_type", ["caption", "carousel"])
def test_non_reels_tasks_do_not_use_reels_system_knowledge(client: TestClient, task_type: str):
    _upload_internal_reels_pack(client)

    response = client.post("/api/v1/agent/chat", json={
        "message": f"Help me with {task_type}",
        "task_type": task_type,
        "user_id": "user-1",
        "auto_sync": False,
    })
    assert response.status_code == 200, response.text

    captured = client.captured_contexts[-1]
    assert captured["playbook_context"] is None


def test_end_user_cannot_inspect_internal_system_knowledge(client: TestClient):
    _upload_internal_reels_pack(client)

    public_route_response = client.get("/api/v1/knowledge-packs/user-1")
    assert public_route_response.status_code == 404

    no_header_response = client.get("/api/v1/internal/knowledge-packs")
    assert no_header_response.status_code == 403


def test_reels_tasks_fallback_when_no_system_knowledge_exists(client: TestClient):
    response = client.post("/api/v1/agent/chat", json={
        "message": "Need reel ideas",
        "task_type": "reel_idea",
        "user_id": "user-1",
        "auto_sync": False,
    })
    assert response.status_code == 200, response.text

    captured = client.captured_contexts[-1]
    assert captured["playbook_context"] is not None
    assert captured["playbook_context"]["used_system_knowledge"] is False
    assert captured["playbook_context"]["retrieved_chunk_count"] == 0


def test_reels_tasks_fallback_when_system_retrieval_fails(client: TestClient, monkeypatch):
    _upload_internal_reels_pack(client)

    def broken_retrieval(**kwargs):
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(agent_route.knowledge_pack_service, "retrieve_system_context", broken_retrieval)

    response = client.post("/api/v1/agent/chat", json={
        "message": "Need reel ideas",
        "task_type": "reel_idea",
        "user_id": "user-1",
        "auto_sync": False,
    })
    assert response.status_code == 200, response.text

    captured = client.captured_contexts[-1]
    assert captured["playbook_context"] is None
