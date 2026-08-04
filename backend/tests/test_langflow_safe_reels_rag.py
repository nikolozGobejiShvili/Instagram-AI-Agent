import base64
import json
import sys
import types
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import agent as agent_route  # noqa: E402
from app.api.routes import knowledge_packs as knowledge_pack_route  # noqa: E402
from app.api.routes import internal_generation_debug as debug_route  # noqa: E402
from app.services.generation_history_service import GenerationHistoryService  # noqa: E402
from app.services.knowledge_pack_service import KnowledgePackService  # noqa: E402
from app.services.langflow_service import LangflowService  # noqa: E402
from app.services.link_context_service import LinkContextService  # noqa: E402
from app.schemas.agent import AgentChatRequest  # noqa: E402
from app.services.deterministic_knowledge_retrieval_service import (  # noqa: E402
    DeterministicKnowledgeRetrievalService,
    KnowledgeRetrievalResult,
    should_use_knowledge_retrieval,
)


INTERNAL_HEADERS = {"X-Internal-Admin-Key": "test-internal-key"}


def _decode_runtime_payload(input_value: str) -> dict:
    assert input_value.startswith("BASE64JSON:")
    encoded_payload = input_value.split("BASE64JSON:", 1)[1]
    decoded_payload = base64.b64decode(encoded_payload).decode("utf-8")
    return json.loads(decoded_payload)


def _main_agent_runtime_payload(input_value: str) -> dict:
    if input_value.lstrip().startswith("{"):
        return json.loads(input_value)
    marker = "runtime_context_json:"
    assert marker in input_value
    return json.loads(input_value.split(marker, 1)[1].strip())


def _langflow_payload_response(reply_payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {
                                    "text": json.dumps(reply_payload, ensure_ascii=False)
                                }
                            }
                        }
                    ]
                }
            ]
        },
        request=httpx.Request("POST", "http://127.0.0.1:7860/api/v1/run/test-flow?stream=false"),
    )


def _langflow_text_response(reply_text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {
                                    "text": reply_text
                                }
                            }
                        }
                    ]
                }
            ]
        },
        request=httpx.Request("POST", "http://127.0.0.1:7860/api/v1/run/test-flow?stream=false"),
    )


def _patch_basic_agent_chat_route(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "true")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {})
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {"posts": []})
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {})

    def fake_langflow_run(**kwargs):
        return {
            "reply": "გამარჯობა! შემიძლია Instagram კონტენტში დაგეხმარო.",
            "account_id": kwargs.get("account_id"),
            "model_provider": "langflow",
            "model_name": "test-main-flow",
            "used_langflow": True,
            "prompt_section_names": ["main_agent_runtime_payload"],
        }

    monkeypatch.setattr(agent_route.langflow_service, "run_agent", fake_langflow_run)
    return history_service


def test_langflow_main_agent_sends_only_sanitized_runtime_variables(monkeypatch):
    monkeypatch.setenv("LANGFLOW_FLOW_ID", "test-main-flow")
    monkeypatch.setenv("LANGFLOW_API_KEY", "test-key")

    service = LangflowService()
    captured: list[dict] = []

    def fake_post_run_flow(**kwargs):
        captured.append(kwargs)
        return _langflow_text_response("main agent ok")

    monkeypatch.setattr(service, "_post_run_flow", fake_post_run_flow)

    result = service.run_flat_prompt_agent(
        message="Give me three Reel ideas.",
        task_type="reel_idea",
        account_id="acct-1",
        goal="increase qualified inbound leads from Instagram",
        link="https://www.instagram.com/reel/example/",
        profile_context={
            "brand_name": "Lead Lab",
            "niche": "service marketing",
            "target_audience": "service founders",
            "bio": "We turn weak DMs into qualified leads",
            "content_focus": ["reels", "proof"],
        },
        recent_posts_context={
            "posts": [
                {
                    "post_id": "p1",
                    "content_type": "REEL",
                    "topic": "CTA mistakes",
                    "caption": "RAW_CAPTION_SENTINEL " * 30,
                    "views": 1000,
                    "likes": 90,
                    "comments": 10,
                    "saves": 5,
                }
            ]
        },
        recent_content_context={
            "top_formats": ["Reels"],
            "best_topics": ["CTA mistakes"],
            "notes": ["proof-led content converts better"],
        },
        playbook_context={
            "chunks": [
                {
                    "chunk_label": "private",
                    "text": "RAW_PLAYBOOK_SENTINEL must never be sent",
                }
            ]
        },
    )

    assert result["reply"] == "main agent ok"
    assert result["used_langflow"] is True
    assert result["prompt_section_names"] == ["main_agent_runtime_payload"]

    sent_payload = captured[-1]["payload"]
    assert sent_payload["input_type"] == "chat"
    assert sent_payload["output_type"] == "chat"
    assert "response_language: English" in sent_payload["input_value"]
    assert "SYSTEM ROLE AND RULES" not in sent_payload["input_value"]
    assert "RAW_PLAYBOOK_SENTINEL" not in sent_payload["input_value"]
    assert "RAW_CAPTION_SENTINEL RAW_CAPTION_SENTINEL RAW_CAPTION_SENTINEL" not in sent_payload["input_value"]
    runtime_payload = _main_agent_runtime_payload(sent_payload["input_value"])
    assert set(runtime_payload) == {
        "task_type",
        "user_request",
        "goal",
        "account_id",
        "profile_summary",
        "metrics_summary",
        "compact_recent_posts_summary",
        "language",
        "language_code",
        "response_language",
        "response_language_instruction",
        "link",
        "source_url",
    }
    serialized = json.dumps(runtime_payload, ensure_ascii=False)
    assert "SYSTEM ROLE AND RULES" not in serialized
    assert "RAW_PLAYBOOK_SENTINEL" not in serialized
    assert "RAW_CAPTION_SENTINEL RAW_CAPTION_SENTINEL RAW_CAPTION_SENTINEL" not in serialized
    assert runtime_payload["task_type"] == "reel_idea"
    assert runtime_payload["account_id"] == "acct-1"
    assert runtime_payload["link"] == "https://www.instagram.com/reel/example/"
    assert runtime_payload["source_url"] == "https://www.instagram.com/reel/example/"
    assert runtime_payload["language"] == "en"
    assert runtime_payload["language_code"] == "en"
    assert runtime_payload["response_language"] == "English"
    assert "natural English" in runtime_payload["response_language_instruction"]
    assert runtime_payload["compact_recent_posts_summary"][0]["caption_summary"]


def test_task_based_knowledge_retrieval_routing():
    georgian_greeting = "\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0, \u10e0\u10d0\u10e8\u10d8 \u10d3\u10d0\u10db\u10d4\u10ee\u10db\u10d0\u10e0\u10d4\u10d1\u10d8?"

    assert should_use_knowledge_retrieval("chat", georgian_greeting) is False
    assert should_use_knowledge_retrieval("reel_idea", "hello") is False
    assert should_use_knowledge_retrieval("reel_idea", "Give me three strong Reel ideas") is True
    assert should_use_knowledge_retrieval("reel_script", "Write one sales Reel script") is True
    assert should_use_knowledge_retrieval("reel_feedback", "Review this Reel hook") is True
    assert should_use_knowledge_retrieval("caption", "Write a caption") is False


def test_langflow_main_agent_runtime_payload_adds_knowledge_context_only_when_provided():
    service = LangflowService()

    without_context = service._runtime_payload_for_main_agent(
        task_type="chat",
        message="hello",
        account_id="acct-1",
    )
    with_context = service._runtime_payload_for_main_agent(
        task_type="reel_idea",
        message="Give me three strong Reel ideas.",
        account_id="acct-1",
        knowledge_context="Internal compact guidance. RAW_DOC_SENTINEL should stay compact.",
    )

    assert "knowledge_context" not in without_context
    assert with_context["knowledge_context"].startswith("Internal compact guidance")
    serialized = json.dumps(with_context, ensure_ascii=False)
    assert "SYSTEM ROLE AND RULES" not in serialized
    assert "access_token" not in serialized
    assert "file_path" not in serialized


def test_langflow_main_agent_runtime_payload_uses_georgian_response_language():
    service = LangflowService()

    runtime_payload = service._runtime_payload_for_main_agent(
        task_type="reel_idea",
        message="მომეცი 3 ძლიერი Reels იდეა.",
        account_id="acct-1",
    )

    assert runtime_payload["language"] == "ka"
    assert runtime_payload["language_code"] == "ka"
    assert runtime_payload["response_language"] == "Georgian"
    assert "ქართული" in runtime_payload["response_language_instruction"]
    assert service._serialize_main_agent_runtime_payload(runtime_payload).startswith("პასუხის ენა: ქართული")


def test_langflow_main_agent_runtime_payload_uses_english_response_language():
    service = LangflowService()

    runtime_payload = service._runtime_payload_for_main_agent(
        task_type="reel_script",
        message="Write one strong Reel script for my account.",
        account_id="acct-1",
    )

    assert runtime_payload["language"] == "en"
    assert runtime_payload["language_code"] == "en"
    assert runtime_payload["response_language"] == "English"
    assert "natural English" in runtime_payload["response_language_instruction"]


def test_langflow_main_agent_runtime_payload_uses_russian_response_language():
    service = LangflowService()

    runtime_payload = service._runtime_payload_for_main_agent(
        task_type="reel_feedback",
        message="Проанализируй этот Reel и улучши CTA.",
        account_id="acct-1",
    )

    assert runtime_payload["language"] == "ru"
    assert runtime_payload["language_code"] == "ru"
    assert runtime_payload["response_language"] == "Russian"
    assert "русски" in runtime_payload["response_language_instruction"]
    assert service._serialize_main_agent_runtime_payload(runtime_payload).startswith("Язык ответа: русский")


def test_langflow_main_agent_language_falls_back_to_georgian_profile_context():
    service = LangflowService()

    runtime_payload = service._runtime_payload_for_main_agent(
        task_type="reel_idea",
        message="...",
        account_id="acct-1",
        profile_context={
            "brand_name": "მარიამი",
            "niche": "ინსტაგრამ კონტენტი",
        },
    )

    assert runtime_payload["language_code"] == "ka"
    assert runtime_payload["response_language"] == "Georgian"
    assert "ქართული" in runtime_payload["response_language_instruction"]


def test_langflow_service_sends_only_sanitized_reels_runtime_payload(monkeypatch):
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "true")
    monkeypatch.setenv("LANGFLOW_REELS_GENERATION_FLOW_ID", "test-flow")
    monkeypatch.setenv("LANGFLOW_VECTOR_STORE_PROVIDER", "chroma")

    service = LangflowService()
    captured_payloads: list[dict] = []

    def fake_post_run_flow(**kwargs):
        captured_payloads.append(kwargs["payload"])
        return _langflow_payload_response(
            {
                "reply": "Here are 3 Reel ideas.",
                "structured_output": {
                    "ideas": [
                        {
                            "title": "CTA Audit Reel",
                            "hook": "თუ ხალხი გიყურებს, მაგრამ არ გწერს, CTA-ს პრობლემა გაქვს.",
                            "format_type": "Talking-head breakdown",
                            "main_idea": "Show one CTA mistake and one fix.",
                            "shot_list": ["Hook", "Mistake", "Fix", "CTA"],
                            "why_it_can_work": "It turns a hidden conversion issue into a fast-save lesson.",
                            "cta": "DM მომწერე CTA.",
                        }
                    ]
                },
                "parse_status": "parsed",
                "used_system_knowledge": True,
                "matched_knowledge_domain": "reels",
                "matched_knowledge_pack_ids": ["kp_langflow"],
                "retrieved_chunk_count": 3,
                "retrieved_chunk_titles": ["Module 1", "Module 2", "Module 3"],
                "model_provider": "langflow-openai",
                "model_name": "gpt-4.1-mini",
            }
        )

    monkeypatch.setattr(service, "_post_run_flow", fake_post_run_flow)

    result = service.run_reels_rag_agent(
        message="მომეცი 3 ძლიერი Reels იდეა.",
        task_type="reel_idea",
        account_id="acct-1",
        goal="increase qualified inbound leads from Instagram",
        profile_context={
            "brand_name": "Lead Lab",
            "niche": "service marketing",
            "target_audience": "service founders",
            "brand_voice": "direct",
            "bio": "We turn weak DMs into qualified leads",
            "content_focus": ["reels", "proof"],
            "strengths": ["clear offer"],
            "weak_points": ["slow CTA"],
        },
        recent_posts_context={
            "posts": [
                {
                    "post_id": "p1",
                    "content_type": "REEL",
                    "topic": "CTA mistakes",
                    "caption": "RAW_PLAYBOOK_SENTINEL must never be sent. " * 20,
                    "views": 1000,
                    "likes": 90,
                    "comments": 10,
                    "saves": 5,
                }
            ]
        },
        recent_content_context={
            "top_formats": ["Reels"],
            "best_topics": ["CTA mistakes"],
            "notes": ["proof-led content converts better"],
        },
        reel_context={
            "source": "link",
            "caption": "Test reel caption",
            "analysis_brief": "Check the hook and CTA",
        },
        link_context={
            "link": "https://www.instagram.com/reel/example/",
            "summary": "Public summary",
            "hook_style": "problem-first",
            "content_type": "reel",
            "source_patterns": ["direct hook", "proof moment"],
        },
    )

    assert result["used_langflow"] is True
    assert result["used_system_knowledge"] is True
    assert result["retrieved_chunk_count"] == 3
    assert result["model_provider"] == "langflow-openai"
    assert result["model_name"] == "gpt-4.1-mini"

    sent_payload = captured_payloads[-1]
    assert sent_payload["input_type"] == "text"
    assert sent_payload["output_type"] == "text"
    input_value = sent_payload["input_value"]
    assert "SYSTEM ROLE AND RULES" not in input_value
    assert "RAW_PLAYBOOK_SENTINEL" not in input_value
    assert "Use tension and simplicity." not in input_value
    runtime_payload = _decode_runtime_payload(input_value)
    assert runtime_payload["task_type"] == "reel_idea"
    assert runtime_payload["knowledge_domain"] == "reels"
    assert runtime_payload["vector_store_provider"] == "chroma"
    assert runtime_payload["language"] in {"ka", "en"}
    assert runtime_payload["language_code"] == runtime_payload["language"]
    assert runtime_payload["response_language"] in {"Georgian", "English"}
    assert runtime_payload["response_language_instruction"]
    assert runtime_payload["retrieval_top_k"] == 3
    assert "viral idea mechanics" in runtime_payload["retrieval_query"]
    assert runtime_payload["response_contract"]["ideas"][0]["title"] == "..."
    assert "profile_summary" in runtime_payload
    assert "recent_posts_summary" in runtime_payload
    assert "user_request" in runtime_payload
    assert isinstance(runtime_payload["recent_posts_summary"], list)
    assert runtime_payload["recent_posts_summary"][0]["caption_summary"]
    assert "RAW_PLAYBOOK_SENTINEL" not in runtime_payload["recent_posts_summary"][0]["caption_summary"]


def test_agent_route_uses_safe_langflow_reels_without_local_chunk_injection(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERNAL_ADMIN_KEY", INTERNAL_HEADERS["X-Internal-Admin-Key"])
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "false")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "true")
    monkeypatch.setenv("LANGFLOW_REELS_GENERATION_FLOW_ID", "test-safe-reels-flow")

    knowledge_service = KnowledgePackService(
        data_file=tmp_path / "knowledge_packs.json",
        chunks_file=tmp_path / "knowledge_pack_chunks.json",
        storage_dir=tmp_path / "knowledge_packs_files",
    )
    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(knowledge_pack_route, "knowledge_pack_service", knowledge_service)
    monkeypatch.setattr(agent_route, "knowledge_pack_service", knowledge_service)
    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(debug_route, "generation_history_service", history_service)
    monkeypatch.setattr(knowledge_pack_route.langflow_service, "ingest_system_reels_knowledge", lambda **kwargs: {"ingestion_triggered": True})

    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.connected_accounts_service, "find_user_id_by_account_id", lambda account_id: "user-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {
        "brand_name": "Lead Lab",
        "niche": "service marketing",
        "target_audience": "service founders",
        "brand_voice": "direct",
        "bio": "We turn weak DMs into qualified leads",
        "content_focus": ["reels", "proof"],
        "strengths": ["clear offer"],
        "weak_points": ["slow CTA"],
    })
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {
        "posts": [
            {"post_id": "p1", "content_type": "REEL", "topic": "CTA mistakes", "caption": "Very long caption " * 30, "views": 1000, "likes": 90, "comments": 10, "saves": 5},
            {"post_id": "p2", "content_type": "REEL", "topic": "hook logic", "caption": "Another caption " * 30, "views": 900, "likes": 80, "comments": 8, "saves": 4},
        ]
    })
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {
        "top_formats": ["Reels"],
        "best_topics": ["CTA mistakes", "hook logic"],
        "notes": ["proof-led content converts better"],
    })
    monkeypatch.setattr(agent_route.link_context_service, "extract_context", lambda link: {
        "link": link,
        "detected_platform": "instagram",
        "content_type": "reel",
        "summary": "Public Reel summary",
        "hook_style": "problem-first",
        "source_patterns": ["direct hook", "proof moment"],
    })
    monkeypatch.setattr(agent_route.langflow_service, "run_agent", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Unsafe flat Langflow path should not be called")))
    monkeypatch.setattr(agent_route.llm_service, "run_agent", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Direct LLM path should not be called for safe Langflow reels mode")))
    monkeypatch.setattr(agent_route.knowledge_pack_service, "retrieve_system_context", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Backend must not inject raw reels chunks into Langflow runtime payload")))

    captured_langflow_payloads: list[dict] = []

    def fake_post_run_flow(**kwargs):
        captured_langflow_payloads.append(kwargs["payload"])
        return _langflow_payload_response(
            {
                "reply": "Title:\nCTA Audit Reel\n\n1-second hook:\nთუ ნახვები გაქვს, მაგრამ DM არა, CTA-ს პრობლემა გაქვს.",
                "structured_output": {
                    "ideas": [
                        {
                            "title": "CTA Audit Reel",
                            "hook": "თუ ნახვები გაქვს, მაგრამ DM არა, CTA-ს პრობლემა გაქვს.",
                            "format_type": "Talking-head breakdown",
                            "main_idea": "Show one CTA mistake and one fix.",
                            "shot_list": ["Hook", "Mistake", "Fix", "CTA"],
                            "why_it_can_work": "It turns a hidden conversion issue into a saveable Reel lesson.",
                            "cta": "DM მომწერე CTA.",
                        }
                    ]
                },
                "parse_status": "parsed",
                "used_system_knowledge": True,
                "matched_knowledge_domain": "reels",
                "matched_knowledge_pack_ids": ["kp-safe-langflow"],
                "retrieved_chunk_count": 3,
                "retrieved_chunk_titles": ["Module 1", "Module 2", "Module 3"],
                "model_provider": "langflow-openai",
                "model_name": "gpt-4.1-mini",
            }
        )

    monkeypatch.setattr(agent_route.langflow_service, "_post_run_flow", fake_post_run_flow)

    client = TestClient(app)
    upload_response = client.post(
        "/api/v1/internal/knowledge-packs/upload",
        headers=INTERNAL_HEADERS,
        data={
            "title": "Mariami Reels Playbook",
            "description": "Internal reels methodology",
            "domain": "reels",
            "supported_task_types": "reel_idea,reel_script,reel_feedback",
            "scope": "system",
            "visibility": "internal",
            "status": "active",
        },
        files=[
            (
                "files",
                (
                    "mariami-reels.md",
                    b"# Module 1\nUse tension and simplicity.\n\n# Module 2\nAdapt trends to the niche.",
                    "text/markdown",
                ),
            ),
        ],
    )
    assert upload_response.status_code == 200, upload_response.text

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "მომეცი 3 ძლიერი Reels იდეა ჩემი მიმდინარე Instagram კონტექსტის მიხედვით.",
            "task_type": "reel_idea",
            "user_id": "user-1",
            "goal": "increase qualified inbound leads from Instagram",
            "auto_sync": False,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_type"] == "reel_idea"
    assert payload["parse_status"] == "parsed"
    assert "Use tension and simplicity." not in payload["reply"]
    assert payload["structured_output"]["ideas"][0]["title"] == "CTA Audit Reel"

    sent_input = captured_langflow_payloads[-1]["input_value"]
    assert "SYSTEM ROLE AND RULES" not in sent_input
    assert "Use tension and simplicity." not in sent_input
    assert "Adapt trends to the niche." not in sent_input
    sent_runtime_payload = _decode_runtime_payload(sent_input)
    assert sent_runtime_payload["task_type"] == "reel_idea"
    assert sent_runtime_payload["knowledge_domain"] == "reels"
    assert sent_runtime_payload["retrieval_top_k"] == 3
    assert sent_runtime_payload["response_contract"]["ideas"][0]["cta"] == "..."

    history_item = history_service.get_latest_item(user_id="user-1", task_type="reel_idea")
    assert history_item is not None
    history_json = json.dumps(history_item, ensure_ascii=False)
    assert "Use tension and simplicity." not in history_json
    assert history_item["used_langflow"] is True
    assert history_item["used_system_knowledge"] is True
    assert history_item["matched_knowledge_domain"] == "reels"
    assert history_item["retrieved_chunk_count"] == 3

    debug_response = client.get(
        "/api/v1/internal/generation-debug/latest",
        headers=INTERNAL_HEADERS,
        params={"user_id": "user-1", "task_type": "reel_idea"},
    )
    assert debug_response.status_code == 200, debug_response.text
    debug_payload = debug_response.json()
    assert debug_payload["used_langflow"] is True
    assert debug_payload["matched_knowledge_domain"] == "reels"
    assert debug_payload["retrieved_chunk_count"] == 3


def test_non_reels_tasks_do_not_use_safe_langflow_reels(monkeypatch):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "false")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "true")
    monkeypatch.setenv("LANGFLOW_REELS_GENERATION_FLOW_ID", "test-safe-reels-flow")

    called = {"langflow": False, "direct": False}

    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: "acct-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {})
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {"posts": []})
    monkeypatch.setattr(agent_route.langflow_service, "run_reels_rag_agent", lambda **kwargs: called.__setitem__("langflow", True))

    def fake_direct_run(**kwargs):
        called["direct"] = True
        return {
            "reply": "caption ok",
            "account_id": kwargs.get("account_id"),
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "used_langflow": False,
            "prompt_section_names": ["base_system_instruction", "user_request"],
        }

    monkeypatch.setattr(agent_route.llm_service, "run_agent", fake_direct_run)

    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "Write a caption",
            "task_type": "caption",
            "user_id": "user-1",
            "auto_sync": False,
        },
    )
    assert response.status_code == 200, response.text
    assert called["direct"] is True
    assert called["langflow"] is False


def test_agent_route_skips_deterministic_retrieval_for_chat_greeting(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "true")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {})
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Greeting should not trigger deterministic retrieval")),
    )

    captured_calls: list[dict] = []

    def fake_langflow_run(**kwargs):
        captured_calls.append(kwargs)
        return {
            "reply": "\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0! \u10e8\u10d4\u10db\u10d8\u10eb\u10da\u10d8\u10d0 Instagram \u10d9\u10dd\u10dc\u10e2\u10d4\u10dc\u10e2\u10e8\u10d8 \u10d3\u10d0\u10d2\u10d4\u10ee\u10db\u10d0\u10e0\u10dd.",
            "account_id": kwargs.get("account_id"),
            "model_provider": "langflow",
            "model_name": "test-main-flow",
            "used_langflow": True,
            "prompt_section_names": ["main_agent_runtime_payload"],
        }

    monkeypatch.setattr(agent_route.langflow_service, "run_agent", fake_langflow_run)

    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0, \u10e0\u10d0\u10e8\u10d8 \u10d3\u10d0\u10db\u10d4\u10ee\u10db\u10d0\u10e0\u10d4\u10d1\u10d8?",
            "task_type": "chat",
            "user_id": "user-1",
            "account_id": "acct-1",
            "auto_sync": False,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_type"] == "chat"
    assert payload["knowledge_retrieval_used"] is False
    assert payload["knowledge_retrieved_count"] == 0
    assert payload["parse_status"] == "raw_only"
    assert "knowledge_context" not in captured_calls[-1]


def test_agent_route_uses_deterministic_retrieval_for_reel_idea(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "true")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "acct-1")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {
        "brand_name": "Lead Lab",
        "niche": "service marketing",
        "target_audience": "service founders",
    })
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {
        "posts": [
            {"post_id": "p1", "content_type": "REEL", "topic": "CTA mistakes", "caption": "Long caption " * 20},
        ]
    })
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {
        "best_topics": ["CTA mistakes"],
        "top_formats": ["Reels"],
    })
    monkeypatch.setattr(
        agent_route.knowledge_pack_service,
        "retrieve_system_context",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Main Langflow path should not use legacy local chunks")),
    )
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: KnowledgeRetrievalResult(
            used=True,
            top_k=5,
            retrieved_count=4,
            collection_name="mariami_reels_playbook_v1",
            knowledge_context="Internal compact Mariami strategy guidance. Do not quote this context.",
        ),
    )

    reply = "\n\n".join(
        [
            "IDEA 1\nTitle: CTA Audit\nHook: Your views are not the problem.\nFormat type: Talking head\nMain idea: Show one CTA mistake and the fix.\nShot list:\n1. Hook\n2. Mistake\n3. Fix\nWhy it can work: It makes a hidden conversion issue clear.\nCaption idea: Save this before posting.\nCTA: DM CTA.",
            "IDEA 2\nTitle: First Second Fix\nHook: Most Reels lose people before the point.\nFormat type: Screen recording\nMain idea: Compare a slow opening with a direct one.\nShot list:\n1. Bad opening\n2. Better opening\n3. CTA\nWhy it can work: It gives an instantly usable pattern.\nCaption idea: Use this opening today.\nCTA: DM HOOK.",
            "IDEA 3\nTitle: Proof Before Pitch\nHook: Stop selling before showing proof.\nFormat type: Proof breakdown\nMain idea: Show one proof moment before the offer.\nShot list:\n1. Claim\n2. Proof\n3. Offer\nWhy it can work: It builds trust before asking for action.\nCaption idea: Proof makes the CTA easier.\nCTA: DM PROOF.",
        ]
    )
    captured_calls: list[dict] = []

    def fake_langflow_run(**kwargs):
        captured_calls.append(kwargs)
        return {
            "reply": reply,
            "account_id": kwargs.get("account_id"),
            "model_provider": "langflow",
            "model_name": "test-main-flow",
            "used_langflow": True,
            "prompt_section_names": ["main_agent_runtime_payload", "deterministic_knowledge_context"],
        }

    monkeypatch.setattr(agent_route.langflow_service, "run_agent", fake_langflow_run)

    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "Give me 3 strong Reel ideas for my current Instagram context.",
            "task_type": "reel_idea",
            "user_id": "user-1",
            "account_id": "acct-1",
            "goal": "increase qualified inbound leads from Instagram",
            "auto_sync": False,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["knowledge_retrieval_used"] is True
    assert payload["knowledge_retrieval_top_k"] == 5
    assert payload["knowledge_retrieved_count"] == 4
    assert payload["knowledge_collection_name"] == "mariami_reels_playbook_v1"
    assert payload["parse_status"] == "parsed"
    assert len(payload["structured_output"]["ideas"]) == 3
    assert captured_calls[-1]["knowledge_context"].startswith("Internal compact Mariami strategy")
    assert "file_path" not in json.dumps(payload, ensure_ascii=False)

    history_item = history_service.get_latest_item(user_id="user-1", task_type="reel_idea")
    assert history_item["knowledge_retrieval_used"] is True
    assert history_item["knowledge_retrieved_count"] == 4
    history_json = json.dumps(history_item, ensure_ascii=False)
    assert "Internal compact Mariami strategy" not in history_json
    assert "file_path" not in history_json


def test_default_playbook_lexical_mode_skips_openai_embedding(monkeypatch, tmp_path):
    monkeypatch.delenv("KNOWLEDGE_RETRIEVAL_MODE", raising=False)

    class FakeCollection:
        def get(self, include=None, limit=None):
            if include == ["documents"]:
                return {
                    "documents": [
                        "CTA mistakes: start with the specific pain before the offer.",
                        "Retention: use a first-second pattern interrupt before explaining.",
                    ]
                }
            raise AssertionError("Lexical mode should not request embeddings")

        def query(self, **kwargs):
            raise AssertionError("Lexical mode should not run vector query")

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_collection(self, name):
            assert name == "mariami_reels_playbook_v1"
            return FakeCollection()

    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=FakeClient))

    service = DeterministicKnowledgeRetrievalService(
        chroma_persist_dir=tmp_path,
        collection_name="mariami_reels_playbook_v1",
    )
    monkeypatch.setattr(
        service,
        "_openai_embedding",
        lambda text: (_ for _ in ()).throw(AssertionError("OpenAI embedding should not be attempted")),
    )

    result = service.retrieve(
        task_type="reel_idea",
        message="Give me 3 Reel ideas about CTA mistakes and retention.",
    )

    assert result.used is True
    assert result.collection_name == "mariami_reels_playbook_v1"
    assert result.retrieved_count == 2
    assert "CTA mistakes" in result.knowledge_context


def test_agent_chat_auto_sync_true_skips_sync_when_context_is_fresh(monkeypatch, tmp_path):
    _patch_basic_agent_chat_route(monkeypatch, tmp_path)

    monkeypatch.setattr(
        agent_route.instagram_context_sync_service,
        "get_context_freshness",
        lambda account_id: {
            "account_id": account_id,
            "last_synced_at": "2026-05-02T08:00:00+00:00",
            "context_was_fresh": True,
            "sync_required": False,
            "sync_skipped": True,
            "has_complete_context": True,
            "missing_sections": [],
            "context_age_seconds": 120,
            "stale_reasons": [],
        },
    )
    monkeypatch.setattr(
        agent_route.instagram_context_sync_service,
        "sync",
        lambda user_id, account_id=None: (_ for _ in ()).throw(AssertionError("Fresh context should skip sync")),
    )
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Chat greeting should not trigger retrieval")),
    )

    response = TestClient(app).post(
        "/api/v1/agent/chat",
        json={
            "message": "გამარჯობა, რაში დამეხმარები?",
            "task_type": "chat",
            "user_id": "user-1",
            "account_id": "acct-1",
            "auto_sync": True,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["auto_sync_requested"] is True
    assert payload["auto_sync_performed"] is False
    assert payload["context_fresh"] is True
    assert payload["context_age_seconds"] == 120
    assert payload["knowledge_retrieval_used"] is False


def test_agent_chat_auto_sync_true_runs_sync_when_context_is_missing_or_stale(monkeypatch, tmp_path):
    _patch_basic_agent_chat_route(monkeypatch, tmp_path)

    freshness_states = [
        {
            "account_id": "acct-1",
            "last_synced_at": None,
            "context_was_fresh": False,
            "sync_required": True,
            "sync_skipped": False,
            "has_complete_context": False,
            "missing_sections": ["profile_context"],
            "context_age_seconds": None,
            "stale_reasons": ["missing_context"],
        },
        {
            "account_id": "acct-1",
            "last_synced_at": "2026-05-02T08:00:00+00:00",
            "context_was_fresh": True,
            "sync_required": False,
            "sync_skipped": True,
            "has_complete_context": True,
            "missing_sections": [],
            "context_age_seconds": 0,
            "stale_reasons": [],
        },
    ]
    sync_calls: list[tuple[str, str | None]] = []

    def fake_get_context_freshness(account_id):
        return freshness_states.pop(0)

    def fake_sync(user_id, account_id=None):
        sync_calls.append((user_id, account_id))
        return {"synced": True}

    monkeypatch.setattr(agent_route.instagram_context_sync_service, "get_context_freshness", fake_get_context_freshness)
    monkeypatch.setattr(agent_route.instagram_context_sync_service, "sync", fake_sync)
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Chat greeting should not trigger retrieval")),
    )

    response = TestClient(app).post(
        "/api/v1/agent/chat",
        json={
            "message": "გამარჯობა, რაში დამეხმარები?",
            "task_type": "chat",
            "user_id": "user-1",
            "account_id": "acct-1",
            "auto_sync": True,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert sync_calls == [("user-1", "acct-1")]
    assert payload["auto_sync_requested"] is True
    assert payload["auto_sync_performed"] is True
    assert payload["context_fresh"] is True
    assert payload["context_age_seconds"] == 0


def test_agent_chat_auto_sync_true_second_identical_request_skips_fresh_sync(monkeypatch, tmp_path):
    _patch_basic_agent_chat_route(monkeypatch, tmp_path)

    state = {"fresh": False, "post_sync_reads": 0}
    sync_calls: list[tuple[str, str | None]] = []

    def fake_get_context_freshness(account_id):
        if not state["fresh"]:
            return {
                "account_id": account_id,
                "last_synced_at": None,
                "context_was_fresh": False,
                "sync_required": True,
                "sync_skipped": False,
                "has_complete_context": False,
                "missing_sections": ["profile_context"],
                "context_age_seconds": None,
                "stale_reasons": ["missing_context"],
            }

        if state["post_sync_reads"] > 0:
            state["post_sync_reads"] -= 1
            age_seconds = 0
        else:
            age_seconds = 0

        return {
            "account_id": account_id,
            "last_synced_at": "2026-05-02T08:00:00+00:00",
            "context_was_fresh": True,
            "sync_required": False,
            "sync_skipped": True,
            "has_complete_context": True,
            "missing_sections": [],
            "context_age_seconds": age_seconds,
            "stale_reasons": [],
        }

    def fake_sync(user_id, account_id=None):
        sync_calls.append((user_id, account_id))
        state["fresh"] = True
        state["post_sync_reads"] = 1
        return {"synced": True}

    reel_ideas_reply = "\n\n".join(
        [
            "IDEA 1\nTitle: CTA Audit\nHook: Your views are not the problem.\nFormat type: Talking head\nMain idea: Show one CTA mistake and the fix.\nShot list:\n1. Hook\n2. Mistake\n3. Fix\nWhy it can work: It makes a hidden conversion issue clear.\nCaption idea: Save this before posting.\nCTA: DM CTA.",
            "IDEA 2\nTitle: First Second Fix\nHook: Most Reels lose people before the point.\nFormat type: Screen recording\nMain idea: Compare a slow opening with a direct one.\nShot list:\n1. Bad opening\n2. Better opening\n3. CTA\nWhy it can work: It gives an instantly usable pattern.\nCaption idea: Use this opening today.\nCTA: DM HOOK.",
            "IDEA 3\nTitle: Proof Before Pitch\nHook: Stop selling before showing proof.\nFormat type: Proof breakdown\nMain idea: Show one proof moment before the offer.\nShot list:\n1. Claim\n2. Proof\n3. Offer\nWhy it can work: It builds trust before asking for action.\nCaption idea: Proof makes the CTA easier.\nCTA: DM PROOF.",
        ]
    )

    monkeypatch.setattr(agent_route.instagram_context_sync_service, "get_context_freshness", fake_get_context_freshness)
    monkeypatch.setattr(agent_route.instagram_context_sync_service, "sync", fake_sync)
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: KnowledgeRetrievalResult(
            used=True,
            top_k=5,
            retrieved_count=3,
            collection_name="mariami_reels_playbook_v1",
            knowledge_context="Compact strategy guidance.",
        ),
    )
    monkeypatch.setattr(
        agent_route.langflow_service,
        "run_agent",
        lambda **kwargs: {
            "reply": reel_ideas_reply,
            "account_id": kwargs.get("account_id"),
            "model_provider": "langflow",
            "model_name": "test-main-flow",
            "used_langflow": True,
            "prompt_section_names": ["main_agent_runtime_payload", "deterministic_knowledge_context"],
        },
    )

    client = TestClient(app)
    request_payload = {
        "message": "მომეცი 3 ძლიერი Reels იდეა ჩემი მიმდინარე Instagram კონტექსტის მიხედვით.",
        "task_type": "reel_idea",
        "user_id": "user-1",
        "account_id": "test3",
        "goal": "increase qualified inbound leads from Instagram",
        "auto_sync": True,
    }

    first_response = client.post("/api/v1/agent/chat", json=request_payload)
    second_response = client.post("/api/v1/agent/chat", json=request_payload)

    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text
    first_payload = first_response.json()
    second_payload = second_response.json()

    assert sync_calls == [("user-1", "test3")]
    assert first_payload["auto_sync_requested"] is True
    assert first_payload["auto_sync_performed"] is True
    assert first_payload["context_fresh"] is True
    assert first_payload["context_age_seconds"] == 0
    assert second_payload["auto_sync_requested"] is True
    assert second_payload["auto_sync_performed"] is False
    assert second_payload["context_fresh"] is True
    assert second_payload["context_age_seconds"] > 0
    assert second_payload["knowledge_retrieval_used"] is True
    assert second_payload["parse_status"] == "parsed"
    assert len(second_payload["structured_output"]["ideas"]) == 3


def test_agent_chat_auto_sync_false_never_runs_sync(monkeypatch, tmp_path):
    _patch_basic_agent_chat_route(monkeypatch, tmp_path)

    monkeypatch.setattr(
        agent_route.instagram_context_sync_service,
        "get_context_freshness",
        lambda account_id: (_ for _ in ()).throw(AssertionError("auto_sync=false should not check freshness")),
    )
    monkeypatch.setattr(
        agent_route.instagram_context_sync_service,
        "sync",
        lambda user_id, account_id=None: (_ for _ in ()).throw(AssertionError("auto_sync=false should not run sync")),
    )
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Chat greeting should not trigger retrieval")),
    )

    response = TestClient(app).post(
        "/api/v1/agent/chat",
        json={
            "message": "გამარჯობა, რაში დამეხმარები?",
            "task_type": "chat",
            "user_id": "user-1",
            "account_id": "acct-1",
            "auto_sync": False,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["auto_sync_requested"] is False
    assert payload["auto_sync_performed"] is False
    assert payload["context_fresh"] is None
    assert payload["context_age_seconds"] is None
    assert payload["knowledge_retrieval_used"] is False


def test_agent_chat_request_accepts_link_and_source_url_aliases():
    source_url = "https://www.instagram.com/reel/DXZeHOZoFRd/"

    source_url_payload = AgentChatRequest(
        message="Analyze this Instagram Reel.",
        task_type="link_analysis",
        user_id="user-1",
        source_url=source_url,
    )
    link_payload = AgentChatRequest(
        message="Analyze this Instagram Reel.",
        task_type="link_analysis",
        user_id="user-1",
        link=source_url,
    )

    assert source_url_payload.link == source_url
    assert source_url_payload.source_url == source_url
    assert link_payload.link == source_url
    assert link_payload.source_url == source_url


def test_link_context_service_uses_safe_instagram_fallback_without_fetch_metadata(monkeypatch):
    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, link):
            raise httpx.ConnectError("metadata unavailable", request=httpx.Request("GET", link))

    monkeypatch.setattr("app.services.link_context_service.httpx.Client", FailingClient)

    result = LinkContextService().extract_context("https://www.instagram.com/reel/DXZeHOZoFRd/")

    assert result["source_url"] == "https://www.instagram.com/reel/DXZeHOZoFRd/"
    assert result["detected_platform"] == "instagram"
    assert result["content_type"] == "reel"
    assert result["data_available"] is False
    assert result["data_availability"] == "limited_link_signal"
    assert "Full visual" in result["limitations"][0]
    assert "watched" not in json.dumps(result, ensure_ascii=False).lower()


def test_link_analysis_route_sends_link_context_media_match_and_knowledge_to_main_langflow(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "true")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")
    monkeypatch.setattr(agent_route.langflow_service, "flow_id", "test-main-flow")
    monkeypatch.setattr(agent_route.langflow_service, "api_key", "test-key")

    history_service = GenerationHistoryService()
    history_service.data_file = tmp_path / "generation_history.json"

    monkeypatch.setattr(agent_route, "generation_history_service", history_service)
    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "test3")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {
        "brand_name": "Lead Lab",
        "niche": "service marketing",
        "target_audience": "service founders",
        "brand_voice": "direct",
        "bio": "We turn weak DMs into qualified leads",
        "content_focus": ["reels", "proof"],
        "access_token": "PROFILE_SECRET_SHOULD_NOT_LEAK",
    })
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {
        "posts": [
            {
                "post_id": "p1",
                "content_type": "REEL",
                "topic": "CTA mistakes",
                "caption": "Long caption " * 20,
                "views": 1000,
                "likes": 90,
                "comments": 10,
                "saves": 5,
            },
        ]
    })
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {
        "top_formats": ["Reels"],
        "best_topics": ["CTA mistakes"],
        "weak_ctas": ["follow for more"],
        "notes": ["retention drops after the first second"],
    })
    monkeypatch.setattr(agent_route.link_context_service, "extract_context", lambda link: {
        "link": link,
        "source_url": link,
        "source_type": "instagram_link",
        "detected_platform": "instagram",
        "content_type": "reel",
        "summary": "Public Reel metadata summary",
        "hook_style": "problem-first",
        "business_goal": "engagement",
        "data_available": True,
        "data_availability": "public_metadata",
        "analysis_basis": ["url_structure", "public_page_metadata"],
        "limitations": ["No visual/audio/frame analysis was performed."],
        "source_patterns": ["direct hook", "proof moment"],
        "probable_strengths": ["metadata was fetched"],
        "adaptation_notes": ["adapt the angle"],
        "access_token": "LINK_SECRET_SHOULD_NOT_LEAK",
        "raw_html": "RAW_HTML_SHOULD_NOT_LEAK",
    })
    monkeypatch.setattr(agent_route.instagram_media_service, "get_media", lambda user_id, account_id, limit=25: {
        "items": [
            {
                "media_id": "media-1",
                "account_id": account_id,
                "media_type": "REEL",
                "caption": "Own account Reel caption that should be summarized safely.",
                "permalink": "https://www.instagram.com/reel/DXZeHOZoFRd/",
                "timestamp": "2026-04-27T10:00:00+00:00",
                "like_count": 123,
                "comments_count": 9,
                "is_reel": True,
            }
        ]
    })
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: KnowledgeRetrievalResult(
            used=True,
            top_k=5,
            retrieved_count=2,
            collection_name="mariami_reels_playbook_v1",
            knowledge_context="Internal compact strategy guidance for link analysis.",
        ),
    )

    captured_langflow_payloads: list[dict] = []

    def fake_post_run_flow(**kwargs):
        captured_langflow_payloads.append(kwargs["payload"])
        return _langflow_text_response(
            "What works:\nAvailable link signal and account context point to a direct hook.\n\n"
            "What is weak:\nFull visual/audio data is unavailable.\n\n"
            "Why it may perform:\nThe pattern is easy to understand quickly.\n\n"
            "How to adapt it for the user's account:\nUse the same problem-first structure with a clearer CTA."
        )

    monkeypatch.setattr(agent_route.langflow_service, "_post_run_flow", fake_post_run_flow)

    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "გაანალიზე ეს Instagram Reel და მითხარი რა მუშაობს.",
            "task_type": "link_analysis",
            "user_id": "user-1",
            "account_id": "test3",
            "link": "https://www.instagram.com/reel/DXZeHOZoFRd/",
            "goal": "improve retention and CTA clarity",
            "auto_sync": False,
        },
    )

    assert response.status_code == 200, response.text
    response_payload = response.json()
    assert response_payload["knowledge_retrieval_used"] is True
    assert response_payload["knowledge_retrieved_count"] == 2

    sent_input = captured_langflow_payloads[-1]["input_value"]
    runtime_payload = _main_agent_runtime_payload(sent_input)
    serialized_runtime = json.dumps(runtime_payload, ensure_ascii=False)

    assert runtime_payload["task_type"] == "link_analysis"
    assert runtime_payload["link"] == "https://www.instagram.com/reel/DXZeHOZoFRd/"
    assert runtime_payload["source_url"] == "https://www.instagram.com/reel/DXZeHOZoFRd/"
    assert runtime_payload["link_context"]["detected_platform"] == "instagram"
    assert runtime_payload["link_context"]["content_type"] == "reel"
    assert runtime_payload["link_context"]["data_availability"] == "public_metadata"
    assert runtime_payload["link_context"]["matched_connected_media"]["media_id"] == "media-1"
    assert runtime_payload["target_media_summary"]["media_id"] == "media-1"
    assert runtime_payload["target_media_summary"]["like_count"] == 123
    assert "knowledge_context" in runtime_payload
    assert "PROFILE_SECRET_SHOULD_NOT_LEAK" not in serialized_runtime
    assert "LINK_SECRET_SHOULD_NOT_LEAK" not in serialized_runtime
    assert "RAW_HTML_SHOULD_NOT_LEAK" not in serialized_runtime
    assert "SYSTEM ROLE AND RULES" not in sent_input


def test_reel_feedback_with_link_passes_link_context_to_main_langflow(monkeypatch):
    monkeypatch.setenv("USE_LANGFLOW_FOR_AGENT_CHAT", "true")
    monkeypatch.setenv("USE_LANGFLOW_SAFE_REELS_RAG", "false")

    monkeypatch.setattr(agent_route.billing_service, "enforce_agent_access", lambda user_id, task_type: {"current_plan": "pro"})
    monkeypatch.setattr(agent_route.billing_service, "increment_generation_usage", lambda user_id: None)
    monkeypatch.setattr(agent_route.connected_accounts_service, "resolve_account_id", lambda user_id, account_id=None: account_id or "test3")
    monkeypatch.setattr(agent_route.profile_context_service, "get_context", lambda account_id: {})
    monkeypatch.setattr(agent_route.recent_posts_context_service, "get_context", lambda account_id: {"posts": []})
    monkeypatch.setattr(agent_route.recent_content_context_service, "get_context", lambda account_id: {})
    monkeypatch.setattr(agent_route.instagram_media_service, "get_media", lambda user_id, account_id, limit=25: {"items": []})
    monkeypatch.setattr(agent_route.link_context_service, "extract_context", lambda link: {
        "link": link,
        "source_url": link,
        "source_type": "instagram_link",
        "detected_platform": "instagram",
        "content_type": "reel",
        "summary": "Public Reel metadata unavailable; fallback link signal only.",
        "hook_style": "short visual hook",
        "business_goal": "engagement",
        "data_available": False,
        "data_availability": "limited_link_signal",
        "analysis_basis": ["url_structure"],
        "limitations": ["Full visual/audio/frame analysis is unavailable."],
        "source_patterns": ["likely optimized for fast first-3-second retention"],
        "probable_strengths": ["link structure was recognized"],
        "adaptation_notes": ["adapt to the user's brand voice"],
    })
    monkeypatch.setattr(
        agent_route.knowledge_retrieval_service,
        "retrieve",
        lambda **kwargs: KnowledgeRetrievalResult(
            used=True,
            top_k=5,
            retrieved_count=0,
            collection_name="mariami_reels_playbook_v1",
        ),
    )

    captured_calls: list[dict] = []

    def fake_langflow_run(**kwargs):
        captured_calls.append(kwargs)
        return {
            "reply": (
                "Summary:\nAnalysis is based on the available link signal and account context, not a watched video.\n\n"
                "What works:\nThe Reel link is recognizable as a Reel.\n\n"
                "What hurts:\nFull visual data is unavailable.\n\n"
                "Retention risks:\nThe hook cannot be verified from metadata alone.\n\n"
                "Better hook:\nStart with the specific problem.\n\n"
                "Better CTA:\nUse one direct DM CTA.\n\n"
                "Improved version:\nOpen with the pain point, give one proof moment, and close with one CTA."
            ),
            "account_id": kwargs.get("account_id"),
            "model_provider": "langflow",
            "model_name": "test-main-flow",
            "used_langflow": True,
            "prompt_section_names": ["main_agent_runtime_payload"],
        }

    monkeypatch.setattr(agent_route.langflow_service, "run_agent", fake_langflow_run)

    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "Review this Reel link.",
            "task_type": "reel_feedback",
            "user_id": "user-1",
            "account_id": "test3",
            "link": "https://www.instagram.com/reel/DXZeHOZoFRd/",
            "auto_sync": False,
        },
    )

    assert response.status_code == 200, response.text
    assert captured_calls[-1]["link"] == "https://www.instagram.com/reel/DXZeHOZoFRd/"
    assert captured_calls[-1]["link_context"]["detected_platform"] == "instagram"
    assert captured_calls[-1]["link_context"]["data_available"] is False
    assert captured_calls[-1]["reel_context"]["source"] == "link"
    assert captured_calls[-1]["reel_context"]["permalink"] == "https://www.instagram.com/reel/DXZeHOZoFRd/"
