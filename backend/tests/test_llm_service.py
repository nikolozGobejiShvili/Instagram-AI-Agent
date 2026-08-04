import json
import sys
from pathlib import Path

import httpx
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.llm_service import LLMService, LLMServiceError  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, json=self._payload, headers=self.headers, request=self.request),
            )


def test_llm_service_retries_rate_limit_and_raises_clean_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PRIMARY_LLM_MODEL", "primary-model")
    monkeypatch.delenv("FALLBACK_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("FALLBACK_LLM_MODEL", raising=False)

    service = LLMService()
    sleep_calls: list[float] = []

    monkeypatch.setattr(service, "_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(service, "_jitter", lambda: 0.0)
    monkeypatch.setattr(
        service,
        "_send_openai_request",
        lambda **kwargs: FakeResponse(
            429,
            {"error": {"code": "insufficient_quota", "type": "insufficient_quota", "message": "quota hit"}},
            {"Retry-After": "1.5"},
        ),
    )

    prepared = service._prepare_generation(
        message="მომეცი 3 ძლიერი Reels იდეა.",
        task_type="reel_idea",
        goal="increase qualified inbound leads",
        profile_context={"niche": "beauty"},
        recent_posts_context={"posts": [{"content_type": "REEL", "topic": "hook logic", "caption": "Short caption"}]},
        playbook_context={"chunks": [{"chunk_label": "Module 1", "text": "Use tension and simplicity."}]},
    )

    with pytest.raises(LLMServiceError) as exc_info:
        service._call_openai_with_fallback(
            task_type="reel_idea",
            response_input=prepared["response_input"],
            prompt_token_estimate=prepared["prompt_token_estimate"],
        )

    exc = exc_info.value
    assert exc.code == "llm_rate_limited"
    assert exc.status_code == 503
    assert exc.retry_count == service.MAX_RETRIES
    assert exc.rate_limited is True
    assert exc.prompt_token_estimate == prepared["prompt_token_estimate"]
    assert sleep_calls == [1.5, 1.5, 1.5]


def test_llm_service_uses_fallback_model_after_primary_rate_limit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PRIMARY_LLM_MODEL", "primary-model")
    monkeypatch.setenv("FALLBACK_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "fallback-model")

    service = LLMService()
    call_models: list[str] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(service, "_sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(service, "_jitter", lambda: 0.0)

    def fake_send_openai_request(**kwargs):
        model_name = kwargs["payload"]["model"]
        call_models.append(model_name)
        if model_name == "primary-model":
            return FakeResponse(
                429,
                {"error": {"code": "rate_limit", "type": "rate_limit", "message": "slow down"}},
                {"Retry-After": "0"},
            )

        return FakeResponse(
            200,
            {
                "output_text": json.dumps(
                    {
                        "reply": "Title:\nFallback Reel",
                        "structured_output": {
                            "ideas": [
                                {
                                    "title": "Fallback Reel",
                                    "hook": "Stop wasting your first second.",
                                    "format_type": "Talking-head breakdown",
                                    "main_idea": "Show one mistake and one fix.",
                                    "shot_list": ["Hook", "Proof", "Fix", "CTA"],
                                    "why_it_can_work": "It turns a familiar pain point into a saveable Reel.",
                                    "cta": "DM me REEL.",
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
            },
        )

    monkeypatch.setattr(service, "_send_openai_request", fake_send_openai_request)

    result = service.run_agent(
        message="მომეცი 3 ძლიერი Reels იდეა.",
        task_type="reel_idea",
        goal="increase qualified inbound leads",
        profile_context={"niche": "beauty"},
        recent_posts_context={"posts": [{"content_type": "REEL", "topic": "hook logic", "caption": "Short caption"}]},
        playbook_context={"chunks": [{"chunk_label": "Module 1", "text": "Use tension and simplicity."}]},
    )

    assert result["model_provider"] == "openai"
    assert result["model_name"] == "fallback-model"
    assert result["used_langflow"] is False
    assert result["rate_limited"] is True
    assert result["retry_count"] == service.MAX_RETRIES
    assert result["parse_status"] == "parsed"
    assert result["structured_output"]["ideas"][0]["title"] == "Fallback Reel"
    assert call_models.count("primary-model") == service.MAX_RETRIES + 1
    assert call_models.count("fallback-model") == 1
    assert sleep_calls == [0.0, 0.0, 0.0]


def test_llm_service_compacts_recent_posts_and_limits_playbook_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PRIMARY_LLM_MODEL", "primary-model")

    service = LLMService()
    long_caption = ("LONG_CAPTION_MARKER " * 80) + "FULL_CAPTION_TAIL_LEAK"
    playbook_chunks = []
    for chunk_index in range(1, 8):
        playbook_chunks.append({
            "chunk_id": f"chunk-{chunk_index}",
            "knowledge_pack_id": "kp-test",
            "chunk_label": f"Module {chunk_index}",
            "file_name": "mariami-reels.md",
            "text": f"SENTINEL_CHUNK_{chunk_index} " + ("strategy " * 180),
        })

    prepared = service._prepare_generation(
        message="მომეცი 3 ძლიერი Reels იდეა ჩემი მიმდინარე Instagram კონტექსტის მიხედვით.",
        task_type="reel_idea",
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
        recent_content_context={
            "top_formats": ["Reels"],
            "best_topics": ["CTA mistakes", "hook logic"],
            "notes": ["retention drops after the first second", "proof-led content converts better"],
        },
        recent_posts_context={
            "posts": [
                {"post_id": f"p{index}", "content_type": "REEL", "topic": f"topic {index}", "caption": long_caption, "views": 1000 - index, "likes": 90 - index, "comments": 10, "saves": 5}
                for index in range(1, 8)
            ]
        },
        playbook_context={
            "used_system_knowledge": True,
            "matched_knowledge_domain": "reels",
            "matched_knowledge_pack_ids": ["kp-test"],
            "chunks": playbook_chunks,
        },
    )

    assert 3 <= prepared["playbook_context"]["retrieved_chunk_count"] <= service.MAX_PLAYBOOK_CHUNKS_BY_TASK["reel_idea"]
    assert len(prepared["recent_posts_context"]["posts"]) == service.MAX_RECENT_POSTS_BY_TASK["reel_idea"]
    joined_prompt = "\n".join(
        content_item["text"]
        for message in prepared["response_input"]
        for content_item in message.get("content", [])
    )
    assert "SENTINEL_CHUNK_1" in joined_prompt
    assert "SENTINEL_CHUNK_6" not in joined_prompt
    assert "SENTINEL_CHUNK_7" not in joined_prompt
    assert "FULL_CAPTION_TAIL_LEAK" not in joined_prompt
    assert "caption_summary=" in joined_prompt
    assert "reason_relevant=" in joined_prompt
    assert prepared["prompt_token_estimate"] > 0
