"""Provider abstraction: Anthropic Sonnet 4.6 for text, Gemini for images.

Before this, LLMService.run_agent raised for any provider that was not OpenAI, so
neither wanted model could be attached (audit 2026-08-04, finding 7). These tests
pin the contract using stub clients — no credentials, no network.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import providers  # noqa: E402
from app.services.providers.anthropic_text import AnthropicTextProvider  # noqa: E402
from app.services.providers.base import ProviderError, ProviderNotConfigured  # noqa: E402
from app.services.providers.gemini_image import GeminiImageProvider  # noqa: E402

SECTIONS = [
    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are an agent."}]},
    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Answer in Georgian."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "გააკეთე კარუსელი"}]},
]


class StubAnthropic:
    """Captures the request and returns a canned message."""

    def __init__(self, blocks=None, stop_reason="end_turn"):
        self.captured = None
        self._blocks = blocks if blocks is not None else [SimpleNamespace(type="text", text="ok")]
        self._stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content=self._blocks,
            model=kwargs["model"],
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


@pytest.fixture(autouse=True)
def _clear_registry():
    providers.reset_cache()
    yield
    providers.reset_cache()


# ---------------------------------------------------------------- Anthropic


def test_uses_sonnet_4_6_and_adaptive_thinking():
    client = StubAnthropic()
    AnthropicTextProvider(client=client).generate(
        task_type="carousel", response_input=SECTIONS, max_output_tokens=1200
    )
    req = client.captured

    assert req["model"] == "claude-sonnet-4-6", "model id must carry no date suffix"
    # budget_tokens is deprecated on 4.6; depth is controlled by effort.
    assert req["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in str(req["thinking"])
    # effort is nested inside output_config, not a top-level parameter
    assert req["output_config"]["effort"] == "medium"
    assert "effort" not in req


def test_system_sections_are_split_out_of_the_conversation():
    client = StubAnthropic()
    AnthropicTextProvider(client=client).generate(
        task_type="chat", response_input=SECTIONS, max_output_tokens=700
    )
    req = client.captured

    assert req["system"] == "You are an agent.\n\nAnswer in Georgian."
    assert req["messages"] == [{"role": "user", "content": "გააკეთე კარუსელი"}]


def test_effort_scales_with_task_type():
    provider = AnthropicTextProvider(client=StubAnthropic())
    assert provider.effort_for("chat") == "low"
    assert provider.effort_for("profile_audit") == "high"
    assert provider.effort_for("unknown_task") == "medium"


def test_thinking_blocks_are_not_returned_as_the_reply():
    """Thinking blocks precede text, so content[0] would be reasoning."""
    client = StubAnthropic(blocks=[
        SimpleNamespace(type="thinking", thinking="deliberating..."),
        SimpleNamespace(type="text", text="Slide 1: hook"),
    ])
    result = AnthropicTextProvider(client=client).generate(
        task_type="carousel", response_input=SECTIONS, max_output_tokens=1200
    )

    assert result["reply"] == "Slide 1: hook"
    assert "deliberating" not in result["reply"]
    assert result["model_provider"] == "anthropic"
    assert result["usage"] == {"input_tokens": 11, "output_tokens": 7}


def test_refusal_is_reported_distinctly_from_a_generation_failure():
    client = StubAnthropic(blocks=[], stop_reason="refusal")
    with pytest.raises(ProviderError) as exc:
        AnthropicTextProvider(client=client).generate(
            task_type="chat", response_input=SECTIONS, max_output_tokens=700
        )
    assert exc.value.code == "generation_refused"
    assert exc.value.status_code == 422


def test_missing_api_key_is_a_configuration_error_not_a_model_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured) as exc:
        AnthropicTextProvider().generate(
            task_type="chat", response_input=SECTIONS, max_output_tokens=700
        )
    assert exc.value.missing == "ANTHROPIC_API_KEY"
    assert exc.value.status_code == 503


# ------------------------------------------------------------------- Gemini


class StubGemini:
    def __init__(self, parts):
        self.captured = None
        self.models = SimpleNamespace(generate_content=self._generate)
        self._parts = parts

    def _generate(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=self._parts))]
        )


def _image_part(data):
    return SimpleNamespace(inline_data=SimpleNamespace(data=data))


def test_image_prompt_forbids_rendered_text():
    """Text is composited in code; letters baked into pixels are the failure mode."""
    client = StubGemini([_image_part(b"PNGDATA")])
    provider = GeminiImageProvider(client=client, model_name="test-image-model")

    assert provider.generate_image(prompt="warm minimal studio backdrop") == b"PNGDATA"

    sent = client.captured["contents"].lower()
    assert "do not render any text" in sent
    assert "warm minimal studio backdrop" in sent
    assert client.captured["config"]["image_config"]["aspect_ratio"] == "4:5"


def test_image_is_found_even_when_the_model_narrates_first():
    client = StubGemini([
        SimpleNamespace(inline_data=None, text="Here is your background:"),
        _image_part(b"REALIMAGE"),
    ])
    provider = GeminiImageProvider(client=client, model_name="test-image-model")
    assert provider.generate_image(prompt="x") == b"REALIMAGE"


def test_image_model_id_must_be_configured_not_guessed(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    with pytest.raises(ProviderNotConfigured) as exc:
        GeminiImageProvider(client=StubGemini([])).generate_image(prompt="x")
    assert exc.value.missing == "GEMINI_IMAGE_MODEL"


def test_response_without_an_image_fails_loudly():
    client = StubGemini([SimpleNamespace(inline_data=None, text="I cannot do that")])
    provider = GeminiImageProvider(client=client, model_name="test-image-model")
    with pytest.raises(ProviderError, match="no image"):
        provider.generate_image(prompt="x")


# ----------------------------------------------------------------- registry


def test_registry_resolves_and_caches():
    providers.register_text_provider("stub", lambda: AnthropicTextProvider(client=StubAnthropic()))
    assert providers.get_text_provider("stub") is providers.get_text_provider("stub")
    assert "anthropic" in providers.supported_text_providers()
    assert "gemini" in providers.supported_image_providers()


def test_unknown_provider_is_rejected_clearly():
    with pytest.raises(ProviderError, match="Unsupported text provider"):
        providers.get_text_provider("not-a-provider")


# ------------------------------------------------------- llm_service wiring


def test_llm_service_routes_non_openai_providers_to_the_registry(monkeypatch):
    """The hard raise is gone: a non-OpenAI provider now generates."""
    from app.services.llm_service import LLMService

    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "anthropic")
    stub = StubAnthropic(blocks=[SimpleNamespace(type="text", text="analysed")])
    providers.register_text_provider("anthropic", lambda: AnthropicTextProvider(client=stub))

    result = LLMService().run_agent(message="შეაფასე ჩემი გვერდი", task_type="profile_audit")

    assert result["reply"] == "analysed"
    assert result["model_provider"] == "anthropic"
    assert result["used_langflow"] is False
    assert result["prompt_section_names"], "prompt sections must still be reported"
    # profile_audit is an analysis task — it should not run at the cheapest effort
    assert stub.captured["output_config"]["effort"] == "high"


def test_llm_service_surfaces_provider_errors_as_llm_errors(monkeypatch):
    from app.services.llm_service import LLMService, LLMServiceError

    def explode():
        provider = AnthropicTextProvider(client=StubAnthropic())
        provider.generate = lambda **kwargs: (_ for _ in ()).throw(
            ProviderError("upstream is down", provider="anthropic", status_code=504, code="llm_unreachable")
        )
        return provider

    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "anthropic")
    providers.register_text_provider("anthropic", explode)

    with pytest.raises(LLMServiceError) as exc:
        LLMService().run_agent(message="hi", task_type="chat")

    assert exc.value.status_code == 504
    assert exc.value.code == "llm_unreachable"
    assert exc.value.model_provider == "anthropic"
