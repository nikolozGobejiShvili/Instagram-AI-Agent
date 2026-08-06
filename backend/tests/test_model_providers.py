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


def test_missing_credential_is_a_configuration_error_not_a_model_error(monkeypatch):
    # Both names must be cleared: ANTHROPIC_API_KEY is still accepted as an
    # alias for ANTHROPIC_AUTH_TOKEN (see test_anthropic_setup_token.py).
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured) as exc:
        AnthropicTextProvider().generate(
            task_type="chat", response_input=SECTIONS, max_output_tokens=700
        )
    assert exc.value.missing == "ANTHROPIC_AUTH_TOKEN"
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
    # profile_audit is schema-enforced, so the model answers with the JSON
    # document rather than prose.
    stub = StubAnthropic(blocks=[_schema_reply(reply="analysed", structured_output={"summary": "ok"})])
    providers.register_text_provider("anthropic", lambda: AnthropicTextProvider(client=stub))

    result = LLMService().run_agent(message="შეაფასე ჩემი გვერდი", task_type="profile_audit")

    assert result["reply"] == "analysed"
    assert result["model_provider"] == "anthropic"
    assert result["used_langflow"] is False
    assert result["prompt_section_names"], "prompt sections must still be reported"
    # profile_audit is an analysis task — it should not run at the cheapest effort
    assert stub.captured["output_config"]["effort"] == "high"


CAROUSEL_REPLY = "\n".join([
    "Title: ძლიერი კაუჭის 3 წესი",
    "Slide 1: დაიწყე კითხვით",
    "მაყურებელი პასუხს ეძებს.",
    "Slide 2: აჩვენე შედეგი",
    "ციფრი ჯობია დაპირებას.",
    "Slide 3: ერთი აზრი ერთ სლაიდზე",
    "ორი აზრი ორივეს კლავს.",
    "Final CTA slide: შემინახე მოგვიანებისთვის",
])


SCHEMA = {
    "type": "object",
    "required": ["reply", "structured_output"],
    "properties": {"reply": {"type": "string"}, "structured_output": {"type": "object"}},
}

# Every task whose shape is enforced rather than requested. Without a schema a
# task falls back to heading parsing, which Sonnet 4.6 does not follow -- it
# answers in markdown and the payload comes back empty.
SCHEMA_ENFORCED_TASKS = [
    "carousel",
    "caption",
    "profile_audit",
    "content_plan",
    "performance_summary",
    "reel_idea",
    "reel_script",
    "reel_feedback",
]


def _sample_for(node):
    """Smallest value satisfying ``node``, used to prove the schemas agree."""
    node_type = node.get("type")
    if node_type == "object":
        return {name: _sample_for(sub) for name, sub in node.get("properties", {}).items()}
    if node_type == "array":
        return [_sample_for(node["items"])]
    if node_type == "integer":
        return 1
    if node_type == "boolean":
        return True
    return "x"


# ------------------------------------------------------- schema derivation


@pytest.mark.parametrize("task_type", SCHEMA_ENFORCED_TASKS)
def test_every_structured_task_has_an_enforced_schema(task_type):
    from app.services.llm_service import LLMService

    assert LLMService()._structured_schema(task_type) is not None


@pytest.mark.parametrize("task_type", SCHEMA_ENFORCED_TASKS)
def test_the_enforced_schema_is_narrower_than_the_validating_model(task_type):
    """The property the whole design rests on.

    The provider enforces one schema and ``validate_structured_output_payload``
    then checks another. If the enforced one were the wider of the two, a
    response could satisfy the provider and fail validation -- structured_output
    silently becoming None, which is precisely the bug this path exists to stop.
    Generating from the enforced schema and validating the result asserts the
    direction directly instead of trusting that two definitions were kept in
    step by hand.
    """
    from app.schemas.agent import validate_structured_output_payload
    from app.services.llm_service import LLMService

    schema = LLMService()._structured_schema(task_type)
    sample = _sample_for(schema["properties"]["structured_output"])

    validated, status, _ = validate_structured_output_payload(task_type, "parsed", sample)

    assert validated is not None, f"{task_type}: enforced schema produced a payload validation rejects"
    assert status == "parsed"


@pytest.mark.parametrize("task_type", SCHEMA_ENFORCED_TASKS)
def test_enforced_schemas_leave_no_field_optional(task_type):
    """A partially-filled object must not be a valid answer: the customer is
    paying for every field, and an absent one reads as a rendering bug later."""
    from app.services.llm_service import LLMService

    def assert_tight(node, path="structured_output"):
        if node.get("type") == "object" and "properties" in node:
            assert set(node["required"]) == set(node["properties"]), path
            assert node["additionalProperties"] is False, path
            for name, sub in node["properties"].items():
                assert_tight(sub, f"{path}.{name}")
        elif node.get("type") == "array":
            assert_tight(node["items"], f"{path}[]")

    assert_tight(LLMService()._structured_schema(task_type)["properties"]["structured_output"])


def test_nested_models_are_inlined_rather_than_referenced():
    """Whether a provider follows $defs is not worth discovering in production."""
    import json

    from app.services.llm_service import LLMService

    rendered = json.dumps(LLMService()._structured_schema("content_plan"))

    assert "$ref" not in rendered
    assert "$defs" not in rendered
    # the nested item model still made it through
    items = LLMService()._structured_schema("content_plan")["properties"]["structured_output"]
    assert "topic" in items["properties"]["content_items"]["items"]["properties"]


def test_nullable_fields_collapse_so_the_model_cannot_answer_null():
    """`str | None` on the validating model means "may be absent afterwards",
    not "feel free to skip it"."""
    from app.services.llm_service import LLMService

    summary = LLMService()._structured_schema("profile_audit")["properties"]["structured_output"]["properties"]["summary"]

    assert summary == {"type": "string"}


def _schema_reply(**payload):
    import json

    return SimpleNamespace(type="text", text=json.dumps(payload))


# ------------------------------------------------- schema-constrained output


def test_a_schema_constrains_the_response_instead_of_asking_for_headings():
    """Asking did not work. Given the "Title: / Slide 1:" contract Sonnet 4.6
    answers in markdown, so the shape has to be enforced rather than requested."""
    client = StubAnthropic(blocks=[_schema_reply(reply="ok", structured_output={"slides": []})])

    AnthropicTextProvider(client=client).generate(
        task_type="carousel", response_input=SECTIONS, max_output_tokens=1200, response_schema=SCHEMA
    )

    fmt = client.captured["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": SCHEMA}
    # effort must survive alongside it -- both live in output_config
    assert client.captured["output_config"]["effort"] == "medium"


def test_no_schema_sends_no_format():
    client = StubAnthropic()

    AnthropicTextProvider(client=client).generate(
        task_type="chat", response_input=SECTIONS, max_output_tokens=700
    )

    assert "format" not in client.captured["output_config"]


def test_the_customer_never_sees_the_raw_json():
    """With output_config.format the text block *is* the JSON document, so
    returning it unchanged would put JSON where prose belongs."""
    client = StubAnthropic(blocks=[_schema_reply(
        reply="Here is your carousel.",
        structured_output={"title": "სათაური", "slides": [{"slide_number": 1}]},
    )])

    result = AnthropicTextProvider(client=client).generate(
        task_type="carousel", response_input=SECTIONS, max_output_tokens=1200, response_schema=SCHEMA
    )

    assert result["reply"] == "Here is your carousel."
    assert "structured_output" not in result["reply"]
    assert result["structured_output"]["slides"] == [{"slide_number": 1}]
    assert result["parse_status"] == "parsed"


def test_a_truncated_schema_response_fails_instead_of_showing_json():
    """A cut-off document is a fragment like `{"reply":"Here's your 30-day...`.

    There is no prose to fall back to, so showing it is worse than an honest
    failure -- and usage is charged on success only, so failing means the
    customer is not billed for the fragment.
    """
    client = StubAnthropic(
        blocks=[SimpleNamespace(type="text", text='{"reply":"Here\'s your 30-day plan for')],
        stop_reason="max_tokens",
    )

    with pytest.raises(ProviderError) as exc:
        AnthropicTextProvider(client=client).generate(
            task_type="content_plan", response_input=SECTIONS, max_output_tokens=1200, response_schema=SCHEMA
        )

    # Named distinctly: hitting the ceiling means our own token limit is too low
    # for this task, which is invisible if reported as a generic failure.
    assert exc.value.code == "generation_truncated"


def test_an_unreadable_schema_response_is_reported_as_such():
    client = StubAnthropic(blocks=[SimpleNamespace(type="text", text="not json at all")])

    with pytest.raises(ProviderError) as exc:
        AnthropicTextProvider(client=client).generate(
            task_type="carousel", response_input=SECTIONS, max_output_tokens=1200, response_schema=SCHEMA
        )

    assert exc.value.code == "generation_unreadable"
    assert "not json at all" not in exc.value.safe_message


def test_schema_tasks_get_enough_room_for_their_widest_answer():
    """JSON is more verbose than the prose these limits were tuned for, and a
    truncated document loses the whole generation rather than its last line."""
    from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES
    from app.services.llm_service import LLMService

    service = LLMService()
    # 30 dated items, each carrying four fields, plus lists and a summary
    assert service._max_output_tokens("content_plan") >= 4000
    # every slide carries a headline, body and art direction
    assert service._max_output_tokens("carousel") >= MAX_CAROUSEL_SLIDES * 120


def test_carousel_is_one_of_the_schema_enforced_tasks():
    """The task the schema was missing on. Losing it here puts the headline
    feature back on the model's willingness to follow headings."""
    from app.services.llm_service import LLMService

    schema = LLMService()._structured_schema("carousel")

    assert schema is not None
    slides = schema["properties"]["structured_output"]["properties"]["slides"]
    assert set(slides["items"]["required"]) == {"slide_number", "headline", "body", "image_prompt"}


def test_the_schema_reaches_the_provider(monkeypatch):
    """A schema defined but never passed down would enforce nothing."""
    from app.services.llm_service import LLMService

    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "anthropic")
    stub = StubAnthropic(blocks=[_schema_reply(reply="ok", structured_output={"title": "t", "slides": []})])
    providers.register_text_provider("anthropic", lambda: AnthropicTextProvider(client=stub))

    LLMService().run_agent(message="გააკეთე კარუსელი", task_type="carousel")

    assert "format" in stub.captured["output_config"]


class ProseOnlyProvider:
    """A provider that ignores ``response_schema``.

    The base contract permits this deliberately -- a provider that cannot
    enforce a schema must not fail the request -- so the caller parses prose
    itself. That path was missing entirely: the registry branch returned the
    provider payload untouched, so a carousel job read ``structured_output``,
    found nothing, and failed with "did not produce any slides", naming the
    model rather than the parse that never ran. This reached production.
    """

    name = "prose-only"

    def __init__(self, reply):
        self._reply = reply

    def generate(self, *, task_type, response_input, max_output_tokens, response_schema=None):
        return {"reply": self._reply, "model_provider": self.name, "model_name": "stub"}


def test_a_provider_that_ignores_schemas_still_yields_structured_output(monkeypatch):
    from app.services.llm_service import LLMService

    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "prose-only")
    providers.register_text_provider("prose-only", lambda: ProseOnlyProvider(CAROUSEL_REPLY))

    result = LLMService().run_agent(message="გააკეთე კარუსელი", task_type="carousel")

    structured = result["structured_output"]
    assert structured is not None, "a carousel with no structured output cannot be rendered"
    assert [slide["slide_number"] for slide in structured["slides"]] == [1, 2, 3]
    assert structured["slides"][0]["headline"] == "დაიწყე კითხვით"
    assert result["parse_status"] == "parsed"


def test_an_unparseable_prose_reply_still_returns_the_generation(monkeypatch):
    """Prose that will not parse is still prose. Discarding it because the
    headings were wrong would turn a formatting miss into a refund -- unlike a
    truncated JSON document, where nothing readable is left."""
    from app.services.llm_service import LLMService

    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "prose-only")
    providers.register_text_provider("prose-only", lambda: ProseOnlyProvider("just some prose, no headings"))

    result = LLMService().run_agent(message="hi", task_type="carousel")

    assert result["reply"] == "just some prose, no headings"
    assert result["parse_status"] == "raw_only"
    assert result["structured_output"] is None


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
