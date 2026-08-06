"""What reaches the model: how much retrieved material, and under what label.

Both were wrong in ways that cost quality without ever failing. Material was
written at one size and shown at another, and every retrieved chunk was
announced to the model as reels strategy regardless of what it was about.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.deterministic_knowledge_retrieval_service import (  # noqa: E402
    RAG_TASK_TYPES,
    DeterministicKnowledgeRetrievalService,
)
from app.services.knowledge_pack_service import KnowledgePackService  # noqa: E402
from app.services.langflow_service import LangflowService  # noqa: E402


@pytest.fixture()
def service():
    return DeterministicKnowledgeRetrievalService()


# --------------------------------------------------------------- chunk sizing


def test_retrieval_shows_a_whole_chunk_rather_than_half_of_it(service):
    """Chunks are written at up to 1200 characters and were truncated to 520 on
    the way out, so more than half of everything the operator wrote was
    discarded -- silently, and only at read time."""
    written = KnowledgePackService()._chunk_text("x " * 2000)
    longest = max(len(chunk) for chunk in written)

    assert service.MAX_CHUNK_CHARS >= longest, (
        f"chunks are indexed at up to {longest} chars but shown at {service.MAX_CHUNK_CHARS}"
    )


def test_a_full_chunk_survives_the_context_builder(service):
    chunk = "A" * service.MAX_CHUNK_CHARS
    context = service._build_knowledge_context([chunk], "carousel")

    assert chunk in context, "the chunk was truncated between retrieval and the prompt"


# -------------------------------------------------------------------- budgets


@pytest.mark.parametrize("task_type", sorted(RAG_TASK_TYPES))
def test_every_retrieving_task_has_a_deliberate_budget(task_type, service):
    """A task missing from the table silently falls back to the old single
    number, which is the behaviour this replaced."""
    assert task_type in service.CONTEXT_BUDGET_BY_TASK


def test_long_form_tasks_get_more_room_than_a_caption(service):
    """One budget for everything was generous for a caption, where strategy
    crowds out the request, and mean for the long answers that benefit most."""
    assert service.context_budget("content_plan") > service.context_budget("caption")
    assert service.context_budget("profile_audit") > service.context_budget("caption")


def test_a_budget_fits_at_least_one_whole_chunk(service):
    """A budget below MAX_CHUNK_CHARS would retrieve material and then include
    none of it."""
    for task_type, budget in service.CONTEXT_BUDGET_BY_TASK.items():
        assert budget >= service.MAX_CHUNK_CHARS, task_type


def test_the_budget_actually_bounds_the_context(service):
    chunks = ["B" * service.MAX_CHUNK_CHARS] * 20

    context = service._build_knowledge_context(chunks, "caption")

    assert len(context) <= service.context_budget("caption") + 600  # + heading


# --------------------------------------------------------------------- label


def test_the_context_is_not_announced_as_reels_strategy(service):
    """The heading was hard-coded to "Internal Mariami Reels strategy context",
    so material about carousels or audits was labelled as reels strategy -- a
    description that argues against the content directly beneath it."""
    context = service._build_knowledge_context(["one CTA per carousel"], "carousel")

    assert "Reels strategy" not in context
    assert "carousel" in context


def test_the_material_is_framed_as_proprietary_and_preferred(service):
    """It has to outrank the model's own general knowledge, which is the entire
    reason for uploading it."""
    context = service._build_knowledge_context(["open with tension"], "reel_idea")

    assert "proprietary" in context.lower()
    assert "generic best practice" in context.lower()


def test_the_material_stays_hidden_from_the_customer(service):
    context = service._build_knowledge_context(["open with tension"], "reel_idea")

    assert "never quote" in context.lower()


def test_no_context_produces_no_heading(service):
    """An empty preamble reads to the model as material that exists and is
    blank, which invites it to invent the contents."""
    assert service._build_knowledge_context([], "carousel") is None
    assert service._build_knowledge_context(["   "], "carousel") is None


# ------------------------------------------------------------------ behaviour


def test_the_system_prompt_states_the_agent_is_accountable_not_advisory():
    """The difference the subscription is sold on: it does the work rather than
    describing what a marketer would do."""
    instruction = LangflowService()._base_system_instruction().lower()

    assert "not as an assistant" in instruction
    assert "finished work" in instruction


def test_the_quality_bar_reaches_the_model():
    instruction = LangflowService()._base_system_instruction()

    for line in LangflowService.QUALITY_BAR:
        assert line in instruction


def test_the_quality_bar_forbids_unfilled_placeholders():
    """Copy the user still has to finish is the most common way generated
    marketing output fails to be worth paying for."""
    bar = " ".join(LangflowService.QUALITY_BAR).lower()

    assert "placeholder" in bar
    assert "post as-is" in bar


def test_the_quality_bar_separates_knowledge_from_assumption():
    """A page audit padded with 'probably' looks thorough and asserts nothing."""
    bar = " ".join(LangflowService.QUALITY_BAR).lower()

    assert "assuming" in bar
    assert "probably" in bar


# ------------------------------------------------------- absence in the prompt


def _sections(**kwargs):
    from app.services.llm_service import LLMService

    return LLMService()._build_prompt_sections(
        message="Write a caption for my new mug.", task_type="caption", **kwargs
    )


def test_missing_context_is_stated_once_not_once_per_section():
    """An account with nothing connected used to receive six system sections
    reading "not available" plus a context_notice repeating the same absence --
    absence asserted seven times against two short lines of fact. Told mostly
    what it does not have, the model concludes it has nothing specific and
    writes a template."""
    names = [s["name"] for s in _sections(niche="ceramics studio, Tbilisi", goal="DM orders")]

    for absent in (
        "profile_context",
        "recent_posts_context",
        "link_context",
        "recent_content_context",
        "internal_strategy_context",
        "reel_context",
    ):
        assert absent not in names, f"{absent} was included with nothing in it"
    assert "context_notice" in names, "the single statement of what is missing must remain"


def test_context_that_exists_is_still_included():
    """The filter must key on presence, not remove the feature."""
    names = [s["name"] for s in _sections(profile_context={"username": "ceramics", "followers": 1200})]

    assert "profile_context" in names


def test_the_facts_the_caller_supplied_survive():
    joined = "\n".join(
        s["content"] for s in _sections(niche="ceramics studio, Tbilisi", goal="DM orders")
    )

    assert "ceramics studio, Tbilisi" in joined
    assert "DM orders" in joined


def test_playbook_precedence_rules_are_absent_without_a_playbook():
    """Half the priority rules arbitrated between the playbook and account data
    on every request, including the majority carrying no playbook at all."""
    joined = "\n".join(s["content"] for s in _sections(niche="ceramics"))

    assert "playbook" not in joined.lower()


def test_playbook_precedence_rules_appear_when_there_is_one():
    joined = "\n".join(
        s["content"]
        for s in _sections(
            playbook_context={"chunks": [{"chunk_label": "Hooks", "text": "open with tension"}]}
        )
    )

    assert "playbook" in joined.lower()


@pytest.mark.parametrize("task_type", sorted(RAG_TASK_TYPES))
def test_every_task_instruction_says_what_failure_looks_like(task_type):
    """A one-line instruction produces one-line thinking. Each task names the
    specific way its output goes wrong, because "be professional" yields generic
    professionalism."""
    instruction = LangflowService()._task_instruction(task_type)

    assert len(instruction) > 200, f"{task_type} instruction is too thin to steer anything"
