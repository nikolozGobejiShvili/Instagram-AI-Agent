"""Assembles the account signal and strategy material a generation needs.

This exists because the two entry points had drifted apart. `/agent/chat` loads
the customer's profile, their recent posts and their recent performance before
generating; `/api/v1/generation-jobs` — the asynchronous path the website
actually uses — loaded none of it. Carousels, content plans and audits produced
through the main endpoint therefore ran with no Instagram data at all, which
defeats the product's core claim: that the answer is about *this* account.

Nothing failed while that was true. The prompt builder defaults every context to
None and simply omits the section, so the generation succeeded and read
plausibly — it was just generic. That is why it survived a green test suite.

Both kinds of context are gathered here rather than at each call site so the
task-type gating exists once. Duplicating those sets across routes is how one
copy gains a task type and the other does not, and the symptom is a single
feature quietly answering without data.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Which tasks are improved by which signal. A caption benefits from knowing what
# the account posts; it does not need a month of performance history, and paying
# to assemble it would only crowd the prompt.
TASKS_NEEDING_RECENT_POSTS = {
    "reel_idea",
    "reel_script",
    "reel_feedback",
    "caption",
    "carousel",
    "profile_audit",
    "content_plan",
    "performance_summary",
}
TASKS_NEEDING_RECENT_PERFORMANCE = {
    "content_plan",
    "profile_audit",
    "performance_summary",
    "reel_feedback",
    "link_analysis",
}


class AccountContextService:
    def __init__(
        self,
        *,
        profile_context_service=None,
        recent_posts_context_service=None,
        recent_content_context_service=None,
        knowledge_retrieval_service=None,
    ):
        # Imported lazily and injectable so tests can drive this without the
        # services' own storage or network.
        if profile_context_service is None:
            from app.services.profile_context_service import ProfileContextService

            profile_context_service = ProfileContextService()
        if recent_posts_context_service is None:
            from app.services.recent_posts_context_service import RecentPostsContextService

            recent_posts_context_service = RecentPostsContextService()
        if recent_content_context_service is None:
            from app.services.recent_content_context_service import RecentContentContextService

            recent_content_context_service = RecentContentContextService()
        if knowledge_retrieval_service is None:
            from app.services.deterministic_knowledge_retrieval_service import (
                DeterministicKnowledgeRetrievalService,
            )

            knowledge_retrieval_service = DeterministicKnowledgeRetrievalService()

        self.profile_context_service = profile_context_service
        self.recent_posts_context_service = recent_posts_context_service
        self.recent_content_context_service = recent_content_context_service
        self.knowledge_retrieval_service = knowledge_retrieval_service

    def account_context(self, *, task_type: str, account_id: str | None) -> dict:
        """The customer's own Instagram signal, as run_agent keyword arguments.

        Each lookup is independent: a missing profile must not also cost the
        caller their recent posts. Every failure degrades to None and is logged,
        because generating with partial context is worth more to the customer
        than a 500, and the alternative — failing the job — turns a stale cache
        into a lost generation.
        """
        context: dict = {
            "profile_context": None,
            "recent_posts_context": None,
            "recent_content_context": None,
        }
        if not account_id:
            return context

        context["profile_context"] = self._safely(
            "profile", lambda: self.profile_context_service.get_context(account_id)
        )
        if task_type in TASKS_NEEDING_RECENT_POSTS:
            context["recent_posts_context"] = self._safely(
                "recent posts", lambda: self.recent_posts_context_service.get_context(account_id)
            )
        if task_type in TASKS_NEEDING_RECENT_PERFORMANCE:
            context["recent_content_context"] = self._safely(
                "recent performance", lambda: self.recent_content_context_service.get_context(account_id)
            )
        return context

    def strategy_context(self, *, task_type: str, message: str, goal: str | None, account_context: dict) -> dict | None:
        """Retrieved playbook material, shaped for the prompt builder.

        Returns None when nothing relevant was found. That is a real answer, not
        a failure: the retriever's whole design is that it can decline, and an
        empty block would read to the model as material that exists and is
        blank.
        """
        try:
            result = self.knowledge_retrieval_service.retrieve(
                task_type=task_type,
                message=message,
                goal=goal,
                profile_context=account_context.get("profile_context"),
                recent_posts_context=account_context.get("recent_posts_context"),
                recent_content_context=account_context.get("recent_content_context"),
            )
        except Exception as exc:  # noqa: BLE001 - retrieval must not fail a paid generation
            logger.warning("Knowledge retrieval failed (%s); generating without it", type(exc).__name__)
            return None

        if not result.used or not result.knowledge_context:
            return None

        # Reuses the existing playbook formatter rather than inventing a second
        # rendering of the same idea. One chunk because the retriever has already
        # ranked, filtered and joined what survived.
        return {
            "matched_knowledge_domain": "strategy",
            "chunks": [{"chunk_label": result.collection_name, "text": result.knowledge_context}],
            "retrieved_chunk_count": int(result.retrieved_count or 0),
        }

    def _safely(self, label: str, load):
        try:
            return load()
        except Exception as exc:  # noqa: BLE001 - one missing context is not a failed generation
            logger.warning("Could not load %s context: %s", label, type(exc).__name__)
            return None
