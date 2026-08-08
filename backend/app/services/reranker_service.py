"""Second-stage relevance scoring, so less material reaches the prompt.

Embedding search is good at recall and mediocre at precision: it finds the
neighbourhood of an answer, then orders it by a distance computed without ever
comparing the query and the passage directly. A reranker reads both together and
scores them, which is slower and far more accurate — so the useful shape is
recall wide, rerank, keep few.

That inversion is the point. The instinct when retrieval underperforms is to
raise `top_k` and hand the model more, but every extra passage competes with the
instructions for attention, and a rule that has to share the prompt with three
irrelevant paragraphs is a rule that gets skipped. Fewer, better passages beat
more passages, and this is how you get fewer without losing the right one.

It also makes *nothing* an available answer. Vector search returns the k nearest
neighbours whether or not any of them are relevant, so a question about carousels
retrieves the five least-unrelated reels chunks and injects them. A relevance
floor lets the store stay silent, which is the correct output when the material
does not cover the question.

Voyage, because the embeddings are already Voyage: same key, same vendor, and one
less account to hold.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "rerank-2.5"

# Scores are relative to the query, not calibrated across queries, so this is a
# tuned floor rather than a probability. 0.3 keeps passages that are plausibly
# on-topic and drops the ones vector search only surfaced because something had
# to be fifth. Raise it if the agent starts quoting material that half-fits.
DEFAULT_MIN_RELEVANCE = 0.3

# How many candidates the vector stage hands over. Wide, because recall is the
# job it is good at and the reranker is what makes a long list safe.
DEFAULT_CANDIDATE_POOL = 30


@dataclass(frozen=True)
class RankedPassage:
    index: int
    score: float


class RerankerService:
    def __init__(self, *, model: str | None = None, api_key: str | None = None, timeout: float | None = None):
        self.model = model or os.getenv("KNOWLEDGE_RERANK_MODEL", "").strip() or DEFAULT_MODEL
        self._api_key = api_key if api_key is not None else os.getenv("VOYAGE_API_KEY", "").strip()
        self.base_url = (os.getenv("VOYAGE_BASE_URL", "").strip() or "https://api.voyageai.com/v1").rstrip("/")
        self.timeout = float(timeout or os.getenv("RERANK_TIMEOUT_SECONDS", "20") or 20)
        self.min_relevance = float(
            os.getenv("KNOWLEDGE_MIN_RELEVANCE", "").strip() or DEFAULT_MIN_RELEVANCE
        )

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def rank(self, *, query: str, documents: list[str], top_k: int) -> list[RankedPassage] | None:
        """Score ``documents`` against ``query`` and keep those above the floor.

        Returns ``None`` — not an empty list — when reranking could not run, so
        the caller can tell "nothing was relevant" from "this stage was skipped"
        and fall back to vector order instead of silently dropping every result.
        """
        if not documents or not self._api_key:
            return None

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "query": query,
                        "documents": documents,
                        "model": self.model,
                        "top_k": max(1, int(top_k)),
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            # Degrades rather than fails: retrieval without reranking is worse,
            # but a generation the customer paid for should not be lost to a
            # ranking call.
            logger.warning("Rerank unavailable (%s); falling back to vector order", type(exc).__name__)
            return None

        entries = payload.get("data")
        if not isinstance(entries, list):
            logger.warning("Rerank returned no data; falling back to vector order")
            return None

        ranked = [
            RankedPassage(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in entries
            if isinstance(item, dict) and "index" in item and "relevance_score" in item
        ]
        kept = [p for p in ranked if p.score >= self.min_relevance]

        if ranked and not kept:
            # Worth a line in the log: it means the customer asked about
            # something the uploaded material does not cover, which is useful to
            # know and invisible otherwise.
            logger.info(
                "No passage cleared the relevance floor (best=%.3f, floor=%.2f) — injecting nothing",
                max(p.score for p in ranked),
                self.min_relevance,
            )
        return kept
