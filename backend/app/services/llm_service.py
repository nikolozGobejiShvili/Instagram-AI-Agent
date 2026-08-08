import json
import logging
import math
import os
import random
import re
import time
from typing import Any

import httpx

from app.services.langflow_service import LangflowService


logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    def __init__(
        self,
        safe_message: str,
        *,
        code: str = "generation_failed",
        status_code: int = 502,
        model_provider: str | None = None,
        model_name: str | None = None,
        prompt_token_estimate: int | None = None,
        retry_count: int = 0,
        rate_limited: bool = False,
        prompt_section_names: list[str] | None = None,
    ):
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.code = code
        self.status_code = status_code
        self.model_provider = model_provider
        self.model_name = model_name
        self.prompt_token_estimate = prompt_token_estimate
        self.retry_count = retry_count
        self.rate_limited = rate_limited
        self.prompt_section_names = list(prompt_section_names or [])


class LLMService:
    REELS_TASK_TYPES = {"reel_idea", "reel_script", "reel_feedback"}
    DEFAULT_PRIMARY_MODELS = [
        "gpt-5.2",
        "gpt-4.1-mini",
        "gpt-4o-mini",
    ]
    MAX_RETRIES = 3
    BACKOFF_BASE_SECONDS = 1.0
    BACKOFF_MAX_SECONDS = 8.0
    MAX_SECTION_CHARS = 2200
    MAX_PROFILE_FIELD_CHARS = 220
    MAX_LINK_FIELD_CHARS = 240
    MAX_REEL_FIELD_CHARS = 260
    MAX_PLAYBOOK_CHARS_PER_CHUNK = 680
    MAX_TOTAL_PLAYBOOK_CHARS = 2800
    MAX_RECENT_POST_CAPTION_SUMMARY_CHARS = 150
    MAX_RECENT_POSTS_BY_TASK = {
        "reel_idea": 5,
        "reel_script": 4,
        "reel_feedback": 4,
    }
    MAX_PLAYBOOK_CHUNKS_BY_TASK = {
        "reel_idea": 5,
        "reel_script": 4,
        "reel_feedback": 4,
    }
    # Sized for the widest answer each task can legitimately produce, not the
    # typical one. These were tuned when every task returned prose, where running
    # out of room costs the last paragraph. A schema-enforced task returns JSON,
    # where running out of room costs *everything*: the document is unparseable
    # and the whole generation is lost. content_plan proved it -- 1400 tokens cut
    # a 30-item plan mid-document, and the customer got a fragment of raw JSON.
    MAX_OUTPUT_TOKENS_BY_TASK = {
        "chat": 700,
        "reel_idea": 2200,
        "reel_script": 2200,
        "reel_feedback": 2200,
        "caption": 1200,
        # up to MAX_CAROUSEL_SLIDES slides, each with headline, body and art
        # direction
        "carousel": 3200,
        "profile_audit": 2200,
        # 30 dated items plus mix, hooks, CTAs and a summary
        "content_plan": 4500,
        "link_analysis": 1600,
        "performance_summary": 2000,
    }

    def __init__(self, prompt_builder: LangflowService | None = None):
        self.prompt_builder = prompt_builder or LangflowService()
        self.primary_provider = (
            os.getenv("PRIMARY_LLM_PROVIDER", "").strip()
            or os.getenv("LLM_PROVIDER", "").strip()
            or "openai"
        ).lower()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "90").strip() or "90")
        self.fallback_provider = (os.getenv("FALLBACK_LLM_PROVIDER", "").strip() or "").lower() or None
        self._rng = random.Random()

    def provider_name(self) -> str:
        return self.primary_provider

    def preferred_model_candidates(self) -> list[str]:
        configured_model = (
            os.getenv("PRIMARY_LLM_MODEL", "").strip()
            or os.getenv("LLM_MODEL_NAME", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
        )
        candidates = [configured_model] if configured_model else []
        candidates.extend(self.DEFAULT_PRIMARY_MODELS)

        deduped_candidates = []
        for candidate in candidates:
            normalized_candidate = " ".join(candidate.split()).strip()
            if normalized_candidate and normalized_candidate not in deduped_candidates:
                deduped_candidates.append(normalized_candidate)
        return deduped_candidates

    def fallback_model_candidates(self) -> list[str]:
        fallback_model = os.getenv("FALLBACK_LLM_MODEL", "").strip()
        return [fallback_model] if fallback_model else []

    # Tasks whose shape is enforced by the provider rather than requested in
    # prose. Derived from the Pydantic models that already validate the
    # response: a hand-written copy that drifted would produce output the
    # provider accepts and validation then rejects, leaving structured_output
    # silently None -- exactly the failure this path exists to prevent.
    # The prose the customer reads, kept separate from the data so neither has
    # to be extracted from the other. The description states how the string is
    # rendered rather than listing forbidden syntax: told only "string", the
    # model reached for markdown headings and bold, which the customer then sees
    # as literal asterisks. Describing the surface is what makes the right shape
    # obvious.
    _REPLY_PROPERTY = {
        "type": "string",
        "description": (
            "One or two sentences of plain conversational prose introducing the result. "
            "Displayed as-is in a chat surface that does not render markdown, so any "
            "'#', '*' or table syntax reaches the customer as literal characters. "
            "The findings themselves belong in structured_output; do not repeat them here."
        ),
    }

    _MODEL_BACKED_SCHEMAS = {
        "caption": "CaptionStructuredOutput",
        "profile_audit": "ProfileAuditStructuredOutput",
        "content_plan": "ContentPlanStructuredOutput",
        "performance_summary": "PerformanceSummaryStructuredOutput",
        "link_analysis": "LinkAnalysisStructuredOutput",
        "public_profile_analysis": "PublicProfileAnalysisStructuredOutput",
    }

    @staticmethod
    def _tighten_schema(node: Any, defs: dict[str, Any]) -> Any:
        """Rewrite a Pydantic JSON Schema into a provider-enforceable one.

        Three transforms, each load-bearing:

        * ``$ref`` is resolved inline. Whether a provider follows ``$defs`` is
          not worth discovering in production.
        * ``T | None`` collapses to ``T``. The generation schema is deliberately
          *narrower* than the validating model -- narrowing is always safe
          because anything produced still validates, whereas widening is what
          lets an unvalidatable response through. It also stops the model
          answering ``null`` for a field the customer is paying to have filled.
        * Every property becomes required with ``additionalProperties: false``,
          so a partially-filled object is not a valid answer.
        """
        if isinstance(node, list):
            return [LLMService._tighten_schema(item, defs) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return LLMService._tighten_schema(defs[ref_name], defs)

        if "anyOf" in node:
            variants = [v for v in node["anyOf"] if v.get("type") != "null"]
            if len(variants) == 1:
                return LLMService._tighten_schema(variants[0], defs)
            node = {**node, "anyOf": [LLMService._tighten_schema(v, defs) for v in variants]}

        # `title` and `default` are Pydantic bookkeeping. `default` especially:
        # it tells a model a field may be omitted, which is the opposite of what
        # an enforced schema is for.
        tightened = {
            key: LLMService._tighten_schema(value, defs)
            for key, value in node.items()
            if key not in {"title", "default", "$defs"}
        }

        if tightened.get("type") == "object" and "properties" in tightened:
            tightened["required"] = list(tightened["properties"])
            tightened["additionalProperties"] = False

        return tightened

    @classmethod
    def _schema_from_model(cls, model_name: str) -> dict[str, Any]:
        from app.schemas import agent as agent_schemas

        model = getattr(agent_schemas, model_name)
        raw = model.model_json_schema()
        return cls._tighten_schema(raw, raw.get("$defs", {}))

    def _structured_schema(self, task_type: str) -> dict[str, Any] | None:
        model_name = self._MODEL_BACKED_SCHEMAS.get(task_type)
        if model_name:
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["reply", "structured_output"],
                "properties": {
                    "reply": self._REPLY_PROPERTY,
                    "structured_output": self._schema_from_model(model_name),
                },
            }

        schemas: dict[str, dict[str, Any]] = {
            # Carousel was the one structured task with no schema, so its shape
            # depended on the model following heading instructions ("Title:",
            # "Slide 1:", ...). Sonnet 4.6 does not -- it answers in markdown, the
            # heading parser finds nothing, and the job dies with "did not produce
            # any slides". Enforcing the schema removes the obedience question
            # entirely, and is also what keeps markdown out of the copy.
            "carousel": {
                "type": "object",
                "additionalProperties": False,
                "required": ["reply", "structured_output"],
                "properties": {
                    "reply": self._REPLY_PROPERTY,
                    "structured_output": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "slides", "cta"],
                        "properties": {
                            "title": {"type": "string"},
                            "slides": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["slide_number", "headline", "body", "image_prompt"],
                                    "properties": {
                                        "slide_number": {"type": "integer"},
                                        "headline": {"type": "string"},
                                        "body": {"type": "string"},
                                        # Art direction for the background only.
                                        # Slide copy is composited in code, so a
                                        # prompt asking for text would produce
                                        # letters behind the real letters.
                                        "image_prompt": {"type": "string"},
                                    },
                                },
                            },
                            "cta": {"type": "string"},
                        },
                    },
                },
            },
            "reel_idea": {
                "type": "object",
                "additionalProperties": False,
                "required": ["reply", "structured_output"],
                "properties": {
                    "reply": {"type": "string"},
                    "structured_output": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ideas"],
                        "properties": {
                            "ideas": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "title",
                                        "hook",
                                        "format_type",
                                        "main_idea",
                                        "shot_list",
                                        "why_it_can_work",
                                        "cta",
                                    ],
                                    "properties": {
                                        "title": {"type": "string"},
                                        "hook": {"type": "string"},
                                        "format_type": {"type": "string"},
                                        "main_idea": {"type": "string"},
                                        "shot_list": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "why_it_can_work": {"type": "string"},
                                        "cta": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "reel_script": {
                "type": "object",
                "additionalProperties": False,
                "required": ["reply", "structured_output"],
                "properties": {
                    "reply": {"type": "string"},
                    "structured_output": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "hook", "script_sections", "cta", "full_script"],
                        "properties": {
                            "title": {"type": "string"},
                            "hook": {"type": "string"},
                            "script_sections": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "cta": {"type": "string"},
                            "full_script": {"type": "string"},
                        },
                    },
                },
            },
            "reel_feedback": {
                "type": "object",
                "additionalProperties": False,
                "required": ["reply", "structured_output"],
                "properties": {
                    "reply": {"type": "string"},
                    "structured_output": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "what_works",
                            "what_hurts",
                            "retention_risks",
                            "better_hook",
                            "improved_structure",
                            "better_cta",
                            "improved_version",
                            "summary",
                        ],
                        "properties": {
                            "what_works": {"type": "array", "items": {"type": "string"}},
                            "what_hurts": {"type": "array", "items": {"type": "string"}},
                            "retention_risks": {"type": "array", "items": {"type": "string"}},
                            "better_hook": {"type": "string"},
                            "improved_structure": {"type": "array", "items": {"type": "string"}},
                            "better_cta": {"type": "string"},
                            "improved_version": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                    },
                },
            },
        }
        return schemas.get(task_type)

    def _direct_structured_output_instruction(self, task_type: str) -> str | None:
        if task_type not in self.REELS_TASK_TYPES:
            return None

        return "\n".join([
            "Return data that matches the provider-enforced structured schema exactly.",
            "The reply field must contain the human-readable final answer for the user.",
            "The structured_output field must contain only clean structured data with no extra headings, no next-idea leakage, and no raw internal strategy text.",
        ])

    def _safe_truncate(self, value: str | None, max_chars: int) -> str | None:
        normalized_value = " ".join((value or "").split()).strip()
        if not normalized_value:
            return None
        if len(normalized_value) <= max_chars:
            return normalized_value
        return f"{normalized_value[: max_chars - 3].rstrip()}..."

    def _normalize_string_list(self, values: Any, *, limit: int = 5, item_max_chars: int = 120) -> list[str]:
        if not isinstance(values, list):
            return []

        normalized_values = []
        for value in values:
            normalized_value = self._safe_truncate(str(value or ""), item_max_chars)
            if not normalized_value:
                continue
            normalized_values.append(normalized_value)
            if len(normalized_values) >= limit:
                break
        return normalized_values

    def _summarize_caption_for_context(self, caption: str | None, *, topic: str | None, max_chars: int) -> str | None:
        normalized_caption = " ".join((caption or "").split()).strip()
        if not normalized_caption:
            return None

        words = re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", normalized_caption.lower())
        unique_ratio = (len(set(words)) / len(words)) if words else 1.0
        if len(words) >= 18 and unique_ratio < 0.55:
            if topic:
                return self._safe_truncate(f"Caption about {topic} with repeated phrasing omitted for compact context.", max_chars)
            return self._safe_truncate("Long caption with repeated phrasing omitted for compact context.", max_chars)

        first_sentence = re.split(r"(?<=[.!?])\s+", normalized_caption, maxsplit=1)[0].strip()
        preferred_summary = first_sentence or normalized_caption
        return self._safe_truncate(preferred_summary, max_chars)

    def _compact_profile_context(self, profile_context: dict | None) -> dict | None:
        if not profile_context:
            return None

        return {
            "brand_name": self._safe_truncate(profile_context.get("brand_name"), self.MAX_PROFILE_FIELD_CHARS),
            "niche": self._safe_truncate(profile_context.get("niche"), self.MAX_PROFILE_FIELD_CHARS),
            "target_audience": self._safe_truncate(profile_context.get("target_audience"), self.MAX_PROFILE_FIELD_CHARS),
            "brand_voice": self._safe_truncate(profile_context.get("brand_voice"), self.MAX_PROFILE_FIELD_CHARS),
            "bio": self._safe_truncate(profile_context.get("bio"), self.MAX_PROFILE_FIELD_CHARS),
            "content_focus": self._normalize_string_list(profile_context.get("content_focus"), limit=5),
            "strengths": self._normalize_string_list(profile_context.get("strengths"), limit=4),
            "weak_points": self._normalize_string_list(profile_context.get("weak_points"), limit=4),
        }

    def _compact_recent_content_context(self, recent_content_context: dict | None) -> dict | None:
        if not recent_content_context:
            return None

        return {
            "top_formats": self._normalize_string_list(recent_content_context.get("top_formats"), limit=4),
            "best_topics": self._normalize_string_list(recent_content_context.get("best_topics"), limit=4),
            "weak_topics": self._normalize_string_list(recent_content_context.get("weak_topics"), limit=4),
            "best_ctas": self._normalize_string_list(recent_content_context.get("best_ctas"), limit=4),
            "weak_ctas": self._normalize_string_list(recent_content_context.get("weak_ctas"), limit=4),
            "notes": self._normalize_string_list(recent_content_context.get("notes"), limit=4, item_max_chars=160),
        }

    def _tokenize_for_matching(self, value: str | None) -> list[str]:
        return re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (value or "").lower())

    def _relevance_reason(self, *, task_type: str, content_type: str, topic: str | None, caption_summary: str | None) -> str:
        normalized_content_type = content_type.upper()
        if task_type in self.REELS_TASK_TYPES and "REEL" in normalized_content_type:
            return "Recent Reel format with engagement signals relevant to hook and CTA decisions."
        if topic:
            return f"Topic overlap around {topic}."
        if caption_summary:
            return "Recent post summary relevant to current generation direction."
        return "Recent post sample for current account context."

    def _compact_recent_posts_context(
        self,
        *,
        task_type: str,
        recent_posts_context: dict | None,
        message: str,
        goal: str | None,
    ) -> dict | None:
        if not recent_posts_context:
            return None

        posts = recent_posts_context.get("posts")
        if not isinstance(posts, list) or not posts:
            return {"posts": []}

        max_posts = self.MAX_RECENT_POSTS_BY_TASK.get(task_type, 5)
        query_tokens = set(self._tokenize_for_matching(f"{message}\n{goal or ''}"))
        scored_posts = []

        for index, post in enumerate(posts):
            if not isinstance(post, dict):
                continue

            content_type = self._safe_truncate(post.get("content_type"), 32) or "unknown"
            topic = self._safe_truncate(post.get("topic"), 100)
            caption_summary = self._summarize_caption_for_context(
                post.get("caption"),
                topic=topic,
                max_chars=self.MAX_RECENT_POST_CAPTION_SUMMARY_CHARS,
            )
            views = int(post.get("views") or 0)
            likes = int(post.get("likes") or 0)
            comments = int(post.get("comments") or 0)
            saves = int(post.get("saves") or 0)

            text_blob = " ".join(filter(None, [topic, caption_summary]))
            overlap_score = sum(1 for token in self._tokenize_for_matching(text_blob) if token in query_tokens)
            reels_bonus = 2 if task_type in self.REELS_TASK_TYPES and "REEL" in content_type.upper() else 0
            engagement_score = min(3, (1 if views > 0 else 0) + (1 if likes > 0 else 0) + (1 if comments > 0 or saves > 0 else 0))
            total_score = overlap_score + reels_bonus + engagement_score

            compact_post = {
                "post_id": self._safe_truncate(post.get("post_id"), 64),
                "content_type": content_type,
                "topic": topic,
                "caption_summary": caption_summary,
                "views": views,
                "likes": likes,
                "comments": comments,
                "saves": saves,
                "reason_relevant": self._relevance_reason(
                    task_type=task_type,
                    content_type=content_type,
                    topic=topic,
                    caption_summary=caption_summary,
                ),
                "_original_index": index,
                "_score": total_score,
            }
            scored_posts.append(compact_post)

        scored_posts.sort(key=lambda item: (item["_score"], -item["_original_index"]), reverse=True)
        selected_posts = scored_posts[:max_posts]
        selected_posts.sort(key=lambda item: item["_original_index"])

        return {
            "posts": [
                {
                    "post_id": post.get("post_id"),
                    "content_type": post.get("content_type"),
                    "topic": post.get("topic"),
                    "caption_summary": post.get("caption_summary"),
                    "views": post.get("views"),
                    "likes": post.get("likes"),
                    "comments": post.get("comments"),
                    "saves": post.get("saves"),
                    "reason_relevant": post.get("reason_relevant"),
                }
                for post in selected_posts
            ],
            "selection_summary": f"Selected {len(selected_posts)} recent posts for compact generation context.",
        }

    def _compact_link_context(self, link_context: dict | None) -> dict | None:
        if not link_context:
            return None

        return {
            "link": self._safe_truncate(link_context.get("link"), self.MAX_LINK_FIELD_CHARS),
            "detected_platform": self._safe_truncate(link_context.get("detected_platform"), 32),
            "source_type": self._safe_truncate(link_context.get("source_type"), 80),
            "creator_handle": self._safe_truncate(link_context.get("creator_handle"), 80),
            "summary": self._safe_truncate(link_context.get("summary"), self.MAX_LINK_FIELD_CHARS),
            "content_type": self._safe_truncate(link_context.get("content_type"), 64),
            "hook_style": self._safe_truncate(link_context.get("hook_style"), 120),
            "business_goal": self._safe_truncate(link_context.get("business_goal"), 120),
            "source_patterns": self._normalize_string_list(link_context.get("source_patterns"), limit=4, item_max_chars=120),
            "probable_strengths": self._normalize_string_list(link_context.get("probable_strengths"), limit=4, item_max_chars=120),
            "adaptation_notes": self._normalize_string_list(link_context.get("adaptation_notes"), limit=4, item_max_chars=120),
        }

    def _compact_reel_context(self, reel_context: dict | None) -> dict | None:
        if not reel_context:
            return None

        return {
            "source": self._safe_truncate(reel_context.get("source"), 40),
            "media_id": self._safe_truncate(reel_context.get("media_id"), 80),
            "permalink": self._safe_truncate(reel_context.get("permalink"), self.MAX_REEL_FIELD_CHARS),
            "media_type": self._safe_truncate(reel_context.get("media_type"), 32),
            "caption": self._safe_truncate(reel_context.get("caption"), self.MAX_REEL_FIELD_CHARS),
            "timestamp": self._safe_truncate(reel_context.get("timestamp"), 64),
            "like_count": reel_context.get("like_count"),
            "comments_count": reel_context.get("comments_count"),
            "analysis_brief": self._safe_truncate(reel_context.get("analysis_brief"), self.MAX_REEL_FIELD_CHARS),
        }

    def _compact_playbook_context(self, *, task_type: str, playbook_context: dict | None) -> dict | None:
        if not playbook_context:
            return None

        chunks = playbook_context.get("chunks")
        if not isinstance(chunks, list):
            return {
                "used_system_knowledge": False,
                "matched_knowledge_domain": playbook_context.get("matched_knowledge_domain"),
                "matched_knowledge_pack_ids": list(playbook_context.get("matched_knowledge_pack_ids") or []),
                "retrieved_chunk_count": 0,
                "retrieved_chunk_titles": [],
                "chunks": [],
            }

        max_chunks = self.MAX_PLAYBOOK_CHUNKS_BY_TASK.get(task_type, 4)
        remaining_chars = self.MAX_TOTAL_PLAYBOOK_CHARS
        compacted_chunks = []

        for chunk in chunks[:max_chunks]:
            if not isinstance(chunk, dict):
                continue
            allowed_chars = min(self.MAX_PLAYBOOK_CHARS_PER_CHUNK, remaining_chars)
            if allowed_chars < 120:
                break
            compact_text = self._safe_truncate(chunk.get("text"), allowed_chars)
            if not compact_text:
                continue
            compacted_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "knowledge_pack_id": chunk.get("knowledge_pack_id"),
                "file_name": chunk.get("file_name"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_label": self._safe_truncate(chunk.get("chunk_label"), 120) or self._safe_truncate(chunk.get("file_name"), 120),
                "text": compact_text,
            })
            remaining_chars -= len(compact_text)

        retrieved_chunk_titles = [
            str(chunk.get("chunk_label") or chunk.get("file_name") or f"Chunk {index}")
            for index, chunk in enumerate(compacted_chunks, start=1)
        ]
        return {
            "used_system_knowledge": bool(compacted_chunks),
            "matched_knowledge_domain": playbook_context.get("matched_knowledge_domain") if compacted_chunks else None,
            "matched_knowledge_pack_ids": list(playbook_context.get("matched_knowledge_pack_ids") or []),
            "retrieved_chunk_count": len(compacted_chunks),
            "retrieved_chunk_titles": retrieved_chunk_titles,
            "chunks": compacted_chunks,
        }

    def _format_recent_posts_context(self, recent_posts_context: dict | None) -> str:
        if not recent_posts_context:
            return "Recent posts context: not available"

        posts = recent_posts_context.get("posts", [])
        if not posts:
            return "Recent posts context: no posts available"

        formatted_posts = []
        for index, post in enumerate(posts, start=1):
            formatted_posts.append(
                (
                    f"Post {index}: "
                    f"type={post.get('content_type', '')}, "
                    f"topic={post.get('topic', '')}, "
                    f"caption_summary={post.get('caption_summary', '')}, "
                    f"views={post.get('views', 0)}, "
                    f"likes={post.get('likes', 0)}, "
                    f"comments={post.get('comments', 0)}, "
                    f"saves={post.get('saves', 0)}, "
                    f"reason_relevant={post.get('reason_relevant', '')}"
                )
            )

        selection_summary = self._safe_truncate(recent_posts_context.get("selection_summary"), 140)
        lines = ["Recent posts context:"]
        if selection_summary:
            lines.append(selection_summary)
        lines.extend(formatted_posts)
        return "\n".join(lines)

    def _estimate_tokens_for_text(self, value: str | None) -> int:
        normalized_value = str(value or "")
        if not normalized_value.strip():
            return 0
        return max(1, math.ceil(len(normalized_value) / 4))

    def _estimate_prompt_tokens(self, response_input: list[dict[str, Any]]) -> int:
        return sum(
            self._estimate_tokens_for_text(content_item.get("text"))
            for message in response_input
            for content_item in message.get("content", [])
            if isinstance(content_item, dict)
        )

    def _max_output_tokens(self, task_type: str) -> int:
        return self.MAX_OUTPUT_TOKENS_BY_TASK.get(task_type, 1000)

    def _format_marketing_brief(self, brief_context: dict | None) -> str:
        """Render the customer's brief as a prompt section.

        Returns an empty string when there is no brief, which drops the section
        entirely: a block of empty labels reads to the model as "these are
        unknown" and invites it to invent them.
        """
        if not brief_context:
            return ""

        lines = [
            "This customer's marketing brief. Treat it as the goal every answer serves.",
            "Do not restate it back to the user; use it to decide what to produce.",
        ]
        for label, value in brief_context.items():
            rendered = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            if rendered.strip():
                lines.append(f"- {label}: {rendered}")
        return "\n".join(lines)

    def _format_public_profile_context(self, context: dict | None) -> str:
        """Render the reference account for the model.

        Captions are listed with their engagement so the model can see which
        openings earned attention rather than being told which did. The
        metrics warning rides in the same block as the numbers -- separated, it
        reads as general advice instead of a constraint on this data.
        """
        if not context:
            return ""

        lines = [
            f"Reference account to learn from: @{context.get('handle')}"
            + (f" ({context.get('name')})" if context.get("name") else ""),
            "This is real data from Meta for an account the customer does NOT own.",
        ]
        for label, key in (
            ("Bio", "biography"),
            ("Followers", "followers_count"),
            ("Total posts", "media_count"),
            ("Median likes on recent posts", "median_likes"),
        ):
            if context.get(key) is not None:
                lines.append(f"- {label}: {context[key]}")

        mix = context.get("content_mix") or {}
        if mix:
            lines.append("- Recent content mix: " + ", ".join(f"{k} x{v}" for k, v in mix.items()))

        captions = context.get("recent_captions") or []
        if captions:
            lines.append("Recent posts (caption, format, engagement):")
            for item in captions:
                lines.append(
                    f'  - [{item.get("type")}] {item.get("likes")} likes / {item.get("comments")} comments: '
                    f'"{item.get("caption")}"'
                )

        if context.get("metrics_note"):
            lines.append(context["metrics_note"])
        return "\n".join(lines)

    def _build_prompt_sections(
        self,
        *,
        message: str,
        task_type: str,
        niche: str | None = None,
        target_audience: str | None = None,
        goal: str | None = None,
        link: str | None = None,
        profile_context: dict | None = None,
        link_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
        playbook_context: dict | None = None,
        reel_context: dict | None = None,
        brief_context: dict | None = None,
        public_profile_context: dict | None = None,
        slide_count: int | None = None,
    ) -> list[dict[str, str]]:
        reels_high_priority_instruction = self.prompt_builder._reels_high_priority_instruction(task_type, playbook_context)
        direct_structured_output_instruction = self._direct_structured_output_instruction(task_type)

        sections: list[dict[str, str]] = [
            {"name": "base_system_instruction", "role": "system", "content": self.prompt_builder._base_system_instruction()},
            {"name": "task_instruction", "role": "system", "content": f"Task type: {task_type}\nTask objective: {self.prompt_builder._task_instruction(task_type)}"},
        ]

        if reels_high_priority_instruction:
            sections.append({
                "name": "reels_high_priority_instruction",
                "role": "system",
                "content": reels_high_priority_instruction,
            })

        sections.append({
            "name": "required_output_format",
            "role": "system",
            "content": self.prompt_builder._output_format_instruction(task_type, slide_count),
        })

        if direct_structured_output_instruction:
            sections.append({
                "name": "reels_structured_output_contract",
                "role": "system",
                "content": direct_structured_output_instruction,
            })

        sections.extend([
            {
                "name": "priority_rules",
                "role": "system",
                "content": self.prompt_builder._priority_rules(
                    niche=niche,
                    target_audience=target_audience,
                    goal=goal,
                    has_playbook=bool(playbook_context),
                ),
            },
            {
                "name": "manual_business_brief",
                "role": "system",
                "content": self.prompt_builder._manual_business_brief(
                    niche=niche,
                    target_audience=target_audience,
                    goal=goal,
                ),
            },
            {
                # Placed ahead of the account context so the customer's stated
                # objective frames everything the model reads afterwards, rather
                # than arriving as an afterthought below the profile data.
                "name": "marketing_brief",
                "role": "system",
                "content": self._format_marketing_brief(brief_context),
            },
            {
                "name": "context_notice",
                "role": "system",
                "content": self.prompt_builder._context_notice(
                    profile_context=profile_context,
                    recent_content_context=recent_content_context,
                    recent_posts_context=recent_posts_context,
                    link_context=link_context,
                ),
            },
        ])

        # Context sections are appended only when the context exists.
        #
        # They used to be unconditional, so an account with nothing connected
        # received seven system sections reading "not available" plus a
        # context_notice repeating the same absence in prose -- absence stated
        # eight times against two short lines of actual fact. A live caption
        # request with niche and goal both supplied came back generic and
        # ignored them both, which is the behaviour that prompt invites: told
        # mostly what it does not have, the model concludes it has nothing
        # specific and writes a template.
        #
        # context_notice still states what is missing, once, in one place. The
        # condition is on the context object rather than on the formatted string
        # so this cannot drift when an "unavailable" wording changes.
        optional_context = [
            ("public_profile_context", public_profile_context, self._format_public_profile_context),
            ("profile_context", profile_context, self.prompt_builder._format_profile_context),
            ("recent_posts_context", recent_posts_context, self._format_recent_posts_context),
            ("link_context", link_context, self.prompt_builder._format_link_context),
            ("recent_content_context", recent_content_context, self.prompt_builder._format_recent_content_context),
            ("internal_strategy_context", playbook_context, self.prompt_builder._format_playbook_context),
            ("reel_context", reel_context, self.prompt_builder._format_reel_context),
        ]
        for name, context, formatter in optional_context:
            if not context:
                continue
            sections.append({"name": name, "role": "system", "content": formatter(context)})

        request_overrides = []
        if niche:
            request_overrides.append(f"Requested niche override: {niche}")
        if target_audience:
            request_overrides.append(f"Requested target audience override: {target_audience}")
        if goal:
            request_overrides.append(f"Goal: {goal}")
        if link:
            request_overrides.append(f"Original link: {link}")
        if request_overrides:
            sections.append({
                "name": "request_overrides",
                "role": "system",
                "content": "\n".join(request_overrides),
            })

        sections.append({
            "name": "execution_note",
            "role": "system",
            "content": "Execution note: keep the answer concise, actionable, and ready to use.",
        })
        sections.append({
            "name": "user_request",
            "role": "user",
            "content": message,
        })
        return sections

    def _sections_to_response_input(self, sections: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
        response_input = []
        section_names = []
        for section in sections:
            content = str(section.get("content") or "").strip()
            if not content:
                continue
            if len(content) > self.MAX_SECTION_CHARS:
                content = f"{content[: self.MAX_SECTION_CHARS - 3].rstrip()}..."
            section_names.append(str(section.get("name")))
            response_input.append({
                "type": "message",
                "role": section["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": content,
                    }
                ],
            })
        return response_input, section_names

    def _extract_output_text(self, response_payload: dict[str, Any]) -> str:
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        collected_parts: list[str] = []
        for item in response_payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []):
                if content_item.get("type") in {"output_text", "text"}:
                    text_value = content_item.get("text")
                    if isinstance(text_value, str) and text_value:
                        collected_parts.append(text_value)
                elif content_item.get("type") == "refusal":
                    refusal_text = content_item.get("refusal")
                    if isinstance(refusal_text, str) and refusal_text:
                        collected_parts.append(refusal_text)

        return "\n".join(part for part in collected_parts if part).strip()

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _jitter(self) -> float:
        return self._rng.uniform(0.0, 0.35)

    def _parse_retry_after_seconds(self, response: httpx.Response | None) -> float | None:
        if response is None:
            return None
        retry_after = response.headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            parsed_seconds = float(retry_after)
        except ValueError:
            return None
        return max(0.0, parsed_seconds)

    def _build_backoff_seconds(self, attempt_number: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return retry_after_seconds
        exponential_delay = min(
            self.BACKOFF_MAX_SECONDS,
            self.BACKOFF_BASE_SECONDS * (2 ** max(attempt_number - 1, 0)),
        )
        return exponential_delay + self._jitter()

    def _extract_openai_error_metadata(self, response: httpx.Response | None) -> dict[str, str | int | None]:
        if response is None:
            return {
                "status_code": None,
                "error_code": None,
                "error_type": None,
                "error_message": None,
            }

        response_payload: dict[str, Any] = {}
        try:
            response_payload = response.json()
        except Exception:
            response_payload = {}

        error_payload = response_payload.get("error", {}) if isinstance(response_payload, dict) else {}
        return {
            "status_code": response.status_code,
            "error_code": error_payload.get("code"),
            "error_type": error_payload.get("type"),
            "error_message": self._safe_truncate(str(error_payload.get("message") or ""), 140),
        }

    def _send_openai_request(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(
                url,
                json=payload,
                headers=headers,
            )

    def _call_openai_model(
        self,
        *,
        task_type: str,
        response_input: list[dict[str, Any]],
        model_name: str,
        max_retries: int,
        prompt_token_estimate: int,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise LLMServiceError(
                "AI generation is not configured yet.",
                code="generation_failed",
                status_code=502,
                model_provider="openai",
                model_name=model_name,
                prompt_token_estimate=prompt_token_estimate,
            )

        schema = self._structured_schema(task_type)
        response_url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retry_count = 0

        for attempt_number in range(1, max_retries + 2):
            payload: dict[str, Any] = {
                "model": model_name,
                "input": response_input,
                "max_output_tokens": self._max_output_tokens(task_type),
            }
            if schema:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": f"{task_type}_response",
                        "strict": True,
                        "schema": schema,
                    }
                }

            try:
                response = self._send_openai_request(
                    url=response_url,
                    payload=payload,
                    headers=headers,
                )
                response.raise_for_status()
                response_payload = response.json()
            except httpx.HTTPStatusError as exc:
                response = exc.response
                if response.status_code == 429:
                    error_meta = self._extract_openai_error_metadata(response)
                    logger.warning(
                        "Direct LLM rate limited provider=openai model=%s task_type=%s attempt=%s retry_count=%s error_code=%s error_type=%s",
                        model_name,
                        task_type,
                        attempt_number,
                        retry_count,
                        error_meta.get("error_code"),
                        error_meta.get("error_type"),
                    )
                    if attempt_number <= max_retries:
                        retry_count += 1
                        backoff_seconds = self._build_backoff_seconds(
                            attempt_number,
                            self._parse_retry_after_seconds(response),
                        )
                        self._sleep(backoff_seconds)
                        continue
                    raise LLMServiceError(
                        "AI generation is temporarily rate limited. Please try again shortly.",
                        code="llm_rate_limited",
                        status_code=503,
                        model_provider="openai",
                        model_name=model_name,
                        prompt_token_estimate=prompt_token_estimate,
                        retry_count=retry_count,
                        rate_limited=True,
                    ) from exc

                error_meta = self._extract_openai_error_metadata(response)
                logger.warning(
                    "Direct LLM upstream failure provider=openai model=%s task_type=%s status=%s error_code=%s error_type=%s",
                    model_name,
                    task_type,
                    response.status_code,
                    error_meta.get("error_code"),
                    error_meta.get("error_type"),
                )
                raise LLMServiceError(
                    "AI generation failed. Please try again shortly.",
                    code="generation_failed",
                    status_code=502,
                    model_provider="openai",
                    model_name=model_name,
                    prompt_token_estimate=prompt_token_estimate,
                    retry_count=retry_count,
                    rate_limited=False,
                ) from exc
            except httpx.HTTPError as exc:
                logger.warning(
                    "Direct LLM HTTP transport failure provider=openai model=%s task_type=%s error=%s",
                    model_name,
                    task_type,
                    exc.__class__.__name__,
                )
                raise LLMServiceError(
                    "AI generation failed. Please try again shortly.",
                    code="generation_failed",
                    status_code=502,
                    model_provider="openai",
                    model_name=model_name,
                    prompt_token_estimate=prompt_token_estimate,
                    retry_count=retry_count,
                    rate_limited=False,
                ) from exc

            output_text = self._extract_output_text(response_payload)
            if not output_text:
                raise LLMServiceError(
                    "AI generation failed. Please try again shortly.",
                    code="generation_failed",
                    status_code=502,
                    model_provider="openai",
                    model_name=model_name,
                    prompt_token_estimate=prompt_token_estimate,
                    retry_count=retry_count,
                )

            if schema:
                try:
                    parsed_output = json.loads(output_text)
                except json.JSONDecodeError as exc:
                    raise LLMServiceError(
                        "AI generation returned invalid structured data.",
                        code="generation_failed",
                        status_code=502,
                        model_provider="openai",
                        model_name=model_name,
                        prompt_token_estimate=prompt_token_estimate,
                        retry_count=retry_count,
                    ) from exc

                reply = str(parsed_output.get("reply") or "").strip()
                structured_output = parsed_output.get("structured_output")
                if not reply or not isinstance(structured_output, dict):
                    raise LLMServiceError(
                        "AI generation returned incomplete structured data.",
                        code="generation_failed",
                        status_code=502,
                        model_provider="openai",
                        model_name=model_name,
                        prompt_token_estimate=prompt_token_estimate,
                        retry_count=retry_count,
                    )
                return {
                    "reply": reply,
                    "structured_output": structured_output,
                    "parse_status": "parsed",
                    "model_provider": "openai",
                    "model_name": model_name,
                    "retry_count": retry_count,
                    "rate_limited": retry_count > 0,
                }

            return {
                "reply": output_text,
                "model_provider": "openai",
                "model_name": model_name,
                "retry_count": retry_count,
                "rate_limited": retry_count > 0,
            }

        raise LLMServiceError(
            "AI generation failed. Please try again shortly.",
            code="generation_failed",
            status_code=502,
            model_provider="openai",
            model_name=model_name,
            prompt_token_estimate=prompt_token_estimate,
        )

    def _call_openai_with_fallback(
        self,
        *,
        task_type: str,
        response_input: list[dict[str, Any]],
        prompt_token_estimate: int,
    ) -> dict[str, Any]:
        primary_candidates = self.preferred_model_candidates()
        fallback_candidates = self.fallback_model_candidates()
        last_error: LLMServiceError | None = None
        last_rate_limit_error: LLMServiceError | None = None

        for model_name in primary_candidates:
            try:
                return self._call_openai_model(
                    task_type=task_type,
                    response_input=response_input,
                    model_name=model_name,
                    max_retries=self.MAX_RETRIES,
                    prompt_token_estimate=prompt_token_estimate,
                )
            except LLMServiceError as exc:
                last_error = exc
                if exc.code == "llm_rate_limited":
                    last_rate_limit_error = exc
                    break

        if last_rate_limit_error and self.fallback_provider == "openai":
            for fallback_model in fallback_candidates[:1]:
                try:
                    fallback_result = self._call_openai_model(
                        task_type=task_type,
                        response_input=response_input,
                        model_name=fallback_model,
                        max_retries=1,
                        prompt_token_estimate=prompt_token_estimate,
                    )
                    fallback_result["rate_limited"] = True
                    fallback_result["retry_count"] = max(
                        last_rate_limit_error.retry_count,
                        int(fallback_result.get("retry_count") or 0),
                    )
                    return fallback_result
                except LLMServiceError as exc:
                    last_error = exc
                    if exc.code == "llm_rate_limited":
                        last_rate_limit_error = exc

        if last_rate_limit_error:
            raise last_rate_limit_error
        if last_error:
            raise last_error

        raise LLMServiceError(
            "AI generation failed. Please try again shortly.",
            code="generation_failed",
            status_code=502,
            model_provider="openai",
        )

    def _prepare_generation(
        self,
        *,
        message: str,
        task_type: str,
        niche: str | None = None,
        target_audience: str | None = None,
        goal: str | None = None,
        link: str | None = None,
        profile_context: dict | None = None,
        link_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
        playbook_context: dict | None = None,
        reel_context: dict | None = None,
        brief_context: dict | None = None,
        public_profile_context: dict | None = None,
        slide_count: int | None = None,
    ) -> dict[str, Any]:
        compact_profile_context = self._compact_profile_context(profile_context)
        compact_recent_content_context = self._compact_recent_content_context(recent_content_context)
        compact_recent_posts_context = self._compact_recent_posts_context(
            task_type=task_type,
            recent_posts_context=recent_posts_context,
            message=message,
            goal=goal,
        )
        compact_link_context = self._compact_link_context(link_context)
        compact_playbook_context = self._compact_playbook_context(
            task_type=task_type,
            playbook_context=playbook_context,
        )
        compact_reel_context = self._compact_reel_context(reel_context)

        prompt_sections = self._build_prompt_sections(
            message=message,
            task_type=task_type,
            niche=niche,
            target_audience=target_audience,
            goal=goal,
            link=link,
            profile_context=compact_profile_context,
            link_context=compact_link_context,
            recent_content_context=compact_recent_content_context,
            recent_posts_context=compact_recent_posts_context,
            playbook_context=compact_playbook_context,
            reel_context=compact_reel_context,
            brief_context=brief_context,
            public_profile_context=public_profile_context,
            slide_count=slide_count,
        )
        response_input, prompt_section_names = self._sections_to_response_input(prompt_sections)
        prompt_token_estimate = self._estimate_prompt_tokens(response_input)
        return {
            "response_input": response_input,
            "prompt_section_names": prompt_section_names,
            "prompt_token_estimate": prompt_token_estimate,
            "playbook_context": compact_playbook_context,
            "recent_posts_context": compact_recent_posts_context,
        }

    def _call_registry_provider(
        self,
        *,
        task_type: str,
        response_input: list[dict[str, Any]],
        prompt_section_names: list[str],
        prompt_token_estimate: int,
    ) -> dict[str, Any]:
        """Run generation through a registered non-OpenAI provider.

        Imported inside the method so this module never pulls a vendor SDK at
        import time -- the OpenAI path must keep working with no other provider
        package installed.
        """
        from app.services.providers import ProviderError, get_text_provider

        try:
            provider = get_text_provider(self.primary_provider)
            return provider.generate(
                task_type=task_type,
                response_input=response_input,
                max_output_tokens=self._max_output_tokens(task_type),
                # When a task has a schema the provider enforces it, so the shape
                # stops depending on the model choosing to follow instructions.
                response_schema=self._structured_schema(task_type),
            )
        except ProviderError as exc:
            raise LLMServiceError(
                exc.safe_message,
                code=exc.code,
                status_code=exc.status_code,
                model_provider=exc.provider,
                model_name=exc.model_name,
                prompt_token_estimate=prompt_token_estimate,
                prompt_section_names=list(prompt_section_names),
            ) from exc

    def _normalize_provider_reply(self, task_type: str, response_payload: dict[str, Any]) -> dict[str, Any]:
        """Parse a provider's prose into the structured payload callers expect.

        Imported inside the method to keep this module's import graph flat --
        the formatter imports the response schemas, which import back into the
        service package.

        A parse failure must not lose the generation the customer already paid
        for: the reply is returned as-is with ``parse_status`` saying so, and the
        caller decides whether prose alone is enough.
        """
        from app.services.agent_response_formatter_service import AgentResponseFormatterService

        reply = response_payload.get("reply")
        try:
            normalized = AgentResponseFormatterService().normalize_reply(task_type, reply)
        except Exception:  # noqa: BLE001 - parsing must never sink a paid-for reply
            logger.exception("Could not parse %s output from %s", task_type, self.primary_provider)
            return {"parse_status": "raw_only", "structured_output": None}

        return {
            "reply": normalized.get("reply") if normalized.get("reply") is not None else reply,
            "parse_status": normalized.get("parse_status"),
            "structured_output": normalized.get("structured_output"),
        }

    def run_agent(
        self,
        *,
        message: str,
        task_type: str,
        account_id: str | None = None,
        niche: str | None = None,
        target_audience: str | None = None,
        goal: str | None = None,
        link: str | None = None,
        profile_context: dict | None = None,
        link_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
        playbook_context: dict | None = None,
        reel_context: dict | None = None,
        brief_context: dict | None = None,
        public_profile_context: dict | None = None,
        slide_count: int | None = None,
    ) -> dict[str, Any]:
        prepared_generation = self._prepare_generation(
            message=message,
            task_type=task_type,
            niche=niche,
            target_audience=target_audience,
            goal=goal,
            link=link,
            profile_context=profile_context,
            link_context=link_context,
            recent_content_context=recent_content_context,
            recent_posts_context=recent_posts_context,
            playbook_context=playbook_context,
            reel_context=reel_context,
            brief_context=brief_context,
            public_profile_context=public_profile_context,
            slide_count=slide_count,
        )

        if self.primary_provider != "openai":
            # Non-OpenAI providers are handled by the registry. The prompt is
            # already built above, so every provider sees the same sections and
            # the same task context -- only the wire format differs.
            response_payload = self._call_registry_provider(
                task_type=task_type,
                response_input=prepared_generation["response_input"],
                prompt_section_names=prepared_generation["prompt_section_names"],
                prompt_token_estimate=prepared_generation["prompt_token_estimate"],
            )
            # Registry providers return prose only. The OpenAI path gets
            # ``structured_output`` back from the model itself, so without this a
            # caller that needs the parsed payload -- a carousel job, above all --
            # received a reply and no slides and failed with "did not produce any
            # slides": a message pointing at the model rather than at the parse
            # step that was never run. The synchronous chat route normalises the
            # same way; doing it here means every consumer of run_agent gets one
            # contract instead of each rediscovering this. Tasks whose schema the
            # provider enforced already carry the payload and are left alone.
            normalized_reply = (
                {}
                if response_payload.get("structured_output") is not None
                else self._normalize_provider_reply(task_type, response_payload)
            )
            return {
                **response_payload,
                **normalized_reply,
                "account_id": account_id,
                "used_langflow": False,
                "prompt_section_names": prepared_generation["prompt_section_names"],
                "prompt_token_estimate": prepared_generation["prompt_token_estimate"],
            }

        try:
            response_payload = self._call_openai_with_fallback(
                task_type=task_type,
                response_input=prepared_generation["response_input"],
                prompt_token_estimate=prepared_generation["prompt_token_estimate"],
            )
        except LLMServiceError as exc:
            if exc.prompt_token_estimate is None:
                exc.prompt_token_estimate = prepared_generation["prompt_token_estimate"]
            if not exc.model_provider:
                exc.model_provider = "openai"
            if not exc.prompt_section_names:
                exc.prompt_section_names = list(prepared_generation["prompt_section_names"])
            raise

        return {
            **response_payload,
            "account_id": account_id,
            "used_langflow": False,
            "prompt_section_names": prepared_generation["prompt_section_names"],
            "prompt_token_estimate": prepared_generation["prompt_token_estimate"],
        }
