import base64
import json
import logging
import os
import random
import re
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


class LangflowServiceError(RuntimeError):
    def __init__(
        self,
        safe_message: str,
        *,
        code: str = "generation_failed",
        status_code: int = 502,
        used_langflow: bool = True,
        model_provider: str | None = "langflow",
        model_name: str | None = None,
        retry_count: int = 0,
        rate_limited: bool = False,
        prompt_section_names: list[str] | None = None,
    ):
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.code = code
        self.status_code = status_code
        self.used_langflow = used_langflow
        self.model_provider = model_provider
        self.model_name = model_name
        self.retry_count = retry_count
        self.rate_limited = rate_limited
        self.prompt_section_names = list(prompt_section_names or [])


class LangflowService:
    REELS_TASK_TYPES = {"reel_idea", "reel_script", "reel_feedback"}
    LANGUAGE_NAMES = {
        "ka": "Georgian",
        "en": "English",
        "ru": "Russian",
    }
    MAX_RETRIES = 3
    BACKOFF_BASE_SECONDS = 1.0
    BACKOFF_MAX_SECONDS = 8.0
    MAX_RECENT_POSTS = 5
    MAX_RECENT_POST_CAPTION_SUMMARY_CHARS = 140
    MAX_PROFILE_SUMMARY_CHARS = 420
    MAX_GOAL_CHARS = 220
    MAX_REEL_SUMMARY_CHARS = 220
    MAX_KNOWLEDGE_CONTEXT_CHARS = 2400

    def __init__(self):
        self.base_url = os.getenv("LANGFLOW_BASE_URL", "http://127.0.0.1:7860/api")
        self.flow_id = os.getenv("LANGFLOW_FLOW_ID", "")
        self.api_key = os.getenv("LANGFLOW_API_KEY", "")
        self.reels_generation_flow_id = os.getenv("LANGFLOW_REELS_GENERATION_FLOW_ID", "").strip() or self.flow_id
        self.reels_ingestion_flow_id = os.getenv("LANGFLOW_REELS_INGESTION_FLOW_ID", "").strip()
        self.vector_store_provider = os.getenv("LANGFLOW_VECTOR_STORE_PROVIDER", "chroma").strip() or "chroma"
        self.safe_reels_rag_enabled = os.getenv("USE_LANGFLOW_SAFE_REELS_RAG", "true").strip().lower() == "true"
        self.timeout_seconds = float(os.getenv("LANGFLOW_TIMEOUT_SECONDS", "60").strip() or "60")
        self._rng = random.Random()

    def _current_reels_generation_flow_id(self) -> str:
        return os.getenv("LANGFLOW_REELS_GENERATION_FLOW_ID", "").strip() or self.reels_generation_flow_id or self.flow_id

    def _current_reels_ingestion_flow_id(self) -> str:
        return os.getenv("LANGFLOW_REELS_INGESTION_FLOW_ID", "").strip() or self.reels_ingestion_flow_id

    def _current_vector_store_provider(self) -> str:
        return os.getenv("LANGFLOW_VECTOR_STORE_PROVIDER", "").strip() or self.vector_store_provider

    def _safe_reels_rag_is_enabled(self) -> bool:
        return os.getenv("USE_LANGFLOW_SAFE_REELS_RAG", "true").strip().lower() == "true"

    def _extract_text(self, data: dict) -> str:
        try:
            return data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except Exception:
            pass

        try:
            return data["outputs"][0]["outputs"][0]["messages"][0]["message"]
        except Exception:
            pass

        try:
            return data["outputs"][0]["outputs"][0]["artifacts"]["message"]
        except Exception:
            pass

        return str(data)

    def _task_instruction(self, task_type: str) -> str:
        instructions = {
            "reel_idea": (
                "Generate original, execution-ready Reel ideas specific to this niche, goal and account "
                "context. Each idea must name the hook mechanic it relies on and why that mechanic suits "
                "this audience. An idea the user cannot picture shooting tomorrow is not an idea, it is a "
                "topic -- reject your own output at that level and go concrete."
            ),
            "reel_script": (
                "Write a Reel script ready to record. The first line must create tension before any context "
                "arrives, because the viewer decides during it. Give scene-by-scene direction the user can "
                "follow without interpretation, on-screen text separate from voiceover, and one CTA matched "
                "to the stated goal. Do not write a script whose hook only makes sense after the payoff."
            ),
            "reel_feedback": (
                "Review the target Reel and be useful rather than kind. Name what works, what weakens it, "
                "and where attention most likely drops -- tied to a specific second or element, not to "
                "'the middle'. Finish with a rewritten version, not a description of how it could be "
                "rewritten. If no Reel context was provided, say so and review only what you were given; "
                "never invent metrics."
            ),
            "caption": (
                "Write a caption the user can post without editing. Match the account's voice rather than a "
                "generic brand voice, earn the first line as a hook because the rest is collapsed behind "
                "'more', and close with one CTA that serves the stated goal. No hashtag walls, no emoji "
                "used as punctuation for its own sake."
            ),
            "carousel": (
                "Write carousel copy that argues one idea across the slides rather than listing several. "
                "Slide one must make stopping feel necessary; each slide after it must earn the swipe by "
                "leaving something unresolved; the last one converts. Keep each slide readable at a glance "
                "-- this is rendered onto an image, so a paragraph will not fit. If a source link is "
                "provided, take its structure and angle, never its words."
            ),
            # Judged against the customer's stated objective when one exists, not
            # against generic best practice. A bio that is "weak" in the abstract
            # may be exactly right for the goal, and the reverse is likelier
            # still: a polished profile that sells nothing the customer offers.
            "profile_audit": (
                "Audit the Instagram profile strategically. If a marketing brief is provided, judge every "
                "element against that objective, offer and ideal customer, and say plainly where the profile "
                "works against the stated goal. Without a brief, fall back to general best practice and say "
                "that is what you are doing. Prioritize conversion clarity, niche positioning, bio strength, "
                "and content direction."
            ),
            "content_plan": (
                "Build a 30-day plan that compounds: each week should build on what the previous one "
                "established rather than restarting the pitch. Anchor it to the stated goal and to whatever "
                "performance history exists. Vary format deliberately and say what each piece is for -- "
                "reach, trust, or conversion. A calendar of unrelated topics is a list, not a plan."
            ),
            "public_profile_analysis": (
                "You have been given a real public account's bio, follower count, content mix and recent "
                "captions with their engagement. Work out how that account is *assembled* -- what it posts, "
                "in what proportion, how its captions open, what its CTA pattern is -- and convert the "
                "transferable mechanics into a plan for this customer's niche. "
                "Say plainly which of its tactics only work because of its scale or budget; a small shop "
                "copying a global brand's brand-awareness posts will reach nobody. "
                "You have likes and comments only. Never state or estimate reach, impressions, saves or "
                "audience demographics for an account the customer does not own -- those numbers were not "
                "provided and inventing them is the fastest way to be wrong in public. "
                "Finish with the first three posts they should make, written out, not described."
            ),
            "link_analysis": (
                "Analyse the linked post and extract the transferable mechanic -- the angle, the hook "
                "logic, the pacing, the CTA style -- then convert it into something original for this "
                "account. Say plainly how much you could actually see: a public link exposes very little, "
                "and an analysis that describes engagement or audience you were not given is fabricated. "
                "Never hand back a version of the source with the words swapped."
            ),
            "performance_summary": (
                "Summarize recent Instagram content performance and turn the findings into practical next "
                "actions. If a marketing brief is provided, measure performance against its primary KPI "
                "rather than raw reach: a post with fewer views that moved the stated KPI outperformed a "
                "viral one that did not."
            ),
        }
        return instructions.get(task_type, "Respond as a practical Instagram content assistant.")

    # The bar the answer has to clear before it is returned.
    #
    # This is where "behaves like a professional" is decided. Retrieved material
    # supplies knowledge; it does not change how the model judges its own work,
    # and the model's default is to sound authoritative on thin reasoning. Each
    # line below names a specific way marketing advice fails, because a generic
    # instruction to "be professional" produces generic professionalism.
    QUALITY_BAR = [
        "Before answering, decide what outcome the user is actually paid for -- sales, qualified DMs, "
        "bookings, retention -- and make every element serve it. Reach that does not move that outcome "
        "is not a win, and should not be presented as one.",
        "Be specific to this account. If an answer would work equally well for any account in any niche, "
        "it is not finished: it is a template. Name the niche, the offer, the objection, the moment.",
        "Prefer one strong, fully worked idea over five thin ones, unless the user asked for a quantity.",
        "State the reasoning behind a recommendation in one clause, not a paragraph -- 'because the first "
        "frame has to survive a mute autoplay', not a lecture on retention.",
        "Write copy the user can post as-is. Placeholders like [your product] or 'insert hook here' are a "
        "failure: fill them from the context you have, and if you genuinely cannot, say which single "
        "detail you need.",
        "Separate what you know from what you are assuming. If account data is absent, say the advice is "
        "based on general practice; never present a guess about their metrics, audience, or history as "
        "fact, and never fill an analysis with 'probably' to look thorough.",
        "Do not pad. No restating the question, no summarising what you are about to say, no closing "
        "offers of further help.",
    ]

    def _base_system_instruction(self) -> str:
        return "\n".join([
            "You are Instagram Agent V1 inside Tichu.",
            "Role: act as the digital marketer running this account, not as an assistant describing what a "
            "marketer would do. You are accountable for the result, so you make the call and justify it "
            "briefly rather than listing options for the user to choose between.",
            "Your value over a general-purpose chatbot is that you hold this account's goal, niche, voice "
            "and history, and that you deliver finished work. Behave accordingly.",
            "Always answer in the same language as the user.",
            "Be practical, specific, structured, and action-oriented.",
            *self.QUALITY_BAR,
            "Do not invent account data, metrics, audience insights, or profile details.",
            "If account context is missing, clearly say that the answer is based on general Instagram best practices.",
            "Ask a clarifying question only when absolutely necessary.",
            "Prefer structured outputs over long paragraphs.",
            "Keep answers concise but useful.",
            "Focus on hooks, structure, clarity, audience attention, conversion, and adaptation to the user's goal.",
            "If a source link is provided, identify transferable patterns, not content to copy.",
            "Never tell the user to duplicate another brand literally. Extract the angle, hook logic, structure, pacing, and CTA style, then convert them into original ideas.",
            "If the user asks for a quantity such as 3 or 5 ideas/scripts, deliver exactly that quantity whenever possible.",
            "If the user mixes analysis with generation, do both in a logical order: analyze first, then adapt, then generate.",
            "If the user writes in Georgian, respond in clean natural Georgian script.",
            "When answering in Georgian, avoid mixing English, Russian, Arabic, or transliterated words unless they are unavoidable proper nouns or standard platform terms such as Instagram, Reel, CTA, DM.",
            "When answering in Georgian, keep required English section labels stable, but write the content under each label in natural professional Georgian.",
            "Avoid literal translations, awkward metaphors, and nonsense phrases in Georgian content.",
        ])

    DEFAULT_CAROUSEL_SLIDES = 5

    def _carousel_output_format(self, slide_count: int | None = None) -> str:
        """Build the carousel heading contract for a given slide count.

        The parser accepts up to ``MAX_CAROUSEL_SLIDES``; asking the model for
        more than that would produce slides that are generated, paid for, and
        then silently dropped.
        """
        from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES

        count = slide_count or self.DEFAULT_CAROUSEL_SLIDES
        count = max(2, min(int(count), MAX_CAROUSEL_SLIDES))
        return "\n".join([
            "Title:",
            *[f"Slide {n}:" for n in range(1, count + 1)],
            "Final CTA slide:",
        ])

    def _output_format_instruction(self, task_type: str, slide_count: int | None = None) -> str:
        formats = {
            "reel_idea": "\n".join([
                "Title:",
                "1-second hook:",
                "Format type:",
                "Main idea:",
                "Shot list:",
                "Why it can work:",
                "CTA:",
            ]),
            "reel_script": "\n".join([
                "Hook:",
                "Problem/angle:",
                "Scene-by-scene script:",
                "On-screen text:",
                "Caption:",
                "CTA:",
            ]),
            "reel_feedback": "\n".join([
                "Summary:",
                "What works:",
                "What hurts:",
                "Retention risks:",
                "Better hook:",
                "Improved structure:",
                "Better CTA:",
                "Improved version:",
            ]),
            "caption": "\n".join([
                "Hook:",
                "Caption:",
                "CTA:",
            ]),
            # Slide headings are generated from the requested count rather than
            # fixed at five, so the prompt contract, the parser and the tier
            # limit all agree on how many slides a carousel has. The count
            # arriving here has already been clamped to the customer's plan.
            "carousel": self._carousel_output_format(slide_count),
            "profile_audit": "\n".join([
                "What works:",
                "What is weak:",
                "What to improve first:",
                "Recommended bio direction:",
                "Content direction:",
            ]),
            "content_plan": "\n".join([
                "Week 1:",
                "Week 2:",
                "Week 3:",
                "Week 4:",
                "Plus:",
                "- best content mix",
                "- hook ideas",
                "- CTA ideas",
            ]),
            "link_analysis": "\n".join([
                "What works:",
                "What is weak:",
                "Why it may perform:",
                "How to adapt it for the user's account:",
            ]),
            "performance_summary": "\n".join([
                "What worked:",
                "What did not work:",
                "Content patterns:",
                "Recommended next moves:",
            ]),
        }
        return formats.get(task_type, "Use a structured Instagram strategy format.")

    def _reels_high_priority_instruction(self, task_type: str, playbook_context: dict | None) -> str | None:
        if task_type not in {"reel_idea", "reel_script", "reel_feedback"}:
            return None

        lines = [
            "High-priority reels methodology rules:",
            "Use the internal reels playbook as a primary strategic guide when it is available.",
            "Do not give generic filler or vague advice. Translate strategy into concrete execution.",
            "Reflect Mariami-style methodology through hook logic, trend adaptation, simplicity, visual execution, retention design, and a single clear CTA.",
            "Never output raw playbook text or mention hidden internal knowledge explicitly.",
        ]

        task_specific_lines = {
            "reel_idea": [
                "For reel_idea, deliver concrete Reel concepts instead of broad topics.",
                "Each idea must include a clean title, a 1-second hook, a concrete format type, a main idea, a shot list, a reason it can work, and a direct CTA.",
                "Trend adaptation and niche specificity should be visible inside the hook, main idea, shot list, and why-it-can-work reasoning instead of bloating the structure with extra filler fields.",
                "Prefer concrete formats such as talking-head diagnosis, proof-led tutorial, comment-reply Reel, objection breakdown, screen-record plus voiceover, or myth-versus-reality when they fit the context.",
                "If you use a trend or pattern, adapt it clearly to the user's niche and business goal rather than naming it vaguely.",
            ],
            "reel_script": [
                "For reel_script, prioritize first-second tension, scene-by-scene retention, and a CTA that matches the business goal.",
                "Write the script so it could be recorded immediately with minimal interpretation.",
            ],
            "reel_feedback": [
                "For reel_feedback, diagnose the hook, structure, clarity, retention risk, and CTA before proposing the improved version.",
                "The improved version must be meaningfully stronger, not just lightly reworded.",
            ],
        }
        lines.extend(task_specific_lines.get(task_type, []))

        if not playbook_context or not playbook_context.get("used_system_knowledge"):
            lines.append("No internal reels playbook chunks were retrieved for this request, so fall back safely to the available account context and general Reel strategy best practices.")

        return "\n".join(lines)

    def _reels_structured_output_contract(self, task_type: str) -> str | None:
        contracts = {
            "reel_idea": [
                "After the human-readable answer, append exactly one final line that starts with STRUCTURED_OUTPUT_JSON: followed by a single valid JSON object.",
                "Do not use markdown code fences around the JSON.",
                'The JSON must match this shape: {"ideas":[{"title":"...","hook":"...","format_type":"...","main_idea":"...","shot_list":["..."],"why_it_can_work":"...","cta":"..."}]}',
            ],
            "reel_script": [
                "After the human-readable answer, append exactly one final line that starts with STRUCTURED_OUTPUT_JSON: followed by a single valid JSON object.",
                "Do not use markdown code fences around the JSON.",
                'The JSON must match this shape: {"title":"...","hook":"...","script_sections":["..."],"cta":"...","full_script":"..."}',
            ],
            "reel_feedback": [
                "After the human-readable answer, append exactly one final line that starts with STRUCTURED_OUTPUT_JSON: followed by a single valid JSON object.",
                "Do not use markdown code fences around the JSON.",
                'The JSON must match this shape: {"what_works":["..."],"what_hurts":["..."],"retention_risks":["..."],"better_hook":"...","improved_structure":["..."],"better_cta":"...","improved_version":"...","summary":"..."}',
            ],
        }
        contract_lines = contracts.get(task_type)
        return "\n".join(contract_lines) if contract_lines else None

    def _context_notice(
        self,
        profile_context: dict | None,
        recent_content_context: dict | None,
        recent_posts_context: dict | None,
        link_context: dict | None,
    ) -> str:
        notices = []

        if not profile_context:
            notices.append("Account profile context is missing. Base the answer only on general Instagram best practices unless the user provided manual business details.")
        if not recent_posts_context:
            notices.append("Recent post-level account data is missing. Do not invent post performance.")
        if not recent_content_context:
            notices.append("Recent performance-summary context is missing. Do not invent trends or metrics.")
        if link_context:
            notices.append("A source link context is available. Use it for inspiration, analysis, and adaptation.")

        if not notices:
            notices.append("Relevant account context is available. Use it carefully without inventing extra data.")

        return "\n".join(["Context availability rules:"] + notices)

    def _priority_rules(
        self,
        niche: str | None,
        target_audience: str | None,
        goal: str | None,
        has_playbook: bool = False,
    ) -> str:
        """Precedence between the sources of truth in this request.

        The playbook rules are conditional. Half of these rules used to explain
        how to weigh "expert playbook guidance" on every request, including the
        majority that carry none -- so the model spent its instructions
        arbitrating between sources where only one was present.
        """
        rules = [
            "Priority rules:",
            "1. Follow the user's explicit request first.",
            "2. If manual business details or overrides are provided, they take priority over stored account context.",
            "3. Use stored account context only as support, never as a replacement for the user's current business description.",
        ]

        if has_playbook:
            rules.extend([
                "4. Use the expert playbook for strategic decisions: structure, hook logic, CTA style, audience framing.",
                "5. Where the playbook conflicts with real account data, trust the data for facts and the playbook for strategy.",
                "6. Do not invent guidance the playbook does not contain; rely on the available context instead.",
            ])

        if niche or target_audience or goal:
            rules.append("Manual business overrides are present in this request. Treat them as the active business direction.")

        return "\n".join(rules)

    def _manual_business_brief(
        self,
        niche: str | None,
        target_audience: str | None,
        goal: str | None,
    ) -> str:
        details = []
        if niche:
            details.append(f"Niche: {niche}")
        if target_audience:
            details.append(f"Target audience: {target_audience}")
        if goal:
            details.append(f"Goal: {goal}")

        if not details:
            return "Manual business brief: not provided"

        return "\n".join(["Manual business brief:"] + details)

    def _format_playbook_context(self, playbook_context: dict | None) -> str:
        if not playbook_context or not playbook_context.get("chunks"):
            return "Internal strategy context: not available"

        domain = str(playbook_context.get("matched_knowledge_domain") or "strategy").strip()
        lines = [
            f"Internal {domain} strategy context:",
            "Use the following expert strategic guidance with high priority when it is relevant:",
        ]

        for index, chunk in enumerate(playbook_context.get("chunks", []), start=1):
            chunk_label = chunk.get("chunk_label") or chunk.get("file_name", "source")
            lines.append(
                f"Playbook chunk {index} [{chunk_label}]: {chunk.get('text', '')}"
            )

        return "\n".join(lines)

    def _format_reel_context(self, reel_context: dict | None) -> str:
        if not reel_context:
            return "Target Reel context: not available"

        performance_signals = []
        if reel_context.get("like_count") is not None:
            performance_signals.append(f"likes={reel_context.get('like_count')}")
        if reel_context.get("comments_count") is not None:
            performance_signals.append(f"comments={reel_context.get('comments_count')}")

        lines = [
            "Target Reel context:",
            f"Source: {reel_context.get('source', '')}",
            f"Media ID: {reel_context.get('media_id', '')}",
            f"Permalink: {reel_context.get('permalink', '')}",
            f"Media type: {reel_context.get('media_type', '')}",
            f"Caption: {reel_context.get('caption', '')}",
            f"Timestamp: {reel_context.get('timestamp', '')}",
            f"Performance signals: {', '.join(performance_signals) if performance_signals else 'not available'}",
            f"Feedback brief: {reel_context.get('analysis_brief', '')}",
        ]
        return "\n".join(lines)

    def _format_profile_context(self, profile_context: dict | None) -> str:
        if not profile_context:
            return "Profile context: not available"

        content_focus = ", ".join(profile_context.get("content_focus", []))
        strengths = ", ".join(profile_context.get("strengths", []))
        weak_points = ", ".join(profile_context.get("weak_points", []))

        return "\n".join([
            "Profile context:",
            f"Brand name: {profile_context.get('brand_name', '')}",
            f"Niche: {profile_context.get('niche', '')}",
            f"Target audience: {profile_context.get('target_audience', '')}",
            f"Brand voice: {profile_context.get('brand_voice', '')}",
            f"Bio: {profile_context.get('bio', '')}",
            f"Content focus: {content_focus}",
            f"Strengths: {strengths}",
            f"Weak points: {weak_points}",
        ])

    def _format_link_context(self, link_context: dict | None) -> str:
        if not link_context:
            return "Link context: not available"

        source_patterns = ", ".join(link_context.get("source_patterns", []))
        probable_strengths = ", ".join(link_context.get("probable_strengths", []))
        adaptation_notes = ", ".join(link_context.get("adaptation_notes", []))

        return "\n".join([
            "Link context:",
            f"Link: {link_context.get('link', '')}",
            f"Detected platform: {link_context.get('detected_platform', '')}",
            f"Source type: {link_context.get('source_type', '')}",
            f"Creator handle: {link_context.get('creator_handle', '')}",
            f"Summary: {link_context.get('summary', '')}",
            f"Content type: {link_context.get('content_type', '')}",
            f"Hook style: {link_context.get('hook_style', '')}",
            f"Likely business goal: {link_context.get('business_goal', '')}",
            f"Source patterns: {source_patterns}",
            f"Probable strengths: {probable_strengths}",
            f"Adaptation notes: {adaptation_notes}",
        ])

    def _format_recent_content_context(self, recent_content_context: dict | None) -> str:
        if not recent_content_context:
            return "Recent content context: not available"

        top_formats = ", ".join(recent_content_context.get("top_formats", []))
        best_topics = ", ".join(recent_content_context.get("best_topics", []))
        weak_topics = ", ".join(recent_content_context.get("weak_topics", []))
        best_ctas = ", ".join(recent_content_context.get("best_ctas", []))
        weak_ctas = ", ".join(recent_content_context.get("weak_ctas", []))
        notes = ", ".join(recent_content_context.get("notes", []))

        return "\n".join([
            "Recent content context:",
            f"Top formats: {top_formats}",
            f"Best topics: {best_topics}",
            f"Weak topics: {weak_topics}",
            f"Best CTAs: {best_ctas}",
            f"Weak CTAs: {weak_ctas}",
            f"Notes: {notes}",
        ])

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
                    f"id={post.get('post_id', '')}, "
                    f"type={post.get('content_type', '')}, "
                    f"topic={post.get('topic', '')}, "
                    f"caption={post.get('caption', '')}, "
                    f"views={post.get('views', 0)}, "
                    f"likes={post.get('likes', 0)}, "
                    f"comments={post.get('comments', 0)}, "
                    f"saves={post.get('saves', 0)}"
                )
            )

        return "\n".join([
            "Recent posts context:",
            *formatted_posts,
        ])

    def _build_input_message(
        self,
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
    ) -> str:
        reels_high_priority_instruction = self._reels_high_priority_instruction(task_type, playbook_context)
        reels_structured_output_contract = self._reels_structured_output_contract(task_type)
        parts = [
            "SYSTEM ROLE AND RULES:",
            self._base_system_instruction(),
            "",
            f"Task type: {task_type}",
            f"Task objective: {self._task_instruction(task_type)}",
        ]

        if reels_high_priority_instruction:
            parts.extend([
                reels_high_priority_instruction,
                self._format_playbook_context(playbook_context),
            ])

        parts.extend([
            "Required output format:",
            self._output_format_instruction(task_type),
        ])

        if reels_structured_output_contract:
            parts.append(reels_structured_output_contract)

        parts.extend([
            self._priority_rules(
                niche=niche,
                target_audience=target_audience,
                goal=goal,
            ),
            self._manual_business_brief(
                niche=niche,
                target_audience=target_audience,
                goal=goal,
            ),
            self._context_notice(
                profile_context=profile_context,
                recent_content_context=recent_content_context,
                recent_posts_context=recent_posts_context,
                link_context=link_context,
            ),
            self._format_profile_context(profile_context),
            self._format_recent_posts_context(recent_posts_context),
            self._format_link_context(link_context),
            self._format_recent_content_context(recent_content_context),
            self._format_reel_context(reel_context),
            f"User request: {message}",
        ])

        if not reels_high_priority_instruction:
            parts.append(self._format_playbook_context(playbook_context))

        if niche:
            parts.append(f"Requested niche override: {niche}")

        if target_audience:
            parts.append(f"Requested target audience override: {target_audience}")

        if goal:
            parts.append(f"Goal: {goal}")

        if link:
            parts.append(f"Original link: {link}")

        parts.append("Execution note: keep the answer concise, actionable, and ready to use.")

        return "\n".join(parts)

    def _safe_truncate(self, value: str | None, max_chars: int) -> str | None:
        normalized_value = " ".join((value or "").split()).strip()
        if not normalized_value:
            return None
        if len(normalized_value) <= max_chars:
            return normalized_value
        return f"{normalized_value[: max_chars - 3].rstrip()}..."

    def _normalize_string_list(self, values: object, *, limit: int = 5, item_max_chars: int = 120) -> list[str]:
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

    def _tokenize_for_matching(self, value: str | None) -> list[str]:
        return re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (value or "").lower())

    def _detect_language_code_from_text(self, value: str | None) -> str | None:
        if re.search(r"[\u10A0-\u10FF]", value or ""):
            return "ka"
        if re.search(r"[\u0400-\u04FF]", value or ""):
            return "ru"
        if re.search(r"[A-Za-z]", value or ""):
            return "en"
        return None

    def _detect_language_code(self, value: str | None) -> str:
        return self._detect_language_code_from_text(value) or "en"

    def _resolve_language_code(
        self,
        message: str | None,
        *,
        profile_context: dict | None = None,
        recent_content_context: dict | None = None,
    ) -> str:
        message_language = self._detect_language_code_from_text(message)
        if message_language:
            return message_language

        context_text = "\n".join(
            part for part in [
                self._compact_profile_summary(profile_context),
                self._compact_recent_content_summary(recent_content_context),
            ]
            if part
        )
        if self._detect_language_code_from_text(context_text) == "ka":
            return "ka"
        return "en"

    def _response_language_name(self, language_code: str) -> str:
        return self.LANGUAGE_NAMES.get(language_code, self.LANGUAGE_NAMES["en"])

    def _response_language_instruction(self, response_language: str) -> str:
        if response_language == "Georgian":
            return (
                "პასუხის ენა: ქართული. უპასუხე ბუნებრივი ქართულით. "
                "არ გადახვიდე ინგლისურ ან რუსულ ტექსტზე, გარდა სტანდარტული პლატფორმის ტერმინებისა: Instagram, Reel, CTA, DM. "
                "Keep required English section labels stable, but write the content under each label in natural professional Georgian. Avoid literal translations, awkward metaphors, and nonsense phrases."
            )
        if response_language == "Russian":
            return (
                "Язык ответа: русский. Отвечай естественно по-русски. "
                "Не переходи на английский или грузинский текст, кроме стандартных терминов платформы: Instagram, Reel, CTA, DM."
            )
        return (
            "Response language: English. Answer in natural English. "
            "Keep standard platform terms such as Instagram, Reel, CTA, DM as-is."
        )

    def _compact_profile_summary(self, profile_context: dict | None) -> str | None:
        if not profile_context:
            return None

        summary_parts = [
            f"Brand: {self._safe_truncate(profile_context.get('brand_name'), 120)}" if profile_context.get("brand_name") else None,
            f"Niche: {self._safe_truncate(profile_context.get('niche'), 120)}" if profile_context.get("niche") else None,
            f"Target audience: {self._safe_truncate(profile_context.get('target_audience'), 120)}" if profile_context.get("target_audience") else None,
            f"Brand voice: {self._safe_truncate(profile_context.get('brand_voice'), 120)}" if profile_context.get("brand_voice") else None,
            f"Bio: {self._safe_truncate(profile_context.get('bio'), 140)}" if profile_context.get("bio") else None,
        ]
        content_focus = self._normalize_string_list(profile_context.get("content_focus"), limit=4)
        strengths = self._normalize_string_list(profile_context.get("strengths"), limit=3)
        weak_points = self._normalize_string_list(profile_context.get("weak_points"), limit=3)
        if content_focus:
            summary_parts.append(f"Content focus: {', '.join(content_focus)}")
        if strengths:
            summary_parts.append(f"Strengths: {', '.join(strengths)}")
        if weak_points:
            summary_parts.append(f"Weak points: {', '.join(weak_points)}")
        return self._safe_truncate(" | ".join(part for part in summary_parts if part), self.MAX_PROFILE_SUMMARY_CHARS)

    def _compact_recent_content_summary(self, recent_content_context: dict | None) -> str | None:
        if not recent_content_context:
            return None

        summary_parts = []
        for label, key in [
            ("Top formats", "top_formats"),
            ("Best topics", "best_topics"),
            ("Weak topics", "weak_topics"),
            ("Best CTAs", "best_ctas"),
            ("Weak CTAs", "weak_ctas"),
            ("Notes", "notes"),
        ]:
            values = self._normalize_string_list(recent_content_context.get(key), limit=4, item_max_chars=120)
            if values:
                summary_parts.append(f"{label}: {', '.join(values)}")
        return self._safe_truncate(" | ".join(summary_parts), 420)

    def _compact_recent_posts_summary(
        self,
        *,
        recent_posts_context: dict | None,
        message: str,
        goal: str | None,
    ) -> list[dict[str, object]]:
        if not recent_posts_context:
            return []

        posts = recent_posts_context.get("posts")
        if not isinstance(posts, list):
            return []

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
            text_blob = " ".join(part for part in [topic, caption_summary] if part)
            overlap_score = sum(1 for token in self._tokenize_for_matching(text_blob) if token in query_tokens)
            reels_bonus = 2 if "REEL" in content_type.upper() else 0
            engagement_score = min(
                3,
                (1 if int(post.get("views") or 0) > 0 else 0)
                + (1 if int(post.get("likes") or 0) > 0 else 0)
                + (1 if int(post.get("comments") or 0) > 0 or int(post.get("saves") or 0) > 0 else 0),
            )
            scored_posts.append({
                "content_type": content_type,
                "topic": topic,
                "caption_summary": caption_summary,
                "views": int(post.get("views") or 0),
                "likes": int(post.get("likes") or 0),
                "comments": int(post.get("comments") or 0),
                "saves": int(post.get("saves") or 0),
                "_score": overlap_score + reels_bonus + engagement_score,
                "_index": index,
            })

        scored_posts.sort(key=lambda item: (item["_score"], -item["_index"]), reverse=True)
        selected_posts = scored_posts[:self.MAX_RECENT_POSTS]
        selected_posts.sort(key=lambda item: item["_index"])

        return [
            {
                "content_type": post.get("content_type"),
                "topic": post.get("topic"),
                "caption_summary": post.get("caption_summary"),
                "views": post.get("views"),
                "likes": post.get("likes"),
                "comments": post.get("comments"),
                "saves": post.get("saves"),
            }
            for post in selected_posts
        ]

    def _metrics_summary(
        self,
        *,
        recent_posts_context: dict | None,
        recent_content_context: dict | None,
    ) -> str | None:
        summary_parts = []

        recent_content_summary = self._compact_recent_content_summary(recent_content_context)
        if recent_content_summary:
            summary_parts.append(recent_content_summary)

        posts = recent_posts_context.get("posts") if isinstance(recent_posts_context, dict) else None
        if isinstance(posts, list) and posts:
            formats = []
            totals = {"views": 0, "likes": 0, "comments": 0, "saves": 0}
            counted_posts = 0
            for post in posts[: self.MAX_RECENT_POSTS]:
                if not isinstance(post, dict):
                    continue
                counted_posts += 1
                content_type = self._safe_truncate(post.get("content_type"), 32)
                if content_type and content_type not in formats:
                    formats.append(content_type)
                for key in totals:
                    totals[key] += int(post.get(key) or 0)
            if counted_posts:
                summary_parts.append(
                    "Recent post metrics sample: "
                    f"posts={counted_posts}, "
                    f"formats={', '.join(formats) if formats else 'unknown'}, "
                    f"views={totals['views']}, "
                    f"likes={totals['likes']}, "
                    f"comments={totals['comments']}, "
                    f"saves={totals['saves']}"
                )

        return self._safe_truncate(" | ".join(summary_parts), 520)

    def _compact_target_reel_summary(self, reel_context: dict | None) -> dict | None:
        if not reel_context:
            return None

        metrics = reel_context.get("metrics") if isinstance(reel_context.get("metrics"), dict) else {}
        like_count = reel_context.get("like_count")
        comments_count = reel_context.get("comments_count")
        if like_count is None:
            like_count = metrics.get("likes") or metrics.get("like_count")
        if comments_count is None:
            comments_count = metrics.get("comments") or metrics.get("comments_count")

        return {
            "source": self._safe_truncate(reel_context.get("source"), 40),
            "account_id": self._safe_truncate(reel_context.get("account_id"), 120),
            "media_id": self._safe_truncate(reel_context.get("media_id"), 80),
            "permalink": self._safe_truncate(reel_context.get("permalink"), 180),
            "source_url": self._safe_truncate(reel_context.get("source_url"), 180),
            "media_type": self._safe_truncate(reel_context.get("media_type"), 48),
            "content_type": self._safe_truncate(reel_context.get("content_type"), 48),
            "caption_summary": self._safe_truncate(
                reel_context.get("caption_summary") or reel_context.get("caption"),
                self.MAX_REEL_SUMMARY_CHARS,
            ),
            "timestamp": self._safe_truncate(reel_context.get("timestamp"), 64),
            "like_count": like_count,
            "comments_count": comments_count,
            "analysis_brief": self._safe_truncate(reel_context.get("analysis_brief"), self.MAX_REEL_SUMMARY_CHARS),
        }

    def _compact_link_context_summary(self, link_context: dict | None) -> dict | None:
        if not isinstance(link_context, dict):
            return None

        compact_context = {
            "source_url": self._safe_truncate(link_context.get("source_url") or link_context.get("link"), 220),
            "link": self._safe_truncate(link_context.get("link") or link_context.get("source_url"), 220),
            "detected_platform": self._safe_truncate(link_context.get("detected_platform"), 32),
            "source_type": self._safe_truncate(link_context.get("source_type"), 80),
            "creator_handle": self._safe_truncate(link_context.get("creator_handle"), 80),
            "content_type": self._safe_truncate(link_context.get("content_type"), 48),
            "summary": self._safe_truncate(link_context.get("summary"), 260),
            "hook_style": self._safe_truncate(link_context.get("hook_style"), 120),
            "business_goal": self._safe_truncate(link_context.get("business_goal"), 120),
            "data_available": bool(link_context.get("data_available")),
            "data_availability": self._safe_truncate(link_context.get("data_availability"), 80),
            "analysis_basis": self._normalize_string_list(link_context.get("analysis_basis"), limit=4, item_max_chars=80),
            "limitations": self._normalize_string_list(link_context.get("limitations"), limit=3, item_max_chars=180),
            "source_patterns": self._normalize_string_list(link_context.get("source_patterns"), limit=4, item_max_chars=120),
            "probable_strengths": self._normalize_string_list(link_context.get("probable_strengths"), limit=4, item_max_chars=120),
            "adaptation_notes": self._normalize_string_list(link_context.get("adaptation_notes"), limit=4, item_max_chars=120),
        }

        matched_media = link_context.get("matched_connected_media")
        if isinstance(matched_media, dict):
            compact_context["matched_connected_media"] = self._compact_target_reel_summary(matched_media)

        return {
            key: value
            for key, value in compact_context.items()
            if value is not None and value != "" and value != []
        }

    def _estimate_tokens(self, payload: dict) -> int:
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        return max(1, len(serialized_payload) // 4)

    def _serialize_runtime_payload(self, payload: dict) -> str:
        encoded_payload = base64.b64encode(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        return f"BASE64JSON:{encoded_payload}"

    def _serialize_main_agent_runtime_payload(self, payload: dict) -> str:
        response_language = payload.get("response_language") or "English"
        language_code = payload.get("language_code") or "en"
        language_instruction = payload.get("response_language_instruction") or self._response_language_instruction(str(response_language))
        language_header_by_code = {
            "ka": "პასუხის ენა: ქართული",
            "ru": "Язык ответа: русский",
            "en": "Response language: English",
        }
        runtime_json = json.dumps(payload, ensure_ascii=False)
        return "\n".join([
            language_header_by_code.get(str(language_code), f"Response language: {response_language}"),
            f"language_code: {language_code}",
            f"response_language: {response_language}",
            str(language_instruction),
            "",
            "runtime_context_json:",
            runtime_json,
        ])

    def _runtime_payload_for_main_agent(
        self,
        *,
        task_type: str,
        message: str,
        account_id: str | None = None,
        goal: str | None = None,
        link: str | None = None,
        link_context: dict | None = None,
        reel_context: dict | None = None,
        profile_context: dict | None = None,
        recent_posts_context: dict | None = None,
        recent_content_context: dict | None = None,
        knowledge_context: str | None = None,
    ) -> dict:
        user_request = message
        safe_link = self._safe_truncate(link, 220)
        if safe_link:
            user_request = f"{message}\nSource link: {safe_link}"

        language_code = self._resolve_language_code(
            message,
            profile_context=profile_context,
            recent_content_context=recent_content_context,
        )
        response_language = self._response_language_name(language_code)

        runtime_payload = {
            "task_type": task_type,
            "language_code": language_code,
            "response_language": response_language,
            "response_language_instruction": self._response_language_instruction(response_language),
            "user_request": self._safe_truncate(user_request, 900),
            "goal": self._safe_truncate(goal, self.MAX_GOAL_CHARS),
            "account_id": self._safe_truncate(account_id, 120),
            "profile_summary": self._compact_profile_summary(profile_context),
            "metrics_summary": self._metrics_summary(
                recent_posts_context=recent_posts_context,
                recent_content_context=recent_content_context,
            ),
            "compact_recent_posts_summary": self._compact_recent_posts_summary(
                recent_posts_context=recent_posts_context,
                message=message,
                goal=goal,
            ),
            "language": language_code,
        }
        if safe_link:
            runtime_payload["link"] = safe_link
            runtime_payload["source_url"] = safe_link

        compact_link_context = self._compact_link_context_summary(link_context)
        if compact_link_context:
            runtime_payload["link_context"] = compact_link_context

        target_media_summary = self._compact_target_reel_summary(reel_context)
        if target_media_summary:
            runtime_payload["target_media_summary"] = target_media_summary

        safe_knowledge_context = self._safe_truncate(knowledge_context, self.MAX_KNOWLEDGE_CONTEXT_CHARS)
        if safe_knowledge_context:
            runtime_payload["knowledge_context"] = safe_knowledge_context

        return runtime_payload

    def _runtime_payload_for_reels(
        self,
        *,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_posts_context: dict | None = None,
        recent_content_context: dict | None = None,
        reel_context: dict | None = None,
        link_context: dict | None = None,
    ) -> dict:
        link_summary = None
        if link_context:
            link_summary_parts = []
            for key in ["link", "summary", "hook_style", "content_type"]:
                value = self._safe_truncate(link_context.get(key), 180)
                if value:
                    link_summary_parts.append(f"{key}: {value}")
            source_patterns = self._normalize_string_list(link_context.get("source_patterns"), limit=4)
            if source_patterns:
                link_summary_parts.append(f"source_patterns: {', '.join(source_patterns)}")
            link_summary = " | ".join(link_summary_parts) if link_summary_parts else None

        retrieval_hints_by_task = {
            "reel_idea": [
                "viral idea mechanics",
                "trend adaptation",
                "first-second hook",
                "simple format selection",
                "DM-driving CTA",
            ],
            "reel_script": [
                "hook structure",
                "scene flow",
                "retention pacing",
                "shot progression",
                "CTA close",
            ],
            "reel_feedback": [
                "hook diagnosis",
                "retention drop analysis",
                "clarity issues",
                "CTA weakness",
                "improvement framework",
            ],
        }
        retrieval_query_by_task = {
            "reel_idea": f"{task_type} reels ideas viral idea mechanics trend adaptation first seconds hook format library simplicity {message} {goal or ''}",
            "reel_script": f"{task_type} reel structure hook shot list retention CTA narrative flow {message} {goal or ''}",
            "reel_feedback": f"{task_type} hook diagnosis retention structure clarity CTA improvement framework {message} {goal or ''}",
        }
        response_contract_by_task = {
            "reel_idea": {
                "ideas": [
                    {
                        "title": "...",
                        "hook": "...",
                        "format_type": "...",
                        "main_idea": "...",
                        "shot_list": ["..."],
                        "why_it_can_work": "...",
                        "cta": "...",
                    }
                ]
            },
            "reel_script": {
                "script": {
                    "hook": "...",
                    "structure": ["..."],
                    "voiceover": "...",
                    "shot_list": ["..."],
                    "cta": "...",
                },
            },
            "reel_feedback": {
                "feedback": {
                    "what_works": ["..."],
                    "what_hurts": ["..."],
                    "retention_issues": ["..."],
                    "hook_improvement": "...",
                    "cta_improvement": "...",
                    "improved_version": "...",
                },
            },
        }

        language_code = self._resolve_language_code(
            message,
            profile_context=profile_context,
            recent_content_context=recent_content_context,
        )
        response_language = self._response_language_name(language_code)

        return {
            "runtime_schema_version": "reels_rag_v1",
            "task_type": task_type,
            "knowledge_domain": "reels",
            "vector_store_provider": self._current_vector_store_provider(),
            "language_code": language_code,
            "response_language": response_language,
            "response_language_instruction": self._response_language_instruction(response_language),
            "language": language_code,
            "user_request": message,
            "goal": self._safe_truncate(goal, self.MAX_GOAL_CHARS),
            "profile_summary": self._compact_profile_summary(profile_context),
            "recent_posts_summary": self._compact_recent_posts_summary(
                recent_posts_context=recent_posts_context,
                message=message,
                goal=goal,
            ),
            "recent_content_summary": self._compact_recent_content_summary(recent_content_context),
            "target_reel_summary": self._compact_target_reel_summary(reel_context),
            "link_summary": link_summary,
            "retrieval_query": self._safe_truncate(retrieval_query_by_task.get(task_type, message), 420),
            "retrieval_hints": retrieval_hints_by_task.get(task_type, []),
            "retrieval_top_k": {"reel_idea": 3, "reel_script": 4, "reel_feedback": 4}.get(task_type, 4),
            "response_contract": response_contract_by_task.get(task_type),
            "response_style_rules": [
                "Return only the final user-ready answer plus structured_output.",
                "Do not echo internal retrieval chunks.",
                "Do not mention hidden system knowledge explicitly.",
                "Keep the answer strategic, concrete, and non-generic.",
            ],
        }

    def _ingestion_payload_for_reels(
        self,
        *,
        knowledge_pack_id: str,
        title: str,
        description: str | None,
        file_paths: list[str],
        supported_task_types: list[str],
    ) -> dict:
        return {
            "runtime_schema_version": "reels_ingestion_v1",
            "operation": "ingest_system_reels_knowledge",
            "knowledge_pack_id": knowledge_pack_id,
            "source_title": title,
            "description": self._safe_truncate(description, 220),
            "domain": "reels",
            "scope": "system",
            "visibility": "internal",
            "status": "active",
            "supported_task_types": list(supported_task_types),
            "vector_store_provider": self._current_vector_store_provider(),
            "file_paths": list(file_paths),
        }

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

    def _backoff_seconds(self, attempt_number: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return retry_after_seconds
        return min(self.BACKOFF_MAX_SECONDS, self.BACKOFF_BASE_SECONDS * (2 ** max(attempt_number - 1, 0))) + self._jitter()

    def _post_run_flow(self, *, flow_id: str, payload: dict, query_suffix: str = "stream=false") -> httpx.Response:
        url = f"{self.base_url}/v1/run/{flow_id}?{query_suffix}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                },
            )

    def _call_run_flow_with_retries(
        self,
        *,
        flow_id: str,
        payload: dict,
        prompt_section_names: list[str],
    ) -> dict:
        if not flow_id:
            raise LangflowServiceError(
                "Langflow reels flow is not configured yet.",
                status_code=502,
                model_name=flow_id or None,
                prompt_section_names=prompt_section_names,
            )

        if not self.api_key:
            raise LangflowServiceError(
                "Langflow API key is not configured yet.",
                status_code=502,
                model_name=flow_id,
                prompt_section_names=prompt_section_names,
            )

        retry_count = 0
        for attempt_number in range(1, self.MAX_RETRIES + 2):
            try:
                response = self._post_run_flow(
                    flow_id=flow_id,
                    payload=payload,
                )
                response.raise_for_status()
                return response.json(), retry_count
            except httpx.HTTPStatusError as exc:
                response = exc.response
                if response.status_code == 429:
                    logger.warning(
                        "Langflow rate limited flow_id=%s attempt=%s retry_count=%s",
                        flow_id,
                        attempt_number,
                        retry_count,
                    )
                    if attempt_number <= self.MAX_RETRIES:
                        retry_count += 1
                        self._sleep(self._backoff_seconds(attempt_number, self._parse_retry_after_seconds(response)))
                        continue
                    raise LangflowServiceError(
                        "AI generation is temporarily rate limited. Please try again shortly.",
                        code="llm_rate_limited",
                        status_code=503,
                        model_name=flow_id,
                        retry_count=retry_count,
                        rate_limited=True,
                        prompt_section_names=prompt_section_names,
                    ) from exc

                raise LangflowServiceError(
                    "AI generation failed. Please try again shortly.",
                    status_code=502,
                    model_name=flow_id,
                    retry_count=retry_count,
                    prompt_section_names=prompt_section_names,
                ) from exc
            except httpx.ReadTimeout as exc:
                raise LangflowServiceError(
                    "Langflow request timed out",
                    status_code=504,
                    model_name=flow_id,
                    retry_count=retry_count,
                    prompt_section_names=prompt_section_names,
                ) from exc
            except httpx.HTTPError as exc:
                raise LangflowServiceError(
                    "AI generation failed. Please try again shortly.",
                    status_code=502,
                    model_name=flow_id,
                    retry_count=retry_count,
                    prompt_section_names=prompt_section_names,
                ) from exc

        raise LangflowServiceError(
            "AI generation failed. Please try again shortly.",
            status_code=502,
            model_name=flow_id,
            retry_count=retry_count,
            prompt_section_names=prompt_section_names,
        )

    def _extract_json_object(self, text: str) -> dict | None:
        normalized_text = (text or "").strip()
        if not normalized_text:
            return None
        try:
            parsed = json.loads(normalized_text)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def should_use_safe_reels_rag(self, task_type: str) -> bool:
        return bool(
            self._safe_reels_rag_is_enabled()
            and task_type in self.REELS_TASK_TYPES
            and self._current_reels_generation_flow_id()
        )

    def run_reels_rag_agent(
        self,
        *,
        message: str,
        task_type: str,
        account_id: str | None = None,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_posts_context: dict | None = None,
        recent_content_context: dict | None = None,
        reel_context: dict | None = None,
        link_context: dict | None = None,
    ) -> dict:
        runtime_payload = self._runtime_payload_for_reels(
            task_type=task_type,
            message=message,
            goal=goal,
            profile_context=profile_context,
            recent_posts_context=recent_posts_context,
            recent_content_context=recent_content_context,
            reel_context=reel_context,
            link_context=link_context,
        )
        prompt_section_names = ["langflow_runtime_payload"]
        prompt_token_estimate = self._estimate_tokens(runtime_payload)
        generation_flow_id = self._current_reels_generation_flow_id()
        langflow_payload = {
            "input_value": self._serialize_runtime_payload(runtime_payload),
            "input_type": "text",
            "output_type": "text",
            "session_id": f"{account_id or 'default-session'}-{uuid4().hex}",
        }

        data, retry_count = self._call_run_flow_with_retries(
            flow_id=generation_flow_id,
            payload=langflow_payload,
            prompt_section_names=prompt_section_names,
        )
        final_text = self._extract_text(data)
        parsed_payload = self._extract_json_object(final_text)

        if not parsed_payload:
            raise LangflowServiceError(
                "Internal AI flow is not configured correctly yet. Please try again shortly.",
                status_code=502,
                model_name=generation_flow_id,
                retry_count=retry_count,
                rate_limited=retry_count > 0,
                prompt_section_names=prompt_section_names,
            )

        error_payload = parsed_payload.get("error")
        if isinstance(error_payload, dict):
            error_code = str(error_payload.get("code") or "generation_failed")
            safe_message = str(error_payload.get("message") or "AI generation failed. Please try again shortly.")
            status_code = 503 if error_code == "llm_rate_limited" else 502
            raise LangflowServiceError(
                safe_message,
                code=error_code,
                status_code=status_code,
                model_name=parsed_payload.get("model_name") or generation_flow_id,
                retry_count=int(parsed_payload.get("retry_count") or retry_count),
                rate_limited=bool(parsed_payload.get("rate_limited")),
                prompt_section_names=prompt_section_names,
            )

        structured_output = parsed_payload.get("structured_output")
        reply = str(parsed_payload.get("reply") or "").strip()
        if not reply or not isinstance(structured_output, dict):
            raise LangflowServiceError(
                "Internal AI flow is not configured correctly yet. Please try again shortly.",
                status_code=502,
                model_name=generation_flow_id,
                retry_count=retry_count,
                rate_limited=retry_count > 0,
                prompt_section_names=prompt_section_names,
            )

        return {
            "reply": reply,
            "account_id": account_id,
            "structured_output": structured_output,
            "parse_status": parsed_payload.get("parse_status"),
            "used_system_knowledge": bool(parsed_payload.get("used_system_knowledge")),
            "matched_knowledge_domain": parsed_payload.get("matched_knowledge_domain"),
            "matched_knowledge_pack_ids": list(parsed_payload.get("matched_knowledge_pack_ids") or []),
            "retrieved_chunk_count": int(parsed_payload.get("retrieved_chunk_count") or 0),
            "retrieved_chunk_titles": list(parsed_payload.get("retrieved_chunk_titles") or []),
            "used_langflow": True,
            "model_provider": parsed_payload.get("model_provider") or "langflow",
            "model_name": parsed_payload.get("model_name") or generation_flow_id,
            "retry_count": retry_count,
            "rate_limited": retry_count > 0,
            "prompt_section_names": prompt_section_names,
            "prompt_token_estimate": int(parsed_payload.get("prompt_token_estimate") or prompt_token_estimate),
        }

    def ingest_system_reels_knowledge(
        self,
        *,
        knowledge_pack_id: str,
        title: str,
        description: str | None,
        file_paths: list[str],
        supported_task_types: list[str],
    ) -> dict:
        ingestion_flow_id = self._current_reels_ingestion_flow_id()
        if not ingestion_flow_id:
            return {
                "ingestion_triggered": False,
                "ingestion_flow_id": None,
                "vector_store_provider": self._current_vector_store_provider(),
            }

        runtime_payload = self._ingestion_payload_for_reels(
            knowledge_pack_id=knowledge_pack_id,
            title=title,
            description=description,
            file_paths=file_paths,
            supported_task_types=supported_task_types,
        )
        prompt_section_names = ["langflow_ingestion_runtime_payload"]
        langflow_payload = {
            "input_value": self._serialize_runtime_payload(runtime_payload),
            "input_type": "text",
            "output_type": "text",
            "session_id": f"ingestion-{knowledge_pack_id}-{uuid4().hex}",
        }
        data, retry_count = self._call_run_flow_with_retries(
            flow_id=ingestion_flow_id,
            payload=langflow_payload,
            prompt_section_names=prompt_section_names,
        )
        final_text = self._extract_text(data)
        parsed_payload = self._extract_json_object(final_text)
        if isinstance(parsed_payload, dict) and isinstance(parsed_payload.get("error"), dict):
            error_payload = parsed_payload["error"]
            error_code = str(error_payload.get("code") or "generation_failed")
            safe_message = str(error_payload.get("message") or "Internal reels knowledge ingestion failed.")
            status_code = 503 if error_code == "llm_rate_limited" else 502
            raise LangflowServiceError(
                safe_message,
                code=error_code,
                status_code=status_code,
                model_name=ingestion_flow_id,
                retry_count=int(parsed_payload.get("retry_count") or retry_count),
                rate_limited=bool(parsed_payload.get("rate_limited")),
                prompt_section_names=prompt_section_names,
            )

        return {
            "ingestion_triggered": True,
            "ingestion_flow_id": ingestion_flow_id,
            "vector_store_provider": self._current_vector_store_provider(),
            "retry_count": retry_count,
            "chunk_count": int(parsed_payload.get("chunk_count") or 0) if isinstance(parsed_payload, dict) else 0,
            "embeddings_stored": bool(parsed_payload.get("embeddings_stored")) if isinstance(parsed_payload, dict) else False,
            "collection_name": parsed_payload.get("collection_name") if isinstance(parsed_payload, dict) else None,
            "raw_result_preview": self._safe_truncate(final_text, 240),
        }

    def run_flat_prompt_agent(
        self,
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
        knowledge_context: str | None = None,
    ) -> dict:
        if not self.flow_id:
            raise ValueError("LANGFLOW_FLOW_ID is not set")

        if not self.api_key:
            raise ValueError("LANGFLOW_API_KEY is not set")

        runtime_payload = self._runtime_payload_for_main_agent(
            task_type=task_type,
            message=message,
            account_id=account_id,
            goal=goal,
            link=link,
            link_context=link_context,
            reel_context=reel_context,
            profile_context=profile_context,
            recent_content_context=recent_content_context,
            recent_posts_context=recent_posts_context,
            knowledge_context=knowledge_context,
        )
        prompt_section_names = ["main_agent_runtime_payload"]
        if runtime_payload.get("knowledge_context"):
            prompt_section_names.append("deterministic_knowledge_context")
        prompt_token_estimate = self._estimate_tokens(runtime_payload)

        payload = {
            "input_value": self._serialize_main_agent_runtime_payload(runtime_payload),
            "input_type": "chat",
            "output_type": "chat",
            "session_id": f"{account_id or 'default-session'}-{uuid4().hex}",
        }

        data, retry_count = self._call_run_flow_with_retries(
            flow_id=self.flow_id,
            payload=payload,
            prompt_section_names=prompt_section_names,
        )
        final_text = self._extract_text(data)

        return {
            "reply": final_text,
            "account_id": account_id,
            "used_langflow": True,
            "model_provider": "langflow",
            "model_name": self.flow_id,
            "prompt_section_names": prompt_section_names,
            "prompt_token_estimate": prompt_token_estimate,
            "retry_count": retry_count,
            "rate_limited": retry_count > 0,
        }

    def run_agent(
        self,
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
        knowledge_context: str | None = None,
    ) -> dict:
        return self.run_flat_prompt_agent(
            message=message,
            task_type=task_type,
            account_id=account_id,
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
            knowledge_context=knowledge_context,
        )
