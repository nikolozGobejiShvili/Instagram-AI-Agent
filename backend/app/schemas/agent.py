from typing import Any, Literal

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


TaskType = Literal[
    "chat",
    "reel_idea",
    "reel_script",
    "reel_feedback",
    "caption",
    "carousel",
    "profile_audit",
    "content_plan",
    "link_analysis",
    "performance_summary",
]

ParseStatus = Literal["parsed", "partial", "raw_only"]


class ReelIdeaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    hook: str | None = None
    format_type: str | None = None
    main_idea: str | None = None
    shot_list: list[str] = Field(default_factory=list)
    why_it_can_work: str | None = None
    caption_idea: str | None = None
    cta: str | None = None


class ReelIdeaStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ideas: list[ReelIdeaItem] = Field(default_factory=list)


class ReelScriptContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook: str | None = None
    problem_angle: str | None = None
    structure: list[str] = Field(default_factory=list)
    voiceover: str | None = None
    shot_list: list[str] = Field(default_factory=list)
    on_screen_text: list[str] = Field(default_factory=list)
    caption: str | None = None
    cta: str | None = None


class ReelScriptStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: ReelScriptContent


class ReelFeedbackContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what_works: list[str] = Field(default_factory=list)
    what_hurts: list[str] = Field(default_factory=list)
    retention_issues: list[str] = Field(default_factory=list)
    hook_improvement: str | None = None
    cta_improvement: str | None = None
    improved_version: str | None = None


class ReelFeedbackStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: ReelFeedbackContent


class CaptionStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook: str | None = None
    body: str | None = None
    cta: str | None = None
    full_caption: str | None = None


class CarouselSlideItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slide_number: int | None = None
    headline: str | None = None
    body: str | None = None

    # Imagery. All optional and appended after the original three fields, so a
    # text-only carousel and any existing consumer keep working unchanged.
    #
    # image_prompt is art direction for the background only: it must never ask
    # for text in the image. Slide copy is composited in code because image
    # models render text unreliably, Georgian especially.
    image_prompt: str | None = None
    image_url: str | None = None
    template_id: str | None = None


class CarouselStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    slides: list[CarouselSlideItem] = Field(default_factory=list)
    cta: str | None = None


class ProfileAuditStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strengths: list[str] = Field(default_factory=list)
    weak_points: list[str] = Field(default_factory=list)
    quick_fixes: list[str] = Field(default_factory=list)
    recommended_bio_direction: str | None = None
    content_direction: list[str] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)
    summary: str | None = None


class ContentPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day_or_slot: str | None = None
    format: str | None = None
    topic: str | None = None
    goal: str | None = None


class ContentPlanStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_title: str | None = None
    content_items: list[ContentPlanItem] = Field(default_factory=list)
    best_content_mix: list[str] = Field(default_factory=list)
    hook_ideas: list[str] = Field(default_factory=list)
    cta_ideas: list[str] = Field(default_factory=list)
    summary: str | None = None


class LinkAnalysisStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_state: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommended_changes: list[str] = Field(default_factory=list)
    best_cta_direction: str | None = None
    summary: str | None = None


class PerformanceSummaryStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    what_worked: list[str] = Field(default_factory=list)
    what_did_not_work: list[str] = Field(default_factory=list)
    content_patterns: list[str] = Field(default_factory=list)
    best_opportunities: list[str] = Field(default_factory=list)
    recommended_next_moves: list[str] = Field(default_factory=list)
    summary: str | None = None


StructuredOutput = (
    ReelIdeaStructuredOutput
    | ReelScriptStructuredOutput
    | ReelFeedbackStructuredOutput
    | CaptionStructuredOutput
    | CarouselStructuredOutput
    | ProfileAuditStructuredOutput
    | ContentPlanStructuredOutput
    | LinkAnalysisStructuredOutput
    | PerformanceSummaryStructuredOutput
    | dict[str, Any]
)


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Write a conversion-focused caption for my next Instagram post",
                    "task_type": "caption",
                    "user_id": "user-1",
                    "auto_sync": True,
                    "goal": "increase qualified DMs from Instagram",
                },
                {
                    "message": "Write a short reel script based on my current Instagram context",
                    "task_type": "reel_script",
                    "user_id": "user-1",
                    "auto_sync": True,
                },
                {
                    "message": "Review this Reel and give me stronger hook + structure improvements",
                    "task_type": "reel_feedback",
                    "user_id": "user-1",
                    "media_id": "17911223344556677",
                    "auto_sync": True,
                    "goal": "improve retention and qualified DMs",
                },
                {
                    "message": "Analyze this Reel link and rewrite it for my brand",
                    "task_type": "reel_feedback",
                    "user_id": "user-1",
                    "link": "https://www.instagram.com/reel/example/",
                },
                {
                    "message": "Create a 5-slide carousel about common conversion mistakes",
                    "task_type": "carousel",
                    "user_id": "user-1",
                },
                {
                    "message": "Audit my Instagram profile and tell me what to fix first",
                    "task_type": "profile_audit",
                    "user_id": "user-1",
                },
                {
                    "message": "Build a simple 2-week Instagram content plan for lead generation",
                    "task_type": "content_plan",
                    "user_id": "user-1",
                },
                {
                    "message": "Analyze this Instagram post and adapt the pattern for my brand",
                    "task_type": "link_analysis",
                    "user_id": "user-1",
                    "link": "https://www.instagram.com/p/example/",
                },
            ]
        }
    )

    message: str
    task_type: TaskType
    user_id: str | None = None
    account_id: str | None = None
    media_id: str | None = None
    auto_sync: bool = False
    niche: str | None = None
    target_audience: str | None = None
    goal: str | None = None
    link: str | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.message or not self.message.strip():
            raise ValueError("message is required")

        normalized_link = " ".join((self.link or "").split()).strip()
        normalized_source_url = " ".join((self.source_url or "").split()).strip()

        if normalized_link:
            self.link = normalized_link
        if normalized_source_url:
            self.source_url = normalized_source_url
        if normalized_source_url and not normalized_link:
            self.link = normalized_source_url
        if normalized_link and not normalized_source_url:
            self.source_url = normalized_link

        if self.task_type == "link_analysis" and not (self.link or self.source_url):
            raise ValueError("link or source_url is required when task_type is 'link_analysis'")

        return self


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "reply": "Title:\nThe DM Mistake Reel\n\n1-second hook:\nYour Reels do not fail because of views. They fail because the CTA arrives too late.\n\nFormat type:\nTalking-head diagnosis with proof screenshots\n\nMain idea:\nBreak down one CTA mistake that kills qualified DMs, then show the exact fix in one sentence.\n\nShot list:\n1. Direct-to-camera hook\n2. Screenshot of weak CTA\n3. Screenshot of stronger CTA\n4. Final direct CTA\n\nWhy it can work:\nIt uses a fast tension hook, gives proof quickly, and converts a common hidden mistake into a saveable teaching moment.\n\nCTA:\nDM me 'CTA' and I will rewrite your Reel CTA.",
                    "account_id": "test-account-3",
                    "task_type": "reel_idea",
                    "parse_status": "parsed",
                    "structured_output": {
                        "ideas": [
                            {
                                "title": "The DM Mistake Reel",
                                "hook": "Your Reels do not fail because of views. They fail because the CTA arrives too late.",
                                "format_type": "Talking-head diagnosis with proof screenshots",
                                "main_idea": "Break down one CTA mistake that kills qualified DMs, then show the exact fix in one sentence.",
                                "shot_list": [
                                    "Direct-to-camera hook",
                                    "Screenshot of weak CTA",
                                    "Screenshot of stronger CTA",
                                    "Final direct CTA",
                                ],
                                "why_it_can_work": "It uses a fast tension hook, gives proof quickly, and converts a common hidden mistake into a saveable teaching moment.",
                                "cta": "DM me 'CTA' and I will rewrite your Reel CTA.",
                            }
                        ]
                    },
                },
                {
                    "reply": "Hook:\nIf your content looks good but still does not sell...\n\nCaption:\nGood content starts where clarity, tension, and real value meet.\n\nCTA:\nDM me for captions like this.",
                    "account_id": "test-account-3",
                    "task_type": "caption",
                    "parse_status": "parsed",
                    "structured_output": {
                        "hook": "If your content looks good but still does not sell...",
                        "body": "Good content starts where clarity, tension, and real value meet.",
                        "cta": "DM me for captions like this.",
                        "full_caption": "If your content looks good but still does not sell...\n\nGood content starts where clarity, tension, and real value meet.\n\nDM me for captions like this.",
                    },
                },
                {
                    "reply": "What worked:\nStrong hooks\n- Short intros\n\nWhat did not work:\nWeak topic consistency\n\nContent patterns:\nReels outperform static posts\n\nRecommended next moves:\n1. Post 3 reels weekly\n2. Repeat the strongest hook structure",
                    "account_id": "test-account-3",
                    "task_type": "performance_summary",
                    "parse_status": "parsed",
                    "structured_output": {
                        "what_worked": [
                            "Strong hooks",
                            "Short intros",
                        ],
                        "what_did_not_work": [
                            "Weak topic consistency",
                        ],
                        "content_patterns": [
                            "Reels outperform static posts",
                        ],
                        "recommended_next_moves": [
                            "Post 3 reels weekly",
                            "Repeat the strongest hook structure",
                        ],
                        "summary": None,
                    },
                },
                {
                    "reply": "Hook:\nთუ ხალხი გიყურებს, მაგრამ არ გწერს, პრობლემა ხშირად პირველ 2 წამში ან CTA-შია.\n\nStructure:\n1. Open with the pain point\n2. Show one proof moment fast\n3. Deliver one fix\n4. Close with one direct CTA\n\nVoiceover:\nთუ ხალხი გიყურებს, მაგრამ არ გწერს, პრობლემა ხშირად იმაშია, რომ Reel ყურადღებას კი იჭერს, მაგრამ ქმედებამდე ვერ მიჰყავს. ჯერ აჩვენე ტკივილი, შემდეგ მტკიცებულება, მერე ერთი ზუსტი გამოსწორება და ბოლოში ერთი მკაფიო CTA.\n\nShot list:\n1. Direct hook to camera\n2. Screenshot of weak Reel opening\n3. Quick corrected example\n4. Final CTA frame\n\nCTA:\nDM მომწერე 'REEL' და ერთ შენ Reel-ს დაგიშლი.",
                    "account_id": "test-account-3",
                    "task_type": "reel_script",
                    "parse_status": "parsed",
                    "structured_output": {
                        "script": {
                            "hook": "თუ ხალხი გიყურებს, მაგრამ არ გწერს, პრობლემა ხშირად პირველ 2 წამში ან CTA-შია.",
                            "structure": [
                                "Open with the pain point",
                                "Show one proof moment fast",
                                "Deliver one fix",
                                "Close with one direct CTA"
                            ],
                            "voiceover": "თუ ხალხი გიყურებს, მაგრამ არ გწერს, პრობლემა ხშირად იმაშია, რომ Reel ყურადღებას კი იჭერს, მაგრამ ქმედებამდე ვერ მიჰყავს. ჯერ აჩვენე ტკივილი, შემდეგ მტკიცებულება, მერე ერთი ზუსტი გამოსწორება და ბოლოში ერთი მკაფიო CTA.",
                            "shot_list": [
                                "Direct hook to camera",
                                "Screenshot of weak Reel opening",
                                "Quick corrected example",
                                "Final CTA frame"
                            ],
                            "cta": "DM მომწერე 'REEL' და ერთ შენ Reel-ს დაგიშლი."
                        }
                    },
                },
                {
                    "reply": "Summary:\nამ Reel-ს კარგი თემა აქვს, მაგრამ hook ნელა იწყება და CTA საკმარისად პირდაპირი არ არის.\n\nWhat works:\nRelevant topic\nClear visual proof\n\nWhat hurts:\nHook is generic\nMiddle section repeats itself\n\nRetention issues:\nNo first-second pattern interrupt\nMomentum drops in the middle\n\nHook improvement:\nთუ შენი Reel ლამაზია, მაგრამ ხალხი მაინც სქროლავს, მიზეზი ხშირად ეს არის.\n\nCTA improvement:\nDM მომწერე 'REEL' და გაჩვენებ როგორ გადააკეთო ეს opening და CTA.\n\nImproved version:\nStart with the mistake immediately, show one proof moment fast, deliver one fix, then close with one direct DM CTA.",
                    "account_id": "test-account-3",
                    "task_type": "reel_feedback",
                    "parse_status": "parsed",
                    "structured_output": {
                        "feedback": {
                            "what_works": [
                                "Relevant topic",
                                "Clear visual proof"
                            ],
                            "what_hurts": [
                                "Hook is generic",
                                "Middle section repeats itself"
                            ],
                            "retention_issues": [
                                "No first-second pattern interrupt",
                                "Momentum drops in the middle"
                            ],
                            "hook_improvement": "თუ შენი Reel ლამაზია, მაგრამ ხალხი მაინც სქროლავს, მიზეზი ხშირად ეს არის.",
                            "cta_improvement": "DM მომწერე 'REEL' და გაჩვენებ როგორ გადააკეთო ეს opening და CTA.",
                            "improved_version": "Start with the mistake immediately, show one proof moment fast, deliver one fix, then close with one direct DM CTA."
                        }
                    },
                },
                {
                    "reply": "Title:\n3 mistakes that kill conversion\n\nSlide 1:\nMistake 1\nNo clear offer\n\nSlide 2:\nMistake 2\nWeak hook\n\nFinal CTA slide:\nDM me and I will show you what to fix first.",
                    "account_id": "test-account-3",
                    "task_type": "carousel",
                    "parse_status": "parsed",
                    "structured_output": {
                        "title": "3 mistakes that kill conversion",
                        "slides": [
                            {
                                "slide_number": 1,
                                "headline": "Mistake 1",
                                "body": "No clear offer",
                            },
                            {
                                "slide_number": 2,
                                "headline": "Mistake 2",
                                "body": "Weak hook",
                            },
                        ],
                        "cta": "DM me and I will show you what to fix first.",
                    },
                },
                {
                    "reply": "What works:\nStrong visual identity\n\nWhat is weak:\nBio does not explain the offer clearly\n\nWhat to improve first:\nClarify the CTA in the bio\n\nRecommended bio direction:\nState the offer and expected result in one line\n\nContent direction:\nMore educational reels and proof content",
                    "account_id": "test-account-3",
                    "task_type": "profile_audit",
                    "parse_status": "parsed",
                    "structured_output": {
                        "strengths": [
                            "Strong visual identity",
                        ],
                        "weak_points": [
                            "Bio does not explain the offer clearly",
                        ],
                        "quick_fixes": [
                            "Clarify the CTA in the bio",
                        ],
                        "priority_actions": [
                            "State the offer and expected result in one line",
                            "More educational reels and proof content",
                        ],
                        "summary": None,
                    },
                },
                {
                    "reply": "2-week lead generation plan\n\nWeek 1:\nReel about client pain points\nCarousel with a simple framework\n\nWeek 2:\nStory Q&A about objections\n\nPlus:\nRepeat the strongest CTA pattern",
                    "account_id": "test-account-3",
                    "task_type": "content_plan",
                    "parse_status": "parsed",
                    "structured_output": {
                        "plan_title": "2-week lead generation plan",
                        "content_items": [
                            {
                                "day_or_slot": "Week 1",
                                "format": "Reel",
                                "topic": "client pain points",
                                "goal": "drive conversion",
                            },
                            {
                                "day_or_slot": "Week 1",
                                "format": "Carousel",
                                "topic": "a simple framework",
                                "goal": "educate",
                            },
                            {
                                "day_or_slot": "Week 2",
                                "format": "Story",
                                "topic": "Q&A about objections",
                                "goal": "build trust",
                            },
                        ],
                        "summary": "Repeat the strongest CTA pattern",
                    },
                },
                {
                    "reply": "What works:\nClear transformation angle\n\nWhat is weak:\nCTA is too soft\n\nWhy it may perform:\nStrong hook and easy-to-scan message\n\nHow to adapt it for the user's account:\nUse the same direct hook logic, stronger proof, and a DM CTA",
                    "account_id": "test-account-3",
                    "task_type": "link_analysis",
                    "parse_status": "parsed",
                    "structured_output": {
                        "current_state": [
                            "Clear transformation angle",
                            "Strong hook and easy-to-scan message",
                        ],
                        "issues": [
                            "CTA is too soft",
                        ],
                        "recommended_changes": [
                            "Use the same direct hook logic",
                            "Stronger proof",
                            "A DM CTA",
                        ],
                        "best_cta_direction": "A DM CTA",
                        "summary": None,
                    },
                },
            ]
        },
    )

    reply: str
    account_id: str | None = None
    task_type: TaskType
    parse_status: ParseStatus
    structured_output: StructuredOutput | None = None
    knowledge_retrieval_used: bool | None = None
    knowledge_retrieval_top_k: int | None = None
    knowledge_retrieved_count: int = 0
    knowledge_collection_name: str | None = None
    auto_sync_requested: bool = False
    auto_sync_performed: bool = False
    context_fresh: bool | None = None
    context_age_seconds: int | None = None


class AgentTaskCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    structured_output_supported: bool
    fields: list[str] = Field(default_factory=list)


class AgentCapabilitiesFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_sync: bool
    active_account_fallback: bool
    generation_history: bool
    knowledge_packs: bool
    plans_and_limits: bool


class AgentCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "supported_task_types": [
                    {
                        "task_type": "reel_idea",
                        "structured_output_supported": True,
                        "fields": ["ideas"],
                    },
                    {
                        "task_type": "reel_feedback",
                        "structured_output_supported": True,
                        "fields": [
                            "feedback",
                        ],
                    },
                    {
                        "task_type": "caption",
                        "structured_output_supported": True,
                        "fields": ["hook", "body", "cta", "full_caption"],
                    },
                    {
                        "task_type": "performance_summary",
                        "structured_output_supported": True,
                        "fields": [
                            "what_worked",
                            "what_did_not_work",
                            "content_patterns",
                            "recommended_next_moves",
                            "summary",
                        ],
                    },
                ],
                "features": {
                    "auto_sync": True,
                    "active_account_fallback": True,
                    "generation_history": True,
                    "knowledge_packs": False,
                    "plans_and_limits": True,
                },
            }
        },
    )

    supported_task_types: list[AgentTaskCapability] = Field(default_factory=list)
    features: AgentCapabilitiesFeatures


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None

    normalized_value = " ".join(str(value).split()).strip()
    return normalized_value or None


def _coerce_block_text(value: Any) -> str | None:
    if value is None:
        return None

    normalized_lines = []
    for raw_line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        collapsed_line = " ".join(raw_line.split()).strip()
        normalized_lines.append(collapsed_line)

    normalized_value = "\n".join(normalized_lines).strip()
    return normalized_value or None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            normalized = _coerce_text(item)
            if not normalized:
                continue
            normalized = re.sub(r"^[-*]\s*", "", normalized).strip()
            normalized = re.sub(r"^\d+[.)]\s*", "", normalized).strip()
            if normalized:
                cleaned_items.append(normalized)
        return cleaned_items

    normalized_value = _coerce_text(value)
    return [normalized_value] if normalized_value else []


_REEL_SCRIPT_AUX_LABEL_PATTERN = re.compile(
    r"^(?:hook|problem\s*/?\s*angle|problem\s+angle|voice\s*over|voiceover|"
    r"on[-\s]*screen\s+(?:text|copy)|caption|cta)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)

_REEL_SCRIPT_FULL_ANSWER_LABEL_PATTERN = re.compile(
    r"^\s*(?:hook|problem\s*/?\s*angle|problem\s+angle|scene[-\s]*by[-\s]*scene|"
    r"on[-\s]*screen\s+(?:text|copy)|caption|cta)\s*:",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _coerce_reel_script_text(value: Any, *, stop_at_subheading: bool = False) -> str | None:
    cleaned_value = _coerce_block_text(value)
    if not cleaned_value:
        return None

    if stop_at_subheading:
        cleaned_value = re.split(
            r"\n\s*(?:problem\s*/?\s*angle|problem\s+angle|scene[-\s]*by[-\s]*scene(?:\s+script)?|"
            r"structure|voice\s*over|voiceover|on[-\s]*screen\s+(?:text|copy)|caption|cta)\s*:",
            cleaned_value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

    cleaned_value = _REEL_SCRIPT_AUX_LABEL_PATTERN.sub("", cleaned_value).strip()
    return cleaned_value or None


def _coerce_reel_script_items(
    value: Any,
    *,
    drop_aux_labels: bool = False,
    drop_aux_items: bool = False,
) -> list[str]:
    items = _coerce_string_list(value)
    cleaned_items = []
    for item in items:
        cleaned_item = re.sub(r"^(?:scene|shot)\s*\d+\s*[:.)-]\s*", "", item, flags=re.IGNORECASE).strip()
        is_aux_label_item = bool(_REEL_SCRIPT_AUX_LABEL_PATTERN.match(cleaned_item))
        if drop_aux_items and is_aux_label_item:
            continue
        if drop_aux_labels and is_aux_label_item:
            cleaned_item = _REEL_SCRIPT_AUX_LABEL_PATTERN.sub("", cleaned_item).strip()
        if cleaned_item:
            cleaned_items.append(cleaned_item)
    return cleaned_items


def _extract_reel_script_labeled_items(value: Any, label_pattern: str) -> list[str]:
    extracted_items = []
    pattern = re.compile(rf"^(?:{label_pattern})\s*[:\-]\s*", flags=re.IGNORECASE)
    for item in _coerce_string_list(value):
        if not pattern.match(item):
            continue
        cleaned_item = pattern.sub("", item).strip()
        if cleaned_item:
            extracted_items.append(cleaned_item)
    return extracted_items


def _looks_like_full_reel_script(value: str | None) -> bool:
    if not value:
        return False
    return len(_REEL_SCRIPT_FULL_ANSWER_LABEL_PATTERN.findall(value)) >= 3


def _coerce_reel_idea_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    raw_ideas = raw_output.get("ideas")
    if not isinstance(raw_ideas, list):
        raw_ideas = []

    coerced_ideas = []
    coercion_applied = raw_output.get("ideas") != raw_ideas

    for raw_idea in raw_ideas:
        if not isinstance(raw_idea, dict):
            coercion_applied = True
            continue

        coerced_idea = {
            "title": _coerce_text(raw_idea.get("title")),
            "hook": _coerce_text(raw_idea.get("hook")),
            "format_type": _coerce_text(raw_idea.get("format_type")),
            "main_idea": _coerce_text(raw_idea.get("main_idea")),
            "shot_list": _coerce_string_list(raw_idea.get("shot_list")),
            "why_it_can_work": _coerce_block_text(raw_idea.get("why_it_can_work")),
            "caption_idea": _coerce_block_text(raw_idea.get("caption_idea")),
            "cta": _coerce_text(raw_idea.get("cta")),
        }
        if coerced_idea != raw_idea:
            coercion_applied = True

        coerced_ideas.append(coerced_idea)

    return {"ideas": coerced_ideas}, coercion_applied


def _coerce_reel_script_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}

    raw_script = raw_output.get("script")
    if isinstance(raw_script, dict):
        normalized_script = raw_script
        coercion_applied = False
    else:
        normalized_script = {
            "hook": raw_output.get("hook"),
            "problem_angle": raw_output.get("problem_angle") or raw_output.get("problem/angle"),
            "structure": raw_output.get("structure") or raw_output.get("scene_by_scene_script") or raw_output.get("script_sections"),
            "voiceover": raw_output.get("voiceover") or raw_output.get("voice_over"),
            "shot_list": raw_output.get("shot_list") or raw_output.get("scene_by_scene_script") or raw_output.get("script_sections"),
            "on_screen_text": raw_output.get("on_screen_text") or raw_output.get("on_screen_copy"),
            "caption": raw_output.get("caption"),
            "cta": raw_output.get("cta"),
        }
        coercion_applied = True

    problem_angle = (
        _coerce_reel_script_text(normalized_script.get("problem_angle"))
        or next(iter(_extract_reel_script_labeled_items(normalized_script.get("structure"), r"problem\s*/?\s*angle|problem\s+angle")), None)
    )
    on_screen_text = _coerce_reel_script_items(normalized_script.get("on_screen_text"), drop_aux_labels=True)
    if not on_screen_text:
        on_screen_text = _extract_reel_script_labeled_items(
            normalized_script.get("structure"),
            r"on[-\s]*screen\s+(?:text|copy)",
        )

    voiceover = _coerce_reel_script_text(normalized_script.get("voiceover"))
    if _looks_like_full_reel_script(voiceover):
        voiceover = None

    coerced_output = {
        "script": {
            "hook": _coerce_reel_script_text(normalized_script.get("hook"), stop_at_subheading=True),
            "problem_angle": problem_angle,
            "structure": _coerce_reel_script_items(normalized_script.get("structure"), drop_aux_items=True),
            "voiceover": voiceover,
            "shot_list": _coerce_reel_script_items(normalized_script.get("shot_list"), drop_aux_items=True),
            "on_screen_text": on_screen_text,
            "caption": _coerce_reel_script_text(normalized_script.get("caption")),
            "cta": _coerce_reel_script_text(normalized_script.get("cta")),
        }
    }
    return coerced_output, coercion_applied or coerced_output != raw_output


def _coerce_reel_feedback_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}

    raw_feedback = raw_output.get("feedback")
    if isinstance(raw_feedback, dict):
        normalized_feedback = raw_feedback
        coercion_applied = False
    else:
        normalized_feedback = {
            "what_works": raw_output.get("what_works"),
            "what_hurts": raw_output.get("what_hurts"),
            "retention_issues": raw_output.get("retention_issues") or raw_output.get("retention_risks"),
            "hook_improvement": raw_output.get("hook_improvement") or raw_output.get("better_hook"),
            "cta_improvement": raw_output.get("cta_improvement") or raw_output.get("better_cta"),
            "improved_version": raw_output.get("improved_version"),
        }
        coercion_applied = True

    coerced_output = {
        "feedback": {
            "what_works": _coerce_string_list(normalized_feedback.get("what_works")),
            "what_hurts": _coerce_string_list(normalized_feedback.get("what_hurts")),
            "retention_issues": _coerce_string_list(normalized_feedback.get("retention_issues")),
            "hook_improvement": _coerce_text(normalized_feedback.get("hook_improvement")),
            "cta_improvement": _coerce_text(normalized_feedback.get("cta_improvement")),
            "improved_version": _coerce_block_text(normalized_feedback.get("improved_version")),
        }
    }
    return coerced_output, coercion_applied or coerced_output != raw_output


def _coerce_caption_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    coerced_output = {
        "hook": _coerce_text(raw_output.get("hook")),
        "body": _coerce_block_text(raw_output.get("body")),
        "cta": _coerce_text(raw_output.get("cta")),
        "full_caption": _coerce_block_text(raw_output.get("full_caption")),
    }
    return coerced_output, coerced_output != raw_output


def _coerce_carousel_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    raw_slides = raw_output.get("slides")
    if not isinstance(raw_slides, list):
        raw_slides = []

    coerced_slides = []
    coercion_applied = raw_output.get("slides") != raw_slides

    for raw_slide in raw_slides:
        if not isinstance(raw_slide, dict):
            coercion_applied = True
            continue

        slide_number = raw_slide.get("slide_number")
        try:
            resolved_slide_number = int(slide_number) if slide_number is not None else None
        except (TypeError, ValueError):
            resolved_slide_number = None

        coerced_slide = {
            "slide_number": resolved_slide_number,
            "headline": _coerce_text(raw_slide.get("headline")),
            "body": _coerce_block_text(raw_slide.get("body")),
        }
        if coerced_slide != raw_slide:
            coercion_applied = True

        coerced_slides.append(coerced_slide)

    coerced_output = {
        "title": _coerce_text(raw_output.get("title")),
        "slides": coerced_slides,
        "cta": _coerce_text(raw_output.get("cta")),
    }
    return coerced_output, coercion_applied or coerced_output != raw_output


def _coerce_profile_audit_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    coerced_output = {
        "strengths": _coerce_string_list(raw_output.get("strengths")),
        "weak_points": _coerce_string_list(raw_output.get("weak_points")),
        "quick_fixes": _coerce_string_list(raw_output.get("quick_fixes")),
        "recommended_bio_direction": _coerce_block_text(raw_output.get("recommended_bio_direction")),
        "content_direction": _coerce_string_list(raw_output.get("content_direction")),
        "priority_actions": _coerce_string_list(raw_output.get("priority_actions")),
        "summary": _coerce_block_text(raw_output.get("summary")),
    }
    return coerced_output, coerced_output != raw_output


def _coerce_content_plan_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    raw_items = raw_output.get("content_items")
    if not isinstance(raw_items, list):
        raw_items = []

    coerced_items = []
    coercion_applied = raw_output.get("content_items") != raw_items

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            coercion_applied = True
            continue

        coerced_item = {
            "day_or_slot": _coerce_text(raw_item.get("day_or_slot")),
            "format": _coerce_text(raw_item.get("format")),
            "topic": _coerce_text(raw_item.get("topic")),
            "goal": _coerce_text(raw_item.get("goal")),
        }
        if coerced_item != raw_item:
            coercion_applied = True

        coerced_items.append(coerced_item)

    coerced_output = {
        "plan_title": _coerce_text(raw_output.get("plan_title")),
        "content_items": coerced_items,
        "best_content_mix": _coerce_string_list(raw_output.get("best_content_mix")),
        "hook_ideas": _coerce_string_list(raw_output.get("hook_ideas")),
        "cta_ideas": _coerce_string_list(raw_output.get("cta_ideas")),
        "summary": _coerce_block_text(raw_output.get("summary")),
    }
    return coerced_output, coercion_applied or coerced_output != raw_output


def _coerce_link_analysis_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    coerced_output = {
        "current_state": _coerce_string_list(raw_output.get("current_state")),
        "issues": _coerce_string_list(raw_output.get("issues")),
        "recommended_changes": _coerce_string_list(raw_output.get("recommended_changes")),
        "best_cta_direction": _coerce_text(raw_output.get("best_cta_direction")),
        "summary": _coerce_block_text(raw_output.get("summary")),
    }
    return coerced_output, coerced_output != raw_output


def _coerce_performance_summary_structured_output(structured_output: Any) -> tuple[dict[str, Any], bool]:
    raw_output = structured_output if isinstance(structured_output, dict) else {}
    coerced_output = {
        "what_worked": _coerce_string_list(raw_output.get("what_worked")),
        "what_did_not_work": _coerce_string_list(raw_output.get("what_did_not_work")),
        "content_patterns": _coerce_string_list(raw_output.get("content_patterns")),
        "best_opportunities": _coerce_string_list(raw_output.get("best_opportunities")),
        "recommended_next_moves": _coerce_string_list(raw_output.get("recommended_next_moves")),
        "summary": _coerce_text(raw_output.get("summary")),
    }
    return coerced_output, coerced_output != raw_output


def validate_structured_output_payload(
    task_type: str,
    parse_status: ParseStatus | str | None,
    structured_output: Any,
) -> tuple[StructuredOutput | dict[str, Any] | None, ParseStatus, str | None]:
    if structured_output is None:
        return None, "raw_only", None

    strict_models = {
        "reel_idea": (ReelIdeaStructuredOutput, _coerce_reel_idea_structured_output),
        "reel_script": (ReelScriptStructuredOutput, _coerce_reel_script_structured_output),
        "reel_feedback": (ReelFeedbackStructuredOutput, _coerce_reel_feedback_structured_output),
        "caption": (CaptionStructuredOutput, _coerce_caption_structured_output),
        "carousel": (CarouselStructuredOutput, _coerce_carousel_structured_output),
        "profile_audit": (ProfileAuditStructuredOutput, _coerce_profile_audit_structured_output),
        "content_plan": (ContentPlanStructuredOutput, _coerce_content_plan_structured_output),
        "link_analysis": (LinkAnalysisStructuredOutput, _coerce_link_analysis_structured_output),
        "performance_summary": (PerformanceSummaryStructuredOutput, _coerce_performance_summary_structured_output),
    }
    model_and_coercer = strict_models.get(task_type)

    if not model_and_coercer:
        if isinstance(structured_output, dict) and structured_output:
            resolved_parse_status: ParseStatus = parse_status if parse_status in {"parsed", "partial"} else "partial"
            return structured_output, resolved_parse_status, "generic"
        return None, "raw_only", None

    model_class, coercer = model_and_coercer
    coerced_output, _ = coercer(structured_output)

    try:
        validated_output = model_class.model_validate(coerced_output)
    except ValidationError:
        return None, "raw_only", model_class.__name__

    validated_payload = validated_output.model_dump()
    has_meaningful_content = any(validated_payload.values())
    if not has_meaningful_content:
        return None, "raw_only", model_class.__name__

    if parse_status in {"parsed", "partial"}:
        resolved_parse_status: ParseStatus = parse_status
    else:
        resolved_parse_status = "partial"

    return validated_payload, resolved_parse_status, model_class.__name__
