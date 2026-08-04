from pydantic import BaseModel, ConfigDict, Field


class InternalGenerationDebugResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "user-1",
                "task_type": "reel_idea",
                "account_id": "acct-1",
                "used_system_knowledge": True,
                "matched_knowledge_domain": "reels",
                "matched_knowledge_pack_ids": ["kp_123abc456def"],
                "retrieved_chunk_count": 4,
                "retrieved_chunk_titles": [
                    "KNOWLEDGE MODULE 1 VIRAL IDEA MECHANICS",
                    "KNOWLEDGE MODULE 2 TREND STRUCTURE",
                ],
                "knowledge_retrieval_used": True,
                "knowledge_retrieval_top_k": 5,
                "knowledge_retrieved_count": 4,
                "knowledge_collection_name": "mariami_reels_playbook_v1",
                "used_langflow": False,
                "model_provider": "openai",
                "model_name": "gpt-5.2",
                "prompt_section_names": [
                    "base_system_instruction",
                    "task_instruction",
                    "reels_high_priority_instruction",
                    "internal_strategy_context",
                    "user_request",
                ],
                "prompt_token_estimate": 1320,
                "retry_count": 0,
                "rate_limited": False,
                "parse_status": "parsed",
                "created_at": "2026-04-27T12:30:00+00:00",
            }
        },
    )

    user_id: str | None = None
    task_type: str
    account_id: str | None = None
    used_system_knowledge: bool | None = None
    matched_knowledge_domain: str | None = None
    matched_knowledge_pack_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    retrieved_chunk_titles: list[str] = Field(default_factory=list)
    knowledge_retrieval_used: bool | None = None
    knowledge_retrieval_top_k: int | None = None
    knowledge_retrieved_count: int = 0
    knowledge_collection_name: str | None = None
    used_langflow: bool | None = None
    model_provider: str | None = None
    model_name: str | None = None
    prompt_section_names: list[str] = Field(default_factory=list)
    prompt_token_estimate: int | None = None
    retry_count: int = 0
    rate_limited: bool = False
    parse_status: str | None = None
    created_at: str
