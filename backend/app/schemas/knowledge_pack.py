from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent import TaskType


KnowledgePackStatus = Literal["active", "inactive"]
KnowledgePackScope = Literal["system", "user"]
KnowledgePackVisibility = Literal["internal", "user"]


class KnowledgePackFileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    file_name: str
    content_type: str | None = None
    extension: str
    file_size_bytes: int
    chunk_count: int
    uploaded_at: str


class KnowledgePackResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "knowledge_pack_id": "kp_01",
                    "owner_user_id": "system",
                    "title": "Mariami Reels Playbook",
                    "description": "Internal reels methodology for hooks, retention, and CTA clarity.",
                    "scope": "system",
                    "domain": "reels",
                    "visibility": "internal",
                    "status": "active",
                    "supported_task_types": [
                        "reel_idea",
                        "reel_script",
                        "reel_feedback",
                    ],
                    "uploaded_files": [
                        {
                            "file_id": "kpf_01",
                            "file_name": "mariami-reels-playbook.md",
                            "content_type": "text/markdown",
                            "extension": ".md",
                            "file_size_bytes": 18244,
                            "chunk_count": 7,
                            "uploaded_at": "2026-04-21T10:00:00+00:00",
                        }
                    ],
                    "total_chunks": 7,
                    "created_at": "2026-04-21T10:00:00+00:00",
                    "updated_at": "2026-04-21T10:00:00+00:00",
                }
            ]
        },
    )

    knowledge_pack_id: str
    owner_user_id: str
    title: str
    description: str | None = None
    scope: KnowledgePackScope
    domain: str | None = None
    visibility: KnowledgePackVisibility
    status: KnowledgePackStatus
    supported_task_types: list[TaskType] = Field(default_factory=list)
    uploaded_files: list[KnowledgePackFileMetadata] = Field(default_factory=list)
    total_chunks: int = 0
    created_at: str
    updated_at: str


class KnowledgePackListResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "scope": "system",
                "domain": "reels",
                "visibility": "internal",
                "knowledge_packs": [
                    {
                        "knowledge_pack_id": "kp_01",
                        "owner_user_id": "system",
                        "title": "Mariami Reels Playbook",
                        "description": "Internal reels methodology for hooks, retention, and CTA clarity.",
                        "scope": "system",
                        "domain": "reels",
                        "visibility": "internal",
                        "status": "active",
                        "supported_task_types": [
                            "reel_idea",
                            "reel_script",
                            "reel_feedback",
                        ],
                        "uploaded_files": [
                            {
                                "file_id": "kpf_01",
                                "file_name": "mariami-reels-playbook.md",
                                "content_type": "text/markdown",
                                "extension": ".md",
                                "file_size_bytes": 18244,
                                "chunk_count": 7,
                                "uploaded_at": "2026-04-21T10:00:00+00:00",
                            }
                        ],
                        "total_chunks": 7,
                        "created_at": "2026-04-21T10:00:00+00:00",
                        "updated_at": "2026-04-21T10:00:00+00:00",
                    }
                ],
            }
        },
    )

    scope: KnowledgePackScope | None = None
    domain: str | None = None
    visibility: KnowledgePackVisibility | None = None
    knowledge_packs: list[KnowledgePackResponse] = Field(default_factory=list)


class KnowledgePackDeleteResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "knowledge_pack_id": "kp_01",
                "deleted": True,
            }
        },
    )

    knowledge_pack_id: str
    deleted: bool
