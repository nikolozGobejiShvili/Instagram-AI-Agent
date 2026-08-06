"""Teaching material has to survive a deploy, and reach every function.

Two properties, both of which the knowledge system lacked while looking like it
worked:

1. Uploaded packs, their chunks and their source files defaulted to a directory
   inside the container image. Every release wiped them, and nothing said so --
   the agent simply went back to knowing nothing.
2. Ingestion only fired for ``domain == "reels"``. Material uploaded to teach
   carousels, audits or planning was stored and never vectorised, so it sat on
   disk looking uploaded while retrieval found nothing.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.deterministic_knowledge_retrieval_service import RAG_TASK_TYPES  # noqa: E402
from app.services.knowledge_pack_service import KnowledgePackService  # noqa: E402


# ------------------------------------------------------------- persistence


def test_the_store_follows_its_configured_directory(monkeypatch, tmp_path):
    """Without this the store sits inside the image, which a deploy replaces."""
    monkeypatch.setenv("KNOWLEDGE_PACK_DIR", str(tmp_path / "volume"))

    service = KnowledgePackService()

    for path in (service.data_file, service.chunks_file, service.storage_dir):
        assert str(path).startswith(str(tmp_path / "volume")), path


def test_an_unset_directory_still_starts(monkeypatch):
    """A missing variable must not stop the service booting -- it degrades to
    the in-image default, which is wrong but recoverable, rather than crashing
    every request."""
    monkeypatch.delenv("KNOWLEDGE_PACK_DIR", raising=False)

    service = KnowledgePackService()

    assert service.data_file.name == "knowledge_packs.json"


def test_uploaded_material_is_readable_from_a_second_instance(monkeypatch, tmp_path):
    """Proves the material is on disk rather than in process memory: a restarted
    container is a new instance reading the same directory."""
    monkeypatch.setenv("KNOWLEDGE_PACK_DIR", str(tmp_path / "volume"))

    uploaded = KnowledgePackService().upload_pack(
        owner_user_id="system",
        title="Carousel playbook",
        description="how strong carousels are structured",
        supported_task_types=["carousel"],
        uploaded_files=[{
            "file_name": "carousel.md",
            "content": b"# Hooks\nOpen with tension.\n\n# CTA\nOne CTA only.",
        }],
        scope="system",
        domain="carousel",
        visibility="internal",
        status="active",
    )

    reopened = KnowledgePackService()
    packs = reopened.list_packs(scope="system", domain="carousel", visibility="internal")

    assert [p["knowledge_pack_id"] for p in packs["knowledge_packs"]] == [uploaded["knowledge_pack_id"]]


# ------------------------------------------------------------ every function


@pytest.mark.parametrize("task_type", sorted(RAG_TASK_TYPES))
def test_material_can_be_attached_to_every_task_that_retrieves_it(task_type):
    """Retrieval and upload have to agree on the task list.

    A task that retrieves knowledge but cannot have any uploaded is a dead
    lookup; the reverse is material that is stored and never read.
    """
    assert task_type in KnowledgePackService.SUPPORTED_TASK_TYPES


def test_retrieval_covers_more_than_reels():
    """The point of the exercise: the agent is taught for every function, not
    just the three it started with."""
    assert {"carousel", "content_plan", "profile_audit", "performance_summary"} <= RAG_TASK_TYPES
