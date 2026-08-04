"""Carousel pipeline: copy -> backgrounds -> composited slides -> served assets."""
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import carousel_media as media_route  # noqa: E402
from app.services.agent_response_formatter_service import (  # noqa: E402
    MAX_CAROUSEL_SLIDES,
    AgentResponseFormatterService,
)
from app.services.carousel_media_service import CarouselMediaService  # noqa: E402
from app.services.carousel_pipeline_service import CarouselPipelineService  # noqa: E402
from app.services.langflow_service import LangflowService  # noqa: E402

GEORGIAN_HEADLINE = "3 ძლიერი Reels იდეა"


def _png(color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (600, 750), color).save(buffer, format="PNG")
    return buffer.getvalue()


class StubImageProvider:
    def __init__(self, fail_on=()):
        self.prompts = []
        self._fail_on = set(fail_on)

    def generate_image(self, *, prompt, aspect_ratio=None):
        self.prompts.append(prompt)
        if len(self.prompts) in self._fail_on:
            raise RuntimeError("provider exploded")
        return _png()


@pytest.fixture()
def pipeline(tmp_path):
    return CarouselPipelineService(
        media_service=CarouselMediaService(media_dir=tmp_path / "media"),
        image_provider=StubImageProvider(),
    )


def _structured(n=3, with_prompts=True):
    return {
        "title": "კარუსელი",
        "slides": [
            {
                "slide_number": i,
                "headline": f"{GEORGIAN_HEADLINE} {i}",
                "body": "დაიწყე ძლიერი კაუჭით.",
                **({"image_prompt": f"abstract backdrop {i}"} if with_prompts else {}),
            }
            for i in range(1, n + 1)
        ],
        "cta": "მიწერე DM-ში",
    }


# ------------------------------------------------------------ the 5-slide cap


def test_parser_no_longer_stops_at_five_slides():
    """The old range(1, 6) silently discarded slides 6+."""
    body = "Title: ცხრა იდეა\n"
    for n in range(1, 10):
        body += f"Slide {n}:\nსათაური {n}\nტექსტი {n}\n"
    body += "Final CTA slide:\nმიწერე"

    structured, status = AgentResponseFormatterService()._parse_carousel(body)

    assert status == "parsed"
    assert [s["slide_number"] for s in structured["slides"]] == list(range(1, 10))


def test_prompt_contract_matches_the_requested_slide_count():
    service = LangflowService()
    assert service._carousel_output_format(8).count("Slide ") == 8
    # Asking for more than the parser accepts would generate slides that are
    # paid for and then dropped.
    assert service._carousel_output_format(999).count("Slide ") == MAX_CAROUSEL_SLIDES
    assert service._carousel_output_format(1).count("Slide ") == 2


def test_slide_count_is_clamped_by_tier(pipeline):
    assert pipeline.resolve_slide_count(None) == 5
    assert pipeline.resolve_slide_count(8) == 8
    assert pipeline.resolve_slide_count(50) == MAX_CAROUSEL_SLIDES
    assert pipeline.resolve_slide_count(10, tier_maximum=6) == 6


# ----------------------------------------------------------------- rendering


def test_every_slide_gets_a_rendered_asset(pipeline):
    result = pipeline.render_carousel(structured_output=_structured(4))

    assert len(result["slides"]) == 4
    assert result["image_failures"] == []
    for slide in result["slides"]:
        assert slide["image_url"].startswith("/api/v1/carousel-media/")
        assert slide["template_id"] == "default"
        # Original fields must survive — existing consumers read these.
        assert slide["headline"] and slide["body"]


def test_art_direction_reaches_the_image_provider(pipeline):
    pipeline.render_carousel(structured_output=_structured(3))
    assert pipeline._image_provider.prompts == [
        "abstract backdrop 1", "abstract backdrop 2", "abstract backdrop 3"
    ]


def test_a_failed_background_still_produces_a_slide(tmp_path):
    """Losing decoration must not lose the copy the customer paid for."""
    provider = StubImageProvider(fail_on={2})
    pipeline = CarouselPipelineService(
        media_service=CarouselMediaService(media_dir=tmp_path / "media"),
        image_provider=provider,
    )

    result = pipeline.render_carousel(structured_output=_structured(3))

    assert len(result["slides"]) == 3, "a failed background must not drop the slide"
    assert [f["slide_number"] for f in result["image_failures"]] == [2]
    assert all(s["image_url"] for s in result["slides"])


def test_text_only_carousel_when_no_image_provider(tmp_path):
    pipeline = CarouselPipelineService(
        media_service=CarouselMediaService(media_dir=tmp_path / "media"),
        image_provider=None,
    )
    result = pipeline.render_carousel(structured_output=_structured(2), generate_images=False)
    assert len(result["slides"]) == 2
    assert all(s["image_url"] for s in result["slides"])


# -------------------------------------------------------------- media store


def test_stored_asset_round_trips(tmp_path):
    service = CarouselMediaService(media_dir=tmp_path / "media")
    asset_id = service.store(_png())
    assert service.load(asset_id) == _png()
    assert service.exists(asset_id)


def test_path_traversal_ids_are_refused(tmp_path):
    service = CarouselMediaService(media_dir=tmp_path / "media")
    for bad in ["../../etc/passwd", "..\\..\\secret", "a/b", "", "not-a-hex-id"]:
        assert service.load(bad) is None
        assert service.exists(bad) is False


def test_empty_image_is_not_stored(tmp_path):
    with pytest.raises(ValueError):
        CarouselMediaService(media_dir=tmp_path / "media").store(b"")


# ----------------------------------------------------------------- endpoint


def test_endpoint_serves_the_image(tmp_path, monkeypatch):
    service = CarouselMediaService(media_dir=tmp_path / "media")
    monkeypatch.setattr(media_route, "carousel_media_service", service)
    asset_id = service.store(_png())

    response = TestClient(app).get(f"/api/v1/carousel-media/{asset_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]
    assert Image.open(io.BytesIO(response.content)).size == (600, 750)


def test_endpoint_404s_for_unknown_and_malformed_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(media_route, "carousel_media_service", CarouselMediaService(media_dir=tmp_path / "media"))
    client = TestClient(app)

    assert client.get("/api/v1/carousel-media/" + "0" * 32).status_code == 404
    assert client.get("/api/v1/carousel-media/not-a-real-id").status_code == 404


# ------------------------------------------------------------ job wiring


def test_carousel_requests_are_routed_to_the_carousel_job_kind(monkeypatch, tmp_path):
    from app.api.routes import generation_jobs as jobs_route
    from app.services.billing_service import BillingService
    from app.services.job_service import JobService

    job_service = JobService(db_path=tmp_path / "jobs.sqlite3")
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(jobs_route, "job_service", job_service)
    monkeypatch.setattr(jobs_route, "billing_service", billing)
    billing.set_plan("carousel-user", {"current_plan": "pro"})

    client = TestClient(app)
    carousel = client.post("/api/v1/generation-jobs", json={
        "user_id": "carousel-user", "task_type": "carousel", "message": "გააკეთე კარუსელი"})
    caption = client.post("/api/v1/generation-jobs", json={
        "user_id": "carousel-user", "task_type": "caption", "message": "კაპშენი"})

    assert carousel.json()["kind"] == "carousel_generation"
    assert caption.json()["kind"] == "agent_generation"
