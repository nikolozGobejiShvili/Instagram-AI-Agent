"""Carousel slide rendering.

The failure this suite exists to prevent: Pillow does not raise on a missing
glyph, it draws blank boxes. A carousel of Georgian tofu looks like a working
feature to every automated check that only asserts "a PNG came back".
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.carousel_render_service import (  # noqa: E402
    DEFAULT_BOLD_FONT,
    DEFAULT_REGULAR_FONT,
    BrandTemplate,
    CarouselRenderError,
    CarouselRenderService,
)

GEORGIAN_HEADLINE = "3 ძლიერი Reels იდეა"
GEORGIAN_BODY = "დაიწყე ძლიერი კაუჭით და დაასრულე ერთი მკაფიო მოწოდებით."
TOFU_PROBE = "￾"  # permanently unassigned codepoint -> always .notdef


@pytest.fixture()
def service():
    return CarouselRenderService(BrandTemplate())


def _open(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png))
    image.load()
    return image


def test_bundled_fonts_are_present():
    """The container has no Georgian system font — these must ship in the repo."""
    assert DEFAULT_BOLD_FONT.exists(), f"missing bundled font: {DEFAULT_BOLD_FONT}"
    assert DEFAULT_REGULAR_FONT.exists(), f"missing bundled font: {DEFAULT_REGULAR_FONT}"


@pytest.mark.parametrize("font_path", [DEFAULT_BOLD_FONT, DEFAULT_REGULAR_FONT])
def test_every_georgian_glyph_renders_rather_than_tofu(font_path):
    """Compare each glyph against a known-missing codepoint.

    Ink coverage alone proves nothing: a tofu box is ink too.
    """
    font = ImageFont.truetype(str(font_path), 64)

    def bitmap(text):
        img = Image.new("L", (200, 140), 0)
        ImageDraw.Draw(img).text((10, 10), text, font=font, fill=255)
        return img.tobytes()

    tofu = bitmap(TOFU_PROBE)
    blank = Image.new("L", (200, 140), 0).tobytes()

    missing = [
        ch for ch in sorted(set(GEORGIAN_HEADLINE + GEORGIAN_BODY))
        if not ch.isspace() and bitmap(ch) in (tofu, blank)
    ]
    assert not missing, f"{font_path.name} renders tofu for: {''.join(missing)}"


def test_slide_is_a_portrait_png_at_instagram_size(service):
    image = _open(service.render_slide(headline=GEORGIAN_HEADLINE, body=GEORGIAN_BODY))
    assert image.format == "PNG"
    assert image.size == (1080, 1350)


def test_georgian_text_actually_lands_on_the_slide(service):
    """A slide with text must differ from an empty one, and not by tofu."""
    with_text = _open(service.render_slide(headline=GEORGIAN_HEADLINE, body=GEORGIAN_BODY)).convert("L")
    without_text = _open(service.render_slide(headline="", body="")).convert("L")

    changed = sum(1 for a, b in zip(with_text.getdata(), without_text.getdata()) if a != b)
    assert changed > 3000, f"only {changed} pixels changed — text did not render"


def test_headline_is_not_uppercased(service):
    """Georgian is unicameral; .upper() would only affect the Latin half."""
    mixed = _open(service.render_slide(headline="reels იდეა")).convert("L")
    upper = _open(service.render_slide(headline="REELS იდეა")).convert("L")

    assert mixed.tobytes() != upper.tobytes(), "headline appears to be uppercased"


def test_long_georgian_text_stays_inside_the_canvas(service):
    """Georgian sets longer than English — wrapping must measure, not count chars."""
    long_body = " ".join([GEORGIAN_BODY] * 12)
    image = _open(service.render_slide(headline=GEORGIAN_HEADLINE, body=long_body)).convert("L")

    width, height = image.size
    margin = BrandTemplate().margin
    pixels = image.load()

    # Nothing may be drawn in the outer margin strips.
    for y in range(0, height, 7):
        for x in list(range(0, margin // 2)) + list(range(width - margin // 2, width)):
            assert pixels[x, y] < 60, f"text overflowed the margin at ({x}, {y})"


def test_background_image_is_composited_under_the_text(service):
    """A supplied background must change the slide, and text must survive it."""
    red = Image.new("RGB", (800, 1000), (200, 30, 30))
    buffer = io.BytesIO()
    red.save(buffer, format="PNG")

    with_bg = _open(service.render_slide(headline=GEORGIAN_HEADLINE, background=buffer.getvalue()))
    without_bg = _open(service.render_slide(headline=GEORGIAN_HEADLINE))

    assert with_bg.tobytes() != without_bg.tobytes()
    # The scrim darkens the photo, so the mean stays well below the raw red.
    reds = [px[0] for px in with_bg.convert("RGB").getdata()]
    assert 20 < (sum(reds) / len(reds)) < 200


def test_a_corrupt_background_degrades_instead_of_failing(service):
    """The text is the payload; a bad image must not lose the whole slide."""
    png = service.render_slide(headline=GEORGIAN_HEADLINE, background=b"not-an-image")
    assert _open(png).size == (1080, 1350)


def test_slide_counter_is_drawn_when_supplied(service):
    numbered = _open(service.render_slide(headline=GEORGIAN_HEADLINE, slide_number=2, total_slides=7)).convert("L")
    plain = _open(service.render_slide(headline=GEORGIAN_HEADLINE)).convert("L")
    assert numbered.tobytes() != plain.tobytes()


def test_missing_font_fails_loudly(tmp_path):
    """Silence here would ship blank slides."""
    template = BrandTemplate(bold_font_path=tmp_path / "nope.ttf", regular_font_path=tmp_path / "nope.ttf")
    with pytest.raises(CarouselRenderError, match="Georgian coverage"):
        CarouselRenderService(template).render_slide(headline=GEORGIAN_HEADLINE)


def test_brand_colours_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("BRAND_BG_COLOR", "#0A2540")
    assert BrandTemplate.from_env().background_color == (10, 37, 64)

    monkeypatch.setenv("BRAND_BG_COLOR", "not-a-colour")
    assert BrandTemplate.from_env().background_color == BrandTemplate().background_color
