"""Composites carousel slides: Gemini background + text drawn in code.

Image models render text unreliably, and Georgian especially, so the words never
go through the image model. Gemini produces a text-free background and this
service draws the headline and body on top at a known size and position.

Georgian drives three decisions here that a Latin-only renderer would get wrong:

* **No uppercasing.** Georgian Mkhedruli is unicameral -- it has no capital
  letters. ``str.upper()`` is a no-op on Georgian and would silently apply to the
  Latin half of a mixed string only, producing inconsistent typography.
* **Text is measured, never estimated.** Georgian sets longer than English for
  the same meaning, so character-count wrapping overflows. Every line is measured
  with the real font, and the size steps down until the block fits.
* **A bundled font, not a system font.** The container has no Georgian font
  installed; Pillow does not raise on a missing glyph, it draws blank boxes. The
  font ships in the repo.
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
DEFAULT_BOLD_FONT = ASSETS_DIR / "NotoSansGeorgian-Bold.ttf"
DEFAULT_REGULAR_FONT = ASSETS_DIR / "NotoSansGeorgian-Regular.ttf"


class CarouselRenderError(RuntimeError):
    """A slide could not be composited."""


@dataclass(frozen=True)
class BrandTemplate:
    """Visual specification for a carousel slide.

    Instagram portrait at 1080x1350. Colours are placeholders until the owner
    supplies brand values -- override via the environment rather than editing
    this file.
    """

    width: int = 1080
    height: int = 1350
    margin: int = 96
    background_color: tuple[int, int, int] = (18, 18, 20)
    headline_color: tuple[int, int, int] = (255, 255, 255)
    body_color: tuple[int, int, int] = (228, 228, 232)
    accent_color: tuple[int, int, int] = (255, 184, 76)
    headline_size: int = 78
    body_size: int = 42
    min_headline_size: int = 40
    min_body_size: int = 26
    line_spacing: float = 1.28
    # Darkens a photographic background so light text stays legible on it.
    scrim_opacity: int = 140
    bold_font_path: Path = field(default_factory=lambda: DEFAULT_BOLD_FONT)
    regular_font_path: Path = field(default_factory=lambda: DEFAULT_REGULAR_FONT)

    @classmethod
    def from_env(cls) -> "BrandTemplate":
        def color(name: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
            raw = os.getenv(name, "").strip().lstrip("#")
            if len(raw) != 6:
                return fallback
            try:
                return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
            except ValueError:
                return fallback

        def path(name: str, fallback: Path) -> Path:
            raw = os.getenv(name, "").strip()
            return Path(raw) if raw else fallback

        return cls(
            background_color=color("BRAND_BG_COLOR", cls.background_color),
            headline_color=color("BRAND_HEADLINE_COLOR", cls.headline_color),
            body_color=color("BRAND_BODY_COLOR", cls.body_color),
            accent_color=color("BRAND_ACCENT_COLOR", cls.accent_color),
            bold_font_path=path("BRAND_BOLD_FONT", DEFAULT_BOLD_FONT),
            regular_font_path=path("BRAND_REGULAR_FONT", DEFAULT_REGULAR_FONT),
        )


class CarouselRenderService:
    def __init__(self, template: BrandTemplate | None = None):
        self.template = template or BrandTemplate.from_env()

    # ------------------------------------------------------------------
    # fonts
    # ------------------------------------------------------------------
    def _load_font(self, path: Path, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError as exc:
            raise CarouselRenderError(
                f"Could not load font '{path}'. A font with Georgian coverage must be "
                "present; the default ships in backend/app/assets/fonts."
            ) from exc

    def _text_width(self, draw: ImageDraw.ImageDraw, text: str, font) -> int:
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left

    def _line_height(self, font) -> int:
        ascent, descent = font.getmetrics()
        return int((ascent + descent) * self.template.line_spacing)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
        """Greedy wrap on measured width.

        Wrapping by character count would overflow on Georgian, which sets wider
        than English for equivalent content.
        """
        lines: list[str] = []
        for paragraph in text.splitlines():
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if self._text_width(draw, candidate, font) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines

    def _fit(self, draw, text, font_path: Path, start_size: int, min_size: int, max_width: int, max_height: int):
        """Step the font size down until the wrapped block fits its box."""
        size = start_size
        while size >= min_size:
            font = self._load_font(font_path, size)
            lines = self._wrap(draw, text, font, max_width)
            if len(lines) * self._line_height(font) <= max_height:
                return font, lines
            size -= 2

        font = self._load_font(font_path, min_size)
        lines = self._wrap(draw, text, font, max_width)
        allowed = max(1, max_height // self._line_height(font))
        if len(lines) > allowed:
            # Truncating beats overflowing off-canvas; the ellipsis makes the cut
            # visible rather than silently losing the tail.
            lines = lines[:allowed]
            lines[-1] = lines[-1].rstrip() + "…"
        return font, lines

    # ------------------------------------------------------------------
    # background
    # ------------------------------------------------------------------
    def _background(self, background: bytes | None) -> Image.Image:
        template = self.template
        canvas = Image.new("RGB", (template.width, template.height), template.background_color)
        if not background:
            return canvas

        try:
            source = Image.open(io.BytesIO(background))
            source.load()
            source = source.convert("RGB")
        except Exception:
            # A broken background must not fail the whole slide -- the text is
            # the payload, the image is decoration.
            logger.warning("Unreadable carousel background; falling back to solid colour")
            return canvas

        canvas.paste(self._cover(source, template.width, template.height))

        scrim = Image.new("RGBA", canvas.size, (0, 0, 0, template.scrim_opacity))
        return Image.alpha_composite(canvas.convert("RGBA"), scrim).convert("RGB")

    def _cover(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Scale-and-crop to fill, preserving aspect ratio."""
        scale = max(width / image.width, height / image.height)
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def render_slide(
        self,
        *,
        headline: str,
        body: str = "",
        slide_number: int | None = None,
        total_slides: int | None = None,
        background: bytes | None = None,
    ) -> bytes:
        """Composite one slide and return PNG bytes."""
        template = self.template
        canvas = self._background(background)
        draw = ImageDraw.Draw(canvas)

        max_width = template.width - (template.margin * 2)
        content_top = template.margin
        content_bottom = template.height - template.margin

        # Slide counter, drawn first so the text block can start beneath it.
        if slide_number is not None:
            counter = f"{slide_number}" if not total_slides else f"{slide_number}/{total_slides}"
            counter_font = self._load_font(template.regular_font_path, 34)
            draw.text((template.margin, content_top), counter, font=counter_font, fill=template.accent_color)
            content_top += self._line_height(counter_font) + 24

        available = content_bottom - content_top
        # Headline gets the upper share; body takes what's left.
        headline_box = int(available * (0.45 if body.strip() else 1.0))

        # Deliberately no .upper(): Georgian is unicameral.
        headline_font, headline_lines = self._fit(
            draw, headline.strip(), template.bold_font_path,
            template.headline_size, template.min_headline_size, max_width, headline_box,
        )

        y = content_top
        for line in headline_lines:
            draw.text((template.margin, y), line, font=headline_font, fill=template.headline_color)
            y += self._line_height(headline_font)

        if body.strip():
            y += 28
            body_font, body_lines = self._fit(
                draw, body.strip(), template.regular_font_path,
                template.body_size, template.min_body_size, max_width, content_bottom - y,
            )
            for line in body_lines:
                draw.text((template.margin, y), line, font=body_font, fill=template.body_color)
                y += self._line_height(body_font)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
