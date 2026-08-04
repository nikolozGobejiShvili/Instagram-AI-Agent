"""Storage for rendered carousel images.

Rendered slides are written to disk and served by id. The location is
configurable because the default sits inside the container filesystem, which is
ephemeral on most hosts -- point ``CAROUSEL_MEDIA_DIR`` at a mounted volume in
production or the images vanish on the next deploy.

Assets are addressed by an opaque id rather than a caller-supplied path. The id
is generated here and validated on the way back out, so a request cannot walk
out of the media directory.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

ASSET_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
CONTENT_TYPE = "image/png"


class CarouselMediaService:
    def __init__(self, media_dir: Path | str | None = None):
        default_dir = Path(__file__).resolve().parent.parent / "data" / "carousel_media"
        self.media_dir = Path(media_dir or os.getenv("CAROUSEL_MEDIA_DIR") or default_dir)

    def _path_for(self, asset_id: str) -> Path:
        # Only ids this service generated are accepted. Rejecting the shape up
        # front means a traversal attempt never reaches the filesystem at all.
        if not ASSET_ID_PATTERN.match(asset_id or ""):
            raise ValueError(f"Invalid carousel asset id: {asset_id!r}")
        return self.media_dir / f"{asset_id}.png"

    def store(self, image_bytes: bytes) -> str:
        """Persist an image and return its asset id."""
        if not image_bytes:
            raise ValueError("Refusing to store an empty carousel image")

        asset_id = uuid.uuid4().hex
        path = self._path_for(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        logger.info("Stored carousel asset asset_id=%s bytes=%s", asset_id, len(image_bytes))
        return asset_id

    def load(self, asset_id: str) -> bytes | None:
        try:
            path = self._path_for(asset_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        return path.read_bytes()

    def exists(self, asset_id: str) -> bool:
        try:
            return self._path_for(asset_id).exists()
        except ValueError:
            return False

    def url_for(self, asset_id: str) -> str:
        return f"/api/v1/carousel-media/{asset_id}"
