import os
import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService
from app.services.instagram_profile_service import InstagramProfileService


logger = logging.getLogger(__name__)


class InstagramMediaService:
    def __init__(self):
        self.connected_accounts_service = ConnectedAccountsService()
        self.instagram_connection_service = InstagramConnectionService()
        self.instagram_profile_service = InstagramProfileService()

    def _resolve_account_id(self, user_id: str, account_id: str | None = None) -> str:
        return self.connected_accounts_service.resolve_account_id(user_id, account_id)

    def _ensure_connected(self, user_id: str, account_id: str) -> None:
        self.instagram_connection_service.get_connection_for_meta(user_id, account_id)

    def _get_graph_base_url(self) -> str:
        return self.instagram_profile_service._get_graph_base_url()

    def _meta_get(
        self,
        user_id: str,
        account_id: str,
        path: str,
        access_token: str,
        fields: str,
        extra_params: dict | None = None,
    ) -> dict:
        return self.instagram_connection_service.meta_get(
            user_id=user_id,
            account_id=account_id,
            path=path,
            access_token=access_token,
            fields=fields,
            extra_params=extra_params,
            timeout_detail="Meta Instagram media fetch timed out",
            http_failure_detail="Meta Instagram media fetch failed",
            request_failure_detail="Meta Instagram media request failed",
            invalid_json_detail="Meta Instagram media fetch returned invalid JSON",
        )

    def _safe_int(self, value) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _is_reel_media(
        self,
        *,
        media_type: str | None = None,
        media_product_type: str | None = None,
        permalink: str | None = None,
    ) -> bool:
        normalized_media_type = str(media_type or "").upper()
        normalized_product_type = str(media_product_type or "").upper()
        normalized_permalink = str(permalink or "").lower()

        return (
            normalized_media_type == "REEL"
            or normalized_product_type == "REELS"
            or "/reel/" in normalized_permalink
            or "/reels/" in normalized_permalink
        )

    def _fetch_media_items(self, user_id: str, account_id: str, instagram_user_id: str, access_token: str, limit: int) -> list[dict]:
        media_payload = self._meta_get(
            user_id,
            account_id,
            f"{instagram_user_id}/media",
            access_token,
            "id,caption,media_type,permalink,thumbnail_url,media_url,timestamp",
            {"limit": limit},
        )
        return media_payload.get("data", [])

    def _fetch_media_metrics(
        self,
        user_id: str,
        account_id: str,
        media_id: str,
        access_token: str,
    ) -> tuple[int | None, int | None]:
        try:
            metrics_payload = self._meta_get(
                user_id,
                account_id,
                media_id,
                access_token,
                "like_count,comments_count",
            )
        except HTTPException as exc:
            if self.instagram_connection_service.is_reconnect_required_exception(exc):
                raise
            logger.warning(
                "Instagram media metrics unavailable for media_id=%s status_code=%s",
                media_id,
                exc.status_code,
            )
            return None, None

        return (
            self._safe_int(metrics_payload.get("like_count")),
            self._safe_int(metrics_payload.get("comments_count")),
        )

    def _build_real_items(
        self,
        user_id: str,
        account_id: str,
        access_token: str,
        instagram_user_id: str,
        limit: int,
    ) -> list[dict]:
        raw_items = self._fetch_media_items(user_id, account_id, instagram_user_id, access_token, limit)
        items = []

        for raw_item in raw_items:
            media_id = str(raw_item.get("id") or "")
            like_count, comments_count = (
                self._fetch_media_metrics(user_id, account_id, media_id, access_token)
                if media_id else (None, None)
            )

            items.append({
                "media_id": media_id,
                "account_id": account_id,
                "media_type": str(raw_item.get("media_type") or ""),
                "caption": str(raw_item.get("caption") or ""),
                "permalink": str(raw_item.get("permalink") or ""),
                "thumbnail_url": str(raw_item.get("thumbnail_url")) if raw_item.get("thumbnail_url") else None,
                "media_url": str(raw_item.get("media_url")) if raw_item.get("media_url") else None,
                "timestamp": str(raw_item.get("timestamp") or ""),
                "like_count": like_count,
                "comments_count": comments_count,
            })

        items.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
        return items

    def _build_mock_items(self, account_id: str, limit: int) -> list[dict]:
        seed = sum(ord(char) for char in account_id)
        media_types = ["IMAGE", "VIDEO", "CAROUSEL_ALBUM", "REEL"]
        base_time = datetime.now(timezone.utc)
        items = []

        for index in range(limit):
            media_type = media_types[index % len(media_types)]
            media_number = index + 1
            media_id = f"{account_id}-media-{media_number}"
            slug = account_id.lower().replace(" ", "-").replace("_", "-")
            timestamp = (base_time - timedelta(days=index, hours=index * 2)).isoformat()

            items.append({
                "media_id": media_id,
                "account_id": account_id,
                "media_type": media_type,
                "caption": f"Mock {media_type.lower()} post {media_number} for {account_id}. Hook, insight, and CTA example.",
                "permalink": f"https://instagram.com/p/{slug}-{media_number}",
                "thumbnail_url": f"https://cdn.example.com/{slug}/{media_id}/thumb.jpg" if media_type in {"VIDEO", "REEL", "CAROUSEL_ALBUM"} else None,
                "media_url": f"https://cdn.example.com/{slug}/{media_id}.jpg" if media_type == "IMAGE" else f"https://cdn.example.com/{slug}/{media_id}.mp4",
                "timestamp": timestamp,
                "like_count": 100 + ((seed + media_number * 13) % 900),
                "comments_count": 10 + ((seed + media_number * 7) % 120),
            })

        return items

    def get_media_item(self, user_id: str, media_id: str, account_id: str | None = None) -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)
        normalized_media_id = str(media_id or "").strip()
        if not normalized_media_id:
            raise HTTPException(status_code=400, detail="media_id is required for Reel feedback")

        if os.getenv("META_MOCK_MODE", "").lower() == "true":
            mock_items = self._build_mock_items(effective_account_id, 25)
            matched_item = next((item for item in mock_items if item.get("media_id") == normalized_media_id), None)
            if not matched_item:
                raise HTTPException(
                    status_code=404,
                    detail="Requested Reel was not found in the connected Instagram account",
                )
            matched_item["is_reel"] = self._is_reel_media(
                media_type=matched_item.get("media_type"),
                permalink=matched_item.get("permalink"),
            )
            return matched_item

        connection = self.instagram_connection_service.get_connection_for_meta(user_id, effective_account_id)
        access_token = str(connection.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        metrics_payload = self._meta_get(
            user_id,
            effective_account_id,
            normalized_media_id,
            access_token,
            "id,caption,media_type,media_product_type,permalink,thumbnail_url,media_url,timestamp,like_count,comments_count",
        )
        if not metrics_payload.get("id"):
            raise HTTPException(
                status_code=404,
                detail="Requested Reel was not found in the connected Instagram account",
            )

        return {
            "media_id": str(metrics_payload.get("id") or normalized_media_id),
            "account_id": effective_account_id,
            "media_type": str(metrics_payload.get("media_type") or ""),
            "media_product_type": str(metrics_payload.get("media_product_type") or ""),
            "caption": str(metrics_payload.get("caption") or ""),
            "permalink": str(metrics_payload.get("permalink") or ""),
            "thumbnail_url": str(metrics_payload.get("thumbnail_url")) if metrics_payload.get("thumbnail_url") else None,
            "media_url": str(metrics_payload.get("media_url")) if metrics_payload.get("media_url") else None,
            "timestamp": str(metrics_payload.get("timestamp") or ""),
            "like_count": self._safe_int(metrics_payload.get("like_count")),
            "comments_count": self._safe_int(metrics_payload.get("comments_count")),
            "is_reel": self._is_reel_media(
                media_type=metrics_payload.get("media_type"),
                media_product_type=metrics_payload.get("media_product_type"),
                permalink=metrics_payload.get("permalink"),
            ),
        }

    def get_media(self, user_id: str, account_id: str | None = None, limit: int = 10) -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)
        normalized_limit = max(1, min(limit, 50))

        if os.getenv("META_MOCK_MODE", "").lower() == "true":
            self._ensure_connected(user_id, effective_account_id)
            return {
                "user_id": user_id,
                "account_id": effective_account_id,
                "platform": "instagram",
                "items": self._build_mock_items(effective_account_id, normalized_limit),
            }

        logger.info(
            "Starting real Instagram media fetch for user_id=%s account_id=%s limit=%s",
            user_id,
            effective_account_id,
            normalized_limit,
        )

        connection = self.instagram_connection_service.get_connection_for_meta(user_id, effective_account_id)
        access_token = str(connection.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        instagram_account = self.instagram_profile_service._get_instagram_account_reference(
            user_id,
            effective_account_id,
            access_token,
        )
        instagram_user_id = str(instagram_account.get("instagram_user_id") or "").strip()
        if not instagram_user_id:
            raise HTTPException(
                status_code=404,
                detail="Linked Instagram professional account could not be resolved",
            )

        logger.info(
            "Resolved Instagram account for media fetch account_id=%s instagram_user_id=%s",
            effective_account_id,
            instagram_user_id,
        )

        items = self._build_real_items(
            user_id,
            effective_account_id,
            access_token,
            instagram_user_id,
            normalized_limit,
        )

        logger.info(
            "Fetched Instagram media count=%s for account_id=%s instagram_user_id=%s",
            len(items),
            effective_account_id,
            instagram_user_id,
        )

        return {
            "user_id": user_id,
            "account_id": effective_account_id,
            "platform": "instagram",
            "items": items,
        }
