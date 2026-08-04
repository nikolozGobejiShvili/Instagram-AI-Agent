import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService
from app.services.instagram_profile_service import InstagramProfileService


logger = logging.getLogger(__name__)


class InstagramInsightsService:
    def __init__(self):
        self.connected_accounts_service = ConnectedAccountsService()
        self.instagram_connection_service = InstagramConnectionService()
        self.instagram_profile_service = InstagramProfileService()

    def _resolve_account_id(self, user_id: str, account_id: str | None = None) -> str:
        return self.connected_accounts_service.resolve_account_id(user_id, account_id)

    def _ensure_connected(self, user_id: str, account_id: str) -> None:
        self.instagram_connection_service.get_connection_for_meta(user_id, account_id)

    def _build_mock_insights(self, user_id: str, account_id: str, period: str) -> dict:
        seed = sum(ord(char) for char in f"{user_id}:{account_id}:{period}")
        content_types = ["REEL", "CAROUSEL_ALBUM", "VIDEO", "IMAGE"]

        followers_count = 1000 + (seed % 9000)
        reach = 5000 + (seed % 40000)
        impressions = reach + 2500 + (seed % 12000)
        profile_views = 300 + (seed % 3000)
        website_clicks = 20 + (seed % 500)
        total_likes = 400 + (seed % 6000)
        total_comments = 40 + (seed % 900)
        total_saves = 60 + (seed % 1500)
        total_shares = 30 + (seed % 1200)
        reels_views = 1500 + (seed % 50000)

        return {
            "user_id": user_id,
            "account_id": account_id,
            "platform": "instagram",
            "period": period,
            "followers_count": followers_count,
            "reach": reach,
            "impressions": impressions,
            "profile_views": profile_views,
            "website_clicks": website_clicks,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_saves": total_saves,
            "total_shares": total_shares,
            "reels_views": reels_views,
            "top_content_type": content_types[seed % len(content_types)],
        }

    def _get_graph_base_url(self) -> str:
        return self.instagram_profile_service._get_graph_base_url()

    def _meta_get(
        self,
        user_id: str,
        account_id: str,
        path: str,
        access_token: str,
        fields: str | None = None,
        extra_params: dict | None = None,
    ) -> dict:
        return self.instagram_connection_service.meta_get(
            user_id=user_id,
            account_id=account_id,
            path=path,
            access_token=access_token,
            fields=fields,
            extra_params=extra_params,
            timeout_detail="Meta Instagram insights fetch timed out",
            http_failure_detail="Meta Instagram insights fetch failed",
            request_failure_detail="Meta Instagram insights request failed",
            invalid_json_detail="Meta Instagram insights fetch returned invalid JSON",
        )

    def _safe_int(self, value) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_period_days(self, period: str) -> int:
        normalized = (period or "").strip().lower()
        supported = {"7d": 7, "30d": 30, "90d": 90}

        if normalized in supported:
            return supported[normalized]

        raise HTTPException(
            status_code=400,
            detail="Unsupported insights period. Use one of: 7d, 30d, 90d",
        )

    def _parse_timestamp(self, timestamp: str) -> datetime | None:
        if not timestamp:
            return None

        try:
            return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return None

    def _fetch_account_metric(
        self,
        user_id: str,
        account_id: str,
        instagram_user_id: str,
        access_token: str,
        metric: str,
        days: int,
    ) -> int | None:
        now = datetime.now(timezone.utc)
        since = int((now - timedelta(days=days)).timestamp())
        until = int(now.timestamp())

        try:
            payload = self._meta_get(
                user_id,
                account_id,
                f"{instagram_user_id}/insights",
                access_token,
                fields=None,
                extra_params={
                    "metric": metric,
                    "period": "day",
                    "since": since,
                    "until": until,
                },
            )
        except HTTPException as exc:
            if self.instagram_connection_service.is_reconnect_required_exception(exc):
                raise
            logger.warning(
                "Instagram account insight unavailable metric=%s instagram_user_id=%s status_code=%s",
                metric,
                instagram_user_id,
                exc.status_code,
            )
            return None

        total = 0
        found_numeric = False
        for metric_entry in payload.get("data", []):
            for value_entry in metric_entry.get("values", []):
                value = value_entry.get("value")
                numeric_value = self._safe_int(value)
                if numeric_value is not None:
                    total += numeric_value
                    found_numeric = True

        return total if found_numeric else None

    def _fetch_recent_media_for_period(
        self,
        user_id: str,
        account_id: str,
        instagram_user_id: str,
        access_token: str,
        days: int,
    ) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        items: list[dict] = []
        after_cursor: str | None = None
        max_pages = 10

        for _ in range(max_pages):
            extra_params = {"limit": 50}
            if after_cursor:
                extra_params["after"] = after_cursor

            payload = self._meta_get(
                user_id,
                account_id,
                f"{instagram_user_id}/media",
                access_token,
                "id,caption,media_type,media_product_type,permalink,thumbnail_url,media_url,timestamp,like_count,comments_count",
                extra_params,
            )

            page_items = payload.get("data", [])
            if not page_items:
                break

            keep_fetching = False
            for raw_item in page_items:
                item_timestamp = self._parse_timestamp(str(raw_item.get("timestamp") or ""))
                if item_timestamp is None:
                    continue

                if item_timestamp >= cutoff:
                    items.append(raw_item)
                    keep_fetching = True

            paging = payload.get("paging", {})
            cursors = paging.get("cursors", {})
            after_cursor = cursors.get("after")

            if not after_cursor:
                break

            if not keep_fetching and len(page_items) < 50:
                break

        items.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        return items

    def _fetch_media_insight_metric(
        self,
        user_id: str,
        account_id: str,
        media_id: str,
        access_token: str,
        metric: str,
    ) -> int | None:
        try:
            payload = self._meta_get(
                user_id,
                account_id,
                f"{media_id}/insights",
                access_token,
                fields=None,
                extra_params={"metric": metric},
            )
        except HTTPException as exc:
            if self.instagram_connection_service.is_reconnect_required_exception(exc):
                raise
            logger.warning(
                "Instagram media insight unavailable media_id=%s metric=%s status_code=%s",
                media_id,
                metric,
                exc.status_code,
            )
            return None

        for metric_entry in payload.get("data", []):
            if metric_entry.get("name") != metric:
                continue

            values = metric_entry.get("values", [])
            if not values:
                continue

            numeric_value = self._safe_int(values[0].get("value"))
            if numeric_value is not None:
                return numeric_value

        return None

    def _derive_media_totals(self, user_id: str, account_id: str, media_items: list[dict], access_token: str) -> dict:
        total_likes = 0
        total_comments = 0
        likes_found = False
        comments_found = False

        total_saves = 0
        saves_found = False
        total_shares = 0
        shares_found = False
        reels_views = 0
        reels_views_found = False

        content_type_scores: dict[str, int] = {}

        for media_item in media_items:
            media_id = str(media_item.get("id") or "")
            media_type = str(media_item.get("media_type") or "")
            media_product_type = str(media_item.get("media_product_type") or "")

            like_count = self._safe_int(media_item.get("like_count"))
            comments_count = self._safe_int(media_item.get("comments_count"))

            engagement_score = 0
            if like_count is not None:
                total_likes += like_count
                likes_found = True
                engagement_score += like_count

            if comments_count is not None:
                total_comments += comments_count
                comments_found = True
                engagement_score += comments_count

            if media_id:
                saved_value = self._fetch_media_insight_metric(
                    user_id,
                    account_id,
                    media_id,
                    access_token,
                    "saved",
                )
                if saved_value is not None:
                    total_saves += saved_value
                    saves_found = True
                    engagement_score += saved_value

                shares_value = self._fetch_media_insight_metric(
                    user_id,
                    account_id,
                    media_id,
                    access_token,
                    "shares",
                )
                if shares_value is not None:
                    total_shares += shares_value
                    shares_found = True
                    engagement_score += shares_value

                if media_type == "REEL" or media_product_type.upper() == "REELS":
                    views_value = self._fetch_media_insight_metric(
                        user_id,
                        account_id,
                        media_id,
                        access_token,
                        "views",
                    )
                    if views_value is not None:
                        reels_views += views_value
                        reels_views_found = True
                        engagement_score += views_value

            if media_type:
                content_type_scores[media_type] = content_type_scores.get(media_type, 0) + engagement_score

        top_content_type = None
        if content_type_scores:
            top_content_type = max(
                content_type_scores.items(),
                key=lambda item: item[1],
            )[0]

        return {
            "total_likes": total_likes if likes_found else None,
            "total_comments": total_comments if comments_found else None,
            "total_saves": total_saves if saves_found else None,
            "total_shares": total_shares if shares_found else None,
            "reels_views": reels_views if reels_views_found else None,
            "top_content_type": top_content_type,
        }

    def get_insights(self, user_id: str, account_id: str | None = None, period: str = "30d") -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)

        if os.getenv("META_MOCK_MODE", "").lower() == "true":
            self._ensure_connected(user_id, effective_account_id)
            return self._build_mock_insights(user_id, effective_account_id, period)

        days = self._parse_period_days(period)
        logger.info(
            "Starting real Instagram insights fetch for user_id=%s account_id=%s period=%s",
            user_id,
            effective_account_id,
            period,
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
            "Resolved Instagram account for insights account_id=%s instagram_user_id=%s period=%s",
            effective_account_id,
            instagram_user_id,
            period,
        )

        profile = self.instagram_profile_service._build_real_profile(
            user_id,
            effective_account_id,
            connection,
        )

        account_metrics = {
            "reach": self._fetch_account_metric(user_id, effective_account_id, instagram_user_id, access_token, "reach", days),
            "impressions": self._fetch_account_metric(user_id, effective_account_id, instagram_user_id, access_token, "impressions", days),
            "profile_views": self._fetch_account_metric(user_id, effective_account_id, instagram_user_id, access_token, "profile_views", days),
            "website_clicks": self._fetch_account_metric(user_id, effective_account_id, instagram_user_id, access_token, "website_clicks", days),
        }
        if account_metrics["website_clicks"] is None:
            account_metrics["website_clicks"] = self._fetch_account_metric(
                user_id,
                effective_account_id,
                instagram_user_id,
                access_token,
                "profile_links_taps",
                days,
            )

        fetched_account_metrics = [name for name, value in account_metrics.items() if value is not None]

        media_items = self._fetch_recent_media_for_period(user_id, effective_account_id, instagram_user_id, access_token, days)
        derived_totals = self._derive_media_totals(user_id, effective_account_id, media_items, access_token)
        fetched_derived_metrics = [
            name for name in ["total_likes", "total_comments", "total_saves", "total_shares", "reels_views"]
            if derived_totals.get(name) is not None
        ]

        logger.info(
            "Fetched Instagram insights metrics account_id=%s instagram_user_id=%s account_metrics=%s derived_metrics=%s media_items=%s",
            effective_account_id,
            instagram_user_id,
            ",".join(fetched_account_metrics) if fetched_account_metrics else "none",
            ",".join(fetched_derived_metrics) if fetched_derived_metrics else "none",
            len(media_items),
        )

        return {
            "user_id": user_id,
            "account_id": effective_account_id,
            "platform": "instagram",
            "period": period,
            "followers_count": self._safe_int(profile.get("followers_count")),
            "reach": account_metrics["reach"],
            "impressions": account_metrics["impressions"],
            "profile_views": account_metrics["profile_views"],
            "website_clicks": account_metrics["website_clicks"],
            "total_likes": derived_totals["total_likes"],
            "total_comments": derived_totals["total_comments"],
            "total_saves": derived_totals["total_saves"],
            "total_shares": derived_totals["total_shares"],
            "reels_views": derived_totals["reels_views"],
            "top_content_type": derived_totals["top_content_type"],
        }
