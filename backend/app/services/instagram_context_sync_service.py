import os
import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_context_sync_metadata_service import InstagramContextSyncMetadataService
from app.services.instagram_insights_service import InstagramInsightsService
from app.services.instagram_media_service import InstagramMediaService
from app.services.instagram_profile_service import InstagramProfileService
from app.services.profile_context_service import ProfileContextService
from app.services.recent_content_context_service import RecentContentContextService
from app.services.recent_posts_context_service import RecentPostsContextService


logger = logging.getLogger(__name__)


class InstagramContextSyncService:
    def __init__(self):
        self.connected_accounts_service = ConnectedAccountsService()
        self.instagram_profile_service = InstagramProfileService()
        self.instagram_media_service = InstagramMediaService()
        self.instagram_insights_service = InstagramInsightsService()
        self.profile_context_service = ProfileContextService()
        self.recent_posts_context_service = RecentPostsContextService()
        self.recent_content_context_service = RecentContentContextService()
        self.sync_metadata_service = InstagramContextSyncMetadataService()

    def _resolve_account_id(self, user_id: str, account_id: str | None = None) -> str:
        return self.connected_accounts_service.resolve_account_id(user_id, account_id)

    def _normalize_content_type(self, media_type: str) -> str:
        mapping = {
            "IMAGE": "image",
            "VIDEO": "video",
            "CAROUSEL_ALBUM": "carousel",
            "REEL": "reel",
        }
        return mapping.get(media_type, media_type.lower())

    def _pick_unique(self, values: list[str], limit: int) -> list[str]:
        unique_values = []
        for value in values:
            normalized_value = " ".join((value or "").split()).strip()
            if not normalized_value or normalized_value in unique_values:
                continue
            unique_values.append(normalized_value)
            if len(unique_values) >= limit:
                break
        return unique_values

    def _normalize_text(self, value: str | None) -> str:
        return " ".join((value or "").strip().split())

    def _infer_topic(self, caption: str, media_type: str) -> str:
        normalized_caption = self._normalize_text(caption).lower()
        if not normalized_caption:
            return self._normalize_content_type(media_type)

        keyword_topics = [
            (("before", "after"), "transformation"),
            (("tip", "tips"), "tips"),
            (("routine",), "routine"),
            (("behind", "scenes"), "behind the scenes"),
            (("result", "results"), "results"),
            (("myth", "myths"), "myth busting"),
        ]
        for keywords, topic in keyword_topics:
            if any(keyword in normalized_caption for keyword in keywords):
                return topic

        caption_words = [
            word.strip(".,!?()[]{}:;\"'")
            for word in normalized_caption.split()
            if len(word.strip(".,!?()[]{}:;\"'")) > 3
        ]
        if not caption_words:
            return self._normalize_content_type(media_type)

        return " ".join(caption_words[:2])

    def _infer_cta(self, caption: str) -> str:
        normalized_caption = (caption or "").lower()
        if "save" in normalized_caption:
            return "Save this"
        if "share" in normalized_caption:
            return "Share this"
        if "comment" in normalized_caption:
            return "Leave a comment"
        if "follow" in normalized_caption:
            return "Follow for more"
        return "Learn more"

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _collect_profile_text(self, profile: dict, media_items: list[dict], recent_content_context: dict) -> str:
        text_parts = [
            self._normalize_text(profile.get("display_name")),
            self._normalize_text(profile.get("username")),
            self._normalize_text(profile.get("biography")),
        ]
        text_parts.extend(self._normalize_text(item.get("caption")) for item in media_items[:10])
        text_parts.extend(self._normalize_text(topic) for topic in recent_content_context.get("best_topics", []))
        text_parts.extend(self._normalize_text(topic) for topic in recent_content_context.get("weak_topics", []))
        return " ".join(part for part in text_parts if part).lower()

    def _infer_niche(self, profile: dict, media_items: list[dict], recent_content_context: dict) -> str | None:
        profile_text = self._collect_profile_text(profile, media_items, recent_content_context)
        if not profile_text:
            return None

        niche_keywords = [
            (
                "creator education / personal brand",
                (
                    "creator",
                    "content creator",
                    "personal brand",
                    "monetization",
                    "digital product",
                    "blog",
                    "blogging",
                    "social media",
                    "instagram growth",
                    "monetize",
                    "personal branding",
                    "smm",
                    "monet",
                    "brand deals",
                    "ugc",
                    "b2c creator",
                    "ბლოგ",
                    "მონეტ",
                    "პირადი ბრენდი",
                    "გამომწერი",
                    "კონტენტ",
                ),
            ),
            (
                "beauty / skincare",
                (
                    "skincare",
                    "skin care",
                    "facial",
                    "glow",
                    "beauty studio",
                    "acne",
                    "skin",
                    "esthetician",
                    "კანის",
                    "სახის",
                    "beauty",
                ),
            ),
            (
                "fashion / personal style",
                (
                    "fashion",
                    "style",
                    "outfit",
                    "lookbook",
                    "wardrobe",
                    "styling",
                    "street style",
                ),
            ),
            (
                "food / cafe / hospitality",
                (
                    "coffee",
                    "cafe",
                    "restaurant",
                    "pastry",
                    "menu",
                    "barista",
                    "food",
                    "dessert",
                ),
            ),
            (
                "fitness / wellness",
                (
                    "fitness",
                    "workout",
                    "gym",
                    "wellness",
                    "training",
                    "coach",
                    "nutrition",
                ),
            ),
            (
                "motorsport / formula 1",
                (
                    "formula 1",
                    "f1",
                    "motorsport",
                    "grand prix",
                    "race weekend",
                    "paddock",
                ),
            ),
        ]

        scored_niches = []
        for niche, keywords in niche_keywords:
            score = sum(1 for keyword in keywords if keyword in profile_text)
            if score:
                scored_niches.append((niche, score))

        if not scored_niches:
            return None

        scored_niches.sort(key=lambda item: item[1], reverse=True)
        top_niche, top_score = scored_niches[0]
        if top_score >= 2:
            return top_niche

        return None

    def _derive_target_audience(self, niche: str | None, profile: dict, recent_content_context: dict) -> str | None:
        if niche == "creator education / personal brand":
            return "creators, bloggers, and personal brands who want stronger Instagram growth and monetization"
        if niche == "beauty / skincare":
            return "people looking for skincare results, treatments, and trusted beauty guidance"
        if niche == "fashion / personal style":
            return "style-conscious shoppers looking for inspiration and wearable fashion ideas"
        if niche == "food / cafe / hospitality":
            return "local customers and food lovers looking for taste, atmosphere, and new places to try"
        if niche == "fitness / wellness":
            return "people interested in healthier habits, training guidance, and practical wellness content"
        if niche == "motorsport / formula 1":
            return "Formula 1 fans and motorsport followers looking for news, reactions, and race-related content"

        bio_text = self._normalize_text(profile.get("biography")).lower()
        topic_text = " ".join(recent_content_context.get("best_topics", [])).lower()
        combined_text = f"{bio_text} {topic_text}".strip()

        if self._contains_any(combined_text, ("client", "book", "service", "consultation", "dm")):
            return "people considering a service, consultation, or direct purchase"
        if self._contains_any(combined_text, ("guide", "tip", "tutorial", "how to", "რჩევ")):
            return "people looking for practical Instagram advice and educational content"

        return None

    def _derive_brand_voice(self, profile: dict, media_items: list[dict], recent_content_context: dict) -> str | None:
        bio = self._normalize_text(profile.get("biography"))
        captions = " ".join(self._normalize_text(item.get("caption")) for item in media_items[:10])
        source_text = " ".join(part for part in [bio, captions] if part).lower()

        if not source_text:
            return None

        voice_traits = []
        if self._contains_any(source_text, ("tip", "guide", "tutorial", "how to", "learn", "რჩევ", "როგორ")):
            voice_traits.append("educational")
        if self._contains_any(source_text, ("book", "shop", "dm", "link", "offer", "buy", "შეიძ", "დამიწერ")):
            voice_traits.append("direct")
        if self._contains_any(source_text, ("story", "journey", "my", "i ", "მე ")) or len(bio) > 40:
            voice_traits.append("personal")
        if source_text.count("!") >= 2 or self._contains_any(source_text, ("now", "today", "fast", "big", "best")):
            voice_traits.append("energetic")

        voice_traits.append("clear")

        if recent_content_context.get("best_ctas"):
            voice_traits.append("conversion-aware")

        return ", ".join(self._pick_unique(voice_traits, 4)) if voice_traits else None

    def _build_content_focus(self, media_items: list[dict], recent_content_context: dict) -> list[str]:
        topic_focus = [
            self._infer_topic(item.get("caption", ""), item.get("media_type", ""))
            for item in media_items
        ]
        format_focus = [
            self._normalize_content_type(item.get("media_type", ""))
            for item in media_items
        ]
        recent_topics = recent_content_context.get("best_topics", [])
        recent_formats = recent_content_context.get("top_formats", [])

        return self._pick_unique(topic_focus + recent_topics + format_focus + recent_formats, 5)

    def _build_strengths(self, media_items: list[dict], insights: dict, recent_content_context: dict) -> list[str]:
        strengths = []
        top_content_type = insights.get("top_content_type")
        reach = int(insights.get("reach") or 0)
        reels_views = int(insights.get("reels_views") or 0)
        unique_formats = self._pick_unique(
            [self._normalize_content_type(item.get("media_type", "")) for item in media_items],
            10,
        )

        if top_content_type:
            strengths.append(f"top content type is {top_content_type.lower()}")
        if recent_content_context.get("top_formats"):
            strengths.append(f"strongest recent format is {recent_content_context['top_formats'][0]}")
        if recent_content_context.get("best_topics"):
            strengths.append(f"best recent topic cluster is {recent_content_context['best_topics'][0]}")
        if reach >= 10000:
            strengths.append("strong reach potential")
        if reels_views >= 5000:
            strengths.append("short-form discovery is showing momentum")
        if len(unique_formats) >= 3:
            strengths.append("recent content mix covers multiple useful formats")
        if media_items:
            strengths.append("recent content cadence is established")

        return self._pick_unique(strengths, 5)

    def _build_weak_points(self, media_items: list[dict], insights: dict, recent_content_context: dict) -> list[str]:
        weak_points = []
        website_clicks = int(insights.get("website_clicks") or 0)
        profile_views = int(insights.get("profile_views") or 0)
        unique_formats = self._pick_unique(
            [self._normalize_content_type(item.get("media_type", "")) for item in media_items],
            10,
        )

        if recent_content_context.get("weak_ctas"):
            weak_points.append(f"CTA pattern to improve: {recent_content_context['weak_ctas'][0]}")
        else:
            weak_points.append("CTA clarity can be improved")
        if recent_content_context.get("weak_topics"):
            weak_points.append(f"topic area to improve: {recent_content_context['weak_topics'][0]}")
        else:
            weak_points.append("topic clustering can be more intentional")
        if website_clicks < 100:
            weak_points.append("profile-to-click conversion path is still weak")
        if len(unique_formats) < 2:
            weak_points.append("format variety is limited")
        else:
            weak_points.append("content series structure is still heuristic-based")
        if profile_views < 500:
            weak_points.append("profile curiosity signals are still modest")

        return self._pick_unique(weak_points, 5)

    def _build_brand_name(self, profile: dict) -> str:
        display_name = self._normalize_text(profile.get("display_name"))
        username = self._normalize_text(profile.get("username"))
        return display_name or username or "Unknown Brand"

    def _build_profile_context(self, account_id: str, profile: dict, media_items: list[dict], insights: dict, recent_content_context: dict) -> dict:
        niche = self._infer_niche(profile, media_items, recent_content_context)
        target_audience = self._derive_target_audience(niche, profile, recent_content_context)
        brand_voice = self._derive_brand_voice(profile, media_items, recent_content_context)

        return {
            "account_id": account_id,
            "brand_name": self._build_brand_name(profile),
            "niche": niche,
            "target_audience": target_audience,
            "brand_voice": brand_voice,
            "bio": self._normalize_text(profile.get("biography")),
            "content_focus": self._build_content_focus(media_items, recent_content_context),
            "strengths": self._build_strengths(media_items, insights, recent_content_context),
            "weak_points": self._build_weak_points(media_items, insights, recent_content_context),
        }

    def _profile_context_fields_replaced(self, stored_context: dict | None, new_context: dict) -> list[str]:
        if not stored_context:
            return []

        tracked_fields = [
            "brand_name",
            "niche",
            "target_audience",
            "brand_voice",
            "bio",
            "content_focus",
            "strengths",
            "weak_points",
        ]
        return [
            field_name
            for field_name in tracked_fields
            if stored_context.get(field_name) != new_context.get(field_name)
        ]

    def _log_profile_context_regeneration(self, account_id: str, replaced_fields: list[str]) -> None:
        logger.info(
            "Regenerated profile_context for account_id=%s using sources=profile,media,insights stale_values_replaced=%s replaced_fields=%s",
            account_id,
            bool(replaced_fields),
            ",".join(replaced_fields) if replaced_fields else "none",
        )

    def _build_recent_posts_context(self, account_id: str, media_items: list[dict]) -> dict:
        posts = []

        for item in media_items:
            likes = int(item.get("like_count") or 0)
            comments = int(item.get("comments_count") or 0)
            media_type = item.get("media_type", "")
            content_type = self._normalize_content_type(media_type)
            multiplier = 20 if content_type in {"reel", "video"} else 12
            estimated_saves = max(5, (likes // 4) + (comments // 2))
            estimated_views = max(100, likes * multiplier + comments * 15 + estimated_saves * 8)

            posts.append({
                "post_id": item.get("media_id", ""),
                "content_type": content_type,
                "topic": self._infer_topic(item.get("caption", ""), media_type),
                "caption": item.get("caption", ""),
                "views": estimated_views,
                "likes": likes,
                "comments": comments,
                "saves": estimated_saves,
            })

        return {
            "account_id": account_id,
            "posts": posts,
        }

    def _build_recent_content_context(self, account_id: str, media_items: list[dict], insights: dict) -> dict:
        format_scores: dict[str, int] = {}
        topic_scores: dict[str, int] = {}
        cta_scores: dict[str, int] = {}

        for item in media_items:
            content_type = self._normalize_content_type(item.get("media_type", ""))
            topic = self._infer_topic(item.get("caption", ""), item.get("media_type", ""))
            cta = self._infer_cta(item.get("caption", ""))
            likes = int(item.get("like_count") or 0)
            comments = int(item.get("comments_count") or 0)
            score = likes + (comments * 3)

            format_scores[content_type] = format_scores.get(content_type, 0) + score
            topic_scores[topic] = topic_scores.get(topic, 0) + score
            cta_scores[cta] = cta_scores.get(cta, 0) + score

        sorted_formats = sorted(format_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_topics = sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_ctas = sorted(cta_scores.items(), key=lambda item: item[1], reverse=True)

        top_formats = [item[0] for item in sorted_formats[:3]]
        best_topics = [item[0] for item in sorted_topics[:3]]
        weak_topics = [item[0] for item in sorted_topics[-2:]] if len(sorted_topics) > 2 else ["generic topics"]
        best_ctas = [item[0] for item in sorted_ctas[:3]]
        weak_ctas = [item[0] for item in sorted_ctas[-2:]] if len(sorted_ctas) > 2 else ["Generic CTA"]

        notes = [
            f"top content type from insights: {insights.get('top_content_type', 'unknown').lower()}",
            f"reach over selected period: {insights.get('reach', 0)}",
            f"reels views over selected period: {insights.get('reels_views', 0)}",
            f"profile views over selected period: {insights.get('profile_views', 0)}",
            f"website clicks over selected period: {insights.get('website_clicks', 0)}",
        ]

        return {
            "account_id": account_id,
            "top_formats": top_formats or ["reel"],
            "best_topics": best_topics or ["content themes still stabilizing"],
            "weak_topics": weak_topics,
            "best_ctas": best_ctas or ["Learn more"],
            "weak_ctas": weak_ctas,
            "notes": self._pick_unique(notes, 5),
        }

    def _get_context_max_age_minutes(self) -> int:
        raw_value = os.getenv("INSTAGRAM_CONTEXT_MAX_AGE_MINUTES", "").strip()
        default_value = 30

        if not raw_value:
            return default_value

        try:
            parsed_value = int(raw_value)
        except ValueError:
            return default_value

        return max(parsed_value, 1)

    def _get_context_freshness_ttl_seconds(self) -> int:
        raw_value = os.getenv("INSTAGRAM_CONTEXT_FRESHNESS_TTL_SECONDS", "").strip()
        if raw_value:
            try:
                parsed_value = int(raw_value)
            except ValueError:
                parsed_value = 0
            if parsed_value > 0:
                return parsed_value

        return self._get_context_max_age_minutes() * 60

    def _parse_synced_at(self, value: str | None) -> datetime | None:
        if not value:
            return None

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def get_context_freshness(self, account_id: str) -> dict:
        profile_context = self.profile_context_service.get_stored_context(account_id)
        recent_posts_context = self.recent_posts_context_service.get_stored_context(account_id)
        recent_content_context = self.recent_content_context_service.get_stored_context(account_id)
        metadata = self.sync_metadata_service.get_metadata(account_id) or {}
        last_synced_at = metadata.get("last_synced_at")
        max_age_minutes = self._get_context_max_age_minutes()
        ttl_seconds = self._get_context_freshness_ttl_seconds()

        missing_sections = []
        if profile_context is None:
            missing_sections.append("profile_context")
        if recent_posts_context is None:
            missing_sections.append("recent_posts_context")
        if recent_content_context is None:
            missing_sections.append("recent_content_context")

        has_complete_context = not missing_sections
        parsed_last_synced_at = self._parse_synced_at(str(last_synced_at) if last_synced_at else None)
        now = datetime.now(timezone.utc)
        context_age_seconds = None
        if parsed_last_synced_at is not None:
            context_age_seconds = max(0, int((now - parsed_last_synced_at).total_seconds()))

        stale_reasons = []
        context_was_fresh = False
        if missing_sections:
            stale_reasons.append("missing_context")
        if not last_synced_at:
            stale_reasons.append("missing_last_synced_at")
        elif parsed_last_synced_at is None:
            stale_reasons.append("invalid_last_synced_at")
        elif has_complete_context:
            max_age = timedelta(seconds=ttl_seconds)
            context_was_fresh = now - parsed_last_synced_at <= max_age
            if not context_was_fresh:
                stale_reasons.append("stale")

        if not has_complete_context:
            context_was_fresh = False

        sync_required = not context_was_fresh

        return {
            "account_id": account_id,
            "last_synced_at": str(last_synced_at) if last_synced_at else None,
            "context_was_fresh": context_was_fresh,
            "sync_required": sync_required,
            "sync_skipped": context_was_fresh,
            "has_complete_context": has_complete_context,
            "missing_sections": missing_sections,
            "max_age_minutes": max_age_minutes,
            "ttl_seconds": ttl_seconds,
            "context_age_seconds": context_age_seconds,
            "stale_reasons": stale_reasons,
        }

    def sync(self, user_id: str, account_id: str | None = None) -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)

        profile = self.instagram_profile_service.get_profile(user_id, effective_account_id)
        media = self.instagram_media_service.get_media(user_id, effective_account_id, limit=10)
        insights = self.instagram_insights_service.get_insights(user_id, effective_account_id, period="30d")
        media_items = media.get("items", [])

        recent_content_context = self._build_recent_content_context(
            effective_account_id,
            media_items,
            insights,
        )
        recent_posts_context = self._build_recent_posts_context(
            effective_account_id,
            media_items,
        )
        profile_context = self._build_profile_context(
            effective_account_id,
            profile,
            media_items,
            insights,
            recent_content_context,
        )
        stored_profile_context = self.profile_context_service.get_stored_context(effective_account_id)
        replaced_fields = self._profile_context_fields_replaced(stored_profile_context, profile_context)

        self._log_profile_context_regeneration(effective_account_id, replaced_fields)

        self.profile_context_service.save_context(profile_context)
        self.recent_posts_context_service.save_context(recent_posts_context)
        self.recent_content_context_service.save_context(recent_content_context)
        sync_metadata = self.sync_metadata_service.mark_synced(effective_account_id)

        return {
            "user_id": user_id,
            "account_id": effective_account_id,
            "synced": True,
            "last_synced_at": sync_metadata.get("last_synced_at"),
            "saved_sections": [
                "profile_context",
                "recent_posts_context",
                "recent_content_context",
            ],
        }
