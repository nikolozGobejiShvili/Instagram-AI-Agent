"""Reading a public Instagram business account through Meta's business_discovery.

The customer's ask is "build my sports shop the way Nike's page is built" — so
the agent has to see how a strong account is actually assembled: what it posts,
in what mix, how the captions open, what gets engagement.

Scraping the public page cannot do this and never could. A live check returned
the string "Instagram" — the login wall Meta serves to unauthenticated
datacenter requests — so the old link-analysis path was analysing an empty page
and inferring a "hook style" from a keyword match against nothing.

business_discovery is the supported route: it returns a public business or
creator account's follower count, media count and recent media (caption,
like/comment counts, media type, permalink, timestamp) to an app that holds an
Instagram Business account of its own. That last part is why this reads from the
*customer's* connection — the call is made as them, against a target they name.

Two limits worth knowing before building on this:

* Business and creator accounts only. A personal account is invisible to
  business_discovery, and no amount of retrying changes that — so a lookup that
  fails this way says so rather than reporting "not found".
* Meta returns no per-post reach or impressions for accounts you do not own.
  Engagement here means likes and comments, and anything the agent says about
  reach for a competitor would be invented.
"""
from __future__ import annotations

import logging
import re

from fastapi import HTTPException

from app.services.instagram_connection_service import InstagramConnectionService

logger = logging.getLogger(__name__)

# Instagram handles: letters, digits, periods and underscores, max 30.
HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,30}$")

MEDIA_FIELDS = "id,caption,like_count,comments_count,media_type,media_product_type,permalink,timestamp"
DEFAULT_MEDIA_LIMIT = 25


class PublicProfileService:
    def __init__(self, connection_service: InstagramConnectionService | None = None):
        self.connection_service = connection_service or InstagramConnectionService()

    @staticmethod
    def normalize_handle(raw: str) -> str:
        """Accept what a customer will actually paste.

        People give a handle, an @handle, or the profile URL they copied from the
        address bar. Rejecting two of those three would read as the feature being
        broken.
        """
        value = (raw or "").strip()
        if not value:
            raise HTTPException(status_code=400, detail="An Instagram handle is required")

        if "instagram.com" in value.lower():
            path = value.split("instagram.com", 1)[1].lstrip("/")
            value = path.split("/", 1)[0].split("?", 1)[0]

        value = value.lstrip("@").strip()
        if not HANDLE_PATTERN.match(value):
            raise HTTPException(
                status_code=400,
                detail=f"'{raw}' is not a valid Instagram handle",
            )
        return value

    def fetch(self, *, user_id: str, account_id: str, handle: str, media_limit: int = DEFAULT_MEDIA_LIMIT) -> dict:
        """Public profile and recent media for ``handle``.

        Read through the customer's own connection because business_discovery is
        scoped to the calling Instagram Business account; there is no anonymous
        form of this call.
        """
        target = self.normalize_handle(handle)
        connection = self.connection_service.get_connection_for_meta(user_id, account_id)
        access_token = str(connection.get("access_token") or "")
        ig_user_id = str(connection.get("instagram_business_account_id") or connection.get("account_id") or "").strip()
        if not ig_user_id:
            raise HTTPException(
                status_code=400,
                detail="The connected account has no Instagram Business account id; reconnect it.",
            )

        # business_discovery is a nested field on the caller's own IG user node,
        # not a standalone endpoint. The whole request is one field expression.
        fields = (
            f"business_discovery.username({target})"
            f"{{username,name,biography,website,followers_count,media_count,profile_picture_url,"
            f"media.limit({int(media_limit)}){{{MEDIA_FIELDS}}}}}"
        )

        payload = self.connection_service.meta_get(
            user_id=user_id,
            account_id=account_id,
            path=ig_user_id,
            access_token=access_token,
            fields=fields,
            timeout_detail="Public profile lookup timed out",
            http_failure_detail=f"Instagram could not return data for '{target}'",
            request_failure_detail="Public profile request failed",
            invalid_json_detail="Public profile lookup returned invalid JSON",
        )

        discovered = (payload or {}).get("business_discovery")
        if not isinstance(discovered, dict):
            # Named precisely: the usual cause is a personal account, which no
            # retry will fix, and "not found" would send the customer looking for
            # a typo that is not there.
            raise HTTPException(
                status_code=404,
                detail=(
                    f"'{target}' could not be read. business_discovery only returns Instagram "
                    "Business and Creator accounts — personal accounts are not available."
                ),
            )

        return self._shape(discovered, target)

    def _shape(self, discovered: dict, handle: str) -> dict:
        media_items = ((discovered.get("media") or {}).get("data")) or []
        posts = [
            {
                "post_id": str(item.get("id") or ""),
                "caption": item.get("caption"),
                "media_type": item.get("media_type"),
                "product_type": item.get("media_product_type"),
                "like_count": item.get("like_count"),
                "comment_count": item.get("comments_count"),
                "permalink": item.get("permalink"),
                "posted_at": item.get("timestamp"),
            }
            for item in media_items
            if isinstance(item, dict)
        ]

        return {
            "handle": discovered.get("username") or handle,
            "name": discovered.get("name"),
            "biography": discovered.get("biography"),
            "website": discovered.get("website"),
            "followers_count": discovered.get("followers_count"),
            "media_count": discovered.get("media_count"),
            "profile_picture_url": discovered.get("profile_picture_url"),
            "posts": posts,
            # Stated in the payload rather than left for the model to assume.
            # Meta returns no reach or impressions for an account you do not own,
            # and a competitor analysis quoting reach would be inventing it.
            "available_metrics": ["like_count", "comment_count"],
            "unavailable_metrics": ["reach", "impressions", "saves", "shares", "audience demographics"],
        }

    def as_prompt_context(self, profile: dict) -> dict | None:
        """Compact the profile into what the model should reason over.

        Captions are the substance here — they carry the hook style, the voice
        and the CTA pattern the customer wants translated to their niche — so
        they survive at length while the ids and urls, which the model cannot use
        for anything, are dropped.
        """
        if not profile:
            return None

        posts = profile.get("posts") or []
        engagements = [p["like_count"] for p in posts if isinstance(p.get("like_count"), int)]
        type_mix: dict[str, int] = {}
        for post in posts:
            key = str(post.get("product_type") or post.get("media_type") or "UNKNOWN")
            type_mix[key] = type_mix.get(key, 0) + 1

        return {
            "handle": profile.get("handle"),
            "name": profile.get("name"),
            "biography": profile.get("biography"),
            "followers_count": profile.get("followers_count"),
            "media_count": profile.get("media_count"),
            "content_mix": type_mix,
            "median_likes": sorted(engagements)[len(engagements) // 2] if engagements else None,
            "recent_captions": [
                {
                    "caption": (p.get("caption") or "")[:600],
                    "type": p.get("product_type") or p.get("media_type"),
                    "likes": p.get("like_count"),
                    "comments": p.get("comment_count"),
                }
                for p in posts[:15]
                if p.get("caption")
            ],
            "metrics_note": (
                "Only likes and comments are available for an account this customer does not own. "
                "Do not state or estimate reach, impressions, saves or audience demographics."
            ),
        }
