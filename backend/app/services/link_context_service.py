import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse


class LinkContextService:
    def _detect_platform(self, link: str) -> str:
        if "instagram.com" in urlparse(link).netloc.lower():
            return "instagram"
        return "unknown"

    def _path_parts(self, link: str) -> list[str]:
        parsed = urlparse(link)
        return [part for part in parsed.path.split("/") if part]

    def _detect_content_type(self, link: str) -> str:
        parts = self._path_parts(link)
        if not parts:
            return "unknown"

        if parts[0] in {"reel", "reels"}:
            return "reel"
        if parts[0] in {"p", "tv"}:
            return "post"
        if parts[0] not in {"stories", "explore"}:
            return "profile"
        return "unknown"

    def _safe_text(self, value: str | None) -> str:
        if not value:
            return ""
        return " ".join(value.split()).strip()

    def _extract_instagram_handle(self, link: str) -> str | None:
        parts = self._path_parts(link)
        if not parts:
            return None

        reserved_roots = {"reel", "reels", "p", "tv", "stories", "explore"}
        if parts[0] in reserved_roots:
            return None

        return parts[0]

    def _infer_hook_style(self, content_type: str, summary: str) -> str:
        normalized_summary = summary.lower()

        if any(keyword in normalized_summary for keyword in {"how to", "step", "tips", "guide"}):
            return "educational hook"
        if any(keyword in normalized_summary for keyword in {"before", "after", "result", "transformation"}):
            return "result-driven hook"
        if any(keyword in normalized_summary for keyword in {"mistake", "wrong", "stop", "avoid"}):
            return "pattern interrupt hook"
        if content_type == "reel":
            return "short visual hook"
        if content_type == "post":
            return "static or carousel hook"
        if content_type == "profile":
            return "brand positioning hook"
        return "unknown"

    def _infer_business_goal(self, summary: str, content_type: str) -> str:
        normalized_summary = summary.lower()

        if any(keyword in normalized_summary for keyword in {"shop", "buy", "order", "sale", "discount"}):
            return "sales"
        if any(keyword in normalized_summary for keyword in {"follow", "community", "audience"}):
            return "audience growth"
        if any(keyword in normalized_summary for keyword in {"learn", "tips", "guide", "educat"}):
            return "education"
        if content_type == "profile":
            return "brand positioning"
        return "engagement"

    def _build_pattern_hints(self, content_type: str, summary: str) -> list[str]:
        hints = []
        normalized_summary = summary.lower()

        if content_type == "reel":
            hints.append("likely optimized for fast first-3-second retention")
        if content_type == "profile":
            hints.append("profile-level positioning can guide tone and offer direction")
        if any(keyword in normalized_summary for keyword in {"tips", "how to", "guide"}):
            hints.append("educational framing is likely part of the content pattern")
        if any(keyword in normalized_summary for keyword in {"result", "before", "after"}):
            hints.append("proof or transformation framing may be part of the creative angle")
        if any(keyword in normalized_summary for keyword in {"new", "launch", "drop"}):
            hints.append("novelty-based hook may be driving attention")

        return hints[:4]

    def extract_context(self, link: str) -> dict:
        detected_platform = self._detect_platform(link)
        content_type = self._detect_content_type(link)
        creator_handle = self._extract_instagram_handle(link)

        title = ""
        og_title = ""
        og_description = ""
        summary = ""
        source_type = "instagram_link" if detected_platform == "instagram" else "external_link"
        hook_style = "unknown"
        business_goal = "unknown"
        source_patterns = []
        probable_strengths = []
        adaptation_notes = []
        data_available = False
        data_availability = "limited_link_signal"
        limitations = [
            "Full visual, audio, profile, and performance data is unavailable from this public link context.",
            "Analysis must be based only on the URL signal, public metadata if available, and the user's account context.",
        ]
        analysis_basis = ["url_structure"]

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }

            with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
                response = client.get(link)
                response.raise_for_status()

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            title_tag = soup.find("title")
            if title_tag:
                title = self._safe_text(title_tag.get_text())

            og_title_tag = soup.find("meta", property="og:title")
            if og_title_tag:
                og_title = self._safe_text(og_title_tag.get("content"))

            og_description_tag = soup.find("meta", property="og:description")
            if og_description_tag:
                og_description = self._safe_text(og_description_tag.get("content"))

            summary_parts = [part for part in [og_title, og_description, title] if part]
            summary = " | ".join(summary_parts)

            if not summary:
                summary = "Public page fetched, but only limited metadata was available."
                data_availability = "limited_public_metadata"
            else:
                data_available = True
                data_availability = "public_metadata"

            hook_style = self._infer_hook_style(content_type, summary)
            business_goal = self._infer_business_goal(summary, content_type)
            source_patterns = self._build_pattern_hints(content_type, summary)
            analysis_basis = ["url_structure", "public_page_metadata"]

            probable_strengths = [
                "metadata was fetched from the public page",
                "content type was detected from the link",
                "basic hook/context inference is possible",
            ]
            if creator_handle:
                probable_strengths.append("source account handle was inferred from the URL")
            if business_goal != "unknown":
                probable_strengths.append(f"probable business goal was inferred as {business_goal}")

            adaptation_notes = [
                "adapt the core angle to the user's niche",
                "match the final output to the user's brand voice",
                "keep one clear CTA instead of copying the source literally",
            ]

        except Exception as exc:
            summary = "Could not fetch public metadata from the link. Safe fallback mode used."
            hook_style = self._infer_hook_style(content_type, summary)
            business_goal = self._infer_business_goal(summary, content_type)
            source_patterns = self._build_pattern_hints(content_type, summary)
            probable_strengths = [
                "link structure was recognized",
                "platform detection still worked",
            ]
            if creator_handle:
                probable_strengths.append("source account handle was inferred from the URL")
            adaptation_notes = [
                "add a short manual description of what you liked about the source",
                "if public metadata is weak, mention the niche and the key style to emulate",
                "adapt the final answer to the user's brand voice and sales goal",
            ]

        return {
            "link": link,
            "source_url": link,
            "source_type": source_type,
            "detected_platform": detected_platform,
            "creator_handle": creator_handle,
            "summary": summary,
            "content_type": content_type,
            "hook_style": hook_style,
            "business_goal": business_goal,
            "data_available": data_available,
            "data_availability": data_availability,
            "limitations": limitations,
            "analysis_basis": analysis_basis,
            "source_patterns": source_patterns,
            "probable_strengths": probable_strengths,
            "adaptation_notes": adaptation_notes,
        }
