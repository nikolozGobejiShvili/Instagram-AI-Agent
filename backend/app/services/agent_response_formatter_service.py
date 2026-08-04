import json
import logging
import re

from app.schemas.agent import validate_structured_output_payload


logger = logging.getLogger(__name__)

# Instagram accepts at most 20 media in one carousel, so this is the hard
# product-wide ceiling. The per-request count comes from the caller and is
# bounded by the plan tier; this only stops the parser short of an absurd value.
MAX_CAROUSEL_SLIDES = 20


class AgentResponseFormatterService:
    def __init__(self):
        self._structured_output_marker = "STRUCTURED_OUTPUT_JSON:"
        self._task_sections = {
            "reel_idea": [
                "Title:",
                "1-second hook:",
                "Hook:",
                "Format type:",
                "Trend or pattern used:",
                "Niche adaptation:",
                "Main idea:",
                "Shot list:",
                "Why it can work:",
                "Caption idea:",
                "CTA:",
            ],
            "reel_script": [
                "Hook:",
                "Problem/angle:",
                "Problem / angle:",
                "Problem angle:",
                "Scene-by-scene script:",
                "Scene by scene script:",
                "Structure:",
                "Voiceover:",
                "Voice over:",
                "Shot list:",
                "On-screen text:",
                "On screen text:",
                "Caption:",
                "CTA:",
            ],
            "reel_feedback": [
                "Summary:",
                "What works:",
                "What works / რა მუშაობს:",
                "რა მუშაობს:",
                "რა არის ძლიერი:",
                "რა არის კარგი:",
                "რას აკეთებს ეს Reel კარგად:",
                "რა მუშაობს კარგად:",
                "კარგად მუშაობს:",
                "ძლიერი მხარეები:",
                "What hurts:",
                "What is weak:",
                "What is weak / რა სუსტია:",
                "რა ასუსტებს:",
                "რა სუსტია:",
                "რა ასუსტებს შედეგს:",
                "რა ასუსტებს ამ Reel-ს:",
                "რა ასუსტებს ვიდეოს:",
                "რა არ მუშაობს:",
                "რა უშლის შედეგს:",
                "სუსტი მხარეები:",
                "Retention risks:",
                "Retention issues:",
                "Retention issues / რიტენშენის პრობლემა:",
                "Retention:",
                "რიტენშენის პრობლემა:",
                "რიტენშენის პრობლემები:",
                "Retention პრობლემები:",
                "Retention-ის პრობლემები:",
                "Retention-ის პრობლემა:",
                "Retention / ყურადღების შენარჩუნება:",
                "Retention / ყურადღების შენარჩუნების პრობლემა:",
                "ყურადღების შენარჩუნების პრობლემა:",
                "ყურადღების შენარჩუნების პრობლემები:",
                "სად იკარგება ყურადღება:",
                "სად იკარგება ყურადღება / retention:",
                "სად იკარგება retention:",
                "სად ვკარგავთ ყურადღებას:",
                "სად იკარგება მაყურებელი:",
                "ყურადღება სად იკარგება:",
                "Better hook:",
                "Hook improvement:",
                "Hook improvement / Hook-ის გაუმჯობესება:",
                "Hook-ის გაუმჯობესება:",
                "ჰუკის გაუმჯობესება:",
                "უკეთესი ჰუკი:",
                "უკეთესი hook:",
                "ახალი ჰუკი:",
                "ახალი hook:",
                "როგორ გავაუმჯობესოთ ჰუკი:",
                "როგორ გავაუმჯობესოთ hook:",
                "ჰუკი როგორ გაუმჯობესდეს:",
                "Improved structure:",
                "Better CTA:",
                "CTA improvement:",
                "CTA improvement / CTA-ის გაუმჯობესება:",
                "CTA clarity:",
                "CTA-ს გაუმჯობესება:",
                "CTA-ის გაუმჯობესება:",
                "CTA-ს სიცხადე:",
                "CTA-ის სიცხადე:",
                "CTA-ის სიცხადის გაუმჯობესება:",
                "CTA როგორ გაუმჯობესდეს:",
                "როგორ გავაუმჯობესოთ CTA:",
                "უკეთესი CTA:",
                "ახალი CTA:",
                "Improved version:",
                "Improved version / გაუმჯობესებული ვერსია:",
                "გაუმჯობესებული ვერსია:",
                "გაუმჯობესებული სცენარი:",
                "გაუმჯობესებული Reel სცენარი:",
                "გაუმჯობესებული ვარიანტი:",
                "ახალი ვერსია:",
                "ახალი სცენარი:",
                "საბოლოო ვერსია:",
                "საბოლოო სცენარი:",
                "გადაწერილი სცენარი:",
                "სცენარის ახალი ვერსია:",
                "სცენარის გაუმჯობესება:",
                "სცენარის გაუმჯობესებული ვერსია:",
                "როგორ გავაუმჯობესო სცენარი:",
                "როგორ გავაუმჯობესოთ სცენარი:",
                "როგორ უნდა გაუმჯობესო სცენარი:",
                "როგორ უნდა გაუმჯობესდეს სცენარი:",
            ],
            "carousel": [
                "Title:",
                # Generated rather than listed: the slide count is driven by the
                # request and the plan tier, so a hardcoded list silently capped
                # every carousel at five regardless of what was asked for.
                *[f"Slide {n}:" for n in range(1, MAX_CAROUSEL_SLIDES + 1)],
                "Final CTA slide:",
            ],
            "caption": [
                "Hook:",
                "Caption:",
                "CTA:",
            ],
            "profile_audit": [
                "Summary:",
                "What works:",
                "რა მუშაობს:",
                "What is weak:",
                "რა ასუსტებს:",
                "რა სუსტია:",
                "What to improve first:",
                "პირველ რიგში რა გავასწორო:",
                "Recommended bio direction:",
                "ბიოს მიმართულება:",
                "Content direction:",
                "კონტენტის მიმართულება:",
                "Next 3 actions:",
                "შემდეგი 3 მოქმედება:",
                "Next actions:",
                "შემდეგი ნაბიჯები:",
            ],
            "content_plan": [
                "Summary:",
                "Week 1:",
                "Week 2:",
                "Week 3:",
                "Week 4:",
                "Best content mix:",
                "საუკეთესო კონტენტ-მიქსი:",
                "Hook ideas:",
                "ჰუკის იდეები:",
                "CTA ideas:",
                "CTA იდეები:",
                "Plus:",
            ],
            "link_analysis": [
                "What works:",
                "What is weak:",
                "Why it may perform:",
                "How to adapt it for the user's account:",
            ],
            "performance_summary": [
                "Summary:",
                "What worked:",
                "რა იმუშავა:",
                "What did not work:",
                "რა არ იმუშავა:",
                "Content patterns:",
                "კონტენტის პატერნები:",
                "Best opportunities:",
                "საუკეთესო შესაძლებლობები:",
                "Recommended next moves:",
                "Next actions:",
                "შემდეგი ნაბიჯები:",
            ],
        }

    def normalize_reply(self, task_type: str, reply: str | None) -> dict:
        if reply is None:
            return {
                "reply": None,
                "parse_status": "raw_only",
                "structured_output": None,
                "cleanup_applied": False,
                "schema_applied": None,
            }

        try:
            reply_without_embedded_json, embedded_structured_output, extraction_applied = self._extract_embedded_structured_output(
                task_type,
                reply,
            )
            formatted_reply, cleanup_applied = self.format_reply(task_type, reply_without_embedded_json)
            parse_source = formatted_reply or reply_without_embedded_json or reply
            parsed_structured_output = embedded_structured_output
            parsed_status = "parsed" if embedded_structured_output is not None else "raw_only"

            if parsed_structured_output is None:
                parsed_structured_output, parsed_status = self._parse_task_output(task_type, parse_source)

            structured_output, parse_status, schema_applied = validate_structured_output_payload(
                task_type,
                parsed_status,
                parsed_structured_output,
            )
            if embedded_structured_output is not None and structured_output is None:
                parsed_structured_output, parsed_status = self._parse_task_output(task_type, parse_source)
                structured_output, parse_status, schema_applied = validate_structured_output_payload(
                    task_type,
                    parsed_status,
                    parsed_structured_output,
                )
            logger.info(
                "Normalized agent reply task_type=%s cleanup_applied=%s parsed_status_before_validation=%s parse_status=%s schema_applied=%s",
                task_type,
                cleanup_applied or extraction_applied,
                parsed_status,
                parse_status,
                schema_applied or "none",
            )
            return {
                "reply": formatted_reply or reply_without_embedded_json or reply,
                "parse_status": parse_status,
                "structured_output": structured_output,
                "cleanup_applied": cleanup_applied or extraction_applied,
                "schema_applied": schema_applied,
            }
        except Exception as exc:
            logger.warning(
                "Agent reply normalization failed task_type=%s error=%s",
                task_type,
                exc,
            )
            return {
                "reply": reply,
                "parse_status": "raw_only",
                "structured_output": None,
                "cleanup_applied": False,
                "schema_applied": None,
            }

    def _extract_embedded_structured_output(self, task_type: str, reply: str) -> tuple[str, dict | None, bool]:
        if task_type not in {"reel_idea", "reel_script", "reel_feedback"}:
            return reply, None, False

        marker_index = reply.rfind(self._structured_output_marker)
        if marker_index < 0:
            return reply, None, False

        candidate_reply = reply[:marker_index].rstrip()
        candidate_payload = reply[marker_index + len(self._structured_output_marker):].strip()
        extracted_payload = self._parse_embedded_json_object(candidate_payload)
        if not isinstance(extracted_payload, dict):
            return reply, None, False

        return candidate_reply or reply, extracted_payload, True

    def _parse_embedded_json_object(self, value: str) -> dict | None:
        candidate = value.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()

        first_brace_index = candidate.find("{")
        if first_brace_index < 0:
            return None

        candidate = candidate[first_brace_index:]
        try:
            parsed_value, end_index = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None

        trailing_text = candidate[end_index:].strip()
        if trailing_text and trailing_text != "```":
            return None

        return parsed_value if isinstance(parsed_value, dict) else None

    def format_reply(self, task_type: str, reply: str | None) -> tuple[str | None, bool]:
        if reply is None:
            return None, False

        normalized_reply = self._prepare_task_text(task_type, reply)
        formatted_reply = self._format_by_task_type(task_type, normalized_reply)
        final_reply = self._cleanup_spacing(formatted_reply)
        return final_reply, final_reply != reply

    def _normalize_reply(self, reply: str) -> str:
        normalized_reply = self._decode_escaped_text(reply.strip())
        normalized_reply = normalized_reply.replace("\r\n", "\n").replace("\r", "\n")
        normalized_reply = self._strip_markdown_noise(normalized_reply)
        normalized_reply = self._strip_separator_lines(normalized_reply)

        normalized_lines = []
        for line in normalized_reply.split("\n"):
            if not line.strip():
                normalized_lines.append("")
                continue

            collapsed_line = re.sub(r"[ \t]+", " ", line).strip()
            normalized_lines.append(collapsed_line)

        return self._cleanup_spacing("\n".join(normalized_lines))

    def _prepare_task_text(self, task_type: str, reply: str) -> str:
        normalized_reply = self._normalize_reply(reply)
        headings = self._task_sections.get(task_type, [])
        normalized_reply = self._insert_line_breaks_before_headings(normalized_reply, headings)
        return self._cleanup_spacing(normalized_reply)

    def _strip_markdown_noise(self, text: str) -> str:
        cleaned_text = text.replace("\u200b", "")
        cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned_text)
        cleaned_text = re.sub(r"__(.*?)__", r"\1", cleaned_text)
        cleaned_text = re.sub(r"`([^`]+)`", r"\1", cleaned_text)
        return cleaned_text

    def _strip_separator_lines(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            if re.fullmatch(r"\s*(?:[-_*]{3,}|[=]{3,})\s*", line):
                lines.append("")
                continue
            lines.append(line)
        return "\n".join(lines)

    def _insert_line_breaks_before_headings(self, text: str, headings: list[str]) -> str:
        prepared_text = text
        for heading in sorted(headings, key=len, reverse=True):
            escaped_heading = re.escape(heading)
            heading_guard = r"(?<!1-second)(?<!1-second )" if heading.lower() == "hook:" else ""
            prepared_text = re.sub(
                rf"(?<!^)(?<!\n){heading_guard}\s*(\*\*{escaped_heading}\*\*|{escaped_heading})",
                r"\n\n\1",
                prepared_text,
                flags=re.IGNORECASE,
            )
        return prepared_text

    def _decode_escaped_text(self, value: str) -> str:
        if not value:
            return value

        if value.startswith('"') and value.endswith('"') and any(token in value for token in ('\\n', '\\"', '\\t')):
            try:
                decoded_value = json.loads(value)
                if isinstance(decoded_value, str):
                    return decoded_value
            except Exception:
                pass

        if "\n" not in value and value.count("\\n") >= 2:
            return (
                value.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
            )

        return value

    def _format_by_task_type(self, task_type: str, text: str) -> str:
        headings = self._task_sections.get(task_type)
        if not headings:
            return text

        preamble, sections = self._extract_sections(text, headings)
        if len(sections) < 2:
            return text

        formatted_sections = [self._render_section(section["heading"], section["body"]) for section in sections]
        parts = []
        if preamble:
            parts.append(preamble)
        parts.extend(part for part in formatted_sections if part)
        return "\n\n".join(parts)

    def _parse_task_output(self, task_type: str, text: str) -> tuple[dict | None, str]:
        parsers = {
            "reel_idea": self._parse_reel_ideas,
            "reel_script": self._parse_reel_script,
            "reel_feedback": self._parse_reel_feedback,
            "carousel": self._parse_carousel,
            "caption": self._parse_caption,
            "profile_audit": self._parse_profile_audit,
            "content_plan": self._parse_content_plan,
            "link_analysis": self._parse_link_analysis,
            "performance_summary": self._parse_performance_summary,
        }
        parser = parsers.get(task_type)
        if not parser:
            return None, "raw_only"

        return parser(text)

    def _extract_sections(self, text: str, headings: list[str]) -> tuple[str, list[dict]]:
        pattern = self._build_heading_pattern(headings)
        matches = list(pattern.finditer(text))
        if not matches:
            return self._clean_value_text(text, headings=headings, preserve_newlines=True) or "", []

        preamble = self._clean_value_text(
            text[:matches[0].start()],
            headings=headings,
            preserve_newlines=True,
        ) or ""
        sections = []
        for index, match in enumerate(matches):
            section_start = match.end()
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[section_start:section_end].strip(" \n\t:-")
            sections.append({
                "heading": self._canonical_heading(match.group("heading"), headings),
                "body": self._normalize_section_body(body, headings),
            })

        return preamble, sections

    def _build_heading_pattern(self, headings: list[str]) -> re.Pattern[str]:
        variants = []
        for heading in sorted(headings, key=len, reverse=True):
            escaped_heading = (
                rf"{re.escape(heading[:-1])}\s*:?"
                if heading.endswith(":") and re.search(r"[\u10A0-\u10FF]", heading)
                else re.escape(heading)
            )
            variants.append(rf"\*\*{escaped_heading}\*\*")
            variants.append(escaped_heading)

        return re.compile(
            rf"^[ \t>*#-]*(?:\d+[.)]\s*)?(?:[^\w\u10A0-\u10FF\n]+)?\s*(?P<heading>{'|'.join(variants)})",
            flags=re.IGNORECASE | re.MULTILINE,
        )

    def _canonical_heading(self, raw_heading: str, headings: list[str]) -> str:
        normalized_heading = raw_heading.replace("*", "").strip()
        for heading in headings:
            if heading.lower().rstrip(":") == normalized_heading.lower().rstrip(":"):
                return heading
        return normalized_heading

    def _render_section(self, heading: str, body: str) -> str:
        if not body:
            return heading
        return f"{heading}\n{body}"

    def _clean_heading_prefixes(self, value: str, headings: list[str] | None = None) -> str:
        cleaned_value = value
        for heading in headings or []:
            escaped_heading = re.escape(heading)
            cleaned_value = re.sub(
                rf"^(?:\*\*{escaped_heading}\*\*|{escaped_heading})\s*",
                "",
                cleaned_value,
                flags=re.IGNORECASE,
            )
        return cleaned_value

    def _strip_leading_noise(self, value: str, *, strip_numbering: bool = True) -> str:
        cleaned_value = value.strip()
        cleaned_value = re.sub(r"^[>\-–—*•]+\s*", "", cleaned_value)
        cleaned_value = re.sub(r"^(?:[👉✅⚠️🔥📌📍✨🚀🎯💡💬📝🏁⚡️❗‼️☑️✅☝️]+)\s*", "", cleaned_value)
        if strip_numbering:
            cleaned_value = re.sub(r"^\d+[.)]\s*", "", cleaned_value)
        return cleaned_value.strip()

    def _clean_value_text(
        self,
        value: str | None,
        *,
        headings: list[str] | None = None,
        preserve_newlines: bool = False,
        strip_numbering: bool = False,
    ) -> str | None:
        if value is None:
            return None

        cleaned_value = self._strip_markdown_noise(value)
        cleaned_value = self._strip_separator_lines(cleaned_value)
        cleaned_value = cleaned_value.replace("\r\n", "\n").replace("\r", "\n")

        cleaned_lines = []
        for raw_line in cleaned_value.split("\n"):
            line = raw_line.strip()
            if not line:
                if preserve_newlines and cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            line = self._clean_heading_prefixes(line, headings)
            line = self._strip_leading_noise(line, strip_numbering=strip_numbering)
            line = re.sub(r"[ \t]+", " ", line).strip(" -:\t")
            if not line:
                continue
            cleaned_lines.append(line)

        if preserve_newlines:
            result = "\n".join(cleaned_lines).strip()
        else:
            result = " ".join(line for line in cleaned_lines if line).strip()
        return result or None

    def _clean_reel_idea_field_text(
        self,
        value: str | None,
        *,
        preserve_newlines: bool = False,
        strip_numbering: bool = False,
    ) -> str | None:
        if value is None:
            return None

        cleaned_lines = []
        for raw_line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            normalized_line = raw_line.strip()
            if re.fullmatch(r"(?:#{1,6}\s*)?reel\s+\d+\s*:?", normalized_line, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"(?:#{1,6}\s*)?idea\s+\d+\s*:?", normalized_line, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"(?:#{1,6}\s*)?summary\s*:?", normalized_line, flags=re.IGNORECASE):
                continue
            cleaned_lines.append(raw_line)

        return self._clean_value_text(
            "\n".join(cleaned_lines),
            preserve_newlines=preserve_newlines,
            strip_numbering=strip_numbering,
        )

    def _normalize_section_body(self, body: str, headings: list[str] | None = None) -> str:
        normalized_body = body.strip()
        normalized_body = normalized_body.replace("\r\n", "\n").replace("\r", "\n")

        if "\n" not in normalized_body and normalized_body.count(" - ") >= 2:
            normalized_body = normalized_body.replace(" - ", "\n- ")

        cleaned_body = self._clean_value_text(
            normalized_body,
            headings=headings,
            preserve_newlines=True,
            strip_numbering=False,
        )
        return self._cleanup_spacing(cleaned_body or "")

    def _cleanup_spacing(self, text: str) -> str:
        cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def _derive_title(self, body: str | None, fallback: str) -> str:
        source_text = self._clean_value_text(body) or ""
        if not source_text:
            return fallback

        source_text = source_text.split("\n", 1)[0].strip(" -:")
        short_title = source_text.split(".", 1)[0].strip()
        if len(short_title) > 70:
            short_title = f"{short_title[:67].rstrip()}..."
        return short_title or fallback

    def _split_listish_text(self, body: str) -> list[str]:
        normalized_body = self._cleanup_spacing(body.strip())
        if not normalized_body:
            return []

        candidates = []
        lines = [line.strip() for line in normalized_body.split("\n") if line.strip()]
        for line in lines:
            numbered_parts = re.split(r"(?=\d+[.)]\s)", line)
            if len(numbered_parts) > 1:
                candidates.extend(part.strip() for part in numbered_parts if part.strip())
                continue

            if " • " in line:
                candidates.extend(part.strip() for part in line.split(" • ") if part.strip())
                continue

            if line.count(" - ") >= 1:
                candidates.extend(part.strip() for part in line.split(" - ") if part.strip())
                continue

            if "; " in line:
                candidates.extend(part.strip() for part in line.split("; ") if part.strip())
                continue

            candidates.append(line)

        cleaned_items = []
        for candidate in candidates:
            cleaned_candidate = self._clean_value_text(candidate, strip_numbering=True)
            if cleaned_candidate and re.fullmatch(
                r"(?:what\s+works|what\s+is\s+weak|retention\s+issues|hook\s+improvement|cta\s+improvement|improved\s+version)\s*/?",
                cleaned_candidate,
                flags=re.IGNORECASE,
            ):
                continue
            if cleaned_candidate:
                cleaned_items.append(cleaned_candidate)

        return cleaned_items

    def _clean_reel_script_items(self, body: str) -> list[str]:
        cleaned_items = []
        for item in self._split_listish_text(body):
            cleaned_item = re.sub(r"^(?:scene|shot)\s*\d+\s*[:.)-]\s*", "", item, flags=re.IGNORECASE).strip()
            cleaned_item = re.sub(
                r"^(?:scene[-\s]*by[-\s]*scene(?:\s+script)?|structure|shot\s+list)\s*[:\-]\s*",
                "",
                cleaned_item,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return cleaned_items

    def _section_body_from_sections(self, sections: list[dict], aliases: list[str]) -> str:
        for section in sections:
            if section.get("heading") in aliases and section.get("body"):
                return str(section["body"])
        return ""

    def _section_bodies_from_sections(self, sections: list[dict], aliases: list[str]) -> list[str]:
        return [
            str(section["body"])
            for section in sections
            if section.get("heading") in aliases and section.get("body")
        ]

    def _reel_script_candidate_lines(self, body: str) -> list[str]:
        normalized_body = self._cleanup_spacing(str(body or "").replace("\r\n", "\n").replace("\r", "\n"))
        if not normalized_body:
            return []

        candidates = []
        for line in [line.strip() for line in normalized_body.split("\n") if line.strip()]:
            numbered_parts = re.split(r"(?=\d+[.)]\s)", line)
            if len(numbered_parts) > 1:
                candidates.extend(part.strip() for part in numbered_parts if part.strip())
                continue
            candidates.append(line)

        cleaned_lines = []
        for candidate in candidates:
            cleaned_candidate = self._clean_value_text(candidate, strip_numbering=False)
            if cleaned_candidate:
                cleaned_lines.append(cleaned_candidate)
        return cleaned_lines

    def _reel_script_inline_label(self, item: str) -> tuple[str | None, str]:
        label_pattern = re.compile(
            r"^(?P<label>"
            r"scene[-\s]*by[-\s]*scene(?:\s+script)?|structure|shot\s+list|"
            r"on[-\s]*screen\s+(?:text|copy)|voice\s*over|voiceover|"
            r"caption|cta|problem\s*/?\s*angle|problem\s+angle|hook"
            r")\s*(?:\([^)]*\))?\s*[:\-]\s*(?P<body>.*)$",
            flags=re.IGNORECASE,
        )
        label_only_pattern = re.compile(
            r"^(?P<label>"
            r"scene[-\s]*by[-\s]*scene(?:\s+script)?|structure|shot\s+list|"
            r"on[-\s]*screen\s+(?:text|copy)|voice\s*over|voiceover"
            r")\s*(?:\([^)]*\))?\s*$",
            flags=re.IGNORECASE,
        )
        stripped_item = item.strip()
        match = label_pattern.match(stripped_item) or label_only_pattern.match(stripped_item)
        if not match:
            return None, item

        label = re.sub(r"\s+", " ", match.group("label").strip().lower())
        body = (match.groupdict().get("body") or "").strip()
        if label.startswith("on"):
            return "on_screen_text", body
        if label in {"voiceover", "voice over"}:
            return "voiceover", body
        if label.startswith("scene") or label == "structure" or label == "shot list":
            return "structure", body
        return label, body

    def _strip_reel_script_scene_prefix(self, item: str) -> tuple[bool, str]:
        cleaned_item = item.strip()
        scene_pattern = re.compile(
            r"^(?:(?:scene|shot|სცენა|კადრი)\s*\d+|\d+)[.)\s:\-–—]*",
            flags=re.IGNORECASE,
        )
        matched = bool(scene_pattern.match(cleaned_item))
        cleaned_item = scene_pattern.sub("", cleaned_item, count=1).strip()
        cleaned_item = re.sub(
            r"^(?:scene|shot|სცენა|კადრი)\s*\d+[.)\s:\-–—]*",
            "",
            cleaned_item,
            flags=re.IGNORECASE,
        ).strip()
        return matched, cleaned_item

    def _is_reel_script_noise_line(self, item: str) -> bool:
        normalized_item = item.strip().strip(":")
        if not normalized_item:
            return True

        normalized_lower = normalized_item.lower()
        if re.fullmatch(
            r"(?:on[-\s]*screen\s+(?:text|copy)|voice\s*over|voiceover|"
            r"scene[-\s]*by[-\s]*scene(?:\s+script)?|structure|shot\s+list)"
            r"\s*(?:\([^)]*\))?",
            normalized_lower,
            flags=re.IGNORECASE,
        ):
            return True

        noise_markers = [
            "voiceover",
            "voice over",
            "on-screen text",
            "on screen text",
            "tone",
            "rhythm",
            "რიტმი",
            "ტონი",
            "მთელი reel განმავლობაში",
            "მთლიანი reel",
        ]
        return any(marker in normalized_lower for marker in noise_markers)

    def _append_unique_reel_script_item(self, items: list[str], value: str | None) -> None:
        cleaned_value = self._clean_value_text(value, strip_numbering=True)
        if not cleaned_value or self._is_reel_script_noise_line(cleaned_value):
            return
        if cleaned_value not in items:
            items.append(cleaned_value)

    def _parse_reel_script_mixed_block(self, body: str, *, default_channel: str = "structure") -> dict[str, list[str]]:
        parsed = {
            "structure": [],
            "shot_list": [],
            "voiceover_items": [],
            "on_screen_text": [],
        }
        current_channel = default_channel

        for item in self._reel_script_candidate_lines(body):
            scene_matched, scene_body = self._strip_reel_script_scene_prefix(item)
            label, label_body = self._reel_script_inline_label(scene_body if scene_matched else item)

            if scene_matched:
                current_channel = "structure"
                if label in {"voiceover", "on_screen_text"}:
                    current_channel = label
                    target = parsed["voiceover_items"] if label == "voiceover" else parsed["on_screen_text"]
                    self._append_unique_reel_script_item(target, label_body)
                    continue
                scene_text = label_body if label == "structure" else scene_body
                self._append_unique_reel_script_item(parsed["structure"], scene_text)
                self._append_unique_reel_script_item(parsed["shot_list"], scene_text)
                continue

            if label in {"voiceover", "on_screen_text", "structure"}:
                current_channel = label
                if label == "voiceover":
                    self._append_unique_reel_script_item(parsed["voiceover_items"], label_body)
                elif label == "on_screen_text":
                    self._append_unique_reel_script_item(parsed["on_screen_text"], label_body)
                else:
                    self._append_unique_reel_script_item(parsed["structure"], label_body)
                    self._append_unique_reel_script_item(parsed["shot_list"], label_body)
                continue

            if label in {"caption", "cta", "hook"} or (label and label.startswith("problem")):
                continue

            if current_channel == "voiceover":
                self._append_unique_reel_script_item(parsed["voiceover_items"], item)
            elif current_channel == "on_screen_text":
                self._append_unique_reel_script_item(parsed["on_screen_text"], item)
            else:
                self._append_unique_reel_script_item(parsed["structure"], item)
                self._append_unique_reel_script_item(parsed["shot_list"], item)

        return parsed

    def _join_caption_parts(self, hook: str | None, body: str | None, cta: str | None) -> str | None:
        parts = [part for part in [hook, body, cta] if part]
        if not parts:
            return None
        return "\n\n".join(parts)

    def _first_section_body(self, section_map: dict[str, str], aliases: list[str]) -> str:
        for alias in aliases:
            body = section_map.get(alias)
            if body:
                return body
        return ""

    def _extract_labeled_list_items(self, body: str, labels: list[str]) -> list[str]:
        extracted_items = []
        normalized_labels = [label.lower().rstrip(":").strip() for label in labels]
        for item in self._split_listish_text(body):
            normalized_item = item.lower()
            for label in normalized_labels:
                match = re.match(rf"^{re.escape(label)}\s*[:\-]\s*(.+)$", normalized_item, flags=re.IGNORECASE)
                if not match:
                    continue
                cleaned_item = self._clean_value_text(item[match.start(1):])
                if cleaned_item:
                    extracted_items.append(cleaned_item)
                break
        return extracted_items

    def _join_script_parts(self, title: str | None, hook: str | None, script_sections: list[str], cta: str | None) -> str | None:
        parts = []
        first_section = script_sections[0] if script_sections else None
        if title and title != hook and title != first_section:
            parts.append(title)
        if hook:
            parts.append(hook)
        if script_sections:
            parts.append("\n".join(section for section in script_sections if section))
        if cta:
            parts.append(cta)
        if not parts:
            return None
        return "\n\n".join(parts)

    def _split_headline_and_body(self, value: str | None) -> tuple[str | None, str | None]:
        cleaned_value = self._clean_value_text(value, preserve_newlines=True)
        if not cleaned_value:
            return None, None

        parts = [part.strip() for part in cleaned_value.split("\n") if part.strip()]
        if not parts:
            return None, None
        if len(parts) == 1:
            return parts[0], None
        return parts[0], "\n".join(parts[1:]).strip() or None

    def _infer_content_format(self, value: str | None) -> str | None:
        normalized_value = (value or "").lower()
        format_keywords = [
            ("Reel", ("reel", "video", "short-form")),
            ("Carousel", ("carousel", "slides")),
            ("Story", ("story", "stories")),
            ("Post", ("post", "feed post", "static post")),
            ("Live", ("live", "livestream")),
        ]
        for format_name, keywords in format_keywords:
            if any(keyword in normalized_value for keyword in keywords):
                return format_name
        return None

    def _infer_content_goal(self, value: str | None) -> str | None:
        normalized_value = (value or "").lower()
        goal_keywords = [
            ("drive conversion", ("conversion", "dm", "book", "buy", "lead", "sale", "offer", "pain point")),
            ("educate", ("educat", "teach", "framework", "tip", "how to", "mistake")),
            ("build trust", ("trust", "proof", "objection", "testimonial", "story q&a", "q&a")),
            ("increase engagement", ("engagement", "comment", "save", "share", "question")),
            ("build awareness", ("awareness", "introduce", "visibility", "reach")),
        ]
        for goal_name, keywords in goal_keywords:
            if any(keyword in normalized_value for keyword in keywords):
                return goal_name
        return None

    def _extract_best_cta_direction(self, recommended_changes: list[str]) -> str | None:
        cta_keywords = ("cta", "dm", "follow", "save", "comment", "book", "link", "bio")
        for item in recommended_changes:
            normalized_item = (item or "").lower()
            if any(keyword in normalized_item for keyword in cta_keywords):
                return item
        return recommended_changes[0] if recommended_changes else None

    def _strip_format_from_topic(self, value: str | None, format_name: str | None) -> str | None:
        cleaned_value = self._clean_value_text(value)
        if not cleaned_value or not format_name:
            return cleaned_value

        pattern = re.compile(rf"^{re.escape(format_name)}\s+(?:about|on|for|with)\s+", flags=re.IGNORECASE)
        stripped_value = pattern.sub("", cleaned_value).strip()
        return stripped_value or cleaned_value

    def _parse_reel_ideas(self, text: str) -> tuple[dict | None, str]:
        idea_block_output, idea_block_status = self._parse_reel_idea_blocks(text)
        if idea_block_output is not None:
            return idea_block_output, idea_block_status

        headings = self._task_sections["reel_idea"]
        _, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        ideas = []
        current_idea = {}
        idea_index = 1

        for section in sections:
            heading = section["heading"]
            body = section["body"]

            if heading == "Title:":
                if current_idea:
                    ideas.append(self._finalize_reel_idea(current_idea, idea_index))
                    idea_index += 1
                    current_idea = {}
                current_idea["title"] = body or None
                continue

            if heading in {"1-second hook:", "Hook:"}:
                if current_idea.get("hook"):
                    ideas.append(self._finalize_reel_idea(current_idea, idea_index))
                    idea_index += 1
                    current_idea = {}
                current_idea["hook"] = body or None
            elif heading == "Format type:":
                current_idea["format_type"] = body or None
            elif heading == "Main idea:":
                current_idea["main_idea"] = body or None
            elif heading == "Shot list:":
                current_idea["shot_list"] = self._split_listish_text(body)
            elif heading == "Why it can work:":
                current_idea["why_it_can_work"] = body or None
            elif heading == "Caption idea:":
                current_idea["caption_idea"] = body or None
            elif heading == "CTA:":
                current_idea["cta"] = body or None

        if current_idea:
            ideas.append(self._finalize_reel_idea(current_idea, idea_index))

        ideas = [idea for idea in ideas if any([
            idea.get("title"),
            idea.get("hook"),
            idea.get("format_type"),
            idea.get("main_idea"),
            idea.get("shot_list"),
            idea.get("why_it_can_work"),
            idea.get("caption_idea"),
            idea.get("cta"),
        ])]
        if not ideas:
            return None, "raw_only"

        fully_parsed = all(self._is_complete_reel_idea(idea) for idea in ideas)
        return {"ideas": ideas}, "parsed" if fully_parsed else "partial"

    def _parse_reel_idea_blocks(self, text: str) -> tuple[dict | None, str]:
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized_text = re.sub(
            r"(?im)^([ \t>*#-]*(?:\*{1,2})?idea\s+\d+\s*(?:[:.)-])?(?:\*{1,2})?)\s+(Title:)",
            r"\1\n\2",
            normalized_text,
        )
        idea_heading_pattern = re.compile(
            r"^[ \t>*#-]*(?:\*{1,2})?idea\s+\d+\s*(?:[:.)-])?(?:\*{1,2})?\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        matches = list(idea_heading_pattern.finditer(normalized_text))
        if not matches:
            return None, "raw_only"

        ideas = []
        for index, match in enumerate(matches, start=1):
            block_start = match.end()
            block_end = matches[index].start() if index < len(matches) else len(normalized_text)
            block = normalized_text[block_start:block_end].strip()
            if not block:
                continue

            parsed_fields = self._parse_reel_idea_block_fields(block)
            finalized_idea = self._finalize_reel_idea(parsed_fields, index)
            if any([
                finalized_idea.get("title"),
                finalized_idea.get("hook"),
                finalized_idea.get("format_type"),
                finalized_idea.get("main_idea"),
                finalized_idea.get("shot_list"),
                finalized_idea.get("why_it_can_work"),
                finalized_idea.get("caption_idea"),
                finalized_idea.get("cta"),
            ]):
                ideas.append(finalized_idea)

        if not ideas:
            return None, "raw_only"

        complete_ideas = [idea for idea in ideas if self._is_complete_reel_idea(idea, require_caption_idea=True)]
        if len(ideas) == 3 and len(complete_ideas) == 3:
            return {"ideas": ideas}, "parsed"
        if 1 <= len(ideas) <= 2 or complete_ideas:
            return {"ideas": ideas}, "partial"
        return {"ideas": ideas}, "partial"

    def _parse_reel_idea_block_fields(self, block: str) -> dict:
        label_to_field = {
            "title": "title",
            "hook": "hook",
            "1-second hook": "hook",
            "format type": "format_type",
            "main idea": "main_idea",
            "shot list": "shot_list",
            "why it can work": "why_it_can_work",
            "caption idea": "caption_idea",
            "cta": "cta",
        }
        label_pattern = re.compile(
            r"^[ \t>*-]*(?:\*{1,2})?(?P<label>Title|Hook|1-second hook|Format type|Main idea|Shot list|Why it can work|Caption idea|CTA)\s*:\s*(?:\*{1,2})?",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        matches = list(label_pattern.finditer(block))
        parsed_fields: dict[str, object] = {}
        for index, match in enumerate(matches):
            raw_label = match.group("label").strip().lower()
            field_name = label_to_field.get(raw_label)
            if not field_name:
                continue
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            raw_value = block[value_start:value_end].strip()
            if field_name == "shot_list":
                parsed_fields[field_name] = self._split_listish_text(raw_value)
            else:
                parsed_fields[field_name] = raw_value or None
        return parsed_fields

    def _finalize_reel_idea(self, idea: dict, index: int) -> dict:
        title = self._clean_reel_idea_field_text(idea.get("title"))
        hook = self._clean_reel_idea_field_text(idea.get("hook"))
        format_type = self._clean_reel_idea_field_text(idea.get("format_type"))
        main_idea = self._clean_reel_idea_field_text(idea.get("main_idea"), preserve_newlines=True)
        why_it_can_work = self._clean_reel_idea_field_text(idea.get("why_it_can_work"), preserve_newlines=True)
        caption_idea = self._clean_reel_idea_field_text(idea.get("caption_idea"), preserve_newlines=True)
        cta = self._clean_reel_idea_field_text(idea.get("cta"))
        shot_list = [item for item in (self._split_listish_text("\n".join(idea.get("shot_list") or [])) if idea.get("shot_list") else []) if item]
        return {
            "title": title or self._derive_title(main_idea or hook, f"Idea {index}"),
            "hook": hook,
            "format_type": format_type,
            "main_idea": main_idea,
            "shot_list": shot_list,
            "why_it_can_work": why_it_can_work,
            "caption_idea": caption_idea,
            "cta": cta,
        }

    def _is_complete_reel_idea(self, idea: dict, *, require_caption_idea: bool = False) -> bool:
        required_fields = [
            idea.get("title"),
            idea.get("hook"),
            idea.get("format_type"),
            idea.get("main_idea"),
            idea.get("shot_list"),
            idea.get("why_it_can_work"),
            idea.get("cta"),
        ]
        if require_caption_idea:
            required_fields.append(idea.get("caption_idea"))
        return all(required_fields)

    def _parse_performance_summary(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["performance_summary"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        summary_body = self._section_body_from_sections(sections, ["Summary:"])
        structured_output = {
            "what_worked": self._split_listish_text(self._section_body_from_sections(sections, [
                "What worked:",
                "რა იმუშავა:",
            ])),
            "what_did_not_work": self._split_listish_text(self._section_body_from_sections(sections, [
                "What did not work:",
                "რა არ იმუშავა:",
            ])),
            "content_patterns": self._split_listish_text(self._section_body_from_sections(sections, [
                "Content patterns:",
                "კონტენტის პატერნები:",
            ])),
            "best_opportunities": self._split_listish_text(self._section_body_from_sections(sections, [
                "Best opportunities:",
                "საუკეთესო შესაძლებლობები:",
            ])),
            "recommended_next_moves": self._split_listish_text(self._section_body_from_sections(sections, [
                "Recommended next moves:",
                "Next actions:",
                "შემდეგი ნაბიჯები:",
            ])),
            "summary": self._clean_value_text(summary_body or preamble, preserve_newlines=True),
        }

        populated_sections = sum(
            1 for key in ["what_worked", "what_did_not_work", "content_patterns", "best_opportunities", "recommended_next_moves"]
            if structured_output.get(key)
        )
        if populated_sections == 0 and not structured_output.get("summary"):
            return None, "raw_only"

        parse_status = "parsed" if populated_sections >= 3 else "partial"
        return structured_output, parse_status

    def _parse_reel_script(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["reel_script"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        problem_angle = self._clean_value_text(self._section_body_from_sections(sections, [
            "Problem/angle:",
            "Problem / angle:",
            "Problem angle:",
        ]), preserve_newlines=True)

        structure = []
        shot_list = []
        voiceover_items = []
        on_screen_text = []

        scene_bodies = self._section_bodies_from_sections(sections, [
            "Scene-by-scene script:",
            "Scene by scene script:",
            "Structure:",
        ])
        for body in scene_bodies:
            parsed_block = self._parse_reel_script_mixed_block(body, default_channel="structure")
            structure.extend(item for item in parsed_block["structure"] if item not in structure)
            shot_list.extend(item for item in parsed_block["shot_list"] if item not in shot_list)
            voiceover_items.extend(item for item in parsed_block["voiceover_items"] if item not in voiceover_items)
            on_screen_text.extend(item for item in parsed_block["on_screen_text"] if item not in on_screen_text)

        for body in self._section_bodies_from_sections(sections, [
            "Voiceover:",
            "Voice over:",
        ]):
            parsed_block = self._parse_reel_script_mixed_block(body, default_channel="voiceover")
            structure.extend(item for item in parsed_block["structure"] if item not in structure)
            shot_list.extend(item for item in parsed_block["shot_list"] if item not in shot_list)
            voiceover_items.extend(item for item in parsed_block["voiceover_items"] if item not in voiceover_items)
            on_screen_text.extend(item for item in parsed_block["on_screen_text"] if item not in on_screen_text)

        explicit_shot_list = []
        for body in self._section_bodies_from_sections(sections, ["Shot list:"]):
            parsed_block = self._parse_reel_script_mixed_block(body, default_channel="structure")
            explicit_shot_list.extend(item for item in parsed_block["shot_list"] if item not in explicit_shot_list)
            structure.extend(item for item in parsed_block["structure"] if item not in structure)

        for body in self._section_bodies_from_sections(sections, [
            "On-screen text:",
            "On screen text:",
        ]):
            parsed_block = self._parse_reel_script_mixed_block(body, default_channel="on_screen_text")
            structure.extend(item for item in parsed_block["structure"] if item not in structure)
            shot_list.extend(item for item in parsed_block["shot_list"] if item not in shot_list)
            voiceover_items.extend(item for item in parsed_block["voiceover_items"] if item not in voiceover_items)
            on_screen_text.extend(item for item in parsed_block["on_screen_text"] if item not in on_screen_text)

        voiceover = self._clean_value_text("\n".join(voiceover_items), preserve_newlines=True)
        hook = self._clean_value_text(self._section_body_from_sections(sections, ["Hook:"]), preserve_newlines=True)
        caption = self._clean_value_text(self._section_body_from_sections(sections, ["Caption:"]), preserve_newlines=True)
        cta = self._clean_value_text(self._section_body_from_sections(sections, ["CTA:"]), preserve_newlines=True)
        title = self._clean_value_text(preamble) or problem_angle or self._derive_title(hook, "Reel Script")

        structured_output = {
            "script": {
                "hook": hook,
                "problem_angle": problem_angle,
                "structure": structure,
                "voiceover": voiceover,
                "shot_list": explicit_shot_list or shot_list or structure,
                "on_screen_text": on_screen_text,
                "caption": caption,
                "cta": cta,
            }
        }
        populated_sections = sum(
            1 for key in ["hook", "problem_angle", "structure", "voiceover", "shot_list", "on_screen_text", "caption", "cta"]
            if structured_output["script"].get(key)
        )
        return (
            structured_output if populated_sections else None,
            "parsed" if hook and structure and cta else "partial" if populated_sections else "raw_only",
        )

    def _parse_reel_feedback(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["reel_feedback"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        section_map = {section["heading"]: section["body"] for section in sections if section["body"]}
        summary_body = self._first_section_body(section_map, ["Summary:"])
        what_works_body = self._first_section_body(section_map, [
            "What works:",
            "What works / რა მუშაობს:",
            "რა მუშაობს:",
            "რა არის ძლიერი:",
            "რა არის კარგი:",
            "რას აკეთებს ეს Reel კარგად:",
            "რა მუშაობს კარგად:",
            "კარგად მუშაობს:",
            "ძლიერი მხარეები:",
        ])
        what_hurts_body = self._first_section_body(section_map, [
            "What hurts:",
            "What is weak:",
            "What is weak / რა სუსტია:",
            "რა ასუსტებს:",
            "რა სუსტია:",
            "რა ასუსტებს შედეგს:",
            "რა ასუსტებს ამ Reel-ს:",
            "რა ასუსტებს ვიდეოს:",
            "რა არ მუშაობს:",
            "რა უშლის შედეგს:",
            "სუსტი მხარეები:",
        ])
        retention_body = self._first_section_body(section_map, [
            "Retention risks:",
            "Retention issues:",
            "Retention issues / რიტენშენის პრობლემა:",
            "Retention:",
            "რიტენშენის პრობლემა:",
            "რიტენშენის პრობლემები:",
            "Retention პრობლემები:",
            "Retention-ის პრობლემები:",
            "Retention-ის პრობლემა:",
            "Retention / ყურადღების შენარჩუნება:",
            "Retention / ყურადღების შენარჩუნების პრობლემა:",
            "ყურადღების შენარჩუნების პრობლემა:",
            "ყურადღების შენარჩუნების პრობლემები:",
            "სად იკარგება ყურადღება:",
            "სად იკარგება ყურადღება / retention:",
            "სად იკარგება retention:",
            "სად ვკარგავთ ყურადღებას:",
            "სად იკარგება მაყურებელი:",
            "ყურადღება სად იკარგება:",
        ])
        hook_improvement_body = self._first_section_body(section_map, [
            "Better hook:",
            "Hook improvement:",
            "Hook improvement / Hook-ის გაუმჯობესება:",
            "Hook-ის გაუმჯობესება:",
            "ჰუკის გაუმჯობესება:",
            "უკეთესი ჰუკი:",
            "უკეთესი hook:",
            "ახალი ჰუკი:",
            "ახალი hook:",
            "როგორ გავაუმჯობესოთ ჰუკი:",
            "როგორ გავაუმჯობესოთ hook:",
            "ჰუკი როგორ გაუმჯობესდეს:",
        ])
        cta_improvement_body = self._first_section_body(section_map, [
            "Better CTA:",
            "CTA improvement:",
            "CTA improvement / CTA-ის გაუმჯობესება:",
            "CTA clarity:",
            "CTA-ს გაუმჯობესება:",
            "CTA-ის გაუმჯობესება:",
            "CTA-ს სიცხადე:",
            "CTA-ის სიცხადე:",
            "CTA-ის სიცხადის გაუმჯობესება:",
            "CTA როგორ გაუმჯობესდეს:",
            "როგორ გავაუმჯობესოთ CTA:",
            "უკეთესი CTA:",
            "ახალი CTA:",
        ])
        improved_version_body = self._first_section_body(section_map, [
            "Improved version:",
            "Improved version / გაუმჯობესებული ვერსია:",
            "გაუმჯობესებული ვერსია:",
            "გაუმჯობესებული სცენარი:",
            "გაუმჯობესებული Reel სცენარი:",
            "გაუმჯობესებული ვარიანტი:",
            "ახალი ვერსია:",
            "ახალი სცენარი:",
            "საბოლოო ვერსია:",
            "საბოლოო სცენარი:",
            "გადაწერილი სცენარი:",
            "სცენარის ახალი ვერსია:",
            "სცენარის გაუმჯობესება:",
            "სცენარის გაუმჯობესებული ვერსია:",
            "როგორ გავაუმჯობესო სცენარი:",
            "როგორ გავაუმჯობესოთ სცენარი:",
            "როგორ უნდა გაუმჯობესო სცენარი:",
            "როგორ უნდა გაუმჯობესდეს სცენარი:",
        ])
        summary = self._clean_value_text(
            summary_body or preamble,
            headings=headings,
            preserve_newlines=True,
        )
        structured_output = {
            "what_works": self._split_listish_text(what_works_body),
            "what_hurts": self._split_listish_text(what_hurts_body),
            "retention_issues": self._split_listish_text(retention_body),
            "hook_improvement": self._clean_value_text(hook_improvement_body, preserve_newlines=True),
            "improved_structure": self._split_listish_text(section_map.get("Improved structure:", "")),
            "cta_improvement": self._clean_value_text(cta_improvement_body, preserve_newlines=True),
            "improved_version": self._clean_value_text(improved_version_body, preserve_newlines=True),
            "summary": summary,
        }

        populated_sections = sum(
            1 for key in ["what_works", "what_hurts", "retention_issues", "hook_improvement", "cta_improvement", "improved_version"]
            if structured_output.get(key)
        )
        if populated_sections == 0:
            return None, "raw_only"

        return (
            structured_output,
            "parsed" if populated_sections >= 4 else "partial",
        )

    def _parse_carousel(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["carousel"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        section_map = {section["heading"]: section["body"] for section in sections if section["body"]}
        slides = []
        for slide_number in range(1, MAX_CAROUSEL_SLIDES + 1):
            slide_text = section_map.get(f"Slide {slide_number}:")
            if slide_text:
                headline, body = self._split_headline_and_body(slide_text)
                slides.append({
                    "slide_number": slide_number,
                    "headline": headline,
                    "body": body,
                })

        structured_output = {
            "title": self._clean_value_text(section_map.get("Title:")) or self._clean_value_text(preamble),
            "slides": slides,
            "cta": self._clean_value_text(section_map.get("Final CTA slide:"), preserve_newlines=True),
        }
        populated_sections = sum(1 for key in ["title", "slides", "cta"] if structured_output.get(key))
        return (
            structured_output if populated_sections else None,
            "parsed" if structured_output.get("title") and len(slides) >= 2 else "partial" if populated_sections else "raw_only",
        )

    def _parse_caption(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["caption"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            full_caption = self._clean_value_text(text, preserve_newlines=True)
            if not full_caption:
                return None, "raw_only"
            return {
                "hook": None,
                "body": None,
                "cta": None,
                "full_caption": full_caption,
            }, "partial"

        section_map = {section["heading"]: section["body"] for section in sections if section["body"]}
        hook = self._clean_value_text(section_map.get("Hook:"), headings=headings, preserve_newlines=True)
        body = self._clean_value_text(section_map.get("Caption:") or preamble, headings=headings, preserve_newlines=True)
        cta = self._clean_value_text(section_map.get("CTA:"), headings=headings, preserve_newlines=True)
        full_caption = self._join_caption_parts(hook, body, cta) or text.strip() or None
        structured_output = {
            "hook": hook,
            "body": body,
            "cta": cta,
            "full_caption": full_caption,
        }
        populated_sections = sum(1 for key in ["hook", "body", "cta"] if structured_output.get(key))
        return (
            structured_output if populated_sections else None,
            "parsed" if body and full_caption and (hook or cta) else "partial" if structured_output.get("full_caption") else "raw_only",
        )

    def _parse_profile_audit(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["profile_audit"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        summary_body = self._section_body_from_sections(sections, ["Summary:"])

        structured_output = {
            "strengths": self._split_listish_text(self._section_body_from_sections(sections, [
                "What works:",
                "რა მუშაობს:",
            ])),
            "weak_points": self._split_listish_text(self._section_body_from_sections(sections, [
                "What is weak:",
                "რა ასუსტებს:",
                "რა სუსტია:",
            ])),
            "quick_fixes": self._split_listish_text(self._section_body_from_sections(sections, [
                "What to improve first:",
                "პირველ რიგში რა გავასწორო:",
            ])),
            "recommended_bio_direction": self._clean_value_text(self._section_body_from_sections(sections, [
                "Recommended bio direction:",
                "ბიოს მიმართულება:",
            ]), preserve_newlines=True),
            "content_direction": self._split_listish_text(self._section_body_from_sections(sections, [
                "Content direction:",
                "კონტენტის მიმართულება:",
            ])),
            "priority_actions": self._split_listish_text(self._section_body_from_sections(sections, [
                "Next 3 actions:",
                "შემდეგი 3 მოქმედება:",
                "Next actions:",
                "შემდეგი ნაბიჯები:",
            ])),
            "summary": self._clean_value_text(summary_body or preamble, preserve_newlines=True),
        }
        populated_sections = sum(
            1 for key in ["strengths", "weak_points", "quick_fixes", "recommended_bio_direction", "content_direction", "priority_actions"]
            if structured_output.get(key)
        )
        return (
            structured_output if populated_sections else None,
            "parsed" if populated_sections >= 3 else "partial" if populated_sections else "raw_only",
        )

    def _parse_content_plan(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["content_plan"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        content_items = []
        for week_number in range(1, 5):
            week_items = self._split_listish_text(self._section_body_from_sections(sections, [f"Week {week_number}:"]))
            if week_items:
                for week_item in week_items:
                    normalized_topic = self._clean_value_text(week_item)
                    format_name = self._infer_content_format(normalized_topic)
                    content_items.append({
                        "day_or_slot": f"Week {week_number}",
                        "format": format_name,
                        "topic": self._strip_format_from_topic(normalized_topic, format_name),
                        "goal": self._infer_content_goal(normalized_topic),
                    })

        plus_body = self._section_body_from_sections(sections, ["Plus:"])
        best_content_mix = self._split_listish_text(self._section_body_from_sections(sections, [
            "Best content mix:",
            "საუკეთესო კონტენტ-მიქსი:",
        ]))
        hook_ideas = self._split_listish_text(self._section_body_from_sections(sections, [
            "Hook ideas:",
            "ჰუკის იდეები:",
        ]))
        cta_ideas = self._split_listish_text(self._section_body_from_sections(sections, [
            "CTA ideas:",
            "CTA იდეები:",
        ]))
        if plus_body:
            best_content_mix = best_content_mix or self._extract_labeled_list_items(plus_body, [
                "best content mix",
                "საუკეთესო კონტენტ-მიქსი",
            ])
            hook_ideas = hook_ideas or self._extract_labeled_list_items(plus_body, [
                "hook ideas",
                "ჰუკის იდეები",
            ])
            cta_ideas = cta_ideas or self._extract_labeled_list_items(plus_body, [
                "cta ideas",
                "CTA იდეები",
            ])

        summary_body = self._section_body_from_sections(sections, ["Summary:"])
        structured_output = {
            "plan_title": self._clean_value_text(preamble),
            "content_items": content_items,
            "best_content_mix": best_content_mix,
            "hook_ideas": hook_ideas,
            "cta_ideas": cta_ideas,
            "summary": self._clean_value_text(summary_body, preserve_newlines=True),
        }
        populated_sections = sum(
            1 for key in ["plan_title", "content_items", "best_content_mix", "hook_ideas", "cta_ideas", "summary"]
            if structured_output.get(key)
        )
        return (
            structured_output if populated_sections else None,
            "parsed" if len(content_items) >= 2 else "partial" if populated_sections else "raw_only",
        )

    def _parse_link_analysis(self, text: str) -> tuple[dict | None, str]:
        headings = self._task_sections["link_analysis"]
        preamble, sections = self._extract_sections(text, headings)
        if not sections:
            return None, "raw_only"

        section_map = {section["heading"]: section["body"] for section in sections if section["body"]}
        current_state = self._split_listish_text(section_map.get("What works:", ""))
        current_state.extend(self._split_listish_text(section_map.get("Why it may perform:", "")))
        issues = self._split_listish_text(section_map.get("What is weak:", ""))
        recommended_changes = self._split_listish_text(section_map.get("How to adapt it for the user's account:", ""))
        structured_output = {
            "current_state": current_state,
            "issues": issues,
            "recommended_changes": recommended_changes,
            "best_cta_direction": self._extract_best_cta_direction(recommended_changes),
            "summary": self._clean_value_text(preamble, preserve_newlines=True),
        }
        populated_sections = sum(
            1 for key in ["current_state", "issues", "recommended_changes", "best_cta_direction"]
            if structured_output.get(key)
        )
        return (
            structured_output if populated_sections else None,
            "parsed" if populated_sections >= 3 else "partial" if populated_sections else "raw_only",
        )
