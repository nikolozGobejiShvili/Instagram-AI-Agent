from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import httpx
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message


class ReelsRagGeneration(Component):
    display_name = "Reels RAG Generation"
    description = "Retrieve internal reels knowledge from Chroma and generate a production-safe reels answer."
    icon = "Sparkles"
    name = "ReelsRagGeneration"

    MAX_RETRIES = 3
    BACKOFF_BASE_SECONDS = 1.0
    BACKOFF_MAX_SECONDS = 8.0
    EMBEDDING_DIMENSION = 64
    MAX_CHUNKS = 6
    DEFAULT_CHUNKS = 4
    MAX_CHUNK_TEXT_CHARS = 360
    ENV_OVERRIDE_KEYS = {
        "LANGFLOW_VECTOR_STORE_PROVIDER",
        "LANGFLOW_CHROMA_PERSIST_DIR",
        "LANGFLOW_REELS_CHROMA_COLLECTION",
        "LANGFLOW_REELS_MODEL_PROVIDER",
        "LANGFLOW_REELS_MODEL_NAME",
        "FALLBACK_LANGFLOW_REELS_MODEL_PROVIDER",
        "FALLBACK_LANGFLOW_REELS_MODEL_NAME",
    }

    inputs = [
        MessageTextInput(
            name="payload_json",
            display_name="Payload JSON",
            value="",
            required=True,
            info="Sanitized runtime JSON payload from the backend.",
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="build_message"),
    ]

    def build_message(self) -> Message:
        try:
            payload = self._load_payload(self.payload_json)
            self._load_backend_env(payload)
            result = self._run_generation(payload)
            return Message(text=json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            error_payload = {
                "error": {
                    "code": "generation_failed",
                    "message": "Internal reels generation flow failed.",
                },
                "details": self._safe_truncate(str(exc), 240),
            }
            return Message(text=json.dumps(error_payload, ensure_ascii=False))

    def _load_payload(self, value: str) -> dict:
        normalized_value = (value or "").strip()
        if normalized_value.startswith("BASE64JSON:"):
            encoded_payload = normalized_value.split("BASE64JSON:", 1)[1].strip()
            decoded_payload = base64.b64decode(encoded_payload).decode("utf-8")
            payload = json.loads(decoded_payload or "{}")
        else:
            payload = json.loads(normalized_value or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")
        return payload

    def _candidate_backend_env_paths(self) -> list[Path]:
        cwd = Path.cwd().resolve()
        candidates: list[Path] = []
        for parent in [cwd, *cwd.parents]:
            if parent.name == "backend":
                candidates.append(parent / ".env")
                break
            candidates.append(parent / "backend" / ".env")
        candidates.append(Path.home() / "Desktop" / "instagram-agent" / "backend" / ".env")
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _candidate_backend_pack_store_paths(self) -> list[Path]:
        candidates: list[Path] = []
        for env_path in self._candidate_backend_env_paths():
            if env_path.exists():
                candidates.append(env_path.parent / "app" / "data" / "knowledge_packs.json")
        cwd = Path.cwd().resolve()
        for parent in [cwd, *cwd.parents]:
            if parent.name == "backend":
                candidates.append(parent / "app" / "data" / "knowledge_packs.json")
                break
            candidates.append(parent / "backend" / "app" / "data" / "knowledge_packs.json")
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _load_backend_env(self, payload: dict) -> None:
        _ = payload
        for env_path in self._candidate_backend_env_paths():
            if not env_path.exists():
                continue
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (key not in os.environ or key in self.ENV_OVERRIDE_KEYS):
                    os.environ[key] = value
            break

    def _load_active_pack_ids(self, task_type: str) -> set[str]:
        for data_path in self._candidate_backend_pack_store_paths():
            if not data_path.exists():
                continue
            try:
                raw_items = json.loads(data_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw_items, dict):
                continue

            active_pack_ids: set[str] = set()
            for pack in raw_items.values():
                if not isinstance(pack, dict):
                    continue
                if str(pack.get("scope") or "") != "system":
                    continue
                if str(pack.get("domain") or "") != "reels":
                    continue
                if str(pack.get("visibility") or "") != "internal":
                    continue
                if str(pack.get("status") or "") != "active":
                    continue

                supported_task_types = pack.get("supported_task_types") or []
                if isinstance(supported_task_types, list):
                    normalized_task_types = {str(item).strip() for item in supported_task_types if str(item).strip()}
                    if normalized_task_types and task_type not in normalized_task_types:
                        continue

                knowledge_pack_id = str(pack.get("knowledge_pack_id") or "").strip()
                if knowledge_pack_id:
                    active_pack_ids.add(knowledge_pack_id)

            if active_pack_ids:
                return active_pack_ids
        return set()

    def _resolve_chroma_dir(self) -> Path:
        configured_path = os.getenv("LANGFLOW_CHROMA_PERSIST_DIR", "").strip()
        if configured_path:
            return Path(configured_path).expanduser().resolve()
        return (Path.home() / "Desktop" / "instagram-agent" / "backend" / "app" / "data" / "langflow_chroma").resolve()

    def _resolve_collection_name(self) -> str:
        return os.getenv("LANGFLOW_REELS_CHROMA_COLLECTION", "reels_system_knowledge").strip() or "reels_system_knowledge"

    def _safe_truncate(self, value: str | None, max_chars: int) -> str | None:
        normalized_value = " ".join((value or "").split()).strip()
        if not normalized_value:
            return None
        if len(normalized_value) <= max_chars:
            return normalized_value
        return f"{normalized_value[: max_chars - 3].rstrip()}..."

    def _normalize_string_list(self, values: object, *, limit: int = 6, max_chars: int = 160) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized_values = []
        for value in values:
            normalized_value = self._safe_truncate(str(value or ""), max_chars)
            if not normalized_value:
                continue
            normalized_values.append(normalized_value)
            if len(normalized_values) >= limit:
                break
        return normalized_values

    def _tokenize(self, value: str | None) -> list[str]:
        return re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (value or "").lower())

    def _embedding_bucket(self, token: str, size: int) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket_index = digest[0] % size
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        magnitude = 1.0 + (digest[2] / 255.0)
        return bucket_index, sign * magnitude

    def _embedding(self, text: str, size: int = EMBEDDING_DIMENSION) -> list[float]:
        vector = [0.0] * size
        for token in self._tokenize(text):
            bucket_index, delta = self._embedding_bucket(token, size)
            vector[bucket_index] += delta
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0:
            return vector
        return [component / norm for component in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _compact_recent_posts_summary(self, payload: dict) -> list[dict]:
        posts = payload.get("recent_posts_summary")
        if not isinstance(posts, list):
            return []
        compact_posts = []
        for post in posts[:5]:
            if not isinstance(post, dict):
                continue
            compact_posts.append({
                "content_type": self._safe_truncate(post.get("content_type"), 40),
                "topic": self._safe_truncate(post.get("topic"), 120),
                "caption_summary": self._safe_truncate(post.get("caption_summary"), 180),
                "views": int(post.get("views") or 0),
                "likes": int(post.get("likes") or 0),
                "comments": int(post.get("comments") or 0),
                "saves": int(post.get("saves") or 0),
            })
        return compact_posts

    def _build_retrieval_text(self, payload: dict) -> str:
        parts = [
            str(payload.get("task_type") or ""),
            str(payload.get("user_request") or ""),
            str(payload.get("goal") or ""),
            str(payload.get("profile_summary") or ""),
            str(payload.get("recent_content_summary") or ""),
            str(payload.get("link_summary") or ""),
            str(payload.get("retrieval_query") or ""),
        ]
        target_reel_summary = payload.get("target_reel_summary")
        if isinstance(target_reel_summary, dict):
            parts.extend(str(target_reel_summary.get(key) or "") for key in ["caption_summary", "analysis_brief", "permalink"])
        recent_posts_summary = self._compact_recent_posts_summary(payload)
        for post in recent_posts_summary[:3]:
            parts.extend(str(post.get(key) or "") for key in ["content_type", "topic", "caption_summary"])
        return "\n".join(part for part in parts if part).strip()

    def _metadata_supports_task_type(self, metadata: dict, task_type: str) -> bool:
        supported_task_types = metadata.get("supported_task_types")
        if isinstance(supported_task_types, str):
            supported_values = [item.strip() for item in supported_task_types.split(",") if item.strip()]
        elif isinstance(supported_task_types, list):
            supported_values = [str(item).strip() for item in supported_task_types if str(item).strip()]
        else:
            supported_values = []
        return not supported_values or task_type in supported_values

    def _score_chunk(
        self,
        *,
        chunk_text: str,
        chunk_metadata: dict,
        chunk_embedding: list[float] | None,
        query_embedding: list[float],
        query_tokens: set[str],
        task_type: str,
    ) -> float:
        lexical_tokens = set(self._tokenize(chunk_text))
        lexical_overlap = len(query_tokens.intersection(lexical_tokens))
        metadata_text = " ".join(
            str(chunk_metadata.get(field) or "")
            for field in ["chunk_title", "source_title", "source_file_name", "description"]
        )
        metadata_overlap = len(query_tokens.intersection(set(self._tokenize(metadata_text))))
        supported_bonus = 3.0 if self._metadata_supports_task_type(chunk_metadata, task_type) else -8.0
        domain_bonus = 1.0 if str(chunk_metadata.get("domain") or "") == "reels" else 0.0
        similarity = self._cosine_similarity(query_embedding, chunk_embedding or [])
        return (lexical_overlap * 2.0) + metadata_overlap + (similarity * 4.0) + supported_bonus + domain_bonus

    def _retrieve_chunks(self, payload: dict) -> dict:
        chroma_dir = self._resolve_chroma_dir()
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=self._resolve_collection_name(),
            metadata={"domain": "reels", "scope": "system", "visibility": "internal"},
        )

        task_type = str(payload.get("task_type") or "").strip()
        active_pack_ids = self._load_active_pack_ids(task_type)
        retrieval_text = self._build_retrieval_text(payload)
        if not retrieval_text:
            return {
                "used_system_knowledge": False,
                "matched_knowledge_domain": None,
                "matched_knowledge_pack_ids": [],
                "retrieved_chunk_count": 0,
                "retrieved_chunk_titles": [],
                "chunks": [],
            }

        query_embedding = self._embedding(retrieval_text)
        requested_top_k = int(payload.get("retrieval_top_k") or self.DEFAULT_CHUNKS)
        top_k = max(1, min(self.MAX_CHUNKS, requested_top_k))
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 3, top_k),
            where={
                "$and": [
                    {"domain": "reels"},
                    {"scope": "system"},
                    {"visibility": "internal"},
                    {"status": "active"},
                ]
            },
            include=["documents", "metadatas", "embeddings"],
        )

        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        embeddings = (raw_results.get("embeddings") or [[]])[0]
        query_tokens = set(self._tokenize(retrieval_text))

        scored_chunks = []
        for document, metadata, embedding in zip(documents, metadatas, embeddings):
            if not isinstance(document, str) or not isinstance(metadata, dict):
                continue
            knowledge_pack_id = str(metadata.get("knowledge_pack_id") or "").strip()
            if active_pack_ids and knowledge_pack_id not in active_pack_ids:
                continue
            if not self._metadata_supports_task_type(metadata, task_type):
                continue
            score = self._score_chunk(
                chunk_text=document,
                chunk_metadata=metadata,
                chunk_embedding=embedding if isinstance(embedding, list) else None,
                query_embedding=query_embedding,
                query_tokens=query_tokens,
                task_type=task_type,
            )
            scored_chunks.append({
                "text": self._safe_truncate(document, self.MAX_CHUNK_TEXT_CHARS) or "",
                "chunk_title": self._safe_truncate(str(metadata.get("chunk_title") or metadata.get("source_title") or "Reels guidance"), 160),
                "knowledge_pack_id": self._safe_truncate(knowledge_pack_id, 120),
                "source_title": self._safe_truncate(str(metadata.get("source_title") or ""), 160),
                "score": score,
            })

        scored_chunks.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        selected_chunks = scored_chunks[:top_k]
        matched_pack_ids = []
        retrieved_chunk_titles = []
        for chunk in selected_chunks:
            knowledge_pack_id = chunk.get("knowledge_pack_id")
            if knowledge_pack_id and knowledge_pack_id not in matched_pack_ids:
                matched_pack_ids.append(knowledge_pack_id)
            chunk_title = chunk.get("chunk_title")
            if chunk_title and chunk_title not in retrieved_chunk_titles:
                retrieved_chunk_titles.append(chunk_title)

        return {
            "used_system_knowledge": bool(selected_chunks),
            "matched_knowledge_domain": "reels" if selected_chunks else None,
            "matched_knowledge_pack_ids": matched_pack_ids,
            "retrieved_chunk_count": len(selected_chunks),
            "retrieved_chunk_titles": retrieved_chunk_titles,
            "chunks": selected_chunks,
        }

    def _reels_task_instructions(self, task_type: str, idea_count: int) -> str:
        task_instructions = {
            "reel_idea": "\n".join([
                f"Generate exactly {idea_count} Reel ideas.",
                "Each idea must feel concrete, niche-aware, easy to record, and strategically adapted to the goal.",
                "Do not give generic filler like 'share value' unless you translate it into an executable concept.",
                "Use the retrieved hidden reels knowledge for hook mechanics, format choice, simplicity, retention, and CTA logic when relevant.",
                "Avoid generic lifestyle or vlog ideas unless the account context clearly points there.",
                "If the goal is qualified inbound leads, prefer diagnosis, objection breakdown, myth-versus-reality, proof-led breakdown, audit-style, before-after, or comment-reply Reel concepts over broad motivational content.",
                "Make the conversion path visible inside the concept itself, not only in the CTA.",
            ]),
            "reel_script": "\n".join([
                "Generate one strong Reel script that is ready to record now.",
                "Prioritize first-second hook strength, scene progression, clarity, and CTA fit.",
                "Keep the structure easy to execute without bloated jargon.",
                "The script should sound like a strong strategist who understands retention and conversion, not like a generic copy template.",
            ]),
            "reel_feedback": "\n".join([
                "Analyze the target Reel carefully before improving it.",
                "Diagnose the hook, retention risk, clarity, and CTA weakness honestly.",
                "The improved version must be meaningfully stronger and more direct.",
                "If account context suggests a lead-generation goal, make the CTA improvement more specific and action-driving.",
            ]),
        }
        return task_instructions.get(task_type, "Generate a strong reels answer.")

    def _reply_format_guidance(self, task_type: str, idea_count: int) -> str:
        reply_guidance = {
            "reel_idea": "\n".join([
                f"In reply, present exactly {idea_count} numbered Reel ideas.",
                "Each idea should feel like a strategist giving a creator a clear concept and shooting direction.",
                "Use clean headings and short sections. Avoid sounding robotic or like a raw JSON dump.",
                "The reply itself must contain the real ideas, not just a one-line summary.",
            ]),
            "reel_script": "\n".join([
                "In reply, present the script in a creator-friendly way with a clear hook, structure, and CTA.",
                "Make the script feel recordable immediately.",
            ]),
            "reel_feedback": "\n".join([
                "In reply, explain what works, what hurts, then show the improved version clearly.",
                "Keep the tone strategic and specific, not vague or sugar-coated.",
            ]),
        }
        return reply_guidance.get(task_type, "")

    def _json_contract(self, task_type: str, idea_count: int) -> str:
        contracts = {
            "reel_idea": (
                "{"
                "\"reply\":\"...\","
                "\"structured_output\":{"
                "\"ideas\":["
                "{\"title\":\"...\",\"hook\":\"...\",\"format_type\":\"...\",\"main_idea\":\"...\",\"shot_list\":[\"...\"],\"why_it_can_work\":\"...\",\"cta\":\"...\"}"
                "]},"
                "\"parse_status\":\"parsed\""
                "}"
            ),
            "reel_script": (
                "{"
                "\"reply\":\"...\","
                "\"structured_output\":{"
                "\"script\":{"
                "\"hook\":\"...\","
                "\"structure\":[\"...\"],"
                "\"voiceover\":\"...\","
                "\"shot_list\":[\"...\"],"
                "\"cta\":\"...\""
                "}},"
                "\"parse_status\":\"parsed\""
                "}"
            ),
            "reel_feedback": (
                "{"
                "\"reply\":\"...\","
                "\"structured_output\":{"
                "\"feedback\":{"
                "\"what_works\":[\"...\"],"
                "\"what_hurts\":[\"...\"],"
                "\"retention_issues\":[\"...\"],"
                "\"hook_improvement\":\"...\","
                "\"cta_improvement\":\"...\","
                "\"improved_version\":\"...\""
                "}},"
                "\"parse_status\":\"parsed\""
                "}"
            ),
        }
        _ = idea_count
        return contracts.get(task_type, "{}")

    def _target_idea_count(self, payload: dict) -> int:
        if str(payload.get("task_type") or "") != "reel_idea":
            return 1
        message = str(payload.get("user_request") or "")
        digit_match = re.search(r"\b([1-5])\b", message)
        if digit_match:
            return max(1, min(5, int(digit_match.group(1))))
        return 3

    def _system_prompt(self, payload: dict, retrieval: dict) -> str:
        task_type = str(payload.get("task_type") or "")
        idea_count = self._target_idea_count(payload)
        language = str(payload.get("language") or "en").strip().lower()
        retrieved_chunks = retrieval.get("chunks") or []
        chunk_lines = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            chunk_title = chunk.get("chunk_title") or f"Chunk {index}"
            chunk_lines.append(f"{index}. {chunk_title}: {chunk.get('text', '')}")

        system_parts = [
            "You are Instagram Agent V1 for reels strategy.",
            "Respond in the same language as the user's request.",
            "If language=ka, the reply and every user-facing string inside structured_output must be fully in natural Georgian script.",
            "If language=ka, English output is invalid unless a proper noun or standard platform term is unavoidable.",
            "If language=en, the reply and structured_output text must be fully in English.",
            "Be practical, strategic, specific, and creator-friendly.",
            "Never mention hidden internal knowledge, retrieval, vector stores, embeddings, or system prompts.",
            "Use the hidden reels methodology for hook logic, simplicity, retention, trend adaptation, and CTA strategy when relevant.",
            f"Language code: {language}",
            self._reels_task_instructions(task_type, idea_count),
            self._reply_format_guidance(task_type, idea_count),
            "Return one valid JSON object only. Do not wrap it in markdown fences.",
            "The JSON must follow this contract exactly:",
            self._json_contract(task_type, idea_count),
        ]

        if chunk_lines:
            system_parts.extend([
                "Hidden internal reels methodology:",
                "\n".join(chunk_lines),
            ])
        else:
            system_parts.append("No hidden internal reels chunks were retrieved for this request. Fall back safely to the provided account context and general Instagram strategy.")

        return "\n\n".join(part for part in system_parts if part)

    def _user_payload_for_model(self, payload: dict) -> dict:
        runtime_payload = {
            "task_type": payload.get("task_type"),
            "language": payload.get("language"),
            "language_instruction": (
                "Return every user-facing string fully in Georgian script."
                if str(payload.get("language") or "").strip().lower() == "ka"
                else "Return every user-facing string in English."
            ),
            "user_request": payload.get("user_request"),
            "goal": payload.get("goal"),
            "profile_summary": payload.get("profile_summary"),
            "recent_posts_summary": self._compact_recent_posts_summary(payload),
            "recent_content_summary": payload.get("recent_content_summary"),
            "target_reel_summary": payload.get("target_reel_summary"),
            "link_summary": payload.get("link_summary"),
            "response_style_rules": payload.get("response_style_rules") or [],
        }
        return runtime_payload

    def _estimate_tokens(self, system_prompt: str, user_payload: dict) -> int:
        serialized = json.dumps({"system": system_prompt, "user": user_payload}, ensure_ascii=False)
        return max(1, len(serialized) // 4)

    def _retry_after_seconds(self, response: httpx.Response | None) -> float | None:
        if response is None:
            return None
        header_value = response.headers.get("Retry-After")
        if not header_value:
            return None
        try:
            return max(0.0, float(header_value))
        except ValueError:
            return None

    def _backoff_seconds(self, attempt_number: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return retry_after_seconds
        rng = random.Random()
        return min(self.BACKOFF_MAX_SECONDS, self.BACKOFF_BASE_SECONDS * (2 ** max(attempt_number - 1, 0))) + rng.uniform(0.0, 0.35)

    def _provider_config(self) -> list[dict]:
        primary_provider = (os.getenv("LANGFLOW_REELS_MODEL_PROVIDER", "groq").strip() or "groq").lower()
        primary_model = os.getenv("LANGFLOW_REELS_MODEL_NAME", "").strip() or (
            "llama-3.3-70b-versatile" if primary_provider == "groq" else "gpt-4.1-mini"
        )
        configs = [{
            "provider": primary_provider,
            "model": primary_model,
        }]

        fallback_provider = (os.getenv("LANGFLOW_REELS_FALLBACK_PROVIDER", "").strip() or "").lower()
        fallback_model = os.getenv("LANGFLOW_REELS_FALLBACK_MODEL_NAME", "").strip()
        if fallback_provider and fallback_model:
            configs.append({
                "provider": fallback_provider,
                "model": fallback_model,
            })

        return configs

    def _provider_request_parts(self, provider: str, model_name: str) -> tuple[str, str]:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")
            return f"{base_url}/chat/completions", api_key

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            _ = model_name
            return "https://api.openai.com/v1/chat/completions", api_key

        raise RuntimeError(f"Unsupported Langflow reels model provider '{provider}'")

    def _max_output_tokens(self, task_type: str) -> int:
        return {
            "reel_idea": 1400,
            "reel_script": 1300,
            "reel_feedback": 1200,
        }.get(task_type, 1200)

    def _call_provider_once(
        self,
        *,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_payload: dict,
        task_type: str,
    ) -> dict:
        url, api_key = self._provider_request_parts(provider, model_name)
        request_payload = {
            "model": model_name,
            "temperature": 0.45,
            "max_tokens": self._max_output_tokens(task_type),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
        return {
            "response": response,
            "request_payload": request_payload,
        }

    def _extract_content(self, response_data: dict) -> str:
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        raise RuntimeError("Model response did not contain a text message")

    def _call_provider_with_retries(
        self,
        *,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_payload: dict,
        task_type: str,
    ) -> tuple[dict, int, bool]:
        retry_count = 0
        rate_limited = False
        for attempt_number in range(1, self.MAX_RETRIES + 2):
            try:
                call_result = self._call_provider_once(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    task_type=task_type,
                )
                response = call_result["response"]
                response.raise_for_status()
                return response.json(), retry_count, rate_limited
            except httpx.HTTPStatusError as exc:
                response = exc.response
                if response.status_code == 429:
                    rate_limited = True
                    if attempt_number <= self.MAX_RETRIES:
                        retry_count += 1
                        time.sleep(self._backoff_seconds(attempt_number, self._retry_after_seconds(response)))
                        continue
                    raise RuntimeError(json.dumps({
                        "error": {
                            "code": "llm_rate_limited",
                            "message": "AI generation is temporarily rate limited. Please try again shortly.",
                        },
                        "retry_count": retry_count,
                        "rate_limited": True,
                        "model_provider": provider,
                        "model_name": model_name,
                    }, ensure_ascii=False)) from exc
                raise RuntimeError(
                    json.dumps({
                        "error": {
                            "code": "generation_failed",
                            "message": "Internal reels generation failed.",
                        },
                        "retry_count": retry_count,
                        "rate_limited": rate_limited,
                        "model_provider": provider,
                        "model_name": model_name,
                    }, ensure_ascii=False)
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    json.dumps({
                        "error": {
                            "code": "generation_failed",
                            "message": "Internal reels generation failed.",
                        },
                        "retry_count": retry_count,
                        "rate_limited": rate_limited,
                        "model_provider": provider,
                        "model_name": model_name,
                    }, ensure_ascii=False)
                ) from exc

        raise RuntimeError("Internal reels generation failed.")

    def _generate_model_response(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        task_type: str,
    ) -> tuple[dict, str, str, int, bool]:
        provider_errors = []
        for provider_config in self._provider_config():
            provider = provider_config["provider"]
            model_name = provider_config["model"]
            try:
                raw_data, retry_count, rate_limited = self._call_provider_with_retries(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    task_type=task_type,
                )
                content = self._extract_content(raw_data)
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Model did not return a JSON object")
                return parsed, provider, model_name, retry_count, rate_limited
            except Exception as exc:
                provider_errors.append(self._safe_truncate(str(exc), 240))
                continue

        fallback_message = provider_errors[-1] if provider_errors else "Internal reels generation failed."
        try:
            parsed_error = json.loads(fallback_message)
            if isinstance(parsed_error, dict) and isinstance(parsed_error.get("error"), dict):
                return parsed_error, "unknown", "unknown", int(parsed_error.get("retry_count") or 0), bool(parsed_error.get("rate_limited"))
        except Exception:
            pass
        raise RuntimeError(fallback_message)

    def _coerce_list(self, values: object, *, limit: int = 8, max_chars: int = 220) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized_values = []
        for value in values:
            normalized = self._safe_truncate(str(value or ""), max_chars)
            if not normalized:
                continue
            normalized_values.append(normalized)
            if len(normalized_values) >= limit:
                break
        return normalized_values

    def _normalize_structured_output(self, task_type: str, structured_output: dict, *, idea_count: int) -> dict:
        if task_type == "reel_idea":
            raw_ideas = structured_output.get("ideas")
            if not isinstance(raw_ideas, list):
                raw_ideas = []
            ideas = []
            for raw_idea in raw_ideas[:idea_count]:
                if not isinstance(raw_idea, dict):
                    continue
                ideas.append({
                    "title": self._safe_truncate(str(raw_idea.get("title") or ""), 160),
                    "hook": self._safe_truncate(str(raw_idea.get("hook") or ""), 220),
                    "format_type": self._safe_truncate(str(raw_idea.get("format_type") or ""), 120),
                    "main_idea": self._safe_truncate(str(raw_idea.get("main_idea") or ""), 320),
                    "shot_list": self._coerce_list(raw_idea.get("shot_list"), limit=6, max_chars=160),
                    "why_it_can_work": self._safe_truncate(str(raw_idea.get("why_it_can_work") or ""), 320),
                    "cta": self._safe_truncate(str(raw_idea.get("cta") or ""), 200),
                })
            if not ideas:
                raise RuntimeError("Reel idea structured output is missing ideas")
            return {"ideas": ideas}

        if task_type == "reel_script":
            raw_script = structured_output.get("script")
            if not isinstance(raw_script, dict):
                raw_script = structured_output
            script = {
                "hook": self._safe_truncate(str(raw_script.get("hook") or ""), 220),
                "structure": self._coerce_list(raw_script.get("structure"), limit=8, max_chars=180),
                "voiceover": self._safe_truncate(str(raw_script.get("voiceover") or ""), 1200),
                "shot_list": self._coerce_list(raw_script.get("shot_list"), limit=8, max_chars=180),
                "cta": self._safe_truncate(str(raw_script.get("cta") or ""), 220),
            }
            if not script["hook"] or not script["voiceover"]:
                raise RuntimeError("Reel script structured output is incomplete")
            return {"script": script}

        if task_type == "reel_feedback":
            raw_feedback = structured_output.get("feedback")
            if not isinstance(raw_feedback, dict):
                raw_feedback = structured_output
            feedback = {
                "what_works": self._coerce_list(raw_feedback.get("what_works"), limit=6, max_chars=180),
                "what_hurts": self._coerce_list(raw_feedback.get("what_hurts"), limit=6, max_chars=180),
                "retention_issues": self._coerce_list(raw_feedback.get("retention_issues"), limit=6, max_chars=180),
                "hook_improvement": self._safe_truncate(str(raw_feedback.get("hook_improvement") or ""), 260),
                "cta_improvement": self._safe_truncate(str(raw_feedback.get("cta_improvement") or ""), 260),
                "improved_version": self._safe_truncate(str(raw_feedback.get("improved_version") or ""), 1400),
            }
            if not feedback["improved_version"]:
                raise RuntimeError("Reel feedback structured output is incomplete")
            return {"feedback": feedback}

        raise RuntimeError(f"Unsupported reels task type '{task_type}'")

    def _is_georgian(self, payload: dict) -> bool:
        return str(payload.get("language") or "").strip().lower() == "ka"

    def _synthesize_reply_from_output(self, *, task_type: str, payload: dict, structured_output: dict) -> str:
        use_georgian = self._is_georgian(payload)

        if task_type == "reel_idea":
            intro = (
                "აქ არის 3 უფრო ძლიერი Reel იდეა შენი მიმდინარე Instagram კონტექსტისა და lead-generation მიზნის მიხედვით:"
                if use_georgian
                else "Here are 3 stronger Reel ideas adapted to your current Instagram context and lead-generation goal:"
            )
            sections = [intro]
            for index, idea in enumerate(structured_output.get("ideas", []), start=1):
                if use_georgian:
                    section = "\n".join([
                        f"{index}. {idea.get('title') or 'Reel idea'}",
                        f"Hook: {idea.get('hook') or ''}",
                        f"Format: {idea.get('format_type') or ''}",
                        f"Main idea: {idea.get('main_idea') or ''}",
                        "Shot list:",
                        *[f"- {item}" for item in idea.get("shot_list", [])],
                        f"Why it can work: {idea.get('why_it_can_work') or ''}",
                        f"CTA: {idea.get('cta') or ''}",
                    ])
                else:
                    section = "\n".join([
                        f"{index}. {idea.get('title') or 'Reel idea'}",
                        f"Hook: {idea.get('hook') or ''}",
                        f"Format: {idea.get('format_type') or ''}",
                        f"Main idea: {idea.get('main_idea') or ''}",
                        "Shot list:",
                        *[f"- {item}" for item in idea.get("shot_list", [])],
                        f"Why it can work: {idea.get('why_it_can_work') or ''}",
                        f"CTA: {idea.get('cta') or ''}",
                    ])
                sections.append(section)
            return "\n\n".join(section for section in sections if section)

        if task_type == "reel_script":
            script = structured_output.get("script", {})
            return "\n\n".join([
                "აქ არის უფრო ძლიერი Reel სცენარი:" if use_georgian else "Here is a stronger Reel script:",
                f"Hook: {script.get('hook') or ''}",
                "Structure:",
                *[f"- {item}" for item in script.get("structure", [])],
                f"Voiceover: {script.get('voiceover') or ''}",
                "Shot list:",
                *[f"- {item}" for item in script.get("shot_list", [])],
                f"CTA: {script.get('cta') or ''}",
            ])

        if task_type == "reel_feedback":
            feedback = structured_output.get("feedback", {})
            return "\n\n".join([
                "აქ არის Reel-ის უფრო სტრატეგიული გარჩევა და გაუმჯობესებული ვერსია:" if use_georgian else "Here is a more strategic Reel review and improved version:",
                "What works:",
                *[f"- {item}" for item in feedback.get("what_works", [])],
                "What hurts:",
                *[f"- {item}" for item in feedback.get("what_hurts", [])],
                "Retention issues:",
                *[f"- {item}" for item in feedback.get("retention_issues", [])],
                f"Hook improvement: {feedback.get('hook_improvement') or ''}",
                f"CTA improvement: {feedback.get('cta_improvement') or ''}",
                f"Improved version: {feedback.get('improved_version') or ''}",
            ])

        return ""

    def _run_generation(self, payload: dict) -> dict:
        task_type = str(payload.get("task_type") or "").strip()
        if task_type not in {"reel_idea", "reel_script", "reel_feedback"}:
            raise ValueError(f"Unsupported reels task type '{task_type}'")

        retrieval_error = None
        try:
            retrieval = self._retrieve_chunks(payload)
        except Exception as exc:
            retrieval = {
                "used_system_knowledge": False,
                "matched_knowledge_domain": None,
                "matched_knowledge_pack_ids": [],
                "retrieved_chunk_count": 0,
                "retrieved_chunk_titles": [],
                "chunks": [],
            }
            retrieval_error = self._safe_truncate(str(exc), 220)

        idea_count = self._target_idea_count(payload)
        system_prompt = self._system_prompt(payload, retrieval)
        user_payload = self._user_payload_for_model(payload)
        prompt_token_estimate = self._estimate_tokens(system_prompt, user_payload)

        model_result, model_provider, model_name, retry_count, rate_limited = self._generate_model_response(
            system_prompt=system_prompt,
            user_payload=user_payload,
            task_type=task_type,
        )

        if isinstance(model_result.get("error"), dict):
            error_payload = dict(model_result)
            error_payload.setdefault("used_system_knowledge", retrieval.get("used_system_knowledge"))
            error_payload.setdefault("matched_knowledge_domain", retrieval.get("matched_knowledge_domain"))
            error_payload.setdefault("matched_knowledge_pack_ids", retrieval.get("matched_knowledge_pack_ids"))
            error_payload.setdefault("retrieved_chunk_count", retrieval.get("retrieved_chunk_count"))
            error_payload.setdefault("retrieved_chunk_titles", retrieval.get("retrieved_chunk_titles"))
            error_payload.setdefault("model_provider", model_provider)
            error_payload.setdefault("model_name", model_name)
            error_payload.setdefault("prompt_token_estimate", prompt_token_estimate)
            error_payload.setdefault("retry_count", retry_count)
            error_payload.setdefault("rate_limited", rate_limited)
            return error_payload

        reply = self._safe_truncate(str(model_result.get("reply") or ""), 12000)
        if not reply:
            raise RuntimeError("Model response did not include reply")

        structured_output = model_result.get("structured_output")
        if not isinstance(structured_output, dict):
            raise RuntimeError("Model response did not include structured_output")

        normalized_output = self._normalize_structured_output(
            task_type,
            structured_output,
            idea_count=idea_count,
        )
        if len(reply.split()) < 20:
            synthesized_reply = self._synthesize_reply_from_output(
                task_type=task_type,
                payload=payload,
                structured_output=normalized_output,
            )
            if synthesized_reply:
                reply = synthesized_reply
        result = {
            "reply": reply,
            "structured_output": normalized_output,
            "parse_status": "parsed",
            "used_system_knowledge": bool(retrieval.get("used_system_knowledge")),
            "matched_knowledge_domain": retrieval.get("matched_knowledge_domain"),
            "matched_knowledge_pack_ids": list(retrieval.get("matched_knowledge_pack_ids") or []),
            "retrieved_chunk_count": int(retrieval.get("retrieved_chunk_count") or 0),
            "retrieved_chunk_titles": list(retrieval.get("retrieved_chunk_titles") or []),
            "model_provider": model_provider,
            "model_name": model_name,
            "prompt_token_estimate": prompt_token_estimate,
            "retry_count": retry_count,
            "rate_limited": rate_limited,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if retrieval_error:
            result["retrieval_warning"] = retrieval_error
        return result
