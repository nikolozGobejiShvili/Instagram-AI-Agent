import io
import json
import logging
import re
import shutil
from hashlib import sha256
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - graceful fallback if dependency is unavailable
    PdfReader = None

try:
    from docx import Document
except Exception:  # pragma: no cover - graceful fallback if dependency is unavailable
    Document = None


logger = logging.getLogger(__name__)


class KnowledgePackService:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
    SUPPORTED_TASK_TYPES = {
        "reel_idea",
        "reel_script",
        "reel_feedback",
        "caption",
        "carousel",
        "profile_audit",
        "content_plan",
        "link_analysis",
        "performance_summary",
    }
    REELS_TASK_TYPES = {
        "reel_idea",
        "reel_script",
        "reel_feedback",
    }
    SUPPORTED_SCOPES = {"system", "user"}
    SUPPORTED_VISIBILITIES = {"internal", "user"}
    SUPPORTED_STATUSES = {"active", "inactive"}
    STOP_WORDS = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "your",
        "have",
        "will",
        "into",
        "about",
        "them",
        "they",
        "what",
        "when",
        "where",
        "then",
        "than",
        "also",
        "more",
        "best",
        "only",
        "just",
        "very",
        "into",
        "over",
        "user",
        "account",
        "instagram",
        "content",
        "post",
        "posts",
        "idea",
        "ideas",
        "make",
        "used",
        "using",
        "yourself",
        "áƒ›áƒáƒ’áƒ áƒáƒ›",
        "áƒ áƒáƒªáƒ",
        "áƒ£áƒœáƒ“áƒ",
        "áƒ áƒáƒ›",
        "áƒ—áƒ£",
        "áƒáƒ áƒ˜áƒ¡",
        "áƒ“áƒ",
        "áƒ”áƒ¡",
        "áƒáƒ áƒ",
        "áƒ áƒ",
        "áƒ›áƒ˜áƒ¡áƒ˜",
        "áƒ›áƒ”áƒ¢áƒ˜",
    }
    TASK_KEYWORDS = {
        "reel_idea": [
            "reel ideas",
            "viral idea mechanics",
            "trend adaptation",
            "hook",
            "format library",
            "simplicity",
            "first seconds",
            "retention",
        ],
        "reel_script": [
            "reel structure",
            "hook",
            "shot list",
            "retention",
            "cta",
            "narrative flow",
            "scene pacing",
            "visual proof",
        ],
        "reel_feedback": [
            "hook diagnosis",
            "retention",
            "structure",
            "clarity",
            "cta",
            "improvement framework",
            "diagnosis",
            "rewrite",
        ],
        "caption": ["caption", "hook", "cta", "body", "conversion"],
        "carousel": ["carousel", "slide", "headline", "cta", "framework"],
        "profile_audit": ["profile", "bio", "audit", "positioning", "cta"],
        "content_plan": ["plan", "calendar", "content", "topic", "goal", "format"],
        "link_analysis": ["analysis", "adapt", "pattern", "cta", "hook"],
        "performance_summary": ["performance", "summary", "patterns", "worked", "cta"],
    }
    REELS_TOP_K_BY_TASK = {
        "reel_idea": 5,
        "reel_script": 4,
        "reel_feedback": 4,
    }
    EMBEDDING_DIMENSION = 64
    EMBEDDING_VERSION = "local_hash_v1"

    def __init__(
        self,
        data_file: Path | None = None,
        chunks_file: Path | None = None,
        storage_dir: Path | None = None,
    ):
        data_root = Path(__file__).resolve().parent.parent / "data"
        self.data_file = data_file or data_root / "knowledge_packs.json"
        self.chunks_file = chunks_file or data_root / "knowledge_pack_chunks.json"
        self.storage_dir = storage_dir or data_root / "knowledge_packs_files"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_text(self, value: str | None) -> str | None:
        normalized_value = " ".join((value or "").split()).strip()
        return normalized_value or None

    def _normalize_scope(self, scope: str | None, *, default: str = "user") -> str:
        normalized_scope = self._normalize_text(scope) or default
        if normalized_scope not in self.SUPPORTED_SCOPES:
            raise HTTPException(status_code=400, detail=f"Unsupported knowledge scope '{normalized_scope}'")
        return normalized_scope

    def _normalize_visibility(self, visibility: str | None, *, default: str) -> str:
        normalized_visibility = self._normalize_text(visibility) or default
        if normalized_visibility not in self.SUPPORTED_VISIBILITIES:
            raise HTTPException(status_code=400, detail=f"Unsupported knowledge visibility '{normalized_visibility}'")
        return normalized_visibility

    def _normalize_status(self, status: str | None, *, default: str = "inactive") -> str:
        normalized_status = self._normalize_text(status) or default
        if normalized_status not in self.SUPPORTED_STATUSES:
            raise HTTPException(status_code=400, detail=f"Unsupported knowledge pack status '{normalized_status}'")
        return normalized_status

    def _normalize_domain(self, domain: str | None) -> str | None:
        normalized_domain = self._normalize_text(domain)
        return normalized_domain.lower() if normalized_domain else None

    def _normalize_supported_task_types(
        self,
        supported_task_types: list[str] | None,
        *,
        strict: bool = True,
    ) -> list[str]:
        if not supported_task_types:
            return []

        normalized_values = []
        for task_type in supported_task_types:
            normalized_task_type = self._normalize_text(task_type)
            if not normalized_task_type:
                continue
            if normalized_task_type not in self.SUPPORTED_TASK_TYPES:
                if strict:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported task type '{normalized_task_type}' for knowledge pack",
                    )
                continue
            if normalized_task_type not in normalized_values:
                normalized_values.append(normalized_task_type)

        return normalized_values

    def _sanitize_file_name(self, file_name: str) -> str:
        sanitized_name = Path(file_name).name.strip()
        if not sanitized_name:
            raise HTTPException(status_code=400, detail="Knowledge pack upload failed: file name is missing")
        return sanitized_name

    def _clean_extracted_text(self, value: str) -> str:
        normalized_text = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u0000", "")
        normalized_text = normalized_text.replace("\u200b", "")
        normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
        normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
        return normalized_text.strip()

    def _extract_chunk_label(self, text: str) -> str | None:
        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = re.sub(r"^[#>*\-\d.\s]+", "", raw_line).strip(" :-\t")
            if not line:
                continue
            return line[:120].rstrip()
        return None

    def _extract_text_from_txt(self, content: bytes) -> str:
        return self._clean_extracted_text(content.decode("utf-8", errors="ignore"))

    def _extract_text_from_pdf(self, content: bytes) -> str:
        if PdfReader is None:
            raise HTTPException(status_code=400, detail="PDF parsing dependency is unavailable")

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return self._clean_extracted_text("\n\n".join(pages))

    def _extract_text_from_docx(self, content: bytes) -> str:
        if Document is None:
            raise HTTPException(status_code=400, detail="DOCX parsing dependency is unavailable")

        document = Document(io.BytesIO(content))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return self._clean_extracted_text("\n\n".join(paragraphs))

    def _extract_text_from_file(self, file_name: str, content: bytes) -> str:
        extension = Path(file_name).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported knowledge pack file type '{extension or 'unknown'}' for file '{file_name}'",
            )

        try:
            if extension in {".txt", ".md"}:
                return self._extract_text_from_txt(content)
            if extension == ".pdf":
                return self._extract_text_from_pdf(content)
            if extension == ".docx":
                return self._extract_text_from_docx(content)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Knowledge pack document parsing failed for file '{file_name}': {exc}",
            ) from exc

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported knowledge pack file type '{extension}' for file '{file_name}'",
        )

    def _chunk_text(self, text: str, max_chars: int = 1200) -> list[str]:
        if not text:
            return []

        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        if not blocks:
            return []

        chunks = []
        current_blocks: list[str] = []
        current_length = 0

        for block in blocks:
            if len(block) > max_chars:
                if current_blocks:
                    chunks.append("\n\n".join(current_blocks).strip())
                    current_blocks = []
                    current_length = 0

                for start_index in range(0, len(block), max_chars - 180):
                    chunk = block[start_index:start_index + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
                continue

            projected_length = current_length + len(block) + (2 if current_blocks else 0)
            if current_blocks and projected_length > max_chars:
                chunks.append("\n\n".join(current_blocks).strip())
                overlap_blocks = current_blocks[-1:]
                current_blocks = overlap_blocks + [block]
                current_length = sum(len(item) for item in current_blocks) + (2 * (len(current_blocks) - 1))
                continue

            current_blocks.append(block)
            current_length = projected_length

        if current_blocks:
            chunks.append("\n\n".join(current_blocks).strip())

        return [chunk for chunk in chunks if chunk]

    def _embedding_bucket(self, token: str) -> int:
        return int(sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.EMBEDDING_DIMENSION

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [round(value / magnitude, 6) for value in vector]

    def _build_local_embedding_from_tokens(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self.EMBEDDING_DIMENSION
        for token, count in Counter(tokens).items():
            vector[self._embedding_bucket(token)] += float(count)
        return self._normalize_vector(vector)

    def _build_local_embedding(self, text: str) -> list[float]:
        return self._build_local_embedding_from_tokens(self._tokenize(text))

    def _top_retrieval_terms(self, text: str, *, limit: int = 12) -> list[str]:
        token_counter = Counter(self._tokenize(text))
        return [token for token, _ in token_counter.most_common(limit)]

    def _normalize_chunk_record(self, chunk: dict, pack: dict | None = None) -> tuple[dict, bool]:
        normalized_chunk = dict(chunk)
        chunk_text = str(normalized_chunk.get("text") or "").strip()
        mutated = False

        if not normalized_chunk.get("chunk_label"):
            normalized_chunk["chunk_label"] = self._extract_chunk_label(chunk_text)
            mutated = True

        expected_embedding = normalized_chunk.get("embedding")
        if not isinstance(expected_embedding, list) or len(expected_embedding) != self.EMBEDDING_DIMENSION:
            normalized_chunk["embedding"] = self._build_local_embedding(chunk_text)
            normalized_chunk["embedding_version"] = self.EMBEDDING_VERSION
            mutated = True
        elif normalized_chunk.get("embedding_version") != self.EMBEDDING_VERSION:
            normalized_chunk["embedding_version"] = self.EMBEDDING_VERSION
            mutated = True

        retrieval_terms = normalized_chunk.get("retrieval_terms")
        if not isinstance(retrieval_terms, list) or not retrieval_terms:
            normalized_chunk["retrieval_terms"] = self._top_retrieval_terms(chunk_text)
            mutated = True

        if pack:
            expected_fields = {
                "scope": pack.get("scope"),
                "domain": pack.get("domain"),
                "visibility": pack.get("visibility"),
                "supported_task_types": list(pack.get("supported_task_types", [])),
                "knowledge_pack_title": pack.get("title"),
                "source_label": normalized_chunk.get("source_label") or f"{pack.get('title')} :: {normalized_chunk.get('file_name')}",
            }
            for field_name, field_value in expected_fields.items():
                if normalized_chunk.get(field_name) != field_value:
                    normalized_chunk[field_name] = field_value
                    mutated = True

        return normalized_chunk, mutated

    def _normalize_pack_record(self, pack_id: str, pack: dict) -> dict:
        raw_scope = self._normalize_text(str(pack.get("scope") or ""))
        inferred_scope = raw_scope or ("system" if str(pack.get("owner_user_id") or "").strip() == "system" else "user")
        scope = inferred_scope if inferred_scope in self.SUPPORTED_SCOPES else "user"

        supported_task_types = self._normalize_supported_task_types(
            pack.get("supported_task_types"),
            strict=False,
        )
        inferred_domain = None
        if supported_task_types and all(task_type in self.REELS_TASK_TYPES for task_type in supported_task_types):
            inferred_domain = "reels"
        domain = self._normalize_domain(pack.get("domain")) or inferred_domain

        default_visibility = "internal" if scope == "system" else "user"
        raw_visibility = self._normalize_text(str(pack.get("visibility") or "")) or default_visibility
        visibility = raw_visibility if raw_visibility in self.SUPPORTED_VISIBILITIES else default_visibility

        raw_status = self._normalize_text(str(pack.get("status") or "")) or "inactive"
        status = raw_status if raw_status in self.SUPPORTED_STATUSES else "inactive"

        owner_user_id = self._normalize_text(str(pack.get("owner_user_id") or "")) or ("system" if scope == "system" else "unknown")
        created_at = self._normalize_text(str(pack.get("created_at") or "")) or self._now().isoformat()
        updated_at = self._normalize_text(str(pack.get("updated_at") or "")) or created_at

        uploaded_files = pack.get("uploaded_files")
        if not isinstance(uploaded_files, list):
            uploaded_files = []

        try:
            total_chunks = int(pack.get("total_chunks") or 0)
        except (TypeError, ValueError):
            total_chunks = 0

        return {
            "knowledge_pack_id": self._normalize_text(str(pack.get("knowledge_pack_id") or "")) or pack_id,
            "owner_user_id": owner_user_id,
            "title": self._normalize_text(str(pack.get("title") or "")) or "Untitled Knowledge Pack",
            "description": self._normalize_text(pack.get("description")),
            "scope": scope,
            "domain": domain,
            "visibility": visibility,
            "status": status,
            "supported_task_types": supported_task_types,
            "uploaded_files": uploaded_files,
            "total_chunks": total_chunks,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _load_packs(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        if not isinstance(raw_items, dict):
            return {}

        normalized_items = {}
        mutated = False
        for pack_id, pack in raw_items.items():
            if not isinstance(pack, dict):
                mutated = True
                continue

            normalized_pack = self._normalize_pack_record(str(pack_id), pack)
            normalized_items[normalized_pack["knowledge_pack_id"]] = normalized_pack
            if pack != normalized_pack or str(pack_id) != normalized_pack["knowledge_pack_id"]:
                mutated = True

        if mutated:
            self._save_packs(normalized_items)

        return normalized_items

    def _save_packs(self, items: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _load_chunks(self) -> dict:
        if not self.chunks_file.exists():
            return {}

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        if not isinstance(raw_items, dict):
            return {}

        packs = self._load_packs()
        normalized_items = {}
        mutated = False

        for knowledge_pack_id, chunk_items in raw_items.items():
            if not isinstance(chunk_items, list):
                mutated = True
                continue

            pack = packs.get(str(knowledge_pack_id))
            normalized_chunk_items = []
            for chunk in chunk_items:
                if not isinstance(chunk, dict):
                    mutated = True
                    continue
                normalized_chunk, chunk_mutated = self._normalize_chunk_record(chunk, pack=pack)
                if chunk_mutated:
                    mutated = True
                normalized_chunk_items.append(normalized_chunk)

            normalized_items[str(knowledge_pack_id)] = normalized_chunk_items

        if mutated:
            self._save_chunks(normalized_items)

        return normalized_items

    def _save_chunks(self, items: dict) -> None:
        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _serialize_pack(self, pack: dict) -> dict:
        return {
            "knowledge_pack_id": pack["knowledge_pack_id"],
            "owner_user_id": pack["owner_user_id"],
            "title": pack["title"],
            "description": pack.get("description"),
            "scope": pack.get("scope"),
            "domain": pack.get("domain"),
            "visibility": pack.get("visibility"),
            "status": pack["status"],
            "supported_task_types": list(pack.get("supported_task_types", [])),
            "uploaded_files": list(pack.get("uploaded_files", [])),
            "total_chunks": int(pack.get("total_chunks", 0)),
            "created_at": pack["created_at"],
            "updated_at": pack["updated_at"],
        }

    def _sort_packs(self, packs: list[dict]) -> list[dict]:
        return sorted(
            packs,
            key=lambda item: (item.get("updated_at", ""), item.get("created_at", "")),
            reverse=True,
        )

    def _resolve_pack_group_key(self, pack: dict) -> tuple[str, str | None, str]:
        if pack.get("scope") == "system":
            return (
                str(pack.get("scope") or "system"),
                self._normalize_domain(pack.get("domain")),
                str(pack.get("visibility") or "internal"),
            )

        return (
            str(pack.get("scope") or "user"),
            str(pack.get("owner_user_id") or ""),
            str(pack.get("visibility") or "user"),
        )

    def upload_pack(
        self,
        *,
        owner_user_id: str,
        title: str,
        description: str | None = None,
        supported_task_types: list[str] | None = None,
        uploaded_files: list[dict],
        scope: str = "user",
        domain: str | None = None,
        visibility: str | None = None,
        status: str = "inactive",
    ) -> dict:
        normalized_scope = self._normalize_scope(scope)
        normalized_domain = self._normalize_domain(domain)
        normalized_status = self._normalize_status(status)
        normalized_owner_user_id = self._normalize_text(owner_user_id)
        normalized_title = self._normalize_text(title)
        normalized_description = self._normalize_text(description)
        normalized_supported_task_types = self._normalize_supported_task_types(supported_task_types)
        normalized_visibility = self._normalize_visibility(
            visibility,
            default="internal" if normalized_scope == "system" else "user",
        )

        if normalized_scope == "system":
            normalized_owner_user_id = "system"
            if normalized_domain != "reels":
                raise HTTPException(
                    status_code=400,
                    detail="System knowledge domain 'reels' is required in phase 1",
                )
            if not normalized_supported_task_types:
                normalized_supported_task_types = sorted(self.REELS_TASK_TYPES)
            elif any(task_type not in self.REELS_TASK_TYPES for task_type in normalized_supported_task_types):
                raise HTTPException(
                    status_code=400,
                    detail="System reels knowledge only supports reel_idea, reel_script, and reel_feedback in phase 1",
                )
        elif not normalized_owner_user_id:
            raise HTTPException(status_code=400, detail="owner_user_id is required for knowledge pack upload")

        if not normalized_title:
            raise HTTPException(status_code=400, detail="title is required for knowledge pack upload")
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="Knowledge pack upload requires at least one file")

        knowledge_pack_id = f"kp_{uuid4().hex[:12]}"
        now_iso = self._now().isoformat()
        pack_storage_dir = self.storage_dir / knowledge_pack_id
        files_metadata = []
        chunks = []
        file_artifacts = []

        for raw_file in uploaded_files:
            file_name = self._sanitize_file_name(str(raw_file.get("file_name") or raw_file.get("filename") or ""))
            content = raw_file.get("content")
            if not isinstance(content, (bytes, bytearray)):
                raise HTTPException(
                    status_code=400,
                    detail=f"Knowledge pack upload failed for file '{file_name}'",
                )

            content_bytes = bytes(content)
            if not content_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Knowledge pack upload failed: file '{file_name}' is empty",
                )
            extracted_text = self._extract_text_from_file(file_name, content_bytes)
            chunk_texts = self._chunk_text(extracted_text)
            if not chunk_texts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Knowledge pack document parsing failed for file '{file_name}': no usable text was extracted",
                )

            file_id = f"kpf_{uuid4().hex[:12]}"
            extension = Path(file_name).suffix.lower()
            stored_file_name = f"{file_id}{extension}"
            files_metadata.append({
                "file_id": file_id,
                "file_name": file_name,
                "content_type": raw_file.get("content_type"),
                "extension": extension,
                "file_size_bytes": len(content_bytes),
                "chunk_count": len(chunk_texts),
                "uploaded_at": now_iso,
            })
            file_artifacts.append({
                "stored_file_name": stored_file_name,
                "content": content_bytes,
            })

            for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                chunks.append({
                    "chunk_id": f"kpc_{uuid4().hex[:12]}",
                    "knowledge_pack_id": knowledge_pack_id,
                    "file_id": file_id,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "chunk_label": self._extract_chunk_label(chunk_text),
                    "scope": normalized_scope,
                    "domain": normalized_domain,
                    "visibility": normalized_visibility,
                    "supported_task_types": list(normalized_supported_task_types),
                    "knowledge_pack_title": normalized_title,
                    "source_label": f"{normalized_title} :: {file_name}",
                    "embedding_version": self.EMBEDDING_VERSION,
                    "embedding": self._build_local_embedding(chunk_text),
                    "retrieval_terms": self._top_retrieval_terms(chunk_text),
                    "text": chunk_text,
                })

        pack = {
            "knowledge_pack_id": knowledge_pack_id,
            "owner_user_id": normalized_owner_user_id,
            "title": normalized_title,
            "description": normalized_description,
            "scope": normalized_scope,
            "domain": normalized_domain,
            "visibility": normalized_visibility,
            "status": normalized_status,
            "supported_task_types": normalized_supported_task_types,
            "uploaded_files": files_metadata,
            "total_chunks": len(chunks),
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        packs = self._load_packs()
        all_chunks = self._load_chunks()

        try:
            pack_storage_dir.mkdir(parents=True, exist_ok=True)
            for file_artifact in file_artifacts:
                (pack_storage_dir / file_artifact["stored_file_name"]).write_bytes(file_artifact["content"])

            packs[knowledge_pack_id] = pack
            if normalized_status == "active":
                target_group = self._resolve_pack_group_key(pack)
                for candidate_pack_id, candidate_pack in packs.items():
                    if candidate_pack_id == knowledge_pack_id:
                        continue
                    if self._resolve_pack_group_key(candidate_pack) != target_group:
                        continue
                    candidate_pack["status"] = "inactive"
                    candidate_pack["updated_at"] = now_iso

            all_chunks[knowledge_pack_id] = chunks
            self._save_packs(packs)
            self._save_chunks(all_chunks)
        except HTTPException:
            raise
        except Exception as exc:
            shutil.rmtree(pack_storage_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Knowledge pack upload failed: {exc}") from exc

        logger.info(
            "Uploaded knowledge pack knowledge_pack_id=%s scope=%s domain=%s visibility=%s files=%s chunks=%s",
            knowledge_pack_id,
            normalized_scope,
            normalized_domain,
            normalized_visibility,
            len(files_metadata),
            len(chunks),
        )
        return self._serialize_pack(pack)

    def list_packs(
        self,
        *,
        owner_user_id: str | None = None,
        scope: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
    ) -> dict:
        packs = self._load_packs()
        normalized_owner_user_id = self._normalize_text(owner_user_id)
        normalized_scope = self._normalize_scope(scope, default="user") if scope is not None else None
        normalized_domain = self._normalize_domain(domain)
        normalized_visibility = self._normalize_visibility(visibility, default="user") if visibility is not None else None

        matched_packs = []
        for pack in packs.values():
            if normalized_owner_user_id and pack.get("owner_user_id") != normalized_owner_user_id:
                continue
            if normalized_scope and pack.get("scope") != normalized_scope:
                continue
            if normalized_domain and pack.get("domain") != normalized_domain:
                continue
            if normalized_visibility and pack.get("visibility") != normalized_visibility:
                continue
            matched_packs.append(self._serialize_pack(pack))

        return {
            "scope": normalized_scope,
            "domain": normalized_domain,
            "visibility": normalized_visibility,
            "knowledge_packs": self._sort_packs(matched_packs),
        }

    def activate_pack(self, knowledge_pack_id: str) -> dict:
        packs = self._load_packs()
        pack = packs.get(knowledge_pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Knowledge pack was not found")

        now_iso = self._now().isoformat()
        target_group = self._resolve_pack_group_key(pack)
        for candidate_pack_id, candidate_pack in packs.items():
            if self._resolve_pack_group_key(candidate_pack) != target_group:
                continue
            candidate_pack["status"] = "active" if candidate_pack_id == knowledge_pack_id else "inactive"
            candidate_pack["updated_at"] = now_iso

        self._save_packs(packs)
        logger.info(
            "Activated knowledge pack knowledge_pack_id=%s scope=%s domain=%s",
            knowledge_pack_id,
            pack.get("scope"),
            pack.get("domain"),
        )
        return self._serialize_pack(packs[knowledge_pack_id])

    def deactivate_pack(self, knowledge_pack_id: str) -> dict:
        packs = self._load_packs()
        pack = packs.get(knowledge_pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Knowledge pack was not found")

        pack["status"] = "inactive"
        pack["updated_at"] = self._now().isoformat()
        packs[knowledge_pack_id] = pack
        self._save_packs(packs)
        logger.info(
            "Deactivated knowledge pack knowledge_pack_id=%s scope=%s domain=%s",
            knowledge_pack_id,
            pack.get("scope"),
            pack.get("domain"),
        )
        return self._serialize_pack(pack)

    def delete_pack(self, knowledge_pack_id: str) -> dict:
        packs = self._load_packs()
        if knowledge_pack_id not in packs:
            raise HTTPException(status_code=404, detail="Knowledge pack was not found")

        packs.pop(knowledge_pack_id, None)
        chunks = self._load_chunks()
        chunks.pop(knowledge_pack_id, None)
        self._save_packs(packs)
        self._save_chunks(chunks)
        shutil.rmtree(self.storage_dir / knowledge_pack_id, ignore_errors=True)
        logger.info("Deleted knowledge pack knowledge_pack_id=%s", knowledge_pack_id)
        return {
            "knowledge_pack_id": knowledge_pack_id,
            "deleted": True,
        }

    def get_pack_file_paths(self, knowledge_pack_id: str) -> list[str]:
        packs = self._load_packs()
        pack = packs.get(knowledge_pack_id)
        if not pack:
            raise HTTPException(status_code=404, detail="Knowledge pack was not found")

        resolved_paths = []
        for file_metadata in pack.get("uploaded_files", []):
            file_id = str(file_metadata.get("file_id") or "").strip()
            extension = str(file_metadata.get("extension") or "").strip()
            if not file_id:
                continue
            stored_path = self.storage_dir / knowledge_pack_id / f"{file_id}{extension}"
            if stored_path.exists():
                resolved_paths.append(str(stored_path.resolve()))
        return resolved_paths

    def get_active_pack(self, owner_user_id: str) -> dict | None:
        packs = self._load_packs()
        active_packs = [
            pack for pack in packs.values()
            if (
                pack.get("owner_user_id") == owner_user_id
                and pack.get("scope") == "user"
                and pack.get("visibility") == "user"
                and pack.get("status") == "active"
            )
        ]
        if not active_packs:
            return None

        active_packs = self._sort_packs(active_packs)
        return active_packs[0]

    def get_active_system_packs(self, *, domain: str, task_type: str | None = None) -> list[dict]:
        normalized_domain = self._normalize_domain(domain)
        packs = self._load_packs()
        matched_packs = []
        for pack in packs.values():
            if pack.get("scope") != "system":
                continue
            if pack.get("visibility") != "internal":
                continue
            if pack.get("status") != "active":
                continue
            if normalized_domain and pack.get("domain") != normalized_domain:
                continue
            supported_task_types = pack.get("supported_task_types", [])
            if task_type and supported_task_types and task_type not in supported_task_types:
                continue
            matched_packs.append(pack)
        return self._sort_packs(matched_packs)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (text or "").lower())
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in self.STOP_WORDS
        ]

    def _cosine_similarity(self, query_embedding: list[float], chunk_embedding: list[float]) -> float:
        if not query_embedding or not chunk_embedding:
            return 0.0
        return sum(query_value * chunk_value for query_value, chunk_value in zip(query_embedding, chunk_embedding))

    def _build_retrieval_query(
        self,
        *,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
    ) -> str:
        parts = [message, task_type]
        parts.extend(self.TASK_KEYWORDS.get(task_type, []))

        if goal:
            parts.append(goal)

        if profile_context:
            parts.extend(filter(None, [
                profile_context.get("niche"),
                profile_context.get("target_audience"),
                profile_context.get("brand_voice"),
                " ".join(profile_context.get("content_focus", [])),
                " ".join(profile_context.get("strengths", [])),
                " ".join(profile_context.get("weak_points", [])),
            ]))

        if recent_content_context:
            parts.extend(filter(None, [
                " ".join(recent_content_context.get("top_formats", [])),
                " ".join(recent_content_context.get("best_topics", [])),
                " ".join(recent_content_context.get("weak_topics", [])),
                " ".join(recent_content_context.get("best_ctas", [])),
                " ".join(recent_content_context.get("weak_ctas", [])),
                " ".join(recent_content_context.get("notes", [])),
            ]))

        if recent_posts_context:
            topic_samples = [
                str(post.get("topic") or "")
                for post in recent_posts_context.get("posts", [])[:5]
                if post.get("topic")
            ]
            if topic_samples:
                parts.append(" ".join(topic_samples))

        return "\n".join(part for part in parts if part)

    def _score_chunk(
        self,
        query_counter: Counter,
        query_tokens: list[str],
        chunk_text: str,
        *,
        task_type: str,
        query_embedding: list[float],
        chunk_embedding: list[float] | None,
        retrieval_terms: list[str] | None,
    ) -> float:
        chunk_tokens = self._tokenize(chunk_text)
        if not chunk_tokens:
            return 0.0

        chunk_counter = Counter(chunk_tokens)
        overlap_score = sum(
            min(query_counter[token], chunk_counter.get(token, 0))
            for token in query_counter
            if token in chunk_counter
        )
        normalized_chunk_text = chunk_text.lower()
        phrase_bonus = 0.0
        for token in query_tokens:
            if len(token) >= 6 and token in normalized_chunk_text:
                phrase_bonus += 0.15

        task_bonus = 0.0
        for keyword in self.TASK_KEYWORDS.get(task_type, []):
            normalized_keyword = keyword.lower()
            if normalized_keyword in normalized_chunk_text:
                task_bonus += 0.3 if " " in normalized_keyword else 0.1

        embedding_score = self._cosine_similarity(query_embedding, chunk_embedding or [])
        retrieval_term_bonus = 0.0
        for retrieval_term in retrieval_terms or []:
            if retrieval_term in query_counter:
                retrieval_term_bonus += 0.2

        return overlap_score + phrase_bonus + task_bonus + retrieval_term_bonus + (embedding_score * 3.5)

    def _select_top_chunks(
        self,
        *,
        candidate_packs: list[dict],
        query_tokens: list[str],
        top_k: int,
        task_type: str,
    ) -> tuple[list[dict], list[str]]:
        if not candidate_packs or not query_tokens:
            return [], []

        all_chunks = self._load_chunks()
        query_counter = Counter(query_tokens)
        query_embedding = self._build_local_embedding_from_tokens(query_tokens)
        scored_chunks = []
        for pack in candidate_packs:
            pack_id = str(pack.get("knowledge_pack_id") or "")
            for chunk in all_chunks.get(pack_id, []):
                chunk_text = str(chunk.get("text") or "")
                score = self._score_chunk(
                    query_counter,
                    query_tokens,
                    chunk_text,
                    task_type=task_type,
                    query_embedding=query_embedding,
                    chunk_embedding=chunk.get("embedding"),
                    retrieval_terms=chunk.get("retrieval_terms"),
                )
                if score <= 0:
                    continue
                scored_chunks.append((
                    score,
                    {
                        **chunk,
                        "chunk_label": chunk.get("chunk_label") or self._extract_chunk_label(chunk_text),
                        "scope": chunk.get("scope") or pack.get("scope"),
                        "domain": chunk.get("domain") or pack.get("domain"),
                        "visibility": chunk.get("visibility") or pack.get("visibility"),
                    },
                ))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        matched_pack_ids = []
        for chunk in selected_chunks:
            matched_pack_id = self._normalize_text(str(chunk.get("knowledge_pack_id") or ""))
            if matched_pack_id and matched_pack_id not in matched_pack_ids:
                matched_pack_ids.append(matched_pack_id)

        return selected_chunks, matched_pack_ids

    def retrieve_context(
        self,
        *,
        owner_user_id: str,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
        top_k: int = 4,
    ) -> dict:
        active_pack = self.get_active_pack(owner_user_id)
        if not active_pack:
            return {
                "used_knowledge_pack": False,
                "knowledge_pack_id": None,
                "knowledge_pack_title": None,
                "retrieved_chunks_count": 0,
                "knowledge_applied": False,
                "chunks": [],
            }

        if active_pack.get("supported_task_types") and task_type not in active_pack.get("supported_task_types", []):
            return {
                "used_knowledge_pack": False,
                "knowledge_pack_id": None,
                "knowledge_pack_title": None,
                "retrieved_chunks_count": 0,
                "knowledge_applied": False,
                "chunks": [],
            }

        query_text = self._build_retrieval_query(
            task_type=task_type,
            message=message,
            goal=goal,
            profile_context=profile_context,
            recent_content_context=recent_content_context,
            recent_posts_context=recent_posts_context,
        )
        query_tokens = self._tokenize(query_text)
        selected_chunks, _ = self._select_top_chunks(
            candidate_packs=[active_pack],
            query_tokens=query_tokens,
            top_k=top_k,
            task_type=task_type,
        )
        return {
            "used_knowledge_pack": bool(selected_chunks),
            "knowledge_pack_id": active_pack["knowledge_pack_id"] if selected_chunks else None,
            "knowledge_pack_title": active_pack["title"] if selected_chunks else None,
            "retrieved_chunks_count": len(selected_chunks),
            "knowledge_applied": bool(selected_chunks),
            "chunks": selected_chunks,
        }

    def retrieve_system_context(
        self,
        *,
        domain: str,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_content_context: dict | None = None,
        recent_posts_context: dict | None = None,
        top_k: int = 4,
    ) -> dict:
        candidate_packs = self.get_active_system_packs(domain=domain, task_type=task_type)
        if not candidate_packs:
            return {
                "used_system_knowledge": False,
                "matched_knowledge_domain": None,
                "matched_knowledge_pack_ids": [],
                "retrieved_chunk_count": 0,
                "retrieved_chunk_titles": [],
                "chunks": [],
            }

        query_text = self._build_retrieval_query(
            task_type=task_type,
            message=message,
            goal=goal,
            profile_context=profile_context,
            recent_content_context=recent_content_context,
            recent_posts_context=recent_posts_context,
        )
        query_tokens = self._tokenize(query_text)
        resolved_top_k = max(top_k, self.REELS_TOP_K_BY_TASK.get(task_type, top_k))
        selected_chunks, matched_pack_ids = self._select_top_chunks(
            candidate_packs=candidate_packs,
            query_tokens=query_tokens,
            top_k=resolved_top_k,
            task_type=task_type,
        )
        retrieved_chunk_titles = [
            str(chunk.get("chunk_label") or chunk.get("file_name") or f"Chunk {index}")
            for index, chunk in enumerate(selected_chunks, start=1)
        ]
        logger.info(
            "Retrieved system knowledge chunks domain=%s task_type=%s matched_packs=%s chunks=%s labels=%s",
            domain,
            task_type,
            ",".join(matched_pack_ids) if matched_pack_ids else "none",
            len(selected_chunks),
            " | ".join(retrieved_chunk_titles[:5]) if retrieved_chunk_titles else "none",
        )
        return {
            "used_system_knowledge": bool(selected_chunks),
            "matched_knowledge_domain": domain if selected_chunks else None,
            "matched_knowledge_pack_ids": matched_pack_ids,
            "retrieved_chunk_count": len(selected_chunks),
            "retrieved_chunk_titles": retrieved_chunk_titles,
            "chunks": selected_chunks,
        }
