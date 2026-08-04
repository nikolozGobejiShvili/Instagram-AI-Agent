from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from docx import Document
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
from pypdf import PdfReader


class ReelsKnowledgeIngestion(Component):
    display_name = "Reels Knowledge Ingestion"
    description = "Ingest internal reels files into a hidden Chroma vector store."
    icon = "Database"
    name = "ReelsKnowledgeIngestion"

    inputs = [
        MessageTextInput(
            name="payload_json",
            display_name="Payload JSON",
            value="",
            required=True,
            info="JSON payload with file_paths, knowledge_pack_id, and reels metadata.",
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="build_message"),
    ]

    def build_message(self) -> Message:
        try:
            payload = self._load_payload(self.payload_json)
            self._load_backend_env(payload)
            summary = self._ingest_payload(payload)
            return Message(text=json.dumps(summary, ensure_ascii=False))
        except Exception as exc:
            error_payload = {
                "error": {
                    "code": "knowledge_ingestion_failed",
                    "message": "Internal reels knowledge ingestion failed.",
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

    def _candidate_backend_env_paths(self, payload: dict) -> list[Path]:
        candidates: list[Path] = []
        file_paths = payload.get("file_paths")
        if isinstance(file_paths, list):
            for raw_path in file_paths:
                try:
                    file_path = Path(str(raw_path)).resolve()
                except Exception:
                    continue
                for parent in [file_path.parent, *file_path.parents]:
                    if parent.name == "backend":
                        candidates.append(parent / ".env")
                        break

        cwd = Path.cwd().resolve()
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

    def _candidate_backend_data_paths(self, payload: dict) -> list[Path]:
        candidates: list[Path] = []
        for env_path in self._candidate_backend_env_paths(payload):
            if env_path.exists():
                candidates.append(env_path.parent / "app" / "data" / "knowledge_pack_chunks.json")
        cwd = Path.cwd().resolve()
        for parent in [cwd, *cwd.parents]:
            if parent.name == "backend":
                candidates.append(parent / "app" / "data" / "knowledge_pack_chunks.json")
                break
            candidates.append(parent / "backend" / "app" / "data" / "knowledge_pack_chunks.json")
        deduped: list[Path] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _load_backend_env(self, payload: dict) -> None:
        for env_path in self._candidate_backend_env_paths(payload):
            if not env_path.exists():
                continue
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            break

    def _resolve_chroma_dir(self, payload: dict) -> Path:
        configured_path = os.getenv("LANGFLOW_CHROMA_PERSIST_DIR", "").strip()
        if configured_path:
            return Path(configured_path).expanduser().resolve()

        file_paths = payload.get("file_paths")
        if isinstance(file_paths, list) and file_paths:
            base_file = Path(str(file_paths[0])).resolve()
            for parent in [base_file.parent, *base_file.parents]:
                if parent.name == "backend":
                    return (parent / "app" / "data" / "langflow_chroma").resolve()

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

    def _tokenize(self, value: str | None) -> list[str]:
        return re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (value or "").lower())

    def _embedding_bucket(self, token: str, size: int) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket_index = digest[0] % size
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        magnitude = 1.0 + (digest[2] / 255.0)
        return bucket_index, sign * magnitude

    def _embedding(self, text: str, size: int = 64) -> list[float]:
        vector = [0.0] * size
        for token in self._tokenize(text):
            bucket_index, delta = self._embedding_bucket(token, size)
            vector[bucket_index] += delta
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0:
            return vector
        return [component / norm for component in vector]

    def _extract_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        if suffix == ".docx":
            document = Document(str(file_path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        raise ValueError(f"Unsupported reels knowledge file type: {suffix}")

    def _load_prechunked_backend_chunks(self, payload: dict) -> list[dict]:
        knowledge_pack_id = str(payload.get("knowledge_pack_id") or "").strip()
        if not knowledge_pack_id:
            return []

        for data_path in self._candidate_backend_data_paths(payload):
            if not data_path.exists():
                continue
            try:
                raw_items = json.loads(data_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            chunk_items = raw_items.get(knowledge_pack_id)
            if not isinstance(chunk_items, list):
                continue
            normalized_chunks = []
            for chunk in chunk_items:
                if not isinstance(chunk, dict):
                    continue
                chunk_text = str(chunk.get("text") or "").strip()
                if not chunk_text:
                    continue
                normalized_chunks.append({
                    "chunk_index": int(chunk.get("chunk_index") or len(normalized_chunks)),
                    "chunk_title": str(chunk.get("chunk_label") or chunk.get("source_label") or chunk.get("file_name") or "Reels guidance").strip(),
                    "text": chunk_text,
                    "file_name": str(chunk.get("file_name") or ""),
                })
            if normalized_chunks:
                return normalized_chunks
        return []

    def _chunk_text(self, text: str, *, file_name: str) -> list[dict]:
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized_text.split("\n")]
        sections: list[tuple[str, list[str]]] = []
        current_title = file_name
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue

            heading_match = re.match(r"^(#{1,6}\s*)?(KNOWLEDGE MODULE\s+\d+.*|[A-Z][A-Z0-9 .,:()'/-]{8,})$", stripped)
            if heading_match and current_lines:
                sections.append((current_title, current_lines))
                current_title = stripped.lstrip("# ").strip()
                current_lines = []
                continue
            if heading_match and not current_lines:
                current_title = stripped.lstrip("# ").strip()
                continue

            current_lines.append(stripped)

        if current_lines:
            sections.append((current_title, current_lines))

        chunk_records: list[dict] = []
        chunk_index = 0
        for section_title, section_lines in sections:
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                continue

            paragraphs = [paragraph.strip() for paragraph in section_text.split("\n\n") if paragraph.strip()]
            buffer = ""
            for paragraph in paragraphs:
                candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
                if len(candidate) <= 950:
                    buffer = candidate
                    continue
                if buffer:
                    chunk_records.append({
                        "chunk_index": chunk_index,
                        "chunk_title": section_title,
                        "text": buffer,
                    })
                    chunk_index += 1
                if len(paragraph) <= 950:
                    buffer = paragraph
                    continue
                start_index = 0
                while start_index < len(paragraph):
                    sliced = paragraph[start_index:start_index + 850].strip()
                    if sliced:
                        chunk_records.append({
                            "chunk_index": chunk_index,
                            "chunk_title": section_title,
                            "text": sliced,
                        })
                        chunk_index += 1
                    start_index += 730
                buffer = ""
            if buffer:
                chunk_records.append({
                    "chunk_index": chunk_index,
                    "chunk_title": section_title,
                    "text": buffer,
                })
                chunk_index += 1

        return chunk_records

    def _ingest_payload(self, payload: dict) -> dict:
        knowledge_pack_id = str(payload.get("knowledge_pack_id") or "").strip()
        source_title = str(payload.get("source_title") or payload.get("title") or "").strip()
        description = self._safe_truncate(str(payload.get("description") or ""), 220)
        file_paths = payload.get("file_paths")
        supported_task_types = payload.get("supported_task_types") or ["reel_idea", "reel_script", "reel_feedback"]
        if not knowledge_pack_id:
            raise ValueError("knowledge_pack_id is required")
        if not isinstance(file_paths, list) or not file_paths:
            raise ValueError("file_paths are required")

        chroma_dir = self._resolve_chroma_dir(payload)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=self._resolve_collection_name(),
            metadata={"domain": "reels", "scope": "system", "visibility": "internal"},
        )

        existing = collection.get(where={"knowledge_pack_id": knowledge_pack_id})
        existing_ids = list(existing.get("ids") or []) if isinstance(existing, dict) else []
        if existing_ids:
            collection.delete(ids=existing_ids)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []
        chunk_titles: list[str] = []

        ingested_at = datetime.now(timezone.utc).isoformat()
        prechunked_chunks = self._load_prechunked_backend_chunks(payload)
        if prechunked_chunks:
            ingested_at = datetime.now(timezone.utc).isoformat()
            for chunk in prechunked_chunks:
                chunk_text = str(chunk["text"]).strip()
                file_name = str(chunk.get("file_name") or "knowledge-pack")
                chunk_id = f"{knowledge_pack_id}:{file_name}:{chunk['chunk_index']}"
                ids.append(chunk_id)
                documents.append(chunk_text)
                chunk_title = self._safe_truncate(str(chunk["chunk_title"]), 160) or file_name
                chunk_titles.append(chunk_title)
                metadatas.append({
                    "knowledge_pack_id": knowledge_pack_id,
                    "source_title": source_title or file_name,
                    "source_file_name": file_name,
                    "chunk_title": chunk_title,
                    "chunk_index": int(chunk["chunk_index"]),
                    "domain": "reels",
                    "scope": "system",
                    "visibility": "internal",
                    "status": "active",
                    "supported_task_types": ",".join(str(item) for item in supported_task_types),
                    "description": description or "",
                    "ingested_at": ingested_at,
                })
                embeddings.append(self._embedding(chunk_text))
        else:
            for raw_path in file_paths:
                file_path = Path(str(raw_path)).resolve()
                if not file_path.exists():
                    raise FileNotFoundError(f"Knowledge file not found: {file_path}")

                extracted_text = self._extract_text_from_file(file_path)
                chunks = self._chunk_text(extracted_text, file_name=file_path.name)
                for chunk in chunks:
                    chunk_text = str(chunk["text"]).strip()
                    if not chunk_text:
                        continue
                    chunk_id = f"{knowledge_pack_id}:{file_path.name}:{chunk['chunk_index']}"
                    ids.append(chunk_id)
                    documents.append(chunk_text)
                    chunk_title = self._safe_truncate(str(chunk["chunk_title"]), 160) or file_path.stem
                    chunk_titles.append(chunk_title)
                    metadatas.append({
                        "knowledge_pack_id": knowledge_pack_id,
                        "source_title": source_title or file_path.stem,
                        "source_file_name": file_path.name,
                        "chunk_title": chunk_title,
                        "chunk_index": int(chunk["chunk_index"]),
                        "domain": "reels",
                        "scope": "system",
                        "visibility": "internal",
                        "status": "active",
                        "supported_task_types": ",".join(str(item) for item in supported_task_types),
                        "description": description or "",
                        "ingested_at": ingested_at,
                    })
                    embeddings.append(self._embedding(chunk_text))

        if not ids:
            raise ValueError("No reels knowledge chunks were created from the provided files")

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return {
            "ingestion_ok": True,
            "knowledge_pack_id": knowledge_pack_id,
            "vector_store_provider": "chroma",
            "collection_name": self._resolve_collection_name(),
            "chunk_count": len(ids),
            "embeddings_stored": True,
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "chunk_titles": chunk_titles[:8],
        }
