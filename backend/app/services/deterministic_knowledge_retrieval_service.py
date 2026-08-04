import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


RAG_TASK_TYPES = {
    "reel_idea",
    "reel_script",
    "reel_feedback",
    "carousel",
    "content_plan",
    "link_analysis",
    "performance_summary",
    "profile_audit",
}

GREETING_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "გამარჯობა",
    "როგორ ხარ",
    "მადლობა",
}

CONTENT_INTENT_PATTERNS = {
    "reel",
    "reels",
    "idea",
    "ideas",
    "script",
    "feedback",
    "carousel",
    "plan",
    "profile",
    "audit",
    "analysis",
    "performance",
    "content",
    "cta",
    "იდეა",
    "იდეები",
    "სცენარ",
    "კარუსელ",
    "კონტენტ",
    "გეგმა",
    "პროფილ",
    "ანალიზ",
}


@dataclass(frozen=True)
class KnowledgeRetrievalResult:
    used: bool
    top_k: int | None
    retrieved_count: int
    collection_name: str
    knowledge_context: str | None = None
    error_type: str | None = None


def _normalize_message(value: str | None) -> str:
    normalized_value = " ".join((value or "").lower().split())
    return re.sub(r"[^\w\s\u10A0-\u10FF]", "", normalized_value).strip()


def _is_obvious_greeting(message: str | None) -> bool:
    normalized_message = _normalize_message(message)
    if not normalized_message:
        return False

    if any(pattern in normalized_message for pattern in CONTENT_INTENT_PATTERNS):
        return False

    if normalized_message in GREETING_PATTERNS:
        return True

    words = normalized_message.split()
    if len(words) > 7:
        return False

    return any(pattern in normalized_message for pattern in GREETING_PATTERNS)


def should_use_knowledge_retrieval(task_type: str | None, message: str | None) -> bool:
    if _is_obvious_greeting(message):
        return False

    return (task_type or "").strip() in RAG_TASK_TYPES


class DeterministicKnowledgeRetrievalService:
    DEFAULT_COLLECTION_NAME = "mariami_reels_playbook_v1"
    DEFAULT_RETRIEVAL_MODE = "lexical"
    DEFAULT_TOP_K = 5
    MAX_CONTEXT_CHARS = 2400
    MAX_CHUNK_CHARS = 520

    def __init__(
        self,
        *,
        chroma_persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        top_k: int | None = None,
    ):
        default_chroma_dir = Path(__file__).resolve().parents[1] / "data" / "langflow_chroma"
        configured_chroma_dir = chroma_persist_dir or os.getenv("LANGFLOW_CHROMA_PERSIST_DIR", "").strip()
        self.chroma_persist_dir = Path(configured_chroma_dir).expanduser().resolve() if configured_chroma_dir else default_chroma_dir
        self.collection_name = (collection_name or self.DEFAULT_COLLECTION_NAME).strip() or self.DEFAULT_COLLECTION_NAME
        self.top_k = int(top_k or self.DEFAULT_TOP_K)
        self.embedding_model = (
            os.getenv("KNOWLEDGE_RETRIEVAL_EMBEDDING_MODEL", "").strip()
            or os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small"
        )
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.timeout_seconds = float(os.getenv("KNOWLEDGE_RETRIEVAL_TIMEOUT_SECONDS", "30").strip() or "30")

    def _retrieval_mode(self) -> str:
        configured_mode = os.getenv("KNOWLEDGE_RETRIEVAL_MODE", "").strip().lower()
        if configured_mode in {"lexical", "local"}:
            return "lexical"
        if configured_mode in {"embedding", "vector", "semantic"}:
            return "embedding"
        if self.collection_name == self.DEFAULT_COLLECTION_NAME:
            return self.DEFAULT_RETRIEVAL_MODE
        return "embedding"

    def retrieve(
        self,
        *,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_posts_context: dict | None = None,
        recent_content_context: dict | None = None,
    ) -> KnowledgeRetrievalResult:
        if not should_use_knowledge_retrieval(task_type, message):
            return KnowledgeRetrievalResult(
                used=False,
                top_k=None,
                retrieved_count=0,
                collection_name=self.collection_name,
            )

        try:
            query = self._build_query(
                task_type=task_type,
                message=message,
                goal=goal,
                profile_context=profile_context,
                recent_posts_context=recent_posts_context,
                recent_content_context=recent_content_context,
            )
            documents = self._query_documents(query=query, top_k=self.top_k)
            knowledge_context = self._build_knowledge_context(documents)
            return KnowledgeRetrievalResult(
                used=True,
                top_k=self.top_k,
                retrieved_count=len(documents),
                collection_name=self.collection_name,
                knowledge_context=knowledge_context,
            )
        except Exception as exc:
            logger.warning(
                "Deterministic knowledge retrieval failed task_type=%s collection=%s error_type=%s",
                task_type,
                self.collection_name,
                type(exc).__name__,
            )
            return KnowledgeRetrievalResult(
                used=True,
                top_k=self.top_k,
                retrieved_count=0,
                collection_name=self.collection_name,
                error_type=type(exc).__name__,
            )

    def _build_query(
        self,
        *,
        task_type: str,
        message: str,
        goal: str | None = None,
        profile_context: dict | None = None,
        recent_posts_context: dict | None = None,
        recent_content_context: dict | None = None,
    ) -> str:
        query_parts = [
            task_type,
            message,
            goal or "",
            str((profile_context or {}).get("niche") or ""),
            str((profile_context or {}).get("target_audience") or ""),
        ]

        best_topics = recent_content_context.get("best_topics") if isinstance(recent_content_context, dict) else None
        if isinstance(best_topics, list):
            query_parts.extend(str(topic) for topic in best_topics[:5])

        posts = recent_posts_context.get("posts") if isinstance(recent_posts_context, dict) else None
        if isinstance(posts, list):
            for post in posts[:5]:
                if not isinstance(post, dict):
                    continue
                query_parts.append(str(post.get("topic") or post.get("content_type") or ""))

        return self._safe_truncate(" ".join(part for part in query_parts if part).strip(), 900) or message

    def _query_documents(self, *, query: str, top_k: int) -> list[str]:
        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma_persist_dir))
        collection = client.get_collection(self.collection_name)
        if self._retrieval_mode() == "lexical":
            return self._lexical_query_documents(collection=collection, query=query, top_k=top_k)

        embedding_dimension = self._collection_embedding_dimension(collection)
        try:
            if embedding_dimension == 64:
                query_embedding = self._hash_embedding(query, size=embedding_dimension)
            else:
                query_embedding = self._openai_embedding(query)

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning(
                "Vector knowledge retrieval failed collection=%s error_type=%s; falling back to lexical Chroma retrieval",
                self.collection_name,
                type(exc).__name__,
            )
            return self._lexical_query_documents(collection=collection, query=query, top_k=top_k)

        nested_documents = response.get("documents") or []
        documents = nested_documents[0] if nested_documents and isinstance(nested_documents[0], list) else nested_documents
        return [
            str(document)
            for document in documents
            if isinstance(document, str) and document.strip()
        ][:top_k]

    def _lexical_query_documents(self, *, collection: Any, query: str, top_k: int) -> list[str]:
        response = collection.get(include=["documents"])
        documents = [
            str(document)
            for document in (response.get("documents") or [])
            if isinstance(document, str) and document.strip()
        ]
        if not documents:
            return []

        query_tokens = set(self._tokenize(query))
        scored_documents = []
        for index, document in enumerate(documents):
            document_tokens = self._tokenize(document)
            token_overlap = len(query_tokens.intersection(document_tokens))
            phrase_score = sum(1 for token in query_tokens if token and token in document.lower())
            score = (token_overlap * 3) + phrase_score
            scored_documents.append((score, index, document))

        scored_documents.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        top_documents = [document for score, _, document in scored_documents if score > 0][:top_k]
        if len(top_documents) < top_k:
            seen = set(top_documents)
            for _, _, document in scored_documents:
                if document in seen:
                    continue
                top_documents.append(document)
                seen.add(document)
                if len(top_documents) >= top_k:
                    break
        return top_documents[:top_k]

    def _collection_embedding_dimension(self, collection: Any) -> int:
        response = collection.get(limit=1, include=["embeddings"])
        embeddings = response.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return 1536
        return len(embeddings[0])

    def _openai_embedding(self, text: str) -> list[float]:
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key is missing for deterministic knowledge retrieval")

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": text,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.openai_base_url}/embeddings", json=payload, headers=headers)
            response.raise_for_status()
        response_payload = response.json()
        return list(response_payload["data"][0]["embedding"])

    def _hash_embedding(self, text: str, *, size: int) -> list[float]:
        vector = [0.0] * size
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket_index = digest[0] % size
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[bucket_index] += sign * (1.0 + (digest[2] / 255.0))

        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0:
            return vector
        return [component / norm for component in vector]

    def _build_knowledge_context(self, documents: list[str]) -> str | None:
        context_parts = []
        total_chars = 0
        for index, document in enumerate(documents, start=1):
            cleaned_document = self._safe_truncate(document, self.MAX_CHUNK_CHARS)
            if not cleaned_document:
                continue

            part = f"{index}. {cleaned_document}"
            if total_chars + len(part) > self.MAX_CONTEXT_CHARS:
                break
            context_parts.append(part)
            total_chars += len(part)

        if not context_parts:
            return None

        return "\n".join([
            "Internal Mariami Reels strategy context. Use it only as hidden strategy guidance; do not quote or reveal it.",
            *context_parts,
        ])

    def _tokenize(self, value: str | None) -> list[str]:
        return re.findall(r"[0-9A-Za-z\u10A0-\u10FF]+", (value or "").lower())

    def _safe_truncate(self, value: str | None, max_chars: int) -> str | None:
        normalized_value = " ".join((value or "").split()).strip()
        if not normalized_value:
            return None
        if len(normalized_value) <= max_chars:
            return normalized_value
        return f"{normalized_value[: max_chars - 3].rstrip()}..."
