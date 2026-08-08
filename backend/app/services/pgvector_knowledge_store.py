"""Knowledge chunks and their embeddings, in Postgres.

Chroma was per-container and this system now has two: the backend that answers
requests and the Langflow instance that authors flows. Each would have written to
its own local directory, so material indexed by one was invisible to the other.
A shared table removes the question.

Two properties this is built around:

* **Task filtering happens in the query, not afterwards.** Fetching the globally
  nearest chunks and discarding the ones for other tasks returns fewer results
  than asked for, and sometimes none, while looking like a ranking problem.
* **Rows record the model that embedded them.** Vectors from different models
  are not comparable, but cosine distance between them is still a number, so a
  mismatch produces confident nonsense rather than an error. Filtering on the
  model makes stale rows invisible instead of misleading.

No ANN index yet, deliberately. Exact search over a knowledge base of this size
is milliseconds and always correct; ivfflat needs training data to be built at
all, and both index types trade recall for a speed nobody needs here. Add one
when the row count justifies it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import DEFAULT_CANDIDATE_POOL, RerankerService

logger = logging.getLogger(__name__)

TABLE_NAME = "knowledge_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    chunk_label: str | None
    knowledge_pack_title: str | None
    knowledge_pack_id: str
    score: float


class PgVectorNotConfigured(RuntimeError):
    """No database URL. Retrying will not help until an operator sets one."""


class PgVectorKnowledgeStore:
    def __init__(
        self,
        *,
        dsn: str | None = None,
        embedding_service: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
        candidate_pool: int | None = None,
    ):
        self._dsn = dsn if dsn is not None else os.getenv("KNOWLEDGE_DATABASE_URL", "").strip() or os.getenv("DATABASE_URL", "").strip()
        self.embeddings = embedding_service or EmbeddingService()
        self.reranker = reranker or RerankerService()
        self.candidate_pool = int(
            candidate_pool or os.getenv("KNOWLEDGE_CANDIDATE_POOL", "").strip() or DEFAULT_CANDIDATE_POOL
        )

    def is_configured(self) -> bool:
        return bool(self._dsn) and self.embeddings.is_configured()

    def _connect(self):
        if not self._dsn:
            raise PgVectorNotConfigured(
                "DATABASE_URL is not set. Knowledge indexing and retrieval need a Postgres connection."
            )
        import psycopg

        return psycopg.connect(self._dsn, connect_timeout=15)

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> None:
        """Create the extension and table if absent. Safe to call repeatedly.

        The vector width comes from the active embedding model rather than a
        literal, so changing the model and forgetting the column is impossible --
        Postgres rejects the insert instead of storing something unsearchable.
        """
        dimensions = self.embeddings.dimensions
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id                   TEXT PRIMARY KEY,
                        knowledge_pack_id    TEXT NOT NULL,
                        chunk_index          INTEGER NOT NULL,
                        chunk_label          TEXT,
                        content              TEXT NOT NULL,
                        scope                TEXT NOT NULL DEFAULT 'system',
                        domain               TEXT,
                        visibility           TEXT,
                        knowledge_pack_title TEXT,
                        supported_task_types TEXT[] NOT NULL DEFAULT '{{}}',
                        embedding_model      TEXT NOT NULL,
                        embedding            vector({dimensions}) NOT NULL,
                        created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                # Deleting a pack's rows before reindexing is the common write,
                # and the task-type filter runs on every read.
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {TABLE_NAME}_pack_idx ON {TABLE_NAME} (knowledge_pack_id)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tasks_idx ON {TABLE_NAME} USING GIN (supported_task_types)"
                )
            conn.commit()

    # --------------------------------------------------------------- ingestion
    def index_pack(self, *, knowledge_pack_id: str, chunks: list[dict]) -> int:
        """Embed and store every chunk of one pack, replacing what was there.

        Replace rather than append: re-uploading a pack after an edit would
        otherwise leave the superseded text in the store, still retrievable and
        indistinguishable from the correction.
        """
        if not chunks:
            return 0

        self.ensure_schema()
        # "text" is what KnowledgePackService stores; the others are accepted so a
        # Langflow flow writing through this class does not have to match an
        # internal key name.
        texts = [
            str(chunk.get("text") or chunk.get("chunk_text") or chunk.get("content") or "")
            for chunk in chunks
        ]
        vectors = self.embeddings.embed(texts)

        rows = []
        for chunk, text, vector in zip(chunks, texts, vectors):
            chunk_index = int(chunk.get("chunk_index") or 0)
            rows.append((
                f"{knowledge_pack_id}:{chunk_index}",
                knowledge_pack_id,
                chunk_index,
                chunk.get("chunk_label"),
                text,
                str(chunk.get("scope") or "system"),
                chunk.get("domain"),
                chunk.get("visibility"),
                chunk.get("knowledge_pack_title"),
                list(chunk.get("supported_task_types") or []),
                self.embeddings.model,
                "[" + ",".join(f"{value:.7f}" for value in vector) + "]",
            ))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {TABLE_NAME} WHERE knowledge_pack_id = %s", (knowledge_pack_id,))
                cur.executemany(
                    f"""
                    INSERT INTO {TABLE_NAME} (
                        id, knowledge_pack_id, chunk_index, chunk_label, content,
                        scope, domain, visibility, knowledge_pack_title,
                        supported_task_types, embedding_model, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    rows,
                )
            conn.commit()

        logger.info(
            "Indexed knowledge pack knowledge_pack_id=%s chunks=%s model=%s",
            knowledge_pack_id,
            len(rows),
            self.embeddings.model,
        )
        return len(rows)

    def delete_pack(self, knowledge_pack_id: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {TABLE_NAME} WHERE knowledge_pack_id = %s", (knowledge_pack_id,))
                deleted = cur.rowcount
            conn.commit()
        return deleted

    # --------------------------------------------------------------- retrieval
    def search(self, *, query: str, task_type: str, top_k: int) -> list[RetrievedChunk]:
        """Recall wide by vector, then rerank and keep only what is relevant.

        The single-stage version could not return nothing: ``ORDER BY distance
        LIMIT k`` always yields k rows, so a carousel question retrieved the five
        least-unrelated reels passages and injected them. Every one of those
        competes with the instructions for the model's attention, which is how a
        rule ends up skipped.

        So the vector stage now recalls a wide pool — the thing it is actually
        good at — and a reranker, which reads query and passage together, decides
        what survives. Fewer passages reach the prompt, and when the material
        does not cover the question, none do.
        """
        pool = max(int(top_k), self.candidate_pool)
        query_vector = "[" + ",".join(f"{v:.7f}" for v in self.embeddings.embed_one(query)) + "]"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT content, chunk_label, knowledge_pack_title, knowledge_pack_id,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {TABLE_NAME}
                    WHERE embedding_model = %s
                      AND %s = ANY(supported_task_types)
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, self.embeddings.model, task_type, query_vector, pool),
                )
                rows = cur.fetchall()

        candidates = [
            RetrievedChunk(
                content=row[0],
                chunk_label=row[1],
                knowledge_pack_title=row[2],
                knowledge_pack_id=row[3],
                score=float(row[4]),
            )
            for row in rows
        ]
        if not candidates:
            return []

        ranked = self.reranker.rank(
            query=query,
            documents=[c.content for c in candidates],
            top_k=int(top_k),
        )
        if ranked is None:
            # Reranking was unavailable, not empty. Falling back to vector order
            # is worse retrieval; treating it as "nothing relevant" would be a
            # silent loss of the knowledge base whenever the ranker is down.
            return candidates[: int(top_k)]

        return [
            RetrievedChunk(
                content=candidates[p.index].content,
                chunk_label=candidates[p.index].chunk_label,
                knowledge_pack_title=candidates[p.index].knowledge_pack_title,
                knowledge_pack_id=candidates[p.index].knowledge_pack_id,
                # The reranker's score, not the cosine one: this is what decided
                # inclusion, so it is what any later debugging needs to see.
                score=p.score,
            )
            for p in ranked
            if 0 <= p.index < len(candidates)
        ]

    def stats(self) -> dict:
        """Row counts by embedding model -- the shape of a mismatch.

        More than one model present means part of the store is unreachable to
        search, which is otherwise invisible.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT embedding_model, count(*), count(DISTINCT knowledge_pack_id) "
                    f"FROM {TABLE_NAME} GROUP BY embedding_model"
                )
                rows = cur.fetchall()

        return {
            "active_model": self.embeddings.model,
            "by_model": [
                {"embedding_model": row[0], "chunks": int(row[1]), "packs": int(row[2])} for row in rows
            ],
        }
