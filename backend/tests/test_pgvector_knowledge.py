"""The shared vector store, tested where it fails quietly.

None of these need a database. Every case here is about a mistake that produces
plausible output rather than an error: a model mismatch that still computes a
distance, a task filter applied after the query instead of inside it, a stale
chunk left behind by a re-upload.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import embedding_service as embedding_module  # noqa: E402
from app.services.deterministic_knowledge_retrieval_service import (  # noqa: E402
    DeterministicKnowledgeRetrievalService,
)
from app.services.embedding_service import (  # noqa: E402
    MODEL_DIMENSIONS,
    EmbeddingFailed,
    EmbeddingNotConfigured,
    EmbeddingService,
)
from app.services.pgvector_knowledge_store import (  # noqa: E402
    TABLE_NAME,
    PgVectorKnowledgeStore,
)


class FakeCursor:
    def __init__(self, rows):
        self.executed = []
        self.rowcount = 0
        self._rows = rows

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def executemany(self, sql, rows):
        self.executed.append((" ".join(sql.split()), rows))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeConnection:
    def __init__(self, rows=()):
        self.cursor_obj = FakeCursor(list(rows))
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class StubEmbeddings:
    """Deterministic vectors; the values do not matter, the wiring does."""

    def __init__(self, model="text-embedding-3-small", dimensions=1536):
        self.model = model
        self.dimensions = dimensions
        self.embedded = []

    def is_configured(self):
        return True

    def embed(self, texts):
        self.embedded.append(list(texts))
        return [[0.5] * self.dimensions for _ in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


def _store(rows=()):
    store = PgVectorKnowledgeStore(dsn="postgresql://stub", embedding_service=StubEmbeddings())
    connection = FakeConnection(rows)
    store._connect = lambda: connection
    return store, connection


# ------------------------------------------------------------------ embedding


def test_embedding_dimensions_match_the_named_model():
    assert EmbeddingService(provider="openai", model="text-embedding-3-small", api_key="k").dimensions == 1536
    assert EmbeddingService(provider="openai", model="text-embedding-3-large", api_key="k").dimensions == 3072
    assert EmbeddingService(provider="voyage", model="voyage-4", api_key="k").dimensions == 1024


def test_voyage_is_the_default_provider(monkeypatch):
    """Anthropic publishes no embeddings model and names Voyage as the provider
    to use instead. The deciding factor is Georgian, though: voyage-4 is trained
    for multilingual retrieval and text-embedding-3-small is not."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("KNOWLEDGE_RETRIEVAL_EMBEDDING_MODEL", raising=False)

    service = EmbeddingService(api_key="k")

    assert service.provider == "voyage"
    assert service.model == "voyage-4"


def test_the_error_names_the_key_the_active_provider_needs(monkeypatch):
    """Telling a Voyage deployment to set OPENAI_API_KEY sends the operator to
    the wrong dashboard."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    with pytest.raises(EmbeddingNotConfigured, match="VOYAGE_API_KEY"):
        EmbeddingService(api_key="").embed(["hello"])

    with pytest.raises(EmbeddingNotConfigured, match="OPENAI_API_KEY"):
        EmbeddingService(provider="openai", api_key="").embed(["hello"])


def test_an_unknown_provider_is_refused_rather_than_silently_defaulted():
    with pytest.raises(EmbeddingNotConfigured, match="Unknown embedding provider"):
        EmbeddingService(provider="cohere", api_key="k")


def test_no_texts_needs_no_credential():
    """An empty pack must not demand an API key to do nothing."""
    assert EmbeddingService(api_key="").embed([]) == []


def test_voyage_is_told_whether_it_is_embedding_a_document_or_a_query(monkeypatch):
    """Voyage prepends a different instruction for each, and its docs are
    explicit that omitting it costs retrieval quality -- a store that embedded
    both sides identically would still work, just worse, which is the kind of
    loss nobody notices."""
    sent = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1] * 1024}]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            sent.update(json)
            return Response()

    monkeypatch.setattr(embedding_module.httpx, "Client", lambda **kwargs: Client())
    service = EmbeddingService(provider="voyage", api_key="k")

    service.embed(["stored material"])
    assert sent["input_type"] == "document"

    service.embed_one("a search")
    assert sent["input_type"] == "query"


def test_openai_is_not_sent_a_parameter_it_does_not_accept(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1] * 1536}]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            sent.update(json)
            return Response()

    monkeypatch.setattr(embedding_module.httpx, "Client", lambda **kwargs: Client())

    EmbeddingService(provider="openai", api_key="k").embed(["x"])

    assert "input_type" not in sent


def test_vectors_are_returned_in_the_order_they_were_asked_for(monkeypatch):
    """Callers zip the result against their own list. A provider that reordered
    would attach every vector to the wrong chunk, silently."""
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            # deliberately out of order
            return {"data": [
                {"index": 2, "embedding": [0.3] * 1536},
                {"index": 0, "embedding": [0.1] * 1536},
                {"index": 1, "embedding": [0.2] * 1536},
            ]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(embedding_module.httpx, "Client", lambda **kwargs: Client())

    vectors = EmbeddingService(provider="openai", api_key="k").embed(["a", "b", "c"])

    assert [round(v[0], 1) for v in vectors] == [0.1, 0.2, 0.3]


def test_a_wrong_width_is_rejected_rather_than_stored(monkeypatch):
    """The column is created at the expected width, so a mismatched vector is
    unstorable and unsearchable -- better named here than as a Postgres error."""
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1] * 8}]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(embedding_module.httpx, "Client", lambda **kwargs: Client())

    with pytest.raises(EmbeddingFailed, match="dimension"):
        EmbeddingService(provider="openai", api_key="k").embed(["a"])


# --------------------------------------------------------------------- schema


def test_the_column_width_follows_the_active_model():
    """Hard-coding 1536 would mean changing the model silently produced a column
    the vectors no longer fit."""
    store = PgVectorKnowledgeStore(
        dsn="postgresql://stub",
        embedding_service=StubEmbeddings(model="text-embedding-3-large", dimensions=3072),
    )
    connection = FakeConnection()
    store._connect = lambda: connection

    store.ensure_schema()

    statements = " ".join(sql for sql, _ in connection.cursor_obj.executed)
    assert "vector(3072)" in statements
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements


# ------------------------------------------------------------------ ingestion


def test_reindexing_replaces_a_packs_rows_rather_than_adding_to_them():
    """Re-uploading after an edit would otherwise leave the superseded text in
    the store, still retrievable and indistinguishable from the correction."""
    store, connection = _store()

    store.index_pack(
        knowledge_pack_id="kp_1",
        chunks=[{"chunk_index": 0, "text": "hooks", "supported_task_types": ["carousel"]}],
    )

    statements = [sql for sql, _ in connection.cursor_obj.executed]
    delete_at = next(i for i, sql in enumerate(statements) if sql.startswith("DELETE"))
    insert_at = next(i for i, sql in enumerate(statements) if sql.startswith("INSERT"))
    assert delete_at < insert_at, "the delete must precede the insert or it removes the new rows"


def test_the_stored_row_records_which_model_embedded_it():
    """Vectors from different models are not comparable, but cosine distance
    between them is still a number -- so the model has to be stored to be
    filtered on."""
    store, connection = _store()

    store.index_pack(knowledge_pack_id="kp_1", chunks=[{"chunk_index": 0, "text": "hooks"}])

    rows = next(params for sql, params in connection.cursor_obj.executed if sql.startswith("INSERT"))
    assert "text-embedding-3-small" in rows[0]


def test_the_chunk_text_key_used_by_the_pack_service_is_the_one_read():
    """KnowledgePackService stores chunk bodies under "text". Reading a
    different key would index a column of empty strings, and empty strings embed
    and search perfectly well."""
    store, connection = _store()

    store.index_pack(knowledge_pack_id="kp_1", chunks=[{"chunk_index": 0, "text": "open with tension"}])

    assert store.embeddings.embedded == [["open with tension"]]


def test_an_empty_pack_makes_no_database_call():
    store, connection = _store()

    assert store.index_pack(knowledge_pack_id="kp_1", chunks=[]) == 0
    assert connection.cursor_obj.executed == []


# ------------------------------------------------------------------ retrieval


def test_the_task_filter_is_part_of_the_query():
    """Filtering after the fact means asking for five chunks, getting five, and
    keeping the two that belong to this task -- which reads as a ranking problem
    and is a query problem."""
    store, connection = _store(rows=[("body", "label", "title", "kp_1", 0.9)])

    store.search(query="hooks", task_type="carousel", top_k=5)

    sql, params = connection.cursor_obj.executed[-1]
    assert "= ANY(supported_task_types)" in sql
    assert "carousel" in params


def test_retrieval_ignores_rows_embedded_by_another_model():
    store, connection = _store(rows=[])

    store.search(query="hooks", task_type="carousel", top_k=5)

    sql, params = connection.cursor_obj.executed[-1]
    assert "embedding_model = %s" in sql
    assert "text-embedding-3-small" in params


def test_results_are_ordered_by_distance_not_by_insertion():
    store, connection = _store(rows=[("body", "label", "title", "kp_1", 0.9)])

    store.search(query="hooks", task_type="carousel", top_k=5)

    sql, _ = connection.cursor_obj.executed[-1]
    assert "ORDER BY embedding <=> %s::vector" in sql


# ------------------------------------------------------- retrieval service


def test_the_service_uses_chroma_unless_told_otherwise(monkeypatch):
    """Existing deployments must not switch stores because a new code path
    exists."""
    monkeypatch.delenv("KNOWLEDGE_VECTOR_STORE", raising=False)

    assert DeterministicKnowledgeRetrievalService()._vector_store() == "chroma"


def test_the_service_switches_to_pgvector_on_request(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_VECTOR_STORE", "pgvector")

    service = DeterministicKnowledgeRetrievalService()

    assert service._vector_store() == "pgvector"
    # The debug record has to name the store actually queried, or anyone
    # diagnosing an empty result is sent to the wrong one.
    assert service._store_label() == f"pgvector:{TABLE_NAME}"
