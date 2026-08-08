"""Retrieval has to be able to return nothing.

The single-stage store could not: `ORDER BY distance LIMIT k` always yields k
rows, so a carousel question retrieved the five least-unrelated reels passages
and put them in the prompt. Every one of those competes with the instructions
for attention, which is how a rule ends up skipped — so the defect is not "poor
ranking", it is prompt pollution that scales with the size of the knowledge base.

These cases pin the two-stage behaviour: recall wide by vector, rerank, keep few,
and keep none when none are relevant.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import reranker_service as rerank_module  # noqa: E402
from app.services.deterministic_knowledge_retrieval_service import (  # noqa: E402
    DeterministicKnowledgeRetrievalService,
)
from app.services.pgvector_knowledge_store import PgVectorKnowledgeStore  # noqa: E402
from app.services.reranker_service import RankedPassage, RerankerService  # noqa: E402


class StubEmbeddings:
    model = "voyage-4"
    dimensions = 1024

    def is_configured(self):
        return True

    def embed(self, texts, input_type="document"):
        return [[0.5] * self.dimensions for _ in texts]

    def embed_one(self, text, input_type="query"):
        return [0.5] * self.dimensions


class StubReranker:
    """Returns whatever the test says survived the floor."""

    def __init__(self, result):
        self.result = result
        self.seen = {}

    def rank(self, *, query, documents, top_k):
        self.seen = {"query": query, "documents": documents, "top_k": top_k}
        return self.result


class FakeCursor:
    def __init__(self, rows):
        self.executed = []
        self._rows = rows

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rows(n):
    return [(f"passage {i}", f"label {i}", "pack", "kp_1", 0.9 - i * 0.01) for i in range(n)]


def _store(rows, reranker):
    store = PgVectorKnowledgeStore(
        dsn="postgresql://stub", embedding_service=StubEmbeddings(), reranker=reranker
    )
    connection = FakeConnection(rows)
    store._connect = lambda: connection
    return store, connection


# ------------------------------------------------------------- two stages


def test_the_vector_stage_recalls_more_than_it_returns():
    """Recall is what embeddings are good at; precision is the reranker's job.
    Fetching only the final count would leave the reranker nothing to choose
    from."""
    store, connection = _store(_rows(30), StubReranker([RankedPassage(0, 0.9)]))

    store.search(query="hooks", task_type="carousel", top_k=3)

    _, params = connection.cursor_obj.executed[-1]
    assert params[-1] >= 30, "the candidate pool must be wider than top_k"


def test_only_the_reranked_passages_survive():
    store, _ = _store(_rows(10), StubReranker([RankedPassage(4, 0.81), RankedPassage(1, 0.55)]))

    results = store.search(query="hooks", task_type="carousel", top_k=3)

    assert [r.content for r in results] == ["passage 4", "passage 1"]


def test_the_reported_score_is_the_one_that_decided_inclusion():
    """Returning the cosine score would make a debugging session read the wrong
    number when asking why a passage was kept."""
    store, _ = _store(_rows(10), StubReranker([RankedPassage(4, 0.81)]))

    assert store.search(query="hooks", task_type="carousel", top_k=3)[0].score == pytest.approx(0.81)


def test_nothing_relevant_means_nothing_injected():
    """The behaviour the old store could not produce, and the reason this exists:
    material that does not cover the question must not reach the prompt."""
    store, _ = _store(_rows(10), StubReranker([]))

    assert store.search(query="pricing in Georgia", task_type="carousel", top_k=3) == []


def test_an_unavailable_reranker_falls_back_instead_of_erasing_the_results():
    """`None` means the stage was skipped, not that nothing was relevant.
    Treating it as empty would silently drop the whole knowledge base whenever
    the ranking call failed."""
    store, _ = _store(_rows(10), StubReranker(None))

    results = store.search(query="hooks", task_type="carousel", top_k=3)

    assert [r.content for r in results] == ["passage 0", "passage 1", "passage 2"]


def test_an_empty_store_never_calls_the_reranker():
    reranker = StubReranker([])
    store, _ = _store([], reranker)

    assert store.search(query="hooks", task_type="carousel", top_k=3) == []
    assert reranker.seen == {}


# ------------------------------------------------------------- the floor


def test_passages_below_the_floor_are_dropped(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [
                {"index": 0, "relevance_score": 0.82},
                {"index": 3, "relevance_score": 0.11},
            ]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return Response()

    monkeypatch.setattr(rerank_module.httpx, "Client", lambda **kw: Client())

    kept = RerankerService(api_key="k").rank(query="q", documents=["a", "b", "c", "d"], top_k=3)

    assert [p.index for p in kept] == [0]


def test_no_key_means_the_stage_is_skipped_not_that_nothing_matched():
    assert RerankerService(api_key="").rank(query="q", documents=["a"], top_k=3) is None


def test_a_ranker_failure_degrades_rather_than_losing_the_generation(monkeypatch):
    """A generation the customer paid for must not be lost to a ranking call."""
    import httpx

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(rerank_module.httpx, "Client", lambda **kw: Client())

    assert RerankerService(api_key="k").rank(query="q", documents=["a"], top_k=3) is None


def test_the_floor_is_configurable(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MIN_RELEVANCE", "0.7")

    assert RerankerService(api_key="k").min_relevance == 0.7


# --------------------------------------------------------- prompt budget


def test_fewer_passages_reach_the_prompt_than_before():
    """Five was a hedge against bad ranking. With a reranker the extra two are
    not insurance, they are two more paragraphs competing with the
    instructions."""
    assert DeterministicKnowledgeRetrievalService.DEFAULT_TOP_K == 3


def test_the_candidate_pool_is_wider_than_what_is_kept():
    store = PgVectorKnowledgeStore(dsn="postgresql://stub", embedding_service=StubEmbeddings())

    assert store.candidate_pool > DeterministicKnowledgeRetrievalService.DEFAULT_TOP_K
