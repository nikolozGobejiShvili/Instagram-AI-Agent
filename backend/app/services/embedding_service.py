"""Text embeddings, in one place so writer and reader cannot disagree.

The whole point of a shared vector store is that whatever indexes a chunk and
whatever queries for it produce comparable vectors. Two components that each
pick their own model do not fail loudly -- cosine distance is still computable
between unrelated vector spaces, so search simply returns confident nonsense.

So the model and its dimension count live here as a pair, the dimension is what
the ``vector(N)`` column is created with, and every stored row records the model
that produced it. A mismatch then becomes a row that is filtered out rather than
a result that is quietly wrong.

Anthropic has no embeddings API, which is why this reaches for OpenAI while the
rest of the agent runs on Claude.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Voyage is the default because Anthropic does not publish an embeddings model
# and names Voyage as the provider to use instead ("Anthropic does not offer its
# own embedding model" — platform.claude.com/docs/en/docs/build-with-claude/embeddings).
#
# The deciding factor here is not the recommendation, though: this agent's
# material and queries are largely Georgian, and voyage-4 is trained for
# multilingual retrieval where text-embedding-3-small is not. Retrieval quality
# in Georgian is the whole point of uploading the material.
DEFAULT_PROVIDER = "voyage"
DEFAULT_MODEL_BY_PROVIDER = {
    "voyage": "voyage-4",
    "openai": "text-embedding-3-small",
}

# Native output width of each supported model. The active model's value is used
# to create the vector column, so this table is schema, not trivia.
MODEL_DIMENSIONS = {
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-4-lite": 1024,
    "voyage-3.5": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

PROVIDER_ENDPOINTS = {
    "voyage": ("https://api.voyageai.com/v1", "VOYAGE_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


class EmbeddingNotConfigured(RuntimeError):
    """No embedding credential. Retrying will not help until an operator acts."""


class EmbeddingFailed(RuntimeError):
    """The embedding provider was reached but did not return usable vectors."""


class EmbeddingService:
    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.provider = (
            provider or os.getenv("EMBEDDING_PROVIDER", "").strip().lower() or DEFAULT_PROVIDER
        )
        if self.provider not in PROVIDER_ENDPOINTS:
            raise EmbeddingNotConfigured(
                f"Unknown embedding provider '{self.provider}'. Supported: {', '.join(PROVIDER_ENDPOINTS)}"
            )

        default_url, key_name = PROVIDER_ENDPOINTS[self.provider]
        self.model = (
            model
            or os.getenv("KNOWLEDGE_RETRIEVAL_EMBEDDING_MODEL", "").strip()
            or DEFAULT_MODEL_BY_PROVIDER[self.provider]
        )
        self._key_name = key_name
        self._api_key = api_key if api_key is not None else os.getenv(key_name, "").strip()
        self.base_url = (os.getenv("EMBEDDING_BASE_URL", "").strip() or default_url).rstrip("/")
        self.timeout = float(timeout or os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30") or 30)

    @property
    def dimensions(self) -> int:
        """Width of this model's vectors.

        Unknown models fall back to the default width rather than raising: the
        column is created from this number, so guessing wrong surfaces
        immediately as a dimension error from Postgres instead of silently
        storing a truncated vector.
        """
        return MODEL_DIMENSIONS.get(self.model, MODEL_DIMENSIONS[DEFAULT_MODEL_BY_PROVIDER[self.provider]])

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        """Embed ``texts`` in order, one request.

        Order is the contract: callers zip the result back against their own
        list, so a provider that reordered would silently attach every vector to
        the wrong chunk. The response is sorted by index to guarantee it.

        ``input_type`` matters on Voyage and is ignored by OpenAI. Voyage
        prepends a different instruction for a stored document than for a search
        query, and its own docs are explicit that omitting it costs retrieval
        quality -- so a store that embedded both sides identically would work,
        just worse, which is the kind of loss nobody notices.
        """
        if not texts:
            return []
        if not self._api_key:
            raise EmbeddingNotConfigured(
                f"{self._key_name} is not set. Embeddings are required to index and search knowledge."
            )

        body: dict = {"model": self.model, "input": texts}
        if self.provider == "voyage":
            body["input_type"] = input_type

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingFailed(f"Embedding request failed: {type(exc).__name__}") from exc

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingFailed(
                f"Embedding provider returned {len(data) if isinstance(data, list) else 'no'} "
                f"vectors for {len(texts)} inputs"
            )

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [list(item["embedding"]) for item in ordered]

        width = len(vectors[0])
        if width != self.dimensions:
            # Loud, because the column was created at self.dimensions: storing
            # these would fail anyway, and searching with them would compare
            # incomparable spaces.
            raise EmbeddingFailed(
                f"Model {self.model} returned {width}-dimension vectors, expected {self.dimensions}"
            )
        return vectors

    def embed_one(self, text: str, *, input_type: str = "query") -> list[float]:
        """Defaults to `query`: the single-text call is the search path."""
        return self.embed([text], input_type=input_type)[0]
