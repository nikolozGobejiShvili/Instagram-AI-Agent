"""Test isolation for the backend suite.

The suite previously had no conftest at all, which cost it two properties a test
suite has to have:

1. **Configuration is injected, never inherited.** Every service module calls
   ``load_dotenv()`` at import time (``app/main.py``, ``app/services/langflow_service.py``,
   ``app/services/deterministic_knowledge_retrieval_service.py``), so the test
   process used to pick up ``backend/.env`` -- real Langflow flow ids, provider
   flags and whatever API keys happen to sit in the developer's shell. The suite
   therefore behaved differently on every machine.

2. **No test touches the network.** ``KnowledgePackService`` uploads trigger
   ``LangflowService.ingest_system_reels_knowledge``, which is not stubbed by any
   fixture. With a real ingestion flow id inherited from ``.env`` it opened a
   socket to a local Langflow server; when that server was not running the test
   failed with ``httpx.ConnectError`` -- surfaced to the caller as a misleading
   502 "AI generation failed. Please try again shortly." Nine tests failed this
   way for purely environmental reasons.

Both guarantees are established here, before any test module is imported.
Tests that legitimately need a live dependency can opt out with
``@pytest.mark.allow_network``.
"""
import os
import pathlib
import sys
import tempfile

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Stop backend/.env from leaking into the test process.
# ---------------------------------------------------------------------------
# This must happen before any application module is imported, because those
# modules bind ``load_dotenv`` at their own import time. Patching the attribute
# on the ``dotenv`` package means their ``from dotenv import load_dotenv`` picks
# up the no-op.
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False
if hasattr(dotenv, "main"):
    dotenv.main.load_dotenv = dotenv.load_dotenv


# Ambient shell variables would defeat the point of the patch above, so clear the
# ones that steer runtime behaviour. Tests that need a value set it explicitly
# via ``monkeypatch.setenv``; everything else falls back to the code's own
# documented defaults, which is what makes runs reproducible.
_PURGED_PREFIXES = ("LANGFLOW_", "META_", "PRIMARY_LLM_", "FALLBACK_LLM_", "USE_LANGFLOW_")
_PURGED_NAMES = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "INTERNAL_ADMIN_KEY",
    "KNOWLEDGE_RETRIEVAL_TIMEOUT_SECONDS",
}

for _name in [k for k in os.environ if k.startswith(_PURGED_PREFIXES) or k in _PURGED_NAMES]:
    os.environ.pop(_name, None)


# A deterministic baseline configuration, replacing what used to leak in from
# ``.env``. Only values the suite actually needs are set, and they are obviously
# fake so a real call made with them cannot reach anything.
#
# Flow ids are deliberately left UNSET: ``LangflowService.ingest_system_reels_knowledge``
# short-circuits when there is no ingestion flow id, which is what keeps a
# knowledge-pack upload from reaching for a live Langflow server. Tests that
# exercise a flow set the id they need themselves.
_TEST_STATE_DIR = pathlib.Path(tempfile.mkdtemp(prefix="ig_agent_test_state_"))

_TEST_DEFAULTS = {
    "LANGFLOW_API_KEY": "test-langflow-key",
    "LANGFLOW_BASE_URL": "http://langflow.invalid",
    # Services are constructed at module import time, so these would otherwise
    # open (and migrate into) the real backend/app/data store the moment a test
    # imports the app. Point them at a throwaway directory instead: a test run
    # must not create or mutate production data.
    "BILLING_DB_PATH": str(_TEST_STATE_DIR / "billing.sqlite3"),
    "JOBS_DB_PATH": str(_TEST_STATE_DIR / "jobs.sqlite3"),
    "CAROUSEL_MEDIA_DIR": str(_TEST_STATE_DIR / "carousel_media"),
    # The background worker is started by the app lifespan. Tests step the
    # worker by hand so assertions cannot race a thread.
    "JOB_WORKER_ENABLED": "false",
}
for _key, _value in _TEST_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


# ---------------------------------------------------------------------------
# 2. Fail loudly on any outbound HTTP request.
# ---------------------------------------------------------------------------
# The guard sits on httpx's *real* transport rather than on ``socket``. Blocking
# sockets outright also breaks asyncio's internal self-pipe on Windows
# (ProactorEventLoop), and every genuine outbound call in this codebase goes
# through httpx.
#
# TestClient is unaffected by construction: it drives the app in-process through
# ``starlette.testclient._TestClientTransport``, which is not an
# ``httpx.HTTPTransport`` subclass, so patching that class cannot intercept it.
# Only a call that would really leave the machine trips this guard.
import httpx  # noqa: E402

_real_handle_request = httpx.HTTPTransport.handle_request
_real_handle_async_request = httpx.AsyncHTTPTransport.handle_async_request


class NetworkAccessAttempted(RuntimeError):
    """Raised when a test tries to reach a real network service."""


def _message(request) -> str:
    return (
        f"Outbound network access attempted during a test ({request.method} {request.url}). "
        "Stub the service instead, or mark the test with "
        "@pytest.mark.allow_network if it genuinely needs a live dependency."
    )


def _blocked_handle_request(self, request):
    raise NetworkAccessAttempted(_message(request))


async def _blocked_handle_async_request(self, request):
    raise NetworkAccessAttempted(_message(request))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: permit real outbound connections for this test",
    )


@pytest.fixture(autouse=True)
def _no_network(request):
    """Block real outbound HTTP unless the test opts out."""
    if request.node.get_closest_marker("allow_network"):
        yield
        return

    httpx.HTTPTransport.handle_request = _blocked_handle_request
    httpx.AsyncHTTPTransport.handle_async_request = _blocked_handle_async_request
    try:
        yield
    finally:
        httpx.HTTPTransport.handle_request = _real_handle_request
        httpx.AsyncHTTPTransport.handle_async_request = _real_handle_async_request
