"""The site key, tested against a client that does not send it by default.

``conftest.py`` gives every other test an authenticated ``TestClient`` so the
existing suite did not have to grow a header on several hundred calls. That
default would make this file vacuous, so everything here builds its client
through ``_unauthenticated_testclient_init`` -- the real, unpatched constructor.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient  # noqa: E402

from app.api.site_auth import SITE_KEY_HEADER, is_guarded  # noqa: E402
from app.main import app  # noqa: E402
from conftest import _TEST_SITE_KEY, _unauthenticated_testclient_init  # noqa: E402

# A read that needs no request body, so a rejection cannot be confused with a
# validation failure.
GUARDED_PATH = "/api/v1/billing/plans"


def _raw_client(**kwargs):
    """A client with no site key unless this test supplies one."""
    client = TestClient.__new__(TestClient)
    _unauthenticated_testclient_init(client, app, **kwargs)
    return client


def test_a_request_without_the_key_is_refused():
    response = _raw_client().get(GUARDED_PATH)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "site_auth_required"


def test_a_wrong_key_is_refused():
    response = _raw_client(headers={SITE_KEY_HEADER: "not-the-key"}).get(GUARDED_PATH)

    assert response.status_code == 401


def test_a_key_that_is_merely_a_prefix_is_refused():
    """Guards against a comparison that stops at the shorter string."""
    response = _raw_client(headers={SITE_KEY_HEADER: _TEST_SITE_KEY[:-1]}).get(GUARDED_PATH)

    assert response.status_code == 401


def test_the_right_key_is_accepted():
    response = _raw_client(headers={SITE_KEY_HEADER: _TEST_SITE_KEY}).get(GUARDED_PATH)

    assert response.status_code == 200


def test_an_unconfigured_key_denies_everything(monkeypatch):
    """Fails closed. A missing configuration must be a loud outage, not a
    silently open API on a public domain."""
    monkeypatch.setenv("SITE_API_KEY", "")

    response = _raw_client(headers={SITE_KEY_HEADER: _TEST_SITE_KEY}).get(GUARDED_PATH)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "site_auth_not_configured"


def test_the_refusal_does_not_echo_the_expected_key():
    body = _raw_client().get(GUARDED_PATH).text

    assert _TEST_SITE_KEY not in body


# --------------------------------------------------------------- what is open


def test_health_stays_open_for_the_platform_probe():
    """Railway restarts a container whose healthcheck fails, so guarding this
    would take the service down permanently the moment the key was set."""
    assert _raw_client().get("/health").status_code == 200


def test_carousel_images_stay_open():
    """They load in <img src="...">, which cannot send a custom header. The
    128-bit opaque asset id is the credential instead."""
    response = _raw_client().get("/api/v1/carousel-media/" + "a" * 32)

    # 404 because this id was never minted -- the point is that it reached the
    # route at all rather than being refused by the guard.
    assert response.status_code == 404


# ------------------------------------------------------------ path coverage


@pytest.mark.parametrize("path", ["/health", "/", "/docs", "/openapi.json"])
def test_non_api_paths_are_not_guarded(path):
    assert is_guarded(path) is False


def test_every_mounted_api_route_is_guarded():
    """The reason this is middleware and not a per-router dependency.

    A dependency has to be remembered on each new router, and forgetting it
    ships that router open while everything still appears to work. This asserts
    the property directly against the app's own route table, so a router added
    later cannot quietly escape the guard.
    """
    unguarded = [
        route.path
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/v1") and not is_guarded(route.path)
    ]

    assert unguarded == ["/api/v1/carousel-media/{asset_id}"], (
        f"unexpectedly unguarded API routes: {unguarded}"
    )


def test_admin_routes_require_their_own_key_on_top_of_the_site_key(monkeypatch):
    """A leaked site key must not be enough to grant yourself a plan."""
    monkeypatch.setenv("INTERNAL_ADMIN_KEY", "admin-key")
    client = _raw_client(headers={SITE_KEY_HEADER: _TEST_SITE_KEY})

    response = client.post("/api/v1/billing/plan/someone/set", json={"current_plan": "agency"})

    assert response.status_code == 403
