"""Endpoints that grant entitlements or reset usage must not be open to callers.

Audit of 2026-08-04, finding 2: `POST /api/v1/billing/plan/{user_id}/set` had no
authentication at all, so any caller who could reach the API could put itself on
the top plan. The maintenance jobs were equally open, and one of them resets
every user's usage counter.

These tests pin the mutating surface shut. Reads are deliberately not covered
here -- reading a plan is not privilege escalation, and who may read is part of
the still-open caller-authentication decision.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402

ADMIN_KEY = "test-internal-admin-key"
ADMIN_HEADERS = {"X-Internal-Admin-Key": ADMIN_KEY}

PRIVILEGED_ENDPOINTS = [
    ("post", "/api/v1/billing/plan/priv-user/set", {"current_plan": "agency"}),
    ("post", "/api/v1/billing/usage/priv-user/reset-month", None),
    ("post", "/api/v1/maintenance/run/monthly-usage-reset", None),
    ("post", "/api/v1/maintenance/run/context-freshness-scan", None),
    ("post", "/api/v1/maintenance/run/context-refresh", None),
    ("post", "/api/v1/maintenance/run/connection-health-scan", None),
]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("INTERNAL_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("BILLING_DB_PATH", str(tmp_path / "billing.sqlite3"))
    return TestClient(app)


def _call(client, method, path, body, headers=None):
    kwargs = {"headers": headers} if headers else {}
    if body is not None:
        kwargs["json"] = body
    return getattr(client, method)(path, **kwargs)


@pytest.mark.parametrize("method,path,body", PRIVILEGED_ENDPOINTS)
def test_rejected_without_admin_key(client, method, path, body):
    response = _call(client, method, path, body)
    assert response.status_code == 403, (
        f"{method.upper()} {path} answered {response.status_code} to an unauthenticated caller"
    )


@pytest.mark.parametrize("method,path,body", PRIVILEGED_ENDPOINTS)
def test_rejected_with_wrong_admin_key(client, method, path, body):
    response = _call(client, method, path, body, headers={"X-Internal-Admin-Key": "wrong"})
    assert response.status_code == 403


def test_self_upgrade_is_closed(client):
    """The specific hole from the audit: granting yourself the top plan."""
    before = client.get("/api/v1/billing/plan/priv-user").json()["current_plan"]

    attack = client.post("/api/v1/billing/plan/priv-user/set", json={"current_plan": "agency"})
    assert attack.status_code == 403

    after = client.get("/api/v1/billing/plan/priv-user").json()["current_plan"]
    assert after == before, "an unauthenticated caller changed the plan"
    assert after != "agency"


def test_admin_key_still_allows_the_operation(client):
    """The fix must not break legitimate administration."""
    response = client.post(
        "/api/v1/billing/plan/priv-admin-user/set",
        json={"current_plan": "agency"},
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["current_plan"] == "agency"


def test_fails_closed_when_no_admin_key_is_configured(monkeypatch, tmp_path):
    """With INTERNAL_ADMIN_KEY unset the endpoint must deny, never default open."""
    monkeypatch.delenv("INTERNAL_ADMIN_KEY", raising=False)
    monkeypatch.setenv("BILLING_DB_PATH", str(tmp_path / "billing.sqlite3"))
    unconfigured_client = TestClient(app)

    response = unconfigured_client.post(
        "/api/v1/billing/plan/priv-user/set",
        json={"current_plan": "agency"},
        headers={"X-Internal-Admin-Key": "anything"},
    )
    assert response.status_code == 403
