"""Taking money, and knowing whose account it bought.

These two arrived together because they only work together: granting a plan is
meaningless if the next request can claim to be a different customer, and a
verified customer is pointless if nobody can pay to become one.

Every case here is a way to get paid-for access without paying, or to reach
another customer's account — so each one is written as the attack rather than as
the happy path.
"""
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import generation_jobs as jobs_route  # noqa: E402
from app.api.routes import payments as payments_route  # noqa: E402
from app.api.user_auth import USER_TOKEN_HEADER  # noqa: E402
from app.services.billing_service import BillingService  # noqa: E402
from app.services.job_service import JobService, JobWorker  # noqa: E402
from app.services.payment_service import PaymentService  # noqa: E402
from app.services.user_token_service import UserTokenError, UserTokenService  # noqa: E402

SECRET = "test-user-token-secret"
WEBHOOK_SECRET = "whsec_test_secret"


# ------------------------------------------------------------------- tokens


@pytest.fixture()
def tokens():
    return UserTokenService(secret=SECRET, ttl_seconds=3600)


def test_a_token_round_trips_to_the_user_it_was_issued_for(tokens):
    issued = tokens.issue("user-1")

    assert tokens.verify(issued["user_token"]) == "user-1"
    assert issued["expires_in"] == 3600


def test_a_token_signed_with_another_secret_is_refused(tokens):
    forged = UserTokenService(secret="not-the-secret").issue("user-1")["user_token"]

    with pytest.raises(UserTokenError):
        tokens.verify(forged)


def test_editing_the_user_inside_a_token_invalidates_it(tokens):
    """The payload is readable by anyone holding the token; only the signature
    stops it being rewritten."""
    import base64

    body, signature = tokens.issue("user-1")["user_token"].split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=="))
    payload["sub"] = "user-2"
    tampered = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    with pytest.raises(UserTokenError):
        tokens.verify(f"{tampered}.{signature}")


def test_an_expired_token_is_refused(tokens):
    issued = tokens.issue("user-1", now=time.time() - 7200)

    with pytest.raises(UserTokenError):
        tokens.verify(issued["user_token"])


def test_every_rejection_reads_the_same(tokens):
    """Distinguishing "bad signature" from "expired" from "malformed" tells an
    attacker which part of a forgery to fix next."""
    messages = set()
    for bad in ["", "nonsense", "a.b", tokens.issue("u", now=time.time() - 9999)["user_token"]]:
        try:
            tokens.verify(bad)
        except UserTokenError as exc:
            messages.add(str(exc))

    assert messages == {"Invalid user token"}


def test_the_ttl_is_capped_however_it_is_configured():
    """A year-long token found in a log is a permanent credential."""
    assert UserTokenService(secret=SECRET, ttl_seconds=10**9).ttl_seconds <= 24 * 3600


# --------------------------------------------------------- identity on routes


@pytest.fixture()
def client(monkeypatch, tmp_path):
    job_service = JobService(db_path=tmp_path / "jobs.sqlite3")
    billing_service = BillingService(
        db_path=tmp_path / "billing.sqlite3", legacy_json_file=tmp_path / "absent.json"
    )
    monkeypatch.setattr(jobs_route, "job_service", job_service)
    monkeypatch.setattr(jobs_route, "billing_service", billing_service)
    monkeypatch.setattr(
        jobs_route.llm_service,
        "run_agent",
        lambda **kwargs: {"reply": "ok", "model_provider": "stub", "used_langflow": False},
    )
    monkeypatch.setenv("USER_TOKEN_SECRET", SECRET)

    test_client = TestClient(app)
    test_client.job_service = job_service
    test_client.billing_service = billing_service
    test_client.worker = JobWorker(job_service, handlers=dict(jobs_route.job_worker.handlers))
    return test_client


def _auth(user_id):
    return {USER_TOKEN_HEADER: UserTokenService(secret=SECRET).issue(user_id)["user_token"]}


def _job_body(user_id="user-1"):
    return {"user_id": user_id, "task_type": "content_plan", "message": "გეგმა"}


def test_spending_credits_without_a_token_is_refused(client):
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})

    response = client.post("/api/v1/generation-jobs", json=_job_body())

    assert response.status_code == 401


def test_a_token_for_another_customer_cannot_spend_this_ones_credits(client):
    """The failure this exists for: a website bug sending a stale user_id would
    otherwise charge the wrong account and return its data."""
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})
    client.billing_service.set_plan("user-2", {"current_plan": "pro"})

    response = client.post("/api/v1/generation-jobs", json=_job_body("user-2"), headers=_auth("user-1"))

    assert response.status_code == 403


def test_the_token_decides_the_user_that_is_charged(client):
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})

    response = client.post("/api/v1/generation-jobs", json=_job_body("user-1"), headers=_auth("user-1"))

    assert response.status_code == 202
    assert response.json()["user_id"] == "user-1"


def test_another_customers_job_reads_as_missing_not_forbidden(client):
    """403 would confirm the id exists, turning a failed guess into a result."""
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})
    job_id = client.post(
        "/api/v1/generation-jobs", json=_job_body("user-1"), headers=_auth("user-1")
    ).json()["job_id"]

    response = client.get(f"/api/v1/generation-jobs/{job_id}", headers=_auth("user-2"))

    assert response.status_code == 404


def test_the_list_cannot_be_pointed_at_another_customer(client):
    """Refused rather than quietly answered for the token's own user: a website
    asking for the wrong customer's jobs is broken, and returning a valid-looking
    empty list would hide that."""
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})
    client.post("/api/v1/generation-jobs", json=_job_body("user-1"), headers=_auth("user-1"))

    response = client.get("/api/v1/generation-jobs?user_id=user-1", headers=_auth("user-2"))

    assert response.status_code == 403


def test_the_list_needs_no_user_id_at_all(client):
    """The token already says who is asking, so the parameter is optional and
    the token is what selects the rows."""
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})
    client.post("/api/v1/generation-jobs", json=_job_body("user-1"), headers=_auth("user-1"))

    mine = client.get("/api/v1/generation-jobs", headers=_auth("user-1")).json()
    theirs = client.get("/api/v1/generation-jobs", headers=_auth("user-2")).json()

    assert len(mine["jobs"]) == 1
    assert theirs["jobs"] == []


def test_without_a_secret_the_service_behaves_exactly_as_before(client, monkeypatch):
    """Enforcement is opt-in so the deployed service keeps working while the
    website that will issue tokens does not exist yet."""
    monkeypatch.delenv("USER_TOKEN_SECRET", raising=False)
    client.billing_service.set_plan("user-1", {"current_plan": "pro"})

    response = client.post("/api/v1/generation-jobs", json=_job_body("user-1"))

    assert response.status_code == 202


# ------------------------------------------------------------------ payments


@pytest.fixture()
def payments(monkeypatch, tmp_path):
    service = PaymentService(db_path=tmp_path / "payments.sqlite3", webhook_secret=WEBHOOK_SECRET)
    monkeypatch.setattr(payments_route, "payment_service", service)
    monkeypatch.setenv("STRIPE_PRICE_MAP", "price_creator=creator,price_pro=pro")
    return service


def _signed(payload: dict, *, secret=WEBHOOK_SECRET, timestamp=None):
    body = json.dumps(payload).encode()
    ts = int(timestamp if timestamp is not None else time.time())
    signature = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, {"Stripe-Signature": f"t={ts},v1={signature}"}


def _event(event_id="evt_1", event_type="checkout.session.completed", user_id="user-1", price="price_pro"):
    return {
        "id": event_id,
        "type": event_type,
        "data": {
            "object": {
                "metadata": {"user_id": user_id},
                "items": {"data": [{"price": {"id": price}}]},
            }
        },
    }


def test_an_unsigned_webhook_cannot_grant_a_plan(payments):
    """The URL is guessable and carries no site key, so without the signature
    anyone could POST themselves the agency plan."""
    response = TestClient(app).post("/api/v1/payments/stripe/webhook", json=_event())

    assert response.status_code == 400


def test_a_webhook_signed_with_the_wrong_secret_is_refused(payments):
    body, headers = _signed(_event(), secret="whsec_attacker")

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.status_code == 400


def test_an_old_signature_cannot_be_replayed(payments):
    """The timestamp is inside the signature precisely so a captured request
    stops working."""
    body, headers = _signed(_event(), timestamp=time.time() - 3600)

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.status_code == 400


def test_a_valid_payment_grants_the_plan(payments, monkeypatch, tmp_path):
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(payments_route, "billing_service", billing)
    body, headers = _signed(_event())

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.json()["applied"] is True
    assert billing.get_plan("user-1")["current_plan"] == "pro"


def test_a_redelivered_event_is_applied_once(payments, monkeypatch, tmp_path):
    """Stripe redelivers by design; applying twice would extend a subscription
    nobody paid twice for."""
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(payments_route, "billing_service", billing)
    body, headers = _signed(_event())
    client = TestClient(app)

    first = client.post("/api/v1/payments/stripe/webhook", content=body, headers=headers)
    second = client.post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert first.json()["applied"] is True
    assert second.json()["applied"] is False
    assert second.json()["reason"] == "duplicate"


def test_cancellation_ends_the_plan(payments, monkeypatch, tmp_path):
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(payments_route, "billing_service", billing)
    billing.set_plan("user-1", {"current_plan": "pro"})
    body, headers = _signed(_event(event_id="evt_2", event_type="customer.subscription.deleted"))

    TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    plan = billing.get_plan("user-1")
    assert plan["current_plan"] == "trial"
    assert plan["plan_status"] == "cancelled"


def test_an_unmapped_price_grants_nothing(payments, monkeypatch, tmp_path):
    """Guessing a tier from an unrecognised price is how someone pays for
    creator and receives agency."""
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(payments_route, "billing_service", billing)
    body, headers = _signed(_event(price="price_unknown"))

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.json()["reason"] == "unmapped_price"
    assert billing.get_plan("user-1")["current_plan"] == "trial"


def test_an_event_without_our_user_id_grants_nothing(payments):
    event = _event(event_id="evt_3")
    event["data"]["object"]["metadata"] = {}
    body, headers = _signed(event)

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.json()["reason"] == "missing_user_id"


def test_unrelated_stripe_events_change_nothing(payments):
    body, headers = _signed(_event(event_id="evt_4", event_type="customer.source.updated"))

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.json()["reason"] == "ignored_event_type"


def test_webhooks_are_refused_when_no_signing_secret_is_configured(monkeypatch, tmp_path):
    """Fails closed: an unverified webhook endpoint that grants plans is an open
    one."""
    service = PaymentService(db_path=tmp_path / "p.sqlite3", webhook_secret="")
    monkeypatch.setattr(payments_route, "payment_service", service)
    body, headers = _signed(_event())

    response = TestClient(app).post("/api/v1/payments/stripe/webhook", content=body, headers=headers)

    assert response.status_code == 503
