"""Starting a payment, and re-teaching the agent material it already holds.

Both close a one-way street. The webhook could only react to money that had
already moved, so the website had to build the checkout itself — and the one
field it must not omit is the one the webhook needs. Indexing only ran on
upload, so a pack stored before the embedding key existed was invisible to
retrieval with no way back except deleting and re-uploading the files.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import payments as payments_route  # noqa: E402
from app.api.user_auth import USER_TOKEN_HEADER  # noqa: E402
from app.services.payment_service import PaymentError, PaymentService  # noqa: E402
from app.services.user_token_service import UserTokenService  # noqa: E402

SECRET = "test-user-token-secret"


@pytest.fixture()
def service(monkeypatch, tmp_path):
    monkeypatch.setenv("STRIPE_PRICE_MAP", "price_creator=creator,price_pro=pro,price_agency=agency")
    return PaymentService(db_path=tmp_path / "p.sqlite3", api_key="sk_test_x")


def _stub_stripe(monkeypatch, captured, *, url="https://checkout.stripe.com/c/pay/cs_test_1"):
    import httpx

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "cs_test_1", "url": url}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, endpoint, data=None, headers=None):
            captured["endpoint"] = endpoint
            captured["data"] = data
            captured["headers"] = headers
            return Response()

    monkeypatch.setattr(httpx, "Client", lambda **kw: Client())


# --------------------------------------------------------------- price map


def test_the_plan_to_price_lookup_is_the_reverse_of_the_webhook_lookup(service):
    """Both directions read the same variable, so they cannot disagree about
    which price means which tier."""
    assert service.price_for_plan("pro") == "price_pro"
    assert PaymentService.plan_for_price("price_pro") == "pro"


def test_an_unmapped_plan_is_refused_rather_than_guessed(service):
    """Guessing a price is how a customer pays for a tier they did not pick."""
    with pytest.raises(PaymentError):
        service.create_checkout_session(
            user_id="u", plan="enterprise", success_url="https://s", cancel_url="https://c"
        )


def test_no_api_key_refuses_instead_of_calling_stripe(monkeypatch, tmp_path):
    svc = PaymentService(db_path=tmp_path / "p.sqlite3", api_key="")

    with pytest.raises(PaymentError):
        svc.create_checkout_session(user_id="u", plan="pro", success_url="https://s", cancel_url="https://c")


# ---------------------------------------------------------------- metadata


def test_the_customer_id_is_attached_to_the_session(monkeypatch, service):
    """A session created without it pays real money and grants nothing — the
    webhook can only log it after the fact."""
    captured = {}
    _stub_stripe(monkeypatch, captured)

    service.create_checkout_session(
        user_id="user-1", plan="pro", success_url="https://s", cancel_url="https://c"
    )

    assert captured["data"]["metadata[user_id]"] == "user-1"


def test_the_customer_id_is_also_attached_to_the_subscription(monkeypatch, service):
    """Checkout events carry the session metadata; renewal and cancellation are
    emitted against the subscription and carry only its own. Setting one means
    the first payment works and every event after it is unattributable."""
    captured = {}
    _stub_stripe(monkeypatch, captured)

    service.create_checkout_session(
        user_id="user-1", plan="pro", success_url="https://s", cancel_url="https://c"
    )

    assert captured["data"]["subscription_data[metadata][user_id]"] == "user-1"


def test_the_session_is_a_subscription_not_a_one_off_payment(monkeypatch, service):
    captured = {}
    _stub_stripe(monkeypatch, captured)

    service.create_checkout_session(
        user_id="u", plan="pro", success_url="https://s", cancel_url="https://c"
    )

    assert captured["data"]["mode"] == "subscription"
    assert captured["data"]["line_items[0][price]"] == "price_pro"


def test_a_stripe_failure_is_reported_without_leaking_the_key(monkeypatch, service):
    import httpx

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", lambda **kw: Client())

    with pytest.raises(PaymentError) as exc:
        service.create_checkout_session(
            user_id="u", plan="pro", success_url="https://s", cancel_url="https://c"
        )

    assert "sk_test" not in exc.value.message


# ------------------------------------------------------------------ route


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("STRIPE_PRICE_MAP", "price_pro=pro")
    monkeypatch.setattr(
        payments_route, "payment_service", PaymentService(db_path=tmp_path / "p.sqlite3", api_key="sk_test_x")
    )
    return TestClient(app)


def _auth(user_id):
    return {USER_TOKEN_HEADER: UserTokenService(secret=SECRET).issue(user_id)["user_token"]}


def _body(**over):
    body = {"user_id": "user-1", "plan": "pro", "success_url": "https://s", "cancel_url": "https://c"}
    body.update(over)
    return body


def test_checkout_requires_a_user_token(client):
    assert client.post("/api/v1/payments/checkout", json=_body()).status_code == 401


def test_a_token_for_another_customer_cannot_buy_on_this_ones_account(client):
    """This decides whose account gets the plan."""
    response = client.post("/api/v1/payments/checkout", json=_body(user_id="user-2"), headers=_auth("user-1"))

    assert response.status_code == 403


def test_the_trial_plan_cannot_be_bought(client):
    """Charging for the free tier takes money for something already given."""
    response = client.post("/api/v1/payments/checkout", json=_body(plan="trial"), headers=_auth("user-1"))

    assert response.status_code == 400


def test_an_unknown_plan_is_refused(client):
    response = client.post("/api/v1/payments/checkout", json=_body(plan="platinum"), headers=_auth("user-1"))

    assert response.status_code == 400


def test_a_valid_request_returns_somewhere_to_send_the_customer(monkeypatch, client):
    captured = {}
    _stub_stripe(monkeypatch, captured)

    response = client.post("/api/v1/payments/checkout", json=_body(), headers=_auth("user-1"))

    assert response.status_code == 200
    assert response.json()["checkout_url"].startswith("https://checkout.stripe.com/")


def test_the_token_decides_who_is_buying(monkeypatch, client):
    captured = {}
    _stub_stripe(monkeypatch, captured)

    client.post("/api/v1/payments/checkout", json=_body(), headers=_auth("user-1"))

    assert captured["data"]["metadata[user_id]"] == "user-1"


# ---------------------------------------------------------------- reindex


def test_reindex_is_operator_only():
    """It re-embeds the entire corpus, which costs money at the embedding
    provider."""
    response = TestClient(app).post("/api/v1/internal/knowledge-packs/reindex")

    assert response.status_code == 403
