"""The competitor watch list, and the diff that is the point of it.

A snapshot restates facts. "Nike has 302M followers" changes nothing the
customer does. "Three new posts since Tuesday, the best took four times their
median" is the output worth paying for — so most of these cases are about the
comparison being right, not about storage.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import tracked_accounts as tracked_route  # noqa: E402
from app.api.user_auth import USER_TOKEN_HEADER  # noqa: E402
from app.services.billing_service import BillingService  # noqa: E402
from app.services.tracked_account_service import TrackedAccountService  # noqa: E402
from app.services.user_token_service import UserTokenService  # noqa: E402

SECRET = "test-user-token-secret"
USER = "watch-user"


def _profile(followers=1000, media=10, posts=None, likes=(100, 200, 300)):
    if posts is None:
        posts = [
            {"post_id": f"p{i}", "caption": f"post {i}", "like_count": like, "comment_count": 5,
             "media_type": "VIDEO", "product_type": "REELS", "permalink": None, "posted_at": None}
            for i, like in enumerate(likes, start=1)
        ]
    return {"handle": "nike", "followers_count": followers, "media_count": media, "posts": posts}


@pytest.fixture()
def service(tmp_path):
    return TrackedAccountService(db_path=tmp_path / "tracked.sqlite3")


# ------------------------------------------------------------------ the diff


def test_the_first_check_reports_no_deltas(service):
    """Measuring against zero would produce '+1000 followers', which is
    derivable and useless."""
    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile())

    assert change["is_first_check"] is True
    assert change["followers_change"] is None
    assert change["new_posts"] == []


def test_the_second_check_reports_what_moved(service):
    service.compare_and_store(user_id=USER, handle="nike", profile=_profile(followers=1000))

    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile(followers=1250))

    assert change["is_first_check"] is False
    assert change["followers_change"] == 250
    assert change["since"] is not None


def test_new_posts_are_identified_by_id_not_by_counting(service):
    """A media_count that rose by two could be three posts and a deletion.
    Telling a marketer '2 new posts' who then finds three has given them a
    wrong picture of that competitor's cadence."""
    first = _profile(posts=[{"post_id": "a", "like_count": 10}, {"post_id": "b", "like_count": 20}])
    service.compare_and_store(user_id=USER, handle="nike", profile=first)

    second = _profile(posts=[
        {"post_id": "b", "like_count": 20},
        {"post_id": "c", "like_count": 30},
        {"post_id": "d", "like_count": 40},
    ])
    change = service.compare_and_store(user_id=USER, handle="nike", profile=second)

    assert [p["post_id"] for p in change["new_posts"]] == ["c", "d"]


def test_a_deleted_post_is_not_reported_as_new(service):
    service.compare_and_store(
        user_id=USER, handle="nike", profile=_profile(posts=[{"post_id": "a"}, {"post_id": "b"}])
    )

    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile(posts=[{"post_id": "a"}]))

    assert change["new_posts"] == []


def test_engagement_movement_is_tracked(service):
    service.compare_and_store(user_id=USER, handle="nike", profile=_profile(likes=(100, 200, 300)))

    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile(likes=(400, 500, 600)))

    assert change["median_likes_change"] == 300


def test_an_unknown_number_yields_no_delta_rather_than_zero(service):
    """Reporting 'no change' for a value we never had is a claim, and 0 looks
    like one."""
    service.compare_and_store(user_id=USER, handle="nike", profile=_profile(followers=None))

    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile(followers=500))

    assert change["followers_change"] is None


# ----------------------------------------------------------------- the limit


def test_the_plan_limit_is_enforced(service):
    """The field existed in every tier and was published in the catalogue while
    nothing enforced it."""
    for handle in ("a", "b", "c"):
        service.add(user_id=USER, handle=handle, label=None, limit=3)

    with pytest.raises(Exception) as exc:
        service.add(user_id=USER, handle="d", label=None, limit=3)

    assert getattr(exc.value, "status_code", None) == 403


def test_a_zero_limit_says_the_plan_lacks_the_feature(service):
    """Trial tracks nobody. 'You can track 0 accounts' reads as a bug."""
    with pytest.raises(Exception) as exc:
        service.add(user_id=USER, handle="a", label=None, limit=0)

    assert "not included" in str(exc.value.detail)


def test_re_adding_a_tracked_account_does_not_consume_a_slot(service):
    service.add(user_id=USER, handle="a", label="first", limit=1)

    service.add(user_id=USER, handle="a", label="renamed", limit=1)

    assert [t["label"] for t in service.list_tracked(USER)] == ["renamed"]


def test_one_customers_list_is_invisible_to_another(service):
    service.add(user_id=USER, handle="a", label=None, limit=3)

    assert service.list_tracked("someone-else") == []


def test_untracking_forgets_the_history(service):
    """Keeping snapshots for an account the customer dropped would make a later
    re-add report months of change nobody asked about."""
    service.add(user_id=USER, handle="nike", label=None, limit=3)
    service.compare_and_store(user_id=USER, handle="nike", profile=_profile())
    service.remove(user_id=USER, handle="nike")

    service.add(user_id=USER, handle="nike", label=None, limit=3)
    change = service.compare_and_store(user_id=USER, handle="nike", profile=_profile())

    assert change["is_first_check"] is True


def test_untracking_something_absent_is_a_404(service):
    with pytest.raises(Exception) as exc:
        service.remove(user_id=USER, handle="never-added")

    assert getattr(exc.value, "status_code", None) == 404


# ------------------------------------------------------------------- routes


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_TOKEN_SECRET", SECRET)
    billing = BillingService(db_path=tmp_path / "b.sqlite3", legacy_json_file=tmp_path / "absent.json")
    monkeypatch.setattr(tracked_route, "billing_service", billing)
    monkeypatch.setattr(
        tracked_route, "tracked_account_service", TrackedAccountService(db_path=tmp_path / "t.sqlite3")
    )
    test_client = TestClient(app)
    test_client.billing_service = billing
    return test_client


def _auth(user_id=USER):
    return {USER_TOKEN_HEADER: UserTokenService(secret=SECRET).issue(user_id)["user_token"]}


def test_adding_requires_a_user_token(client):
    response = client.post("/api/v1/tracked-accounts", json={"user_id": USER, "handle": "nike"})

    assert response.status_code == 401


def test_a_handle_is_normalised_however_it_was_pasted(client):
    client.billing_service.set_plan(USER, {"current_plan": "pro"})

    client.post(
        "/api/v1/tracked-accounts",
        json={"user_id": USER, "handle": "https://www.instagram.com/nike/?hl=en"},
        headers=_auth(),
    )
    body = client.post(
        "/api/v1/tracked-accounts", json={"user_id": USER, "handle": "@nike"}, headers=_auth()
    ).json()

    # One entry, not three, or each would accumulate its own history.
    assert [t["handle"] for t in body["tracked_accounts"]] == ["nike"]


def test_the_list_reports_the_limit_alongside_it(client):
    """So a client can show '1 of 10' without a second request."""
    client.billing_service.set_plan(USER, {"current_plan": "pro"})
    client.post("/api/v1/tracked-accounts", json={"user_id": USER, "handle": "nike"}, headers=_auth())

    body = client.get("/api/v1/tracked-accounts", headers=_auth()).json()

    assert body["tracked_accounts_limit"] == BillingService.PLAN_DEFAULTS["pro"]["tracked_accounts_limit"]


def test_the_trial_plan_cannot_track_anyone(client):
    client.billing_service.set_plan(USER, {"current_plan": "trial"})

    response = client.post(
        "/api/v1/tracked-accounts", json={"user_id": USER, "handle": "nike"}, headers=_auth()
    )

    assert response.status_code == 403


def test_the_token_decides_whose_list_is_read(client):
    client.billing_service.set_plan(USER, {"current_plan": "pro"})
    client.post("/api/v1/tracked-accounts", json={"user_id": USER, "handle": "nike"}, headers=_auth())

    other = client.get("/api/v1/tracked-accounts", headers=_auth("other-user")).json()

    assert other["tracked_accounts"] == []
