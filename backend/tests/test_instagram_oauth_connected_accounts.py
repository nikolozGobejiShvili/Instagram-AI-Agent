import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.api.routes import connected_accounts as connected_accounts_route  # noqa: E402
from app.api.routes import instagram_connection as instagram_connection_route  # noqa: E402
from app.api.routes import instagram_insights as instagram_insights_route  # noqa: E402
from app.api.routes import instagram_media as instagram_media_route  # noqa: E402
from app.api.routes import instagram_profile as instagram_profile_route  # noqa: E402
from app.services.connected_accounts_service import ConnectedAccountsService  # noqa: E402
from app.services.instagram_connection_service import InstagramConnectionService  # noqa: E402


def _seed_pending_instagram_connection(service: InstagramConnectionService) -> None:
    service.save_or_update_record({
        "user_id": "user-1",
        "account_id": "test3",
        "state": "state-1",
        "status": "pending",
        "connection_status": "disconnected",
        "platform": "instagram",
        "access_token": None,
        "refresh_token": None,
        "connected_at": None,
        "error_message": None,
        "last_successful_meta_call_at": None,
        "last_failed_meta_call_at": None,
        "last_error_code": None,
        "last_error_message": None,
        "token_last_checked_at": None,
        "requires_reconnect": False,
        "health_checked_at": None,
    })


def test_oauth_callback_mirrors_connected_account_without_exposing_token(monkeypatch, tmp_path):
    monkeypatch.setenv("META_MOCK_MODE", "true")

    instagram_service = InstagramConnectionService()
    instagram_service.data_file = tmp_path / "instagram_connections.json"
    connected_service = ConnectedAccountsService()
    connected_service.data_file = tmp_path / "connected_accounts.json"
    connected_service.save_accounts({
        "user_id": "user-1",
        "accounts": [
            {
                "account_id": "test-account-3",
                "label": "Existing Active",
                "niche": "existing",
                "is_active": True,
            }
        ],
    })
    _seed_pending_instagram_connection(instagram_service)

    monkeypatch.setattr(instagram_connection_route, "instagram_connection_service", instagram_service)
    monkeypatch.setattr(instagram_connection_route, "connected_accounts_service", connected_service)
    monkeypatch.setattr(connected_accounts_route, "connected_accounts_service", connected_service)

    client = TestClient(app)
    response = client.post(
        "/api/v1/instagram-connections/callback",
        json={"state": "state-1", "code": "dummy-code"},
    )

    assert response.status_code == 200, response.text
    response_payload = response.json()
    serialized_response = response.text
    assert "access_token" not in response_payload
    assert "refresh_token" not in response_payload
    assert "mock-token" not in serialized_response

    accounts_payload = client.get("/api/v1/connected-accounts/user-1").json()
    accounts = accounts_payload["accounts"]
    mirrored_account = next(account for account in accounts if account["account_id"] == "test3")
    active_account = next(account for account in accounts if account["account_id"] == "test-account-3")
    assert mirrored_account["label"] == "test3"
    assert mirrored_account["niche"] == ""
    assert mirrored_account["is_active"] is False
    assert active_account["is_active"] is True


def test_oauth_connected_account_can_be_selected_and_resolved_by_instagram_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("META_MOCK_MODE", "true")

    instagram_service = InstagramConnectionService()
    instagram_service.data_file = tmp_path / "instagram_connections.json"
    connected_service = ConnectedAccountsService()
    connected_service.data_file = tmp_path / "connected_accounts.json"
    connected_service.save_accounts({
        "user_id": "user-1",
        "accounts": [],
    })
    _seed_pending_instagram_connection(instagram_service)

    monkeypatch.setattr(instagram_connection_route, "instagram_connection_service", instagram_service)
    monkeypatch.setattr(instagram_connection_route, "connected_accounts_service", connected_service)
    monkeypatch.setattr(connected_accounts_route, "connected_accounts_service", connected_service)

    instagram_profile_route.instagram_profile_service.connected_accounts_service = connected_service
    instagram_profile_route.instagram_profile_service.instagram_connection_service = instagram_service
    instagram_media_route.instagram_media_service.connected_accounts_service = connected_service
    instagram_media_route.instagram_media_service.instagram_connection_service = instagram_service
    instagram_media_route.instagram_media_service.instagram_profile_service.connected_accounts_service = connected_service
    instagram_media_route.instagram_media_service.instagram_profile_service.instagram_connection_service = instagram_service
    instagram_insights_route.instagram_insights_service.connected_accounts_service = connected_service
    instagram_insights_route.instagram_insights_service.instagram_connection_service = instagram_service
    instagram_insights_route.instagram_insights_service.instagram_profile_service.connected_accounts_service = connected_service
    instagram_insights_route.instagram_insights_service.instagram_profile_service.instagram_connection_service = instagram_service

    client = TestClient(app)
    callback_response = client.post(
        "/api/v1/instagram-connections/callback",
        json={"state": "state-1", "code": "dummy-code"},
    )
    assert callback_response.status_code == 200, callback_response.text

    set_active_response = client.post(
        "/api/v1/connected-accounts/user-1/set-active",
        json={"account_id": "test3"},
    )
    assert set_active_response.status_code == 200, set_active_response.text
    assert set_active_response.json()["accounts"][0]["is_active"] is True

    profile_response = client.get("/api/v1/instagram-profile/user-1?account_id=test3")
    media_response = client.get("/api/v1/instagram-media/user-1?account_id=test3&limit=5")
    insights_response = client.get("/api/v1/instagram-insights/user-1?account_id=test3&period=30d")

    assert profile_response.status_code == 200, profile_response.text
    assert media_response.status_code == 200, media_response.text
    assert insights_response.status_code == 200, insights_response.text
    assert profile_response.json()["account_id"] == "test3"
    assert media_response.json()["account_id"] == "test3"
    assert insights_response.json()["account_id"] == "test3"
