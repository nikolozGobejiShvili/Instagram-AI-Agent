import os

from fastapi import HTTPException

from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService


class InstagramProfileService:
    def __init__(self):
        self.connected_accounts_service = ConnectedAccountsService()
        self.instagram_connection_service = InstagramConnectionService()

    def _resolve_account_id(self, user_id: str, account_id: str | None = None) -> str:
        return self.connected_accounts_service.resolve_account_id(user_id, account_id)

    def _get_connected_connection(self, user_id: str, account_id: str) -> dict:
        return self.instagram_connection_service.get_connection_for_meta(user_id, account_id)

    def _build_mock_profile(self, user_id: str, account_id: str) -> dict:
        normalized_account_id = account_id.lower().replace("-", "_").replace(" ", "_")
        display_name = account_id.replace("-", " ").replace("_", " ").title()
        seed = sum(ord(char) for char in f"{user_id}:{account_id}")

        return {
            "user_id": user_id,
            "account_id": account_id,
            "platform": "instagram",
            "instagram_username": f"{normalized_account_id}_ig",
            "display_name": display_name,
            "biography": f"Mock Instagram profile for {display_name}. Content, insights, and niche testing.",
            "followers_count": 1000 + (seed % 9000),
            "following_count": 100 + (seed % 900),
            "media_count": 12 + (seed % 120),
            "is_connected": True,
        }

    def _get_graph_base_url(self) -> str:
        return self.instagram_connection_service._get_graph_base_url()

    def _build_mock_debug_response(self, user_id: str, account_id: str) -> dict:
        return {
            "user_id": user_id,
            "account_id": account_id,
            "facebook_user_checked": False,
            "token_checked": False,
            "granted_permissions": [
                "instagram_basic",
                "instagram_manage_insights",
                "pages_show_list",
            ],
            "pages_found_count": 1,
            "pages": [
                {
                    "page_id": f"{account_id}-mock-page",
                    "page_name": f"{account_id.replace('-', ' ').title()} Page",
                    "has_instagram_business_account": True,
                }
            ],
            "page_discovery_hint": "Mock mode: page discovery is simulated.",
        }

    def _meta_get(self, user_id: str, account_id: str, path: str, access_token: str, fields: str) -> dict:
        return self.instagram_connection_service.meta_get(
            user_id=user_id,
            account_id=account_id,
            path=path,
            access_token=access_token,
            fields=fields,
            timeout_detail="Meta Instagram profile fetch timed out",
            http_failure_detail="Meta Instagram profile fetch failed",
            request_failure_detail="Meta Instagram profile request failed",
            invalid_json_detail="Meta Instagram profile fetch returned invalid JSON",
        )

    def _try_get_facebook_user_identity(self, user_id: str, account_id: str, access_token: str) -> dict | None:
        try:
            return self._meta_get(user_id, account_id, "me", access_token, "id,name")
        except HTTPException as exc:
            if self.instagram_connection_service.is_reconnect_required_exception(exc):
                raise
            return None

    def _try_get_granted_permissions(self, user_id: str, account_id: str, access_token: str) -> tuple[bool, list[str]]:
        try:
            permissions_payload = self._meta_get(user_id, account_id, "me/permissions", access_token, "permission,status")
        except HTTPException as exc:
            if self.instagram_connection_service.is_reconnect_required_exception(exc):
                raise
            return False, []

        granted_permissions = []
        for item in permissions_payload.get("data", []):
            if item.get("status") == "granted" and item.get("permission"):
                granted_permissions.append(str(item["permission"]))

        return True, sorted(granted_permissions)

    def _build_page_discovery_hint(self, diagnostics: dict) -> str:
        granted_permissions = set(diagnostics.get("granted_permissions", []))
        has_pages_show_list = "pages_show_list" in granted_permissions
        has_pages_manage_metadata = "pages_manage_metadata" in granted_permissions

        if not diagnostics.get("token_checked"):
            return "Could not inspect granted token permissions. Reconnect the account and try again."

        if not has_pages_show_list:
            return "The token is missing pages_show_list, so Meta may not return accessible Facebook Pages."

        if diagnostics.get("pages_found_count", 0) == 0:
            return (
                "The token has pages_show_list but Meta returned zero Pages. Check that the Page is enabled "
                "in Facebook Login settings, the same Facebook user has full Page access, and the account was reconnected after changes. "
                f"pages_manage_metadata present: {has_pages_manage_metadata}."
            )

        if not diagnostics.get("instagram_business_account_present_on_any_page"):
            return (
                "Meta returned Pages, but none exposed instagram_business_account. Check that the Instagram account is Business/Creator "
                "and linked to the returned Facebook Page."
            )

        return "Meta returned at least one Page with a linked Instagram business/creator account."

    def _discover_page_access(self, user_id: str, account_id: str, access_token: str) -> dict:
        facebook_user = self._try_get_facebook_user_identity(user_id, account_id, access_token)
        token_checked, granted_permissions = self._try_get_granted_permissions(user_id, account_id, access_token)
        pages_payload = self._meta_get(
            user_id,
            account_id,
            "me/accounts",
            access_token,
            "id,name,instagram_business_account{id,username,name},connected_instagram_account{id,username,name}",
        )
        raw_pages = pages_payload.get("data", [])
        pages = []
        instagram_account_references = []

        for raw_page in raw_pages:
            instagram_account = raw_page.get("instagram_business_account") or raw_page.get("connected_instagram_account")
            page_id = str(raw_page.get("id") or "")
            page_name = str(raw_page.get("name") or "")
            has_instagram_business_account = bool(instagram_account and instagram_account.get("id"))

            pages.append({
                "page_id": page_id,
                "page_name": page_name,
                "has_instagram_business_account": has_instagram_business_account,
            })

            if has_instagram_business_account:
                instagram_account_references.append({
                    "page_id": page_id,
                    "page_name": page_name,
                    "instagram_user_id": str(instagram_account.get("id") or ""),
                    "instagram_username": str(instagram_account.get("username") or ""),
                    "instagram_name": str(instagram_account.get("name") or ""),
                })

        diagnostics = {
            "facebook_user_checked": facebook_user is not None,
            "token_checked": token_checked,
            "granted_permissions": granted_permissions,
            "pages_found_count": len(pages),
            "pages": pages,
            "matched_page_ids": [page["page_id"] for page in pages if page["page_id"]],
            "matched_page_names": [page["page_name"] for page in pages if page["page_name"]],
            "instagram_business_account_present_on_any_page": any(
                page["has_instagram_business_account"] for page in pages
            ),
            "instagram_account_references": instagram_account_references,
        }
        diagnostics["page_discovery_hint"] = self._build_page_discovery_hint(diagnostics)
        return diagnostics

    def _build_discovery_error_detail(self, message: str, diagnostics: dict) -> dict:
        granted_permissions = set(diagnostics["granted_permissions"])
        return {
            "message": message,
            "facebook_user_checked": diagnostics["facebook_user_checked"],
            "token_checked": diagnostics["token_checked"],
            "granted_permissions": diagnostics["granted_permissions"],
            "pages_show_list_present": "pages_show_list" in granted_permissions,
            "pages_manage_metadata_present": "pages_manage_metadata" in granted_permissions,
            "pages_found_count": diagnostics["pages_found_count"],
            "matched_page_ids": diagnostics["matched_page_ids"],
            "matched_page_names": diagnostics["matched_page_names"],
            "instagram_business_account_present_on_any_page": diagnostics["instagram_business_account_present_on_any_page"],
            "page_discovery_hint": diagnostics["page_discovery_hint"],
        }

    def _get_instagram_account_reference(self, user_id: str, account_id: str, access_token: str) -> dict:
        diagnostics = self._discover_page_access(user_id, account_id, access_token)

        if diagnostics["pages_found_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail=self._build_discovery_error_detail(
                    "No accessible Facebook Pages were found for this Meta user token",
                    diagnostics,
                ),
            )

        if not diagnostics["instagram_account_references"]:
            raise HTTPException(
                status_code=400,
                detail=self._build_discovery_error_detail(
                    "No linked Instagram business or creator account was found on the accessible Facebook Pages",
                    diagnostics,
                ),
            )

        return diagnostics["instagram_account_references"][0]

    def _build_real_profile(self, user_id: str, account_id: str, connection: dict) -> dict:
        access_token = str(connection.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        instagram_account = self._get_instagram_account_reference(user_id, account_id, access_token)
        profile_payload = self._meta_get(
            user_id,
            account_id,
            instagram_account["instagram_user_id"],
            access_token,
            "username,name,biography,followers_count,follows_count,media_count,profile_picture_url",
        )

        display_name = (
            str(profile_payload.get("name") or "").strip()
            or instagram_account["instagram_name"]
            or instagram_account["page_name"]
            or account_id
        )

        return {
            "user_id": user_id,
            "account_id": account_id,
            "platform": "instagram",
            "instagram_username": str(profile_payload.get("username") or instagram_account["instagram_username"] or ""),
            "display_name": display_name,
            "biography": str(profile_payload.get("biography") or ""),
            "followers_count": int(profile_payload.get("followers_count") or 0),
            "following_count": int(profile_payload.get("follows_count") or 0),
            "media_count": int(profile_payload.get("media_count") or 0),
            "is_connected": True,
        }

    def get_profile(self, user_id: str, account_id: str | None = None) -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)
        connection = self._get_connected_connection(user_id, effective_account_id)

        if os.getenv("META_MOCK_MODE", "").lower() == "true":
            return self._build_mock_profile(user_id, effective_account_id)

        return self._build_real_profile(user_id, effective_account_id, connection)

    def get_page_diagnostics(self, user_id: str, account_id: str | None = None) -> dict:
        effective_account_id = self._resolve_account_id(user_id, account_id)
        connection = self._get_connected_connection(user_id, effective_account_id)

        if os.getenv("META_MOCK_MODE", "").lower() == "true":
            return self._build_mock_debug_response(user_id, effective_account_id)

        access_token = str(connection.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        diagnostics = self._discover_page_access(user_id, effective_account_id, access_token)
        return {
            "user_id": user_id,
            "account_id": effective_account_id,
            "facebook_user_checked": diagnostics["facebook_user_checked"],
            "token_checked": diagnostics["token_checked"],
            "granted_permissions": diagnostics["granted_permissions"],
            "pages_found_count": diagnostics["pages_found_count"],
            "pages": diagnostics["pages"],
            "page_discovery_hint": diagnostics["page_discovery_hint"],
        }
