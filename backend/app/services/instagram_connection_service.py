import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import HTTPException


logger = logging.getLogger(__name__)


class InstagramConnectionService:
    RECONNECT_REQUIRED_DETAIL = (
        "Instagram account must be reconnected. The saved Meta connection is no longer valid."
    )

    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "instagram_connections.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _is_same_connection(self, item: dict, payload: dict) -> bool:
        return (
            item.get("user_id") == payload.get("user_id")
            and item.get("account_id") == payload.get("account_id")
            and item.get("platform") == payload.get("platform")
        )

    def _infer_connection_status(self, item: dict) -> str:
        status = str(item.get("status") or "").strip().lower()
        connection_status = str(item.get("connection_status") or "").strip().lower()

        if connection_status in {"connected", "stale", "reconnect_required", "disconnected", "failed"}:
            return connection_status

        if item.get("requires_reconnect"):
            return "reconnect_required"
        if status == "connected":
            return "connected"
        if status == "failed":
            return "failed"
        if status == "disconnected":
            return "disconnected"
        return "disconnected"

    def _normalize_connection_record(self, item: dict) -> dict:
        normalized_item = dict(item)
        normalized_item["connection_status"] = self._infer_connection_status(normalized_item)
        normalized_item["requires_reconnect"] = bool(normalized_item.get("requires_reconnect"))

        for field_name in [
            "connected_at",
            "error_message",
            "last_successful_meta_call_at",
            "last_failed_meta_call_at",
            "last_error_code",
            "last_error_message",
            "token_last_checked_at",
            "health_checked_at",
        ]:
            normalized_item[field_name] = normalized_item.get(field_name)

        return normalized_item

    def _to_public_connection_record(self, item: dict) -> dict:
        normalized_item = self._normalize_connection_record(item)
        public_item = dict(normalized_item)
        public_item.pop("access_token", None)
        public_item.pop("refresh_token", None)
        return public_item

    def _load_items(self) -> list[dict]:
        if not self.data_file.exists():
            return []

        with open(self.data_file, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        if not isinstance(raw_items, list):
            return []

        return [self._normalize_connection_record(item) for item in raw_items if isinstance(item, dict)]

    def _save_items(self, items: list[dict]) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _find_latest_connection_index(self, items: list[dict], user_id: str, account_id: str) -> int | None:
        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            if item.get("user_id") == user_id and item.get("account_id") == account_id:
                return index
        return None

    def _extract_meta_error(self, response: httpx.Response) -> tuple[str | None, str]:
        try:
            payload = response.json()
        except ValueError:
            return None, f"Meta returned HTTP {response.status_code}"

        error = payload.get("error")
        if isinstance(error, dict):
            error_code = error.get("code")
            error_message = error.get("message")
            return str(error_code) if error_code is not None else None, str(error_message or f"Meta returned HTTP {response.status_code}")

        if payload.get("error_message"):
            return None, str(payload["error_message"])

        return None, f"Meta returned HTTP {response.status_code}"

    def _is_reconnect_error(self, status_code: int, error_code: str | None, error_message: str) -> bool:
        normalized_message = (error_message or "").lower()
        reconnect_codes = {"102", "190"}
        reconnect_markers = [
            "error validating access token",
            "access token has expired",
            "session has expired",
            "invalid oauth access token",
            "invalid access token",
            "token has expired",
            "access token is invalid",
            "has been revoked",
            "permissions error",
            "permission error",
            "permission has not been granted",
            "not authorized",
        ]

        if error_code in reconnect_codes:
            return True
        if any(marker in normalized_message for marker in reconnect_markers):
            return True
        return status_code == 401

    def _is_temporary_meta_failure(self, status_code: int, error_code: str | None, error_message: str) -> bool:
        normalized_message = (error_message or "").lower()
        temporary_codes = {"1", "2", "4", "17", "341", "613"}
        temporary_markers = [
            "temporarily unavailable",
            "rate limit",
            "please reduce",
            "temporarily blocked",
            "server error",
            "try again later",
            "timeout",
            "timed out",
        ]

        if status_code in {500, 502, 503, 504}:
            return True
        if error_code in temporary_codes:
            return True
        return any(marker in normalized_message for marker in temporary_markers)

    def is_reconnect_required_exception(self, exc: HTTPException) -> bool:
        normalized_detail = str(exc.detail or "").lower()
        return exc.status_code == 403 and (
            "must be reconnected" in normalized_detail
            or "saved meta connection is no longer valid" in normalized_detail
        )

    def _build_reconnect_exception(self, account_id: str) -> HTTPException:
        return HTTPException(
            status_code=403,
            detail=self.RECONNECT_REQUIRED_DETAIL,
            headers={"X-Instagram-Account-Id": account_id},
        )

    def save_or_update_record(self, payload: dict) -> dict:
        payload = self._normalize_connection_record(payload)
        items = self._load_items()
        updated = False

        for index, item in enumerate(items):
            if item.get("state") == payload["state"]:
                items[index] = payload
                updated = True
                break

        if not updated:
            if payload.get("status") == "pending":
                items = [
                    item for item in items
                    if not (
                        self._is_same_connection(item, payload)
                        and item.get("status") in {"pending", "disconnected"}
                    )
                ]

            items.append(payload)

        self._save_items(items)
        return payload

    def get_connections_by_user(self, user_id: str) -> dict:
        latest_connections = [
            self._to_public_connection_record(item)
            for item in self._get_latest_connections_raw(user_id)
        ]
        return {
            "user_id": user_id,
            "connections": latest_connections,
        }

    def _get_latest_connections_raw(self, user_id: str) -> list[dict]:
        items = [item for item in self._load_items() if item.get("user_id") == user_id]
        latest_connections = []
        seen_account_ids = set()

        for item in reversed(items):
            account_id = item.get("account_id")
            if account_id in seen_account_ids:
                continue

            latest_connections.append(self._normalize_connection_record(item))
            seen_account_ids.add(account_id)

        return latest_connections

    def list_connection_user_ids(self) -> list[str]:
        return sorted({
            str(item.get("user_id"))
            for item in self._load_items()
            if item.get("user_id")
        })

    def get_connection_by_state(self, state: str) -> dict | None:
        for item in self._load_items():
            if item.get("state") == state:
                return item
        return None

    def get_latest_connection_by_account(self, user_id: str, account_id: str) -> dict | None:
        items = self._load_items()
        index = self._find_latest_connection_index(items, user_id, account_id)
        if index is None:
            return None
        return items[index]

    def update_connection_by_state(self, state: str, **updates) -> dict:
        items = self._load_items()

        for index, item in enumerate(items):
            if item.get("state") == state:
                updated_item = self._normalize_connection_record({**item, **updates})
                items[index] = updated_item
                self._save_items(items)
                return updated_item

        raise ValueError(f"Connection with state '{state}' was not found")

    def update_latest_connection(self, user_id: str, account_id: str, **updates) -> dict:
        items = self._load_items()
        index = self._find_latest_connection_index(items, user_id, account_id)
        if index is None:
            raise ValueError(f"Connection with account_id '{account_id}' was not found for user '{user_id}'")

        updated_item = self._normalize_connection_record({**items[index], **updates})
        items[index] = updated_item
        self._save_items(items)
        return updated_item

    def _mark_failed_connection(self, state: str, error_message: str) -> dict:
        failure_time = self._now_iso()
        return self.update_connection_by_state(
            state,
            status="failed",
            connection_status="failed",
            access_token=None,
            refresh_token=None,
            connected_at=None,
            error_message=error_message,
            last_failed_meta_call_at=failure_time,
            last_error_code="token_exchange_failed",
            last_error_message=error_message,
            token_last_checked_at=failure_time,
            health_checked_at=failure_time,
            requires_reconnect=False,
        )

    def _mark_meta_success(self, user_id: str, account_id: str) -> dict:
        success_time = self._now_iso()
        return self.update_latest_connection(
            user_id,
            account_id,
            status="connected",
            connection_status="connected",
            requires_reconnect=False,
            last_successful_meta_call_at=success_time,
            token_last_checked_at=success_time,
            health_checked_at=success_time,
            last_error_code=None,
            last_error_message=None,
            error_message=None,
        )

    def _mark_meta_reconnect_required(
        self,
        user_id: str,
        account_id: str,
        *,
        error_code: str | None,
        error_message: str,
    ) -> dict:
        failure_time = self._now_iso()
        return self.update_latest_connection(
            user_id,
            account_id,
            connection_status="reconnect_required",
            requires_reconnect=True,
            last_failed_meta_call_at=failure_time,
            last_error_code=error_code,
            last_error_message=error_message,
            error_message=error_message,
            token_last_checked_at=failure_time,
            health_checked_at=failure_time,
        )

    def _mark_meta_temporary_failure(
        self,
        user_id: str,
        account_id: str,
        *,
        error_code: str | None,
        error_message: str,
    ) -> dict:
        existing_connection = self.get_latest_connection_by_account(user_id, account_id) or {}
        failure_time = self._now_iso()
        connection_status = "stale" if existing_connection.get("last_successful_meta_call_at") else "failed"
        return self.update_latest_connection(
            user_id,
            account_id,
            connection_status=connection_status,
            requires_reconnect=False,
            last_failed_meta_call_at=failure_time,
            last_error_code=error_code,
            last_error_message=error_message,
            error_message=error_message,
            token_last_checked_at=failure_time,
            health_checked_at=failure_time,
        )

    def touch_health_checked(self, user_id: str, account_id: str) -> dict:
        checked_at = self._now_iso()
        return self.update_latest_connection(
            user_id,
            account_id,
            health_checked_at=checked_at,
        )

    def get_connection_for_meta(self, user_id: str, account_id: str) -> dict:
        connection = self.get_latest_connection_by_account(user_id, account_id)
        if not connection or connection.get("status") != "connected":
            raise HTTPException(
                status_code=400,
                detail="Instagram account is not connected for this user/account",
            )

        if connection.get("requires_reconnect") or connection.get("connection_status") == "reconnect_required":
            raise self._build_reconnect_exception(account_id)

        access_token = str(connection.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        return connection

    def _get_token_exchange_config(self) -> dict:
        config = {
            "app_id": os.getenv("META_APP_ID", "").strip(),
            "app_secret": os.getenv("META_APP_SECRET", "").strip(),
            "redirect_uri": os.getenv("META_REDIRECT_URI", "").strip(),
            "token_url": os.getenv("META_TOKEN_URL", "").strip(),
        }
        if not all(config.values()):
            raise RuntimeError("Meta OAuth token config is incomplete")
        return config

    def _get_graph_base_url(self) -> str:
        token_url = os.getenv("META_TOKEN_URL", "").strip()
        token_path = "/oauth/access_token"

        if token_url and token_path in token_url:
            return token_url.rsplit(token_path, 1)[0]

        return "https://graph.facebook.com/v23.0"

    def _extract_token_error_message(self, response: httpx.Response) -> str:
        _, error_message = self._extract_meta_error(response)
        return error_message

    def meta_get(
        self,
        *,
        user_id: str,
        account_id: str,
        path: str,
        access_token: str | None = None,
        fields: str | None = None,
        extra_params: dict | None = None,
        timeout_detail: str,
        http_failure_detail: str,
        request_failure_detail: str,
        invalid_json_detail: str,
    ) -> dict:
        connection = self.get_connection_for_meta(user_id, account_id)
        resolved_access_token = str(access_token or connection.get("access_token") or "").strip()
        if not resolved_access_token:
            raise HTTPException(
                status_code=400,
                detail="Connected Instagram account is missing an access token",
            )

        url = f"{self._get_graph_base_url()}/{path.lstrip('/')}"
        params = {"access_token": resolved_access_token}
        if fields:
            params["fields"] = fields
        if extra_params:
            params.update(extra_params)

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.ReadTimeout as exc:
            self._mark_meta_temporary_failure(
                user_id,
                account_id,
                error_code="timeout",
                error_message=timeout_detail,
            )
            raise HTTPException(status_code=504, detail=timeout_detail) from exc
        except httpx.HTTPStatusError as exc:
            error_code, error_message = self._extract_meta_error(exc.response)
            if self._is_reconnect_error(exc.response.status_code, error_code, error_message):
                self._mark_meta_reconnect_required(
                    user_id,
                    account_id,
                    error_code=error_code,
                    error_message=error_message,
                )
                raise self._build_reconnect_exception(account_id) from exc

            self._mark_meta_temporary_failure(
                user_id,
                account_id,
                error_code=error_code,
                error_message=error_message,
            )
            raise HTTPException(
                status_code=502 if self._is_temporary_meta_failure(exc.response.status_code, error_code, error_message) else 502,
                detail=f"{http_failure_detail}: {error_message}",
            ) from exc
        except httpx.HTTPError as exc:
            self._mark_meta_temporary_failure(
                user_id,
                account_id,
                error_code="request_failed",
                error_message=request_failure_detail,
            )
            raise HTTPException(status_code=502, detail=request_failure_detail) from exc
        except ValueError as exc:
            self._mark_meta_temporary_failure(
                user_id,
                account_id,
                error_code="invalid_json",
                error_message=invalid_json_detail,
            )
            raise HTTPException(status_code=502, detail=invalid_json_detail) from exc

        self._mark_meta_success(user_id, account_id)
        return payload

    def exchange_code_for_connection(self, state: str, code: str) -> dict:
        try:
            config = self._get_token_exchange_config()
        except RuntimeError as exc:
            self._mark_failed_connection(state, str(exc))
            raise

        request_params = {
            "client_id": config["app_id"],
            "client_secret": config["app_secret"],
            "redirect_uri": config["redirect_uri"],
            "code": code,
            "grant_type": "authorization_code",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(config["token_url"], params=request_params)
                response.raise_for_status()
                payload = response.json()
        except httpx.ReadTimeout as exc:
            failure_message = "Meta token exchange timed out"
            self._mark_failed_connection(state, failure_message)
            raise RuntimeError(failure_message) from exc
        except httpx.HTTPStatusError as exc:
            failure_message = f"Meta token exchange failed: {self._extract_token_error_message(exc.response)}"
            self._mark_failed_connection(state, failure_message)
            raise RuntimeError(failure_message) from exc
        except httpx.HTTPError as exc:
            failure_message = "Meta token exchange request failed"
            self._mark_failed_connection(state, failure_message)
            raise RuntimeError(failure_message) from exc
        except ValueError as exc:
            failure_message = "Meta token exchange returned invalid JSON"
            self._mark_failed_connection(state, failure_message)
            raise RuntimeError(failure_message) from exc

        access_token = payload.get("access_token")
        if not access_token:
            failure_message = "Meta token exchange did not return an access token"
            self._mark_failed_connection(state, failure_message)
            raise RuntimeError(failure_message)

        return self.update_connection_by_state(
            state,
            status="connected",
            connection_status="connected",
            access_token=access_token,
            refresh_token=payload.get("refresh_token"),
            connected_at=self._now_iso(),
            error_message=None,
            last_error_code=None,
            last_error_message=None,
            last_failed_meta_call_at=None,
            last_successful_meta_call_at=None,
            token_last_checked_at=None,
            health_checked_at=None,
            requires_reconnect=False,
        )

    def _build_health_record(self, item: dict) -> dict:
        normalized_item = self._normalize_connection_record(item)
        return {
            "user_id": str(normalized_item.get("user_id") or ""),
            "account_id": str(normalized_item.get("account_id") or ""),
            "connection_status": str(normalized_item.get("connection_status") or "disconnected"),
            "requires_reconnect": bool(normalized_item.get("requires_reconnect")),
            "token_last_checked_at": normalized_item.get("token_last_checked_at"),
            "last_successful_meta_call_at": normalized_item.get("last_successful_meta_call_at"),
            "last_failed_meta_call_at": normalized_item.get("last_failed_meta_call_at"),
            "last_error_code": normalized_item.get("last_error_code"),
            "last_error_message": normalized_item.get("last_error_message"),
            "health_checked_at": normalized_item.get("health_checked_at"),
        }

    def get_connection_health(self, user_id: str, account_id: str | None = None) -> dict:
        connections = self._get_latest_connections_raw(user_id)
        if account_id:
            connections = [connection for connection in connections if connection.get("account_id") == account_id]
            if not connections:
                raise HTTPException(
                    status_code=404,
                    detail=f"Connected account '{account_id}' was not found for this user",
                )

        if not connections:
            return {
                "user_id": user_id,
                "connections": [],
            }

        health_records = []
        mock_mode_enabled = os.getenv("META_MOCK_MODE", "").lower() == "true"

        for connection in connections:
            current_account_id = str(connection.get("account_id") or "")
            if not current_account_id:
                continue

            if mock_mode_enabled:
                updated_connection = self.update_latest_connection(
                    user_id,
                    current_account_id,
                    health_checked_at=self._now_iso(),
                    token_last_checked_at=self._now_iso(),
                )
                health_records.append(self._build_health_record(updated_connection))
                continue

            if connection.get("status") != "connected" or not str(connection.get("access_token") or "").strip():
                updated_connection = self.touch_health_checked(user_id, current_account_id)
                health_records.append(self._build_health_record(updated_connection))
                continue

            if connection.get("requires_reconnect") or connection.get("connection_status") == "reconnect_required":
                updated_connection = self.touch_health_checked(user_id, current_account_id)
                health_records.append(self._build_health_record(updated_connection))
                continue

            try:
                self.meta_get(
                    user_id=user_id,
                    account_id=current_account_id,
                    path="me",
                    fields="id,name",
                    timeout_detail="Meta Instagram connection health check timed out",
                    http_failure_detail="Meta Instagram connection health check failed",
                    request_failure_detail="Meta Instagram connection health check request failed",
                    invalid_json_detail="Meta Instagram connection health check returned invalid JSON",
                )
            except HTTPException as exc:
                logger.warning(
                    "Instagram connection health check failed user_id=%s account_id=%s status_code=%s detail=%s",
                    user_id,
                    current_account_id,
                    exc.status_code,
                    exc.detail,
                )

            latest_connection = self.get_latest_connection_by_account(user_id, current_account_id)
            if latest_connection:
                health_records.append(self._build_health_record(latest_connection))

        return {
            "user_id": user_id,
            "connections": health_records,
        }

    def get_reconnect_status(self, user_id: str) -> dict:
        connections = self._get_latest_connections_raw(user_id)
        reconnect_account_ids = [
            str(connection.get("account_id") or "")
            for connection in connections
            if connection.get("requires_reconnect") or connection.get("connection_status") == "reconnect_required"
        ]
        reconnect_account_ids = [account_id for account_id in reconnect_account_ids if account_id]

        if reconnect_account_ids:
            message = "One or more Instagram accounts must be reconnected."
        elif connections:
            message = "All connected Instagram accounts are currently usable."
        else:
            message = "No connected Instagram accounts were found for this user."

        return {
            "user_id": user_id,
            "requires_reconnect": bool(reconnect_account_ids),
            "account_ids": reconnect_account_ids,
            "message": message,
        }
