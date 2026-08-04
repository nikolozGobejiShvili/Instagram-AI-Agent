import json
import logging
from pathlib import Path

from fastapi import HTTPException


logger = logging.getLogger(__name__)


class ConnectedAccountsService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "connected_accounts.json"

    def _load_items(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_items(self, items: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def get_accounts(self, user_id: str) -> dict:
        items = self._load_items()

        return {
            "user_id": user_id,
            "accounts": items.get(user_id, []),
        }

    def find_user_id_by_account_id(self, account_id: str) -> str | None:
        items = self._load_items()

        for user_id, accounts in items.items():
            for account in accounts:
                if str(account.get("account_id")) == str(account_id):
                    return str(user_id)

        return None

    def list_all_accounts(self) -> list[dict]:
        items = self._load_items()
        results = []

        for user_id, accounts in items.items():
            if not isinstance(accounts, list):
                continue

            for account in accounts:
                if not isinstance(account, dict):
                    continue

                results.append({
                    "user_id": str(user_id),
                    "account_id": str(account.get("account_id") or ""),
                    "label": account.get("label"),
                    "niche": account.get("niche"),
                    "is_active": bool(account.get("is_active")),
                })

        return results

    def save_accounts(self, payload: dict) -> dict:
        items = self._load_items()
        user_id = payload["user_id"]

        items[user_id] = payload["accounts"]
        self._save_items(items)

        return {
            "user_id": user_id,
            "accounts": items[user_id],
        }

    def upsert_account(
        self,
        *,
        user_id: str,
        account_id: str,
        label: str | None = None,
        niche: str | None = None,
        is_active: bool | None = None,
    ) -> dict:
        items = self._load_items()
        accounts = items.get(user_id, [])
        normalized_label = " ".join((label or "").split()).strip()
        normalized_niche = " ".join((niche or "").split()).strip()
        found = False

        for account in accounts:
            if str(account.get("account_id")) != account_id:
                if is_active is True:
                    account["is_active"] = False
                continue

            account["account_id"] = account_id
            account["label"] = normalized_label or account.get("label") or account_id
            account["niche"] = normalized_niche or account.get("niche") or ""
            account["is_active"] = bool(is_active) if is_active is not None else bool(account.get("is_active"))
            found = True

        if not found:
            accounts.append({
                "account_id": account_id,
                "label": normalized_label or account_id,
                "niche": normalized_niche,
                "is_active": bool(is_active),
            })

        items[user_id] = accounts
        self._save_items(items)

        return {
            "user_id": user_id,
            "accounts": items[user_id],
        }

    def set_active_account(self, user_id: str, account_id: str) -> dict:
        items = self._load_items()
        accounts = items.get(user_id, [])
        found = False

        for account in accounts:
            is_target = account.get("account_id") == account_id
            account["is_active"] = is_target
            if is_target:
                found = True

        if not found:
            raise ValueError(f"Account '{account_id}' was not found for user '{user_id}'")

        items[user_id] = accounts
        self._save_items(items)

        return {
            "user_id": user_id,
            "accounts": items[user_id],
        }

    def resolve_account_id(self, user_id: str, account_id: str | None = None) -> str:
        accounts_payload = self.get_accounts(user_id)
        accounts = accounts_payload.get("accounts", [])

        logger.info(
            "Resolving connected account user_id=%s provided_account_id=%s accounts_count=%s",
            user_id,
            account_id,
            len(accounts),
        )

        if account_id:
            matched_account = next(
                (account for account in accounts if str(account.get("account_id")) == account_id),
                None,
            )
            if matched_account:
                logger.info(
                    "Resolved connected account user_id=%s provided_account_id=%s resolved_account_id=%s strategy=explicit",
                    user_id,
                    account_id,
                    account_id,
                )
                return account_id

            if not accounts:
                logger.warning(
                    "Failed to resolve connected account user_id=%s provided_account_id=%s reason=no_accounts",
                    user_id,
                    account_id,
                )
                raise HTTPException(status_code=404, detail="No connected accounts found for this user")

            logger.warning(
                "Failed to resolve connected account user_id=%s provided_account_id=%s reason=account_not_found",
                user_id,
                account_id,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Connected account '{account_id}' was not found for this user",
            )

        if not accounts:
            logger.warning(
                "Failed to resolve connected account user_id=%s provided_account_id=%s reason=no_accounts",
                user_id,
                account_id,
            )
            raise HTTPException(status_code=404, detail="No connected accounts found for this user")

        active_accounts = [
            account for account in accounts
            if account.get("is_active") and account.get("account_id")
        ]

        if len(active_accounts) == 1:
            resolved_account_id = str(active_accounts[0]["account_id"])
            logger.info(
                "Resolved connected account user_id=%s provided_account_id=%s resolved_account_id=%s strategy=active",
                user_id,
                account_id,
                resolved_account_id,
            )
            return resolved_account_id

        if len(active_accounts) > 1:
            logger.warning(
                "Failed to resolve connected account user_id=%s provided_account_id=%s reason=multiple_active_accounts",
                user_id,
                account_id,
            )
            raise HTTPException(
                status_code=409,
                detail="Multiple active connected accounts found for this user. Provide account_id explicitly.",
            )

        if len(accounts) == 1 and accounts[0].get("account_id"):
            resolved_account_id = str(accounts[0]["account_id"])
            logger.info(
                "Resolved connected account user_id=%s provided_account_id=%s resolved_account_id=%s strategy=single_connected_account",
                user_id,
                account_id,
                resolved_account_id,
            )
            return resolved_account_id

        logger.warning(
            "Failed to resolve connected account user_id=%s provided_account_id=%s reason=no_active_account_multiple_connected",
            user_id,
            account_id,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "Multiple connected accounts found for this user, but no active account is configured. "
                "Provide account_id explicitly or set an active account."
            ),
        )
