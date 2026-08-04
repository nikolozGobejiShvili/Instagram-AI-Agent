import json
import logging
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


class InstagramContextSyncMetadataService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "instagram_context_sync_metadata.json"

    def _load_items(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_items(self, items: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def get_metadata(self, account_id: str) -> dict | None:
        items = self._load_items()
        return items.get(account_id)

    def mark_synced(self, account_id: str, last_synced_at: str | None = None) -> dict:
        items = self._load_items()
        resolved_last_synced_at = last_synced_at or datetime.now(timezone.utc).isoformat()
        existing_metadata = items.get(account_id) or {}

        items[account_id] = {
            **existing_metadata,
            "account_id": account_id,
            "last_synced_at": resolved_last_synced_at,
            "context_status": "fresh",
            "last_freshness_checked_at": resolved_last_synced_at,
            "stale_reasons": [],
        }
        self._save_items(items)
        logger.info(
            "Updated Instagram context freshness metadata account_id=%s last_synced_at=%s",
            account_id,
            resolved_last_synced_at,
        )
        return items[account_id]

    def mark_freshness(
        self,
        account_id: str,
        *,
        context_status: str,
        stale_reasons: list[str] | None = None,
        checked_at: str | None = None,
    ) -> dict:
        items = self._load_items()
        resolved_checked_at = checked_at or datetime.now(timezone.utc).isoformat()
        existing_metadata = items.get(account_id) or {"account_id": account_id}

        items[account_id] = {
            **existing_metadata,
            "account_id": account_id,
            "context_status": context_status,
            "last_freshness_checked_at": resolved_checked_at,
            "stale_reasons": list(stale_reasons or []),
        }
        self._save_items(items)
        logger.info(
            "Updated Instagram context freshness state account_id=%s context_status=%s",
            account_id,
            context_status,
        )
        return items[account_id]
