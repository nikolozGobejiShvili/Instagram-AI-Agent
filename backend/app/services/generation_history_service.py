import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


logger = logging.getLogger(__name__)


class GenerationHistoryService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "generation_history.json"

    def _load_items(self) -> list[dict]:
        if not self.data_file.exists():
            return []

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_items(self, items: list[dict]) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _normalize_text_preview(self, value: str | None, limit: int) -> str | None:
        if value is None:
            return None

        normalized_value = " ".join(value.split()).strip()
        if len(normalized_value) <= limit:
            return normalized_value

        return f"{normalized_value[: limit - 3]}..."

    def _sanitize_payload(self, payload: dict) -> dict:
        sanitized_payload = dict(payload)
        sanitized_payload["message_preview"] = self._normalize_text_preview(
            sanitized_payload.get("message_preview"),
            220,
        ) or ""
        sanitized_payload["response_preview"] = self._normalize_text_preview(
            sanitized_payload.get("response_preview"),
            320,
        )
        sanitized_payload["error_message"] = self._normalize_text_preview(
            sanitized_payload.get("error_message"),
            320,
        )

        return sanitized_payload

    def save_item(self, payload: dict) -> dict:
        items = self._load_items()
        sanitized_payload = self._sanitize_payload(payload)
        item = {
            "log_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **sanitized_payload,
        }

        items.append(item)
        self._save_items(items)
        logger.info(
            "Saved generation history log_id=%s user_id=%s account_id=%s task_type=%s status=%s",
            item.get("log_id"),
            item.get("user_id"),
            item.get("account_id"),
            item.get("task_type"),
            item.get("status"),
        )

        return item

    def list_items(self) -> list[dict]:
        return list(self._load_items())

    def get_latest_item(
        self,
        *,
        user_id: str | None = None,
        task_type: str | None = None,
        account_id: str | None = None,
    ) -> dict | None:
        items = self._load_items()
        filtered_items = []
        for item in items:
            if user_id and item.get("user_id") != user_id:
                continue
            if task_type and item.get("task_type") != task_type:
                continue
            if account_id and item.get("account_id") != account_id:
                continue
            filtered_items.append(item)

        if not filtered_items:
            return None

        filtered_items.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return filtered_items[0]

    def get_history_by_user(self, user_id: str) -> list[dict]:
        items = self._load_items()
        filtered_items = [item for item in items if item.get("user_id") == user_id]

        return sorted(
            filtered_items,
            key=lambda item: item.get("timestamp", ""),
            reverse=True,
        )
