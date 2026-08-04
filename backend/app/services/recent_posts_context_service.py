import json
from pathlib import Path


class RecentPostsContextService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "recent_posts_contexts.json"

    def _load_items(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_items(self, items: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def get_stored_context(self, account_id: str) -> dict | None:
        items = self._load_items()
        return items.get(account_id)

    def get_context(self, account_id: str) -> dict:
        items = self._load_items()

        default_context = {
            "account_id": account_id,
            "posts": []
        }

        return items.get(account_id, default_context)

    def save_context(self, payload: dict) -> dict:
        items = self._load_items()
        account_id = payload["account_id"]

        items[account_id] = payload
        self._save_items(items)

        return items[account_id]
