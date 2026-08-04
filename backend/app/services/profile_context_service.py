import json
from pathlib import Path


class ProfileContextService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "profile_contexts.json"

    def _load_profiles(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_profiles(self, profiles: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

    def get_stored_context(self, account_id: str) -> dict | None:
        profiles = self._load_profiles()
        return profiles.get(account_id)

    def get_context(self, account_id: str) -> dict:
        profiles = self._load_profiles()

        default_context = {
            "account_id": account_id,
            "brand_name": "Unknown Brand",
            "niche": "unknown",
            "target_audience": "unknown",
            "brand_voice": "not clearly defined yet",
            "bio": "",
            "content_focus": [],
            "strengths": [],
            "weak_points": [],
        }

        return profiles.get(account_id, default_context)

    def save_context(self, payload: dict) -> dict:
        profiles = self._load_profiles()
        account_id = payload["account_id"]

        profiles[account_id] = payload
        self._save_profiles(profiles)

        return profiles[account_id]
