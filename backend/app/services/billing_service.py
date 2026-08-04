import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException

from app.services.connected_accounts_service import ConnectedAccountsService


logger = logging.getLogger(__name__)


class BillingService:
    SUPPORTED_TASK_TYPES = [
        "chat",
        "reel_idea",
        "reel_script",
        "reel_feedback",
        "caption",
        "carousel",
        "profile_audit",
        "content_plan",
        "link_analysis",
        "performance_summary",
    ]

    PLAN_DEFAULTS = {
        "trial": {
            "connected_account_limit": 1,
            "tracked_accounts_limit": 0,
            "monthly_generation_limit": 15,
            "allowed_task_types": ["chat", "reel_idea", "reel_feedback", "caption", "performance_summary", "profile_audit"],
        },
        "creator": {
            "connected_account_limit": 1,
            "tracked_accounts_limit": 3,
            "monthly_generation_limit": 120,
            "allowed_task_types": ["chat", "reel_idea", "reel_script", "reel_feedback", "caption", "performance_summary", "profile_audit"],
        },
        "pro": {
            "connected_account_limit": 2,
            "tracked_accounts_limit": 10,
            "monthly_generation_limit": 400,
            "allowed_task_types": [
                "chat",
                "reel_idea",
                "caption",
                "performance_summary",
                "profile_audit",
                "reel_script",
                "reel_feedback",
                "carousel",
                "content_plan",
                "link_analysis",
            ],
        },
        "agency": {
            "connected_account_limit": 10,
            "tracked_accounts_limit": 50,
            "monthly_generation_limit": 2000,
            "allowed_task_types": SUPPORTED_TASK_TYPES,
        },
    }

    TRIAL_DURATION_DAYS = 14

    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "billing_plans.json"
        self.connected_accounts_service = ConnectedAccountsService()

    def _load_items(self) -> dict:
        if not self.data_file.exists():
            return {}

        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_items(self, items: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _usage_month(self, value: datetime | None = None) -> str:
        resolved_value = value or self._now()
        return resolved_value.strftime("%Y-%m")

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _default_plan_status(self, current_plan: str) -> str:
        return "trial" if current_plan == "trial" else "active"

    def _validate_plan(self, current_plan: str) -> None:
        if current_plan not in self.PLAN_DEFAULTS:
            raise HTTPException(status_code=400, detail=f"Unsupported plan '{current_plan}'")

    def _build_plan_record(
        self,
        *,
        user_id: str,
        current_plan: str,
        plan_status: str | None = None,
        plan_started_at: str | None = None,
        plan_expires_at: str | None = None,
        monthly_generation_used: int = 0,
        usage_month: str | None = None,
    ) -> dict:
        self._validate_plan(current_plan)
        defaults = self.PLAN_DEFAULTS[current_plan]
        now = self._now()
        if plan_started_at and self._parse_datetime(plan_started_at) is None:
            raise HTTPException(status_code=400, detail="plan_started_at must be a valid ISO datetime")
        if plan_expires_at and self._parse_datetime(plan_expires_at) is None:
            raise HTTPException(status_code=400, detail="plan_expires_at must be a valid ISO datetime")

        started_at_dt = self._parse_datetime(plan_started_at) or now
        resolved_plan_status = plan_status or self._default_plan_status(current_plan)
        resolved_expires_at = plan_expires_at

        if not resolved_expires_at and current_plan == "trial":
            resolved_expires_at = (started_at_dt + timedelta(days=self.TRIAL_DURATION_DAYS)).isoformat()

        return {
            "user_id": user_id,
            "current_plan": current_plan,
            "plan_status": resolved_plan_status,
            "plan_started_at": started_at_dt.isoformat(),
            "plan_expires_at": resolved_expires_at,
            "monthly_generation_limit": defaults["monthly_generation_limit"],
            "monthly_generation_used": max(int(monthly_generation_used), 0),
            "connected_account_limit": defaults["connected_account_limit"],
            "tracked_accounts_limit": defaults["tracked_accounts_limit"],
            "allowed_task_types": list(defaults["allowed_task_types"]),
            "usage_month": usage_month or self._usage_month(now),
        }

    def _normalize_record(self, user_id: str, record: dict, *, save: bool = True) -> dict:
        self._validate_plan(record.get("current_plan", "trial"))
        defaults = self.PLAN_DEFAULTS[record["current_plan"]]
        now = self._now()
        current_usage_month = self._usage_month(now)
        changed = False

        if not record.get("plan_started_at"):
            record["plan_started_at"] = now.isoformat()
            changed = True

        if record.get("plan_status") not in {"active", "expired", "cancelled", "trial"}:
            record["plan_status"] = self._default_plan_status(record["current_plan"])
            changed = True

        if record["current_plan"] == "trial" and not record.get("plan_expires_at"):
            started_at = self._parse_datetime(record.get("plan_started_at")) or now
            record["plan_expires_at"] = (started_at + timedelta(days=self.TRIAL_DURATION_DAYS)).isoformat()
            changed = True

        if record.get("usage_month") != current_usage_month:
            record["usage_month"] = current_usage_month
            record["monthly_generation_used"] = 0
            changed = True

        if record.get("monthly_generation_limit") != defaults["monthly_generation_limit"]:
            record["monthly_generation_limit"] = defaults["monthly_generation_limit"]
            changed = True

        if record.get("connected_account_limit") != defaults["connected_account_limit"]:
            record["connected_account_limit"] = defaults["connected_account_limit"]
            changed = True

        if record.get("tracked_accounts_limit") != defaults["tracked_accounts_limit"]:
            record["tracked_accounts_limit"] = defaults["tracked_accounts_limit"]
            changed = True

        if record.get("allowed_task_types") != defaults["allowed_task_types"]:
            record["allowed_task_types"] = list(defaults["allowed_task_types"])
            changed = True

        expires_at = self._parse_datetime(record.get("plan_expires_at"))
        if expires_at and expires_at < now and record.get("plan_status") in {"active", "trial"}:
            record["plan_status"] = "expired"
            changed = True

        if changed and save:
            items = self._load_items()
            items[user_id] = record
            self._save_items(items)

        return record

    def _serialize_plan(self, record: dict) -> dict:
        remaining = max(int(record["monthly_generation_limit"]) - int(record["monthly_generation_used"]), 0)
        return {
            "user_id": record["user_id"],
            "current_plan": record["current_plan"],
            "plan_status": record["plan_status"],
            "plan_started_at": record["plan_started_at"],
            "plan_expires_at": record.get("plan_expires_at"),
            "monthly_generation_limit": int(record["monthly_generation_limit"]),
            "monthly_generation_used": int(record["monthly_generation_used"]),
            "monthly_generation_remaining": remaining,
            "connected_account_limit": int(record["connected_account_limit"]),
            "tracked_accounts_limit": int(record["tracked_accounts_limit"]),
            "allowed_task_types": list(record.get("allowed_task_types", [])),
            "usage_month": str(record.get("usage_month") or self._usage_month()),
        }

    def _get_or_create_record(self, user_id: str) -> dict:
        items = self._load_items()
        record = items.get(user_id)

        if record is None:
            logger.info("Auto-provisioning trial plan for user_id=%s", user_id)
            record = self._build_plan_record(user_id=user_id, current_plan="trial")
            items[user_id] = record
            self._save_items(items)

        return self._normalize_record(user_id, record)

    def get_plan(self, user_id: str) -> dict:
        record = self._get_or_create_record(user_id)
        return self._serialize_plan(record)

    def peek_plan(self, user_id: str) -> dict | None:
        items = self._load_items()
        record = items.get(user_id)
        if record is None:
            return None

        normalized_record = self._normalize_record(user_id, record, save=False)
        return self._serialize_plan(normalized_record)

    def list_user_ids(self) -> list[str]:
        return sorted(self._load_items().keys())

    def set_plan(self, user_id: str, payload: dict) -> dict:
        items = self._load_items()
        existing_record = items.get(user_id)
        normalized_existing = self._normalize_record(user_id, existing_record, save=False) if existing_record else None
        monthly_generation_used = int(normalized_existing.get("monthly_generation_used", 0)) if normalized_existing else 0
        usage_month = normalized_existing.get("usage_month") if normalized_existing else self._usage_month()

        record = self._build_plan_record(
            user_id=user_id,
            current_plan=payload["current_plan"],
            plan_status=payload.get("plan_status"),
            plan_started_at=payload.get("plan_started_at"),
            plan_expires_at=payload.get("plan_expires_at"),
            monthly_generation_used=monthly_generation_used,
            usage_month=usage_month,
        )
        items[user_id] = record
        self._save_items(items)
        logger.info(
            "Updated billing plan user_id=%s current_plan=%s plan_status=%s",
            user_id,
            record["current_plan"],
            record["plan_status"],
        )
        return self._serialize_plan(self._normalize_record(user_id, record))

    def reset_monthly_usage(self, user_id: str) -> dict:
        items = self._load_items()
        record = self._get_or_create_record(user_id)
        record["monthly_generation_used"] = 0
        record["usage_month"] = self._usage_month()
        items[user_id] = record
        self._save_items(items)
        logger.info("Reset monthly generation usage user_id=%s usage_month=%s", user_id, record["usage_month"])
        return {
            "user_id": user_id,
            "current_plan": record["current_plan"],
            "usage_month": record["usage_month"],
            "monthly_generation_used": 0,
            "monthly_generation_limit": int(record["monthly_generation_limit"]),
        }

    def assert_connected_account_limit(self, user_id: str, proposed_count: int | None = None) -> dict:
        plan = self.get_plan(user_id)
        self._assert_plan_active(plan)
        resolved_count = proposed_count if proposed_count is not None else len(
            self.connected_accounts_service.get_accounts(user_id).get("accounts", [])
        )

        if resolved_count > plan["connected_account_limit"]:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Connected account limit reached for current plan. "
                    f"Limit={plan['connected_account_limit']}, current={resolved_count}."
                ),
            )

        return plan

    def _assert_plan_active(self, plan: dict) -> None:
        if plan["plan_status"] == "expired":
            raise HTTPException(
                status_code=403,
                detail=f"Plan expired on {plan.get('plan_expires_at') or 'unknown date'}",
            )

        if plan["plan_status"] not in {"active", "trial"}:
            raise HTTPException(status_code=403, detail="No active plan is available for this user")

    def _assert_task_allowed(self, plan: dict, task_type: str) -> None:
        if task_type not in plan["allowed_task_types"]:
            raise HTTPException(
                status_code=403,
                detail=f"Task type '{task_type}' is not available on the current plan",
            )

    def _assert_generation_limit(self, plan: dict) -> None:
        if int(plan["monthly_generation_used"]) >= int(plan["monthly_generation_limit"]):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Monthly generation limit reached for current plan "
                    f"({plan['monthly_generation_used']}/{plan['monthly_generation_limit']})"
                ),
            )

    def enforce_agent_access(self, user_id: str, task_type: str) -> dict:
        plan = self.get_plan(user_id)
        self._assert_plan_active(plan)
        self._assert_task_allowed(plan, task_type)
        self._assert_generation_limit(plan)
        return plan

    def increment_generation_usage(self, user_id: str) -> dict:
        items = self._load_items()
        record = self._get_or_create_record(user_id)
        record["monthly_generation_used"] = int(record.get("monthly_generation_used", 0)) + 1
        items[user_id] = record
        self._save_items(items)
        logger.info(
            "Incremented monthly generation usage user_id=%s used=%s limit=%s",
            user_id,
            record["monthly_generation_used"],
            record["monthly_generation_limit"],
        )
        return self._serialize_plan(record)
