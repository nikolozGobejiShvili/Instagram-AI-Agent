import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException

from app.services.agent_response_formatter_service import MAX_CAROUSEL_SLIDES
from app.services.connected_accounts_service import ConnectedAccountsService


logger = logging.getLogger(__name__)


class BillingService:
    """Plan entitlements and metered usage.

    Storage is SQLite rather than a JSON file. This is not a style preference:
    usage is the quantity customers are charged against, and the previous
    read-modify-rewrite over a whole JSON document lost 39 of 50 concurrent
    increments and let readers observe the file mid-write (audit, 2026-08-04).
    ``increment_generation_usage`` is now a single atomic statement, so a
    generation is either charged exactly once or not at all.

    The database location is configurable via ``BILLING_DB_PATH`` because the
    default path lives inside the container filesystem, which is ephemeral on
    most hosts -- point it at a mounted volume in production.
    """

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

    # A generation is not a fixed unit of cost. `chat` is one Sonnet call at
    # `effort=low`; `carousel` is one Sonnet call *plus* an image generation per
    # slide, so it costs roughly an order of magnitude more to serve. While every
    # task was charged a single unit, the customers using the headline feature
    # were exactly the ones the plan lost money on. Weights live in one table so
    # pricing stays a single edit rather than a change spread across call sites.
    GENERATION_COST = {
        "carousel": 5,
        "content_plan": 3,
        "profile_audit": 2,
        "link_analysis": 2,
    }
    DEFAULT_GENERATION_COST = 1

    # Trial and creator include `carousel` deliberately. A trial that cannot run
    # the feature the subscription is sold on demonstrates nothing, and a first
    # paid tier without it would mean paying to lose access. The ladder is drawn
    # with credits, slide count and account count instead.
    PLAN_DEFAULTS = {
        "trial": {
            "connected_account_limit": 1,
            "tracked_accounts_limit": 0,
            "monthly_generation_limit": 15,
            "carousel_slide_limit": 5,
            "allowed_task_types": [
                "chat",
                "reel_idea",
                "reel_feedback",
                "caption",
                "performance_summary",
                "profile_audit",
                "carousel",
            ],
        },
        "creator": {
            "connected_account_limit": 1,
            "tracked_accounts_limit": 3,
            "monthly_generation_limit": 120,
            "carousel_slide_limit": 7,
            "allowed_task_types": [
                "chat",
                "reel_idea",
                "reel_script",
                "reel_feedback",
                "caption",
                "performance_summary",
                "profile_audit",
                "carousel",
            ],
        },
        "pro": {
            "connected_account_limit": 2,
            "tracked_accounts_limit": 10,
            "monthly_generation_limit": 400,
            "carousel_slide_limit": 10,
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
            "carousel_slide_limit": MAX_CAROUSEL_SLIDES,
            # Copied, not aliased. Referencing SUPPORTED_TASK_TYPES directly made
            # this the *same list object* as the class constant, so any in-place
            # mutation of a plan's task types would have rewritten the constant
            # for every plan.
            "allowed_task_types": list(SUPPORTED_TASK_TYPES),
        },
    }

    TRIAL_DURATION_DAYS = 14

    @classmethod
    def generation_cost(cls, task_type: str | None) -> int:
        """Credits one run of ``task_type`` consumes."""
        return cls.GENERATION_COST.get(task_type or "", cls.DEFAULT_GENERATION_COST)

    @classmethod
    def plan_catalogue(cls) -> dict:
        """Everything a storefront needs to render pricing, without a customer.

        The tier table only existed inside this module, so a site selling these
        subscriptions had no way to ask what it was selling and would have had to
        hard-code a second copy -- one that silently drifts the first time a limit
        changes here.
        """
        return {
            "plans": [
                {
                    "plan_id": plan_id,
                    "connected_account_limit": defaults["connected_account_limit"],
                    "tracked_accounts_limit": defaults["tracked_accounts_limit"],
                    "monthly_generation_limit": defaults["monthly_generation_limit"],
                    "carousel_slide_limit": defaults["carousel_slide_limit"],
                    "allowed_task_types": list(defaults["allowed_task_types"]),
                    "trial_duration_days": cls.TRIAL_DURATION_DAYS if plan_id == "trial" else None,
                }
                for plan_id, defaults in cls.PLAN_DEFAULTS.items()
            ],
            # Published so a client can show the cost of an action before the
            # customer commits to it, rather than discovering it in the balance.
            "generation_costs": {
                task_type: cls.generation_cost(task_type) for task_type in cls.SUPPORTED_TASK_TYPES
            },
            "default_generation_cost": cls.DEFAULT_GENERATION_COST,
        }

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS billing_plans (
            user_id                 TEXT PRIMARY KEY,
            current_plan            TEXT NOT NULL,
            plan_status             TEXT NOT NULL,
            plan_started_at         TEXT,
            plan_expires_at         TEXT,
            monthly_generation_used INTEGER NOT NULL DEFAULT 0,
            usage_month             TEXT NOT NULL
        )
    """

    def __init__(self, db_path: Path | str | None = None, legacy_json_file: Path | str | None = None):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_path = Path(db_path or os.getenv("BILLING_DB_PATH") or (data_dir / "billing.sqlite3"))
        self.legacy_json_file = Path(legacy_json_file) if legacy_json_file is not None else (data_dir / "billing_plans.json")
        self.connected_accounts_service = ConnectedAccountsService()
        self._initialise_storage()

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None puts transaction control in our hands, so writes
        # can use BEGIN IMMEDIATE and take the write lock up front instead of
        # upgrading mid-transaction and risking SQLITE_BUSY.
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialise_storage(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
        self._migrate_legacy_json()

    def _migrate_legacy_json(self) -> None:
        """Import records from the pre-SQLite JSON store, once.

        ``INSERT OR IGNORE`` makes this idempotent and safe to run from several
        workers starting at the same time. The JSON file is left untouched so it
        remains available as a backup.
        """
        if not self.legacy_json_file.exists():
            return

        try:
            with open(self.legacy_json_file, "r", encoding="utf-8") as f:
                legacy_items = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read legacy billing store %s: %s", self.legacy_json_file, exc)
            return

        if not isinstance(legacy_items, dict) or not legacy_items:
            return

        rows = []
        for user_id, record in legacy_items.items():
            if not isinstance(record, dict):
                continue
            current_plan = record.get("current_plan", "trial")
            if current_plan not in self.PLAN_DEFAULTS:
                logger.warning("Skipping legacy record for %s: unknown plan %r", user_id, current_plan)
                continue
            rows.append((
                user_id,
                current_plan,
                record.get("plan_status") or self._default_plan_status(current_plan),
                record.get("plan_started_at"),
                record.get("plan_expires_at"),
                max(int(record.get("monthly_generation_used") or 0), 0),
                record.get("usage_month") or self._usage_month(),
            ))

        if not rows:
            return

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.executemany(
                "INSERT OR IGNORE INTO billing_plans "
                "(user_id, current_plan, plan_status, plan_started_at, plan_expires_at, "
                " monthly_generation_used, usage_month) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            imported = cursor.rowcount
            conn.execute("COMMIT")

        if imported:
            logger.info("Migrated %s billing record(s) from %s", imported, self.legacy_json_file)

    def _row_to_record(self, row: sqlite3.Row) -> dict:
        """Rehydrate a stored row into the full plan record shape.

        Plan limits and allowed task types are always derived from
        ``PLAN_DEFAULTS`` rather than stored, so a change to the tier table takes
        effect immediately for every existing customer.
        """
        current_plan = row["current_plan"]
        defaults = self.PLAN_DEFAULTS[current_plan]
        return {
            "user_id": row["user_id"],
            "current_plan": current_plan,
            "plan_status": row["plan_status"],
            "plan_started_at": row["plan_started_at"],
            "plan_expires_at": row["plan_expires_at"],
            "monthly_generation_limit": defaults["monthly_generation_limit"],
            "monthly_generation_used": int(row["monthly_generation_used"]),
            "connected_account_limit": defaults["connected_account_limit"],
            "tracked_accounts_limit": defaults["tracked_accounts_limit"],
            "carousel_slide_limit": defaults["carousel_slide_limit"],
            "allowed_task_types": list(defaults["allowed_task_types"]),
            "usage_month": row["usage_month"],
        }

    def _fetch_row(self, conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
        return conn.execute("SELECT * FROM billing_plans WHERE user_id = ?", (user_id,)).fetchone()

    # ------------------------------------------------------------------
    # helpers retained from the JSON implementation
    # ------------------------------------------------------------------
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

    def _new_plan_row(
        self,
        *,
        user_id: str,
        current_plan: str,
        plan_status: str | None = None,
        plan_started_at: str | None = None,
        plan_expires_at: str | None = None,
        monthly_generation_used: int = 0,
        usage_month: str | None = None,
    ) -> tuple:
        self._validate_plan(current_plan)
        now = self._now()
        if plan_started_at and self._parse_datetime(plan_started_at) is None:
            raise HTTPException(status_code=400, detail="plan_started_at must be a valid ISO datetime")
        if plan_expires_at and self._parse_datetime(plan_expires_at) is None:
            raise HTTPException(status_code=400, detail="plan_expires_at must be a valid ISO datetime")

        started_at_dt = self._parse_datetime(plan_started_at) or now
        resolved_expires_at = plan_expires_at
        if not resolved_expires_at and current_plan == "trial":
            resolved_expires_at = (started_at_dt + timedelta(days=self.TRIAL_DURATION_DAYS)).isoformat()

        return (
            user_id,
            current_plan,
            plan_status or self._default_plan_status(current_plan),
            started_at_dt.isoformat(),
            resolved_expires_at,
            max(int(monthly_generation_used), 0),
            usage_month or self._usage_month(now),
        )

    def _normalize_record(self, user_id: str, record: dict, *, save: bool = True) -> dict:
        """Bring a record in line with the current tier table and clock.

        Same rules as before: roll the usage month over, expire a lapsed plan,
        and re-derive limits. Limits and task types are derived on read, so only
        the month rollover and expiry need persisting.
        """
        self._validate_plan(record.get("current_plan", "trial"))
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

        expires_at = self._parse_datetime(record.get("plan_expires_at"))
        if expires_at and expires_at < now and record.get("plan_status") in {"active", "trial"}:
            record["plan_status"] = "expired"
            changed = True

        if changed and save:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE billing_plans SET plan_status = ?, plan_started_at = ?, "
                    "plan_expires_at = ?, monthly_generation_used = ?, usage_month = ? "
                    "WHERE user_id = ?",
                    (
                        record["plan_status"],
                        record["plan_started_at"],
                        record.get("plan_expires_at"),
                        int(record["monthly_generation_used"]),
                        record["usage_month"],
                        user_id,
                    ),
                )

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
            "carousel_slide_limit": int(record["carousel_slide_limit"]),
            "allowed_task_types": list(record.get("allowed_task_types", [])),
            "usage_month": str(record.get("usage_month") or self._usage_month()),
        }

    def _get_or_create_record(self, user_id: str) -> dict:
        with self._connect() as conn:
            row = self._fetch_row(conn, user_id)
            if row is None:
                logger.info("Auto-provisioning trial plan for user_id=%s", user_id)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT OR IGNORE INTO billing_plans "
                    "(user_id, current_plan, plan_status, plan_started_at, plan_expires_at, "
                    " monthly_generation_used, usage_month) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._new_plan_row(user_id=user_id, current_plan="trial"),
                )
                conn.execute("COMMIT")
                row = self._fetch_row(conn, user_id)

        return self._normalize_record(user_id, self._row_to_record(row))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_plan(self, user_id: str) -> dict:
        record = self._get_or_create_record(user_id)
        return self._serialize_plan(record)

    def peek_plan(self, user_id: str) -> dict | None:
        with self._connect() as conn:
            row = self._fetch_row(conn, user_id)
        if row is None:
            return None

        normalized_record = self._normalize_record(user_id, self._row_to_record(row), save=False)
        return self._serialize_plan(normalized_record)

    def list_user_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM billing_plans ORDER BY user_id").fetchall()
        return [row["user_id"] for row in rows]

    def set_plan(self, user_id: str, payload: dict) -> dict:
        with self._connect() as conn:
            existing = self._fetch_row(conn, user_id)
            if existing is not None:
                normalized_existing = self._normalize_record(user_id, self._row_to_record(existing), save=False)
                monthly_generation_used = int(normalized_existing.get("monthly_generation_used", 0))
                usage_month = normalized_existing.get("usage_month")
            else:
                monthly_generation_used = 0
                usage_month = self._usage_month()

            values = self._new_plan_row(
                user_id=user_id,
                current_plan=payload["current_plan"],
                plan_status=payload.get("plan_status"),
                plan_started_at=payload.get("plan_started_at"),
                plan_expires_at=payload.get("plan_expires_at"),
                monthly_generation_used=monthly_generation_used,
                usage_month=usage_month,
            )
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO billing_plans "
                "(user_id, current_plan, plan_status, plan_started_at, plan_expires_at, "
                " monthly_generation_used, usage_month) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                " current_plan=excluded.current_plan, plan_status=excluded.plan_status, "
                " plan_started_at=excluded.plan_started_at, plan_expires_at=excluded.plan_expires_at, "
                " monthly_generation_used=excluded.monthly_generation_used, usage_month=excluded.usage_month",
                values,
            )
            conn.execute("COMMIT")

        logger.info(
            "Updated billing plan user_id=%s current_plan=%s plan_status=%s",
            user_id,
            values[1],
            values[2],
        )
        return self._serialize_plan(self._get_or_create_record(user_id))

    def reset_monthly_usage(self, user_id: str) -> dict:
        record = self._get_or_create_record(user_id)
        usage_month = self._usage_month()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE billing_plans SET monthly_generation_used = 0, usage_month = ? WHERE user_id = ?",
                (usage_month, user_id),
            )
            conn.execute("COMMIT")

        logger.info("Reset monthly generation usage user_id=%s usage_month=%s", user_id, usage_month)
        return {
            "user_id": user_id,
            "current_plan": record["current_plan"],
            "usage_month": usage_month,
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

    def _assert_generation_limit(self, plan: dict, task_type: str | None = None) -> None:
        """Refuse a task the remaining allowance cannot pay for.

        Comparing ``used >= limit`` was only correct while every task cost one
        credit. A customer holding two credits must not be able to start a
        carousel costing five: the charge lands *after* the work succeeds, so the
        overspend would only be discovered once it had already happened.
        """
        cost = self.generation_cost(task_type)
        used = int(plan["monthly_generation_used"])
        limit = int(plan["monthly_generation_limit"])
        if used + cost > limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Monthly generation limit reached for current plan "
                    f"({used}/{limit}); this request costs {cost}"
                ),
            )

    def enforce_agent_access(self, user_id: str, task_type: str) -> dict:
        plan = self.get_plan(user_id)
        self._assert_plan_active(plan)
        self._assert_task_allowed(plan, task_type)
        self._assert_generation_limit(plan, task_type)
        return plan

    def increment_generation_usage(self, user_id: str, task_type: str | None = None) -> dict:
        """Charge one run of ``task_type``, priced by ``GENERATION_COST``.

        The whole operation is one statement inside one transaction, so
        concurrent callers cannot lose each other's increments. The CASE also
        performs the month rollover atomically: without it, two requests either
        side of a month boundary could race and reset a counter that had already
        been incremented.

        ``task_type`` defaults to ``None`` -- charged at the base rate -- so a
        caller that cannot name the task still charges something rather than
        nothing.

        Entitlement is checked at accept time and charged here on success, which
        leaves a window where enough simultaneous requests can overshoot the
        limit. That is deliberate: the alternative is reserving credit up front
        and refunding failures, which bills customers for work they never
        received. The overshoot is bounded by concurrency; a silent charge for a
        failed generation would not be.
        """
        self._get_or_create_record(user_id)  # ensure the row exists
        usage_month = self._usage_month()
        cost = self.generation_cost(task_type)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE billing_plans SET "
                " monthly_generation_used = CASE WHEN usage_month = ? THEN monthly_generation_used + ? ELSE ? END, "
                " usage_month = ? "
                "WHERE user_id = ?",
                (usage_month, cost, cost, usage_month, user_id),
            )
            conn.execute("COMMIT")
            row = self._fetch_row(conn, user_id)

        record = self._row_to_record(row)
        logger.info(
            "Charged generation user_id=%s task_type=%s cost=%s used=%s limit=%s",
            user_id,
            task_type,
            cost,
            record["monthly_generation_used"],
            record["monthly_generation_limit"],
        )
        return self._serialize_plan(record)
