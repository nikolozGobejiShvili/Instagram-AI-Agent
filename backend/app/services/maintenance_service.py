import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from app.services.billing_service import BillingService
from app.services.connected_accounts_service import ConnectedAccountsService
from app.services.instagram_connection_service import InstagramConnectionService
from app.services.instagram_context_sync_metadata_service import InstagramContextSyncMetadataService
from app.services.instagram_context_sync_service import InstagramContextSyncService


logger = logging.getLogger(__name__)


class MaintenanceService:
    def __init__(self):
        self.data_file = Path(__file__).resolve().parent.parent / "data" / "maintenance_status.json"
        self.billing_service = BillingService()
        self.connected_accounts_service = ConnectedAccountsService()
        self.instagram_connection_service = InstagramConnectionService()
        self.instagram_context_sync_service = InstagramContextSyncService()
        self.instagram_context_sync_metadata_service = InstagramContextSyncMetadataService()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _default_status(self) -> dict:
        return {
            "last_monthly_usage_reset_at": None,
            "last_context_freshness_scan_at": None,
            "last_context_refresh_at": None,
            "last_connection_health_scan_at": None,
            "last_job_name": None,
            "last_job_status": None,
            "last_job_started_at": None,
            "last_job_finished_at": None,
            "last_job_summary": None,
        }

    def _load_status(self) -> dict:
        if not self.data_file.exists():
            return self._default_status()

        with open(self.data_file, "r", encoding="utf-8") as f:
            stored_status = json.load(f)

        return {
            **self._default_status(),
            **stored_status,
        }

    def _save_status(self, status: dict) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _build_failure_sample(self, *, user_id: str | None, account_id: str | None, error: object) -> dict:
        return {
            "user_id": user_id,
            "account_id": account_id,
            "error": " ".join(str(error).split())[:240],
        }

    def _finalize_job(self, *, job_name: str, started_at: str, status: str, summary: dict) -> dict:
        finished_at = self._now_iso()
        timestamp_fields = {
            "monthly_usage_reset": "last_monthly_usage_reset_at",
            "context_freshness_scan": "last_context_freshness_scan_at",
            "context_refresh": "last_context_refresh_at",
            "connection_health_scan": "last_connection_health_scan_at",
        }

        stored_status = self._load_status()
        timestamp_field = timestamp_fields.get(job_name)
        if timestamp_field:
            stored_status[timestamp_field] = finished_at

        stored_status["last_job_name"] = job_name
        stored_status["last_job_status"] = status
        stored_status["last_job_started_at"] = started_at
        stored_status["last_job_finished_at"] = finished_at
        stored_status["last_job_summary"] = summary
        self._save_status(stored_status)

        logger.info(
            "Maintenance job completed job_name=%s status=%s summary=%s",
            job_name,
            status,
            summary,
        )
        return {
            "job_name": job_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "summary": summary,
        }

    def get_status(self) -> dict:
        return self._load_status()

    def run_monthly_usage_reset(self) -> dict:
        started_at = self._now_iso()
        logger.info("Starting maintenance job job_name=monthly_usage_reset")

        user_ids = self.billing_service.list_user_ids()
        total_users_reset = 0
        failures = []
        usage_month = datetime.now(timezone.utc).strftime("%Y-%m")

        for user_id in user_ids:
            try:
                self.billing_service.reset_monthly_usage(user_id)
                total_users_reset += 1
            except HTTPException as exc:
                failures.append(self._build_failure_sample(user_id=user_id, account_id=None, error=exc.detail))
            except Exception as exc:
                failures.append(self._build_failure_sample(user_id=user_id, account_id=None, error=exc))

        summary = {
            "usage_month": usage_month,
            "total_users_checked": len(user_ids),
            "total_users_reset": total_users_reset,
            "total_failures": len(failures),
        }
        if failures:
            summary["failure_samples"] = failures[:10]

        return self._finalize_job(
            job_name="monthly_usage_reset",
            started_at=started_at,
            status="success" if not failures else "partial_success",
            summary=summary,
        )

    def _list_connected_accounts_for_scan(self) -> list[dict]:
        connected_accounts = []
        for account in self.connected_accounts_service.list_all_accounts():
            user_id = str(account.get("user_id") or "")
            account_id = str(account.get("account_id") or "")
            if not user_id or not account_id:
                continue

            latest_connection = self.instagram_connection_service.get_latest_connection_by_account(user_id, account_id)
            if not latest_connection or latest_connection.get("status") != "connected":
                continue

            connected_accounts.append({
                **account,
                "connection_status": latest_connection.get("connection_status"),
                "requires_reconnect": bool(latest_connection.get("requires_reconnect")),
            })

        return connected_accounts

    def run_context_freshness_scan(self) -> dict:
        started_at = self._now_iso()
        logger.info("Starting maintenance job job_name=context_freshness_scan")

        accounts = self._list_connected_accounts_for_scan()
        failures = []
        total_fresh = 0
        total_stale = 0
        total_missing = 0

        for account in accounts:
            user_id = str(account.get("user_id") or "")
            account_id = str(account.get("account_id") or "")
            try:
                freshness = self.instagram_context_sync_service.get_context_freshness(account_id)
                if not freshness.get("has_complete_context"):
                    context_status = "missing"
                    total_missing += 1
                elif freshness.get("context_was_fresh"):
                    context_status = "fresh"
                    total_fresh += 1
                else:
                    context_status = "stale"
                    total_stale += 1

                self.instagram_context_sync_metadata_service.mark_freshness(
                    account_id,
                    context_status=context_status,
                    stale_reasons=freshness.get("stale_reasons", []),
                )
            except Exception as exc:
                failures.append(self._build_failure_sample(user_id=user_id, account_id=account_id, error=exc))

        summary = {
            "total_users_checked": len({account["user_id"] for account in accounts}),
            "total_accounts_checked": len(accounts),
            "total_accounts_fresh": total_fresh,
            "total_accounts_marked_stale": total_stale,
            "total_accounts_missing": total_missing,
            "total_failures": len(failures),
        }
        if failures:
            summary["failure_samples"] = failures[:10]

        return self._finalize_job(
            job_name="context_freshness_scan",
            started_at=started_at,
            status="success" if not failures else "partial_success",
            summary=summary,
        )

    def _list_effectively_active_accounts(self) -> list[dict]:
        grouped_accounts: dict[str, list[dict]] = {}
        for account in self.connected_accounts_service.list_all_accounts():
            user_id = str(account.get("user_id") or "")
            account_id = str(account.get("account_id") or "")
            if not user_id or not account_id:
                continue
            grouped_accounts.setdefault(user_id, []).append(account)

        active_accounts = []
        for user_id, accounts in grouped_accounts.items():
            explicitly_active = [account for account in accounts if account.get("is_active")]
            if len(explicitly_active) == 1:
                active_accounts.append(explicitly_active[0])
                continue

            if not explicitly_active and len(accounts) == 1:
                active_accounts.append(accounts[0])

        return active_accounts

    def run_context_refresh(self) -> dict:
        started_at = self._now_iso()
        logger.info("Starting maintenance job job_name=context_refresh")

        accounts = self._list_effectively_active_accounts()
        failures = []
        total_accounts_refreshed = 0
        total_accounts_skipped_fresh = 0
        total_accounts_skipped_not_connected = 0
        total_connections_requiring_reconnect = 0

        for account in accounts:
            user_id = str(account.get("user_id") or "")
            account_id = str(account.get("account_id") or "")
            latest_connection = self.instagram_connection_service.get_latest_connection_by_account(user_id, account_id)

            if not latest_connection or latest_connection.get("status") != "connected":
                total_accounts_skipped_not_connected += 1
                continue

            if latest_connection.get("requires_reconnect") or latest_connection.get("connection_status") == "reconnect_required":
                total_connections_requiring_reconnect += 1
                continue

            try:
                freshness = self.instagram_context_sync_service.get_context_freshness(account_id)
                if not freshness.get("sync_required"):
                    total_accounts_skipped_fresh += 1
                    self.instagram_context_sync_metadata_service.mark_freshness(
                        account_id,
                        context_status="fresh",
                        stale_reasons=freshness.get("stale_reasons", []),
                    )
                    continue

                self.instagram_context_sync_service.sync(user_id, account_id)
                total_accounts_refreshed += 1
            except HTTPException as exc:
                if self.instagram_connection_service.is_reconnect_required_exception(exc):
                    total_connections_requiring_reconnect += 1
                else:
                    failures.append(self._build_failure_sample(user_id=user_id, account_id=account_id, error=exc.detail))
            except Exception as exc:
                failures.append(self._build_failure_sample(user_id=user_id, account_id=account_id, error=exc))

        summary = {
            "total_users_checked": len({account["user_id"] for account in accounts}),
            "total_accounts_checked": len(accounts),
            "total_accounts_refreshed": total_accounts_refreshed,
            "total_accounts_skipped_fresh": total_accounts_skipped_fresh,
            "total_accounts_skipped_not_connected": total_accounts_skipped_not_connected,
            "total_connections_requiring_reconnect": total_connections_requiring_reconnect,
            "total_failures": len(failures),
        }
        if failures:
            summary["failure_samples"] = failures[:10]

        return self._finalize_job(
            job_name="context_refresh",
            started_at=started_at,
            status="success" if not failures else "partial_success",
            summary=summary,
        )

    def run_connection_health_scan(self) -> dict:
        started_at = self._now_iso()
        logger.info("Starting maintenance job job_name=connection_health_scan")

        user_ids = self.instagram_connection_service.list_connection_user_ids()
        failures = []
        total_accounts_checked = 0
        total_connected = 0
        total_stale = 0
        total_failed = 0
        total_disconnected = 0
        total_connections_requiring_reconnect = 0

        for user_id in user_ids:
            try:
                health_payload = self.instagram_connection_service.get_connection_health(user_id)
            except Exception as exc:
                failures.append(self._build_failure_sample(user_id=user_id, account_id=None, error=exc))
                continue

            for connection in health_payload.get("connections", []):
                total_accounts_checked += 1
                connection_status = str(connection.get("connection_status") or "")
                if connection_status == "connected":
                    total_connected += 1
                elif connection_status == "stale":
                    total_stale += 1
                elif connection_status == "failed":
                    total_failed += 1
                elif connection_status == "disconnected":
                    total_disconnected += 1
                elif connection_status == "reconnect_required":
                    total_connections_requiring_reconnect += 1

        summary = {
            "total_users_checked": len(user_ids),
            "total_accounts_checked": total_accounts_checked,
            "total_connected": total_connected,
            "total_connections_requiring_reconnect": total_connections_requiring_reconnect,
            "total_stale": total_stale,
            "total_failed": total_failed,
            "total_disconnected": total_disconnected,
            "total_failures": len(failures),
        }
        if failures:
            summary["failure_samples"] = failures[:10]

        return self._finalize_job(
            job_name="connection_health_scan",
            started_at=started_at,
            status="success" if not failures else "partial_success",
            summary=summary,
        )
