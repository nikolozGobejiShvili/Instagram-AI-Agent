from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MaintenanceJobRunResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "job_name": "monthly_usage_reset",
                    "started_at": "2026-04-22T08:00:00+00:00",
                    "finished_at": "2026-04-22T08:00:01+00:00",
                    "status": "success",
                    "summary": {
                        "usage_month": "2026-04",
                        "total_users_checked": 12,
                        "total_users_reset": 12,
                        "total_failures": 0,
                    },
                },
                {
                    "job_name": "context_refresh",
                    "started_at": "2026-04-22T08:10:00+00:00",
                    "finished_at": "2026-04-22T08:10:12+00:00",
                    "status": "success",
                    "summary": {
                        "total_users_checked": 5,
                        "total_accounts_checked": 7,
                        "total_accounts_refreshed": 3,
                        "total_accounts_skipped_fresh": 2,
                        "total_accounts_skipped_not_connected": 1,
                        "total_connections_requiring_reconnect": 1,
                        "total_failures": 0,
                    },
                },
            ]
        },
    )

    job_name: str
    started_at: str
    finished_at: str
    status: str
    summary: dict[str, Any] = Field(default_factory=dict)


class MaintenanceStatusResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "last_monthly_usage_reset_at": "2026-04-22T08:00:01+00:00",
                "last_context_freshness_scan_at": "2026-04-22T08:05:00+00:00",
                "last_context_refresh_at": "2026-04-22T08:10:12+00:00",
                "last_connection_health_scan_at": "2026-04-22T08:15:00+00:00",
                "last_job_name": "connection_health_scan",
                "last_job_status": "success",
                "last_job_started_at": "2026-04-22T08:14:58+00:00",
                "last_job_finished_at": "2026-04-22T08:15:00+00:00",
                "last_job_summary": {
                    "total_users_checked": 5,
                    "total_accounts_checked": 7,
                    "total_connected": 5,
                    "total_connections_requiring_reconnect": 1,
                    "total_stale": 1,
                    "total_failed": 0,
                    "total_failures": 0,
                },
            }
        },
    )

    last_monthly_usage_reset_at: str | None = None
    last_context_freshness_scan_at: str | None = None
    last_context_refresh_at: str | None = None
    last_connection_health_scan_at: str | None = None
    last_job_name: str | None = None
    last_job_status: str | None = None
    last_job_started_at: str | None = None
    last_job_finished_at: str | None = None
    last_job_summary: dict[str, Any] | None = None
