"""Usage accounting must be exact under concurrency.

This is the metered quantity customers pay against. The audit of 2026-08-04
measured 50 concurrent increments producing a final count of 12 -- 39 lost --
plus 28 JSONDecodeErrors from readers observing a half-written file.

Both failure modes are covered here:

* ``test_concurrent_increments_are_not_lost`` fails on lost updates.
* ``test_store_is_never_observed_truncated`` fails if a concurrent reader ever
  sees the store in an unreadable state.
"""
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.billing_service import BillingService  # noqa: E402

CONCURRENCY = 50
USER = "usage-integrity-user"


@pytest.fixture()
def service(tmp_path):
    return BillingService(db_path=tmp_path / "billing.sqlite3")


def _run_concurrently(fn, count):
    errors = []

    def wrapper():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=wrapper) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_increments_are_not_lost(service):
    """N concurrent generations must be charged exactly N times."""
    service.increment_generation_usage(USER)  # establish the record
    baseline = service.get_plan(USER)["monthly_generation_used"]

    errors = _run_concurrently(lambda: service.increment_generation_usage(USER), CONCURRENCY)
    assert not errors, f"increments raised: {errors[:3]}"

    final = service.get_plan(USER)["monthly_generation_used"]
    assert final == baseline + CONCURRENCY, (
        f"lost {baseline + CONCURRENCY - final} of {CONCURRENCY} increments "
        f"(expected {baseline + CONCURRENCY}, got {final}) -- "
        "usage that customers were charged for went unrecorded"
    )


def test_store_is_never_observed_truncated(service):
    """A reader concurrent with writers must never see a corrupt store."""
    service.increment_generation_usage(USER)

    read_errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                service.get_plan(USER)
            except Exception as exc:  # noqa: BLE001
                read_errors.append(exc)

    reader_threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in reader_threads:
        t.start()
    try:
        write_errors = _run_concurrently(
            lambda: service.increment_generation_usage(USER), CONCURRENCY
        )
    finally:
        stop.set()
        for t in reader_threads:
            t.join(timeout=5)

    assert not write_errors, f"writers raised: {write_errors[:3]}"
    assert not read_errors, (
        f"{len(read_errors)} concurrent reads failed, e.g. {read_errors[0]!r} -- "
        "the store was observed in an unreadable state"
    )


def _write_legacy_store(path, *, usage_month, used=55):
    import json

    path.write_text(
        json.dumps(
            {
                "legacy-user": {
                    "user_id": "legacy-user",
                    "current_plan": "pro",
                    "plan_status": "active",
                    "monthly_generation_limit": 400,
                    "monthly_generation_used": used,
                    "connected_account_limit": 2,
                    "tracked_accounts_limit": 10,
                    "usage_month": usage_month,
                }
            }
        ),
        encoding="utf-8",
    )


def test_existing_json_records_are_migrated(tmp_path):
    """An existing billing_plans.json must carry over, not be silently dropped."""
    from datetime import datetime, timezone

    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    legacy = tmp_path / "billing_plans.json"
    _write_legacy_store(legacy, usage_month=current_month)

    service = BillingService(db_path=tmp_path / "billing.sqlite3", legacy_json_file=legacy)
    plan = service.get_plan("legacy-user")

    assert plan["current_plan"] == "pro"
    assert plan["monthly_generation_used"] == 55, "usage recorded in the current month must survive migration"
    assert plan["connected_account_limit"] == 2
    assert legacy.exists(), "the legacy store must be left in place as a backup"


def test_migrated_usage_resets_when_the_month_has_rolled_over(tmp_path):
    """A count from a previous month must not be charged against the new one."""
    legacy = tmp_path / "billing_plans.json"
    _write_legacy_store(legacy, usage_month="2020-01")

    service = BillingService(db_path=tmp_path / "billing.sqlite3", legacy_json_file=legacy)
    plan = service.get_plan("legacy-user")

    assert plan["current_plan"] == "pro"
    assert plan["monthly_generation_used"] == 0


def test_migration_is_idempotent(tmp_path):
    """Re-running migration must not duplicate or clobber live usage."""
    from datetime import datetime, timezone

    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    legacy = tmp_path / "billing_plans.json"
    db = tmp_path / "billing.sqlite3"
    _write_legacy_store(legacy, usage_month=current_month)

    first = BillingService(db_path=db, legacy_json_file=legacy)
    first.increment_generation_usage("legacy-user")
    used_after_increment = first.get_plan("legacy-user")["monthly_generation_used"]
    assert used_after_increment == 56

    # A second process starting up must not re-import and reset the counter.
    second = BillingService(db_path=db, legacy_json_file=legacy)
    assert second.get_plan("legacy-user")["monthly_generation_used"] == 56
    assert second.list_user_ids() == ["legacy-user"]
