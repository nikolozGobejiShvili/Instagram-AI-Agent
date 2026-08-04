"""Background generation jobs.

Every route handler in this service is ``sync def``, so a request occupies a
threadpool worker until it returns. Carousel generation -- one text call plus one
image generation per slide -- runs for tens of seconds, which as a synchronous
endpoint means gateway timeouts for the caller and threadpool exhaustion for
everyone else (audit, 2026-08-04, finding 5).

Work that cannot finish inside a request therefore goes through here: accepted
immediately, executed in the background, polled for completion.

Storage is SQLite, sharing the reasoning in ``billing_service``: job state has to
survive concurrent writers, and a JSON file cannot do that. ``JOBS_DB_PATH``
makes the location configurable because the default path is inside the container
filesystem, which is ephemeral on most hosts.

Scope: the worker runs in-process. That is correct for a single service
instance; running several replicas would need the claim to move to a real queue
or a leased row, which ``claim_next`` is deliberately shaped to allow.
"""
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

TERMINAL_STATUSES = {SUCCEEDED, FAILED}


class JobService:
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id     TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            kind       TEXT NOT NULL,
            status     TEXT NOT NULL,
            payload    TEXT NOT NULL,
            result     TEXT,
            error      TEXT,
            attempts   INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """
    _INDEX = "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)"

    def __init__(self, db_path: Path | str | None = None):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_path = Path(db_path or os.getenv("JOBS_DB_PATH") or (data_dir / "jobs.sqlite3"))
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
            conn.execute(self._INDEX)

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_job(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "user_id": row["user_id"],
            "kind": row["kind"],
            "status": row["status"],
            "payload": json.loads(row["payload"]),
            "result": json.loads(row["result"]) if row["result"] is not None else None,
            "error": row["error"],
            "attempts": int(row["attempts"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def enqueue(self, *, user_id: str, kind: str, payload: dict) -> dict:
        job_id = uuid.uuid4().hex
        now = self._now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO jobs (job_id, user_id, kind, status, payload, attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (job_id, user_id, kind, QUEUED, json.dumps(payload, ensure_ascii=False), now, now),
            )
            conn.execute("COMMIT")
        logger.info("Enqueued job job_id=%s kind=%s user_id=%s", job_id, kind, user_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row)

    def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def claim_next(self) -> dict | None:
        """Atomically take the oldest queued job.

        The select and the update share one ``BEGIN IMMEDIATE`` transaction, so
        the write lock is held across both. Without that, two workers could read
        the same row before either wrote, and the job would run twice -- which for
        a metered, paid generation means charging the customer twice.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at, rowid LIMIT 1",
                    (QUEUED,),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None

                conn.execute(
                    "UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ? WHERE job_id = ?",
                    (RUNNING, self._now(), row["job_id"]),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            claimed = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()

        return self._row_to_job(claimed)

    def _finish(self, job_id: str, *, status: str, result: dict | None, error: str | None) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                    self._now(),
                    job_id,
                ),
            )
            conn.execute("COMMIT")
        return self.get(job_id)

    def mark_succeeded(self, job_id: str, result: dict) -> dict:
        logger.info("Job succeeded job_id=%s", job_id)
        return self._finish(job_id, status=SUCCEEDED, result=result, error=None)

    def mark_failed(self, job_id: str, error: str) -> dict:
        logger.warning("Job failed job_id=%s error=%s", job_id, error)
        return self._finish(job_id, status=FAILED, result=None, error=error)

    def requeue_stale_running(self) -> int:
        """Return jobs left mid-flight by a crash to the queue.

        The worker runs in-process, so a restart abandons anything that was
        running. Without this they would sit in ``running`` forever and the
        caller would poll a job that nobody is working on.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                (QUEUED, self._now(), RUNNING),
            )
            recovered = cursor.rowcount
            conn.execute("COMMIT")

        if recovered:
            logger.warning("Requeued %s job(s) left running by a previous process", recovered)
        return recovered


class JobWorker:
    """Executes queued jobs.

    ``handlers`` maps a job kind to a callable taking the job payload and
    returning a JSON-serialisable result. Keeping execution behind this registry
    means the queue knows nothing about carousels, models or Instagram.
    """

    def __init__(self, job_service: JobService, handlers: dict, poll_interval: float = 1.0):
        self.job_service = job_service
        self.handlers = handlers
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        """Claim and execute one job. Returns False when the queue is empty."""
        job = self.job_service.claim_next()
        if job is None:
            return False

        handler = self.handlers.get(job["kind"])
        if handler is None:
            # Fail it rather than leave it claimed: a job nobody can execute must
            # reach a terminal state, otherwise the caller polls forever.
            self.job_service.mark_failed(job["job_id"], f"No handler registered for job kind '{job['kind']}'")
            return True

        try:
            result = handler(job["payload"])
        except Exception as exc:  # noqa: BLE001 - any handler failure belongs on the job, not the worker
            logger.exception("Job handler raised job_id=%s kind=%s", job["job_id"], job["kind"])
            self.job_service.mark_failed(job["job_id"], f"{type(exc).__name__}: {exc}")
            return True

        self.job_service.mark_succeeded(job["job_id"], result if result is not None else {})
        return True

    # ------------------------------------------------------------------
    # background loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = self.run_once()
            except Exception:  # noqa: BLE001 - the loop must outlive any single failure
                logger.exception("Job worker loop error")
                did_work = False
            if not did_work:
                self._stop.wait(self.poll_interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.job_service.requeue_stale_running()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._thread.start()
        logger.info("Job worker started")

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("Job worker stopped")
