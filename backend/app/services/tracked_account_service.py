"""A watch list of competitor accounts, and what changed since last time.

`public_profile_analysis` reads one account once. That answers "how is Nike's
page built"; it does not answer "what did my three competitors do this week",
which is the question a marketer actually has on a Monday.

The unit of value here is the **diff**, not the snapshot. "Nike has 302M
followers" is a fact nobody needs restated. "Three new posts since Tuesday, the
best one took four times their median engagement, and it was a Reel" is the
thing that changes what you publish next. So every refresh is stored, and what
is returned is the comparison against the previous one.

New posts are identified by id, not by counting. A media_count that rose by two
could be three posts and one deletion, and a marketer told "2 new posts" who
then finds three has been given a subtly wrong picture of a competitor's cadence.

Storage is SQLite for the same reason billing is: this is a per-customer list
that must survive a deploy, and the JSON read-modify-write pattern used
elsewhere in this codebase already lost concurrent writes once.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class TrackedAccountService:
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS tracked_accounts (
            user_id     TEXT NOT NULL,
            handle      TEXT NOT NULL,
            label       TEXT,
            added_at    TEXT NOT NULL,
            PRIMARY KEY (user_id, handle)
        );
        CREATE TABLE IF NOT EXISTS tracked_snapshots (
            user_id         TEXT NOT NULL,
            handle          TEXT NOT NULL,
            captured_at     TEXT NOT NULL,
            followers_count INTEGER,
            media_count     INTEGER,
            post_ids        TEXT NOT NULL DEFAULT '[]',
            median_likes    INTEGER,
            PRIMARY KEY (user_id, handle, captured_at)
        );
    """

    def __init__(self, db_path: Path | str | None = None):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_path = Path(db_path or os.getenv("TRACKED_DB_PATH") or (data_dir / "tracked_accounts.sqlite3"))
        self._initialise_storage()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialise_storage(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ list
    def add(self, *, user_id: str, handle: str, label: str | None, limit: int) -> dict:
        """Track one account, refusing past the plan's limit.

        The limit is checked inside the write transaction rather than before it:
        two tabs adding a fourth account to a three-account plan would otherwise
        both read three and both insert.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT COUNT(*) AS n FROM tracked_accounts WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
            already = conn.execute(
                "SELECT 1 FROM tracked_accounts WHERE user_id = ? AND handle = ?", (user_id, handle)
            ).fetchone()

            if not already and current >= limit:
                conn.execute("COMMIT")
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"This plan can track {limit} account(s); {current} are already tracked."
                        if limit
                        else "Tracking competitor accounts is not included in this plan."
                    ),
                )

            conn.execute(
                "INSERT INTO tracked_accounts (user_id, handle, label, added_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, handle) DO UPDATE SET label = excluded.label",
                (user_id, handle, label, self._now()),
            )
            conn.execute("COMMIT")

        logger.info("Tracking account user_id=%s handle=%s", user_id, handle)
        return {"handle": handle, "label": label, "tracked": True}

    def remove(self, *, user_id: str, handle: str) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM tracked_accounts WHERE user_id = ? AND handle = ?", (user_id, handle)
            )
            # Snapshots go with it: keeping history for an account the customer
            # stopped following would make a later re-add report months of
            # "change" that they never asked to be told about.
            conn.execute("DELETE FROM tracked_snapshots WHERE user_id = ? AND handle = ?", (user_id, handle))
            removed = cursor.rowcount > 0
            conn.execute("COMMIT")

        if not removed:
            raise HTTPException(status_code=404, detail=f"'{handle}' is not being tracked")
        return {"handle": handle, "tracked": False}

    def list_tracked(self, user_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT handle, label, added_at FROM tracked_accounts WHERE user_id = ? ORDER BY added_at",
                (user_id,),
            ).fetchall()
            tracked = []
            for row in rows:
                last = conn.execute(
                    "SELECT captured_at, followers_count, media_count FROM tracked_snapshots "
                    "WHERE user_id = ? AND handle = ? ORDER BY captured_at DESC LIMIT 1",
                    (user_id, row["handle"]),
                ).fetchone()
                tracked.append({
                    "handle": row["handle"],
                    "label": row["label"],
                    "added_at": row["added_at"],
                    "last_checked_at": last["captured_at"] if last else None,
                    "followers_count": last["followers_count"] if last else None,
                    "media_count": last["media_count"] if last else None,
                })
        return tracked

    def is_tracked(self, *, user_id: str, handle: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM tracked_accounts WHERE user_id = ? AND handle = ?", (user_id, handle)
            ).fetchone() is not None

    # -------------------------------------------------------------- snapshots
    @staticmethod
    def _summarise(profile: dict) -> dict:
        posts = profile.get("posts") or []
        likes = sorted(p["like_count"] for p in posts if isinstance(p.get("like_count"), int))
        return {
            "followers_count": profile.get("followers_count"),
            "media_count": profile.get("media_count"),
            "post_ids": [str(p.get("post_id")) for p in posts if p.get("post_id")],
            "median_likes": likes[len(likes) // 2] if likes else None,
        }

    def compare_and_store(self, *, user_id: str, handle: str, profile: dict) -> dict:
        """Store this reading and return what changed since the previous one.

        A first check reports `is_first_check` rather than inventing deltas
        against zero — "+302,000,000 followers" would be technically derived and
        completely useless.
        """
        summary = self._summarise(profile)

        with self._connect() as conn:
            previous = conn.execute(
                "SELECT captured_at, followers_count, media_count, post_ids, median_likes "
                "FROM tracked_snapshots WHERE user_id = ? AND handle = ? ORDER BY captured_at DESC LIMIT 1",
                (user_id, handle),
            ).fetchone()

            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO tracked_snapshots "
                "(user_id, handle, captured_at, followers_count, media_count, post_ids, median_likes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    handle,
                    self._now(),
                    summary["followers_count"],
                    summary["media_count"],
                    json.dumps(summary["post_ids"]),
                    summary["median_likes"],
                ),
            )
            conn.execute("COMMIT")

        if previous is None:
            return {
                "handle": handle,
                "is_first_check": True,
                "since": None,
                "new_posts": [],
                "followers_change": None,
                "median_likes_change": None,
                **summary,
            }

        seen_before = set(json.loads(previous["post_ids"] or "[]"))
        new_posts = [
            post for post in (profile.get("posts") or []) if str(post.get("post_id")) not in seen_before
        ]

        return {
            "handle": handle,
            "is_first_check": False,
            "since": previous["captured_at"],
            "new_posts": new_posts,
            "followers_change": self._delta(summary["followers_count"], previous["followers_count"]),
            "median_likes_change": self._delta(summary["median_likes"], previous["median_likes"]),
            **summary,
        }

    @staticmethod
    def _delta(current, previous):
        # None rather than 0 when either side is unknown: reporting "no change"
        # for a number we never had is a claim, and 0 would look like one.
        if not isinstance(current, int) or not isinstance(previous, int):
            return None
        return current - previous
