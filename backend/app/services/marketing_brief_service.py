"""The customer's marketing brief — what they are actually trying to achieve.

Before this, ``goal``, ``niche`` and ``target_audience`` arrived as fields on each
agent request and were discarded with it. The customer restated their objective
every time, and nothing they said once shaped anything afterwards.

The brief is stored per (user, account) and injected into every generation, so a
goal stated once steers all later output. Storage is SQLite for the same reason
billing is: an unlocked read-modify-rewrite over a JSON file loses concurrent
edits, and ``BRIEF_DB_PATH`` keeps the location configurable because the default
sits on an ephemeral container filesystem.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Fields the caller may set, in the order they read most naturally in a prompt.
BRIEF_FIELDS = (
    "objective",
    "offer",
    "ideal_customer",
    "funnel_stage",
    "tone_of_voice",
    "primary_kpi",
)

_PROMPT_LABELS = {
    "objective": "Business objective",
    "offer": "Offer being sold",
    "ideal_customer": "Ideal customer",
    "funnel_stage": "Funnel stage to serve",
    "tone_of_voice": "Brand tone of voice",
    "primary_kpi": "Primary KPI",
}


class MarketingBriefService:
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS marketing_briefs (
            user_id         TEXT NOT NULL,
            account_id      TEXT NOT NULL,
            objective       TEXT,
            offer           TEXT,
            ideal_customer  TEXT,
            funnel_stage    TEXT,
            tone_of_voice   TEXT,
            primary_kpi     TEXT,
            topics_to_avoid TEXT NOT NULL DEFAULT '[]',
            updated_at      TEXT NOT NULL,
            PRIMARY KEY (user_id, account_id)
        )
    """

    def __init__(self, db_path: Path | str | None = None):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_path = Path(db_path or os.getenv("BRIEF_DB_PATH") or (data_dir / "marketing_briefs.sqlite3"))
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_brief(self, row: sqlite3.Row | None, user_id: str, account_id: str) -> dict:
        if row is None:
            return {
                "user_id": user_id,
                "account_id": account_id,
                **{field: None for field in BRIEF_FIELDS},
                "topics_to_avoid": [],
                "is_empty": True,
                "updated_at": None,
            }

        try:
            topics = json.loads(row["topics_to_avoid"] or "[]")
        except json.JSONDecodeError:
            topics = []

        brief = {
            "user_id": row["user_id"],
            "account_id": row["account_id"],
            **{field: row[field] for field in BRIEF_FIELDS},
            "topics_to_avoid": topics if isinstance(topics, list) else [],
            "updated_at": row["updated_at"],
        }
        brief["is_empty"] = not any(brief[field] for field in BRIEF_FIELDS) and not brief["topics_to_avoid"]
        return brief

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_brief(self, user_id: str, account_id: str) -> dict:
        """Return the stored brief, or an empty one. Never raises on absence."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM marketing_briefs WHERE user_id = ? AND account_id = ?",
                (user_id, account_id),
            ).fetchone()
        return self._row_to_brief(row, user_id, account_id)

    def save_brief(self, user_id: str, account_id: str, payload: dict) -> dict:
        """Create or update a brief.

        Fields absent from ``payload`` are left as they were, so a caller can fill
        the brief in progressively without having to resend the whole thing.
        Passing an explicit ``None`` clears a field.
        """
        existing = self.get_brief(user_id, account_id)

        merged = {}
        for field in BRIEF_FIELDS:
            value = payload[field] if field in payload else existing.get(field)
            merged[field] = (value or "").strip() or None if isinstance(value, str) else value

        topics = payload.get("topics_to_avoid", existing.get("topics_to_avoid") or [])
        topics = [str(t).strip() for t in (topics or []) if str(t).strip()]

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO marketing_briefs "
                "(user_id, account_id, objective, offer, ideal_customer, funnel_stage, "
                " tone_of_voice, primary_kpi, topics_to_avoid, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, account_id) DO UPDATE SET "
                " objective=excluded.objective, offer=excluded.offer, "
                " ideal_customer=excluded.ideal_customer, funnel_stage=excluded.funnel_stage, "
                " tone_of_voice=excluded.tone_of_voice, primary_kpi=excluded.primary_kpi, "
                " topics_to_avoid=excluded.topics_to_avoid, updated_at=excluded.updated_at",
                (
                    user_id, account_id,
                    merged["objective"], merged["offer"], merged["ideal_customer"],
                    merged["funnel_stage"], merged["tone_of_voice"], merged["primary_kpi"],
                    json.dumps(topics, ensure_ascii=False), self._now(),
                ),
            )
            conn.execute("COMMIT")

        logger.info("Saved marketing brief user_id=%s account_id=%s", user_id, account_id)
        return self.get_brief(user_id, account_id)

    def delete_brief(self, user_id: str, account_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM marketing_briefs WHERE user_id = ? AND account_id = ?",
                (user_id, account_id),
            )
            deleted = cursor.rowcount
            conn.execute("COMMIT")
        return bool(deleted)

    # ------------------------------------------------------------------
    # prompt integration
    # ------------------------------------------------------------------
    def as_prompt_context(self, brief: dict | None) -> dict | None:
        """Reduce a brief to the fields worth spending prompt tokens on.

        Returns None for an empty brief so the caller can omit the section
        entirely rather than send a block of empty labels, which reads to the
        model as "these are unknown" and invites it to invent them.
        """
        if not brief or brief.get("is_empty"):
            return None

        context: dict[str, object] = {}
        for field in BRIEF_FIELDS:
            value = brief.get(field)
            if value:
                context[_PROMPT_LABELS[field]] = value

        topics = brief.get("topics_to_avoid") or []
        if topics:
            context["Topics to avoid"] = list(topics)

        return context or None
