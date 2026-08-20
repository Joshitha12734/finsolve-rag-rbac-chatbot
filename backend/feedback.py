"""
User feedback (👍/👎) on chat answers — the closed loop that lets you
demonstrate an actual improvement cycle: feedback in, retrieval/prompt
changes out, re-evaluate. Separate SQLite file from the audit trail and
workflow requests (same "one file per concern" pattern used throughout
this app), so exporting/analyzing feedback never touches unrelated data.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BASE_DIR

FEEDBACK_DB_PATH = BASE_DIR / "logs" / "feedback.db"

VALID_RATINGS = ("up", "down")
VALID_REASONS = ("wrong_answer", "wrong_source", "outdated", "didnt_answer", "access_issue", "other")


@contextmanager
def _connect():
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(FEEDBACK_DB_PATH))
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                query TEXT NOT NULL,
                answer_preview TEXT NOT NULL,
                rating TEXT NOT NULL,
                reason TEXT,
                route TEXT
            )
            """
        )


def submit_feedback(
    username: str, role: str, query: str, answer: str, rating: str, reason: str | None = None, route: str | None = None
) -> int:
    if rating not in VALID_RATINGS:
        raise ValueError(f"rating must be one of {VALID_RATINGS}")
    if reason is not None and reason not in VALID_REASONS:
        raise ValueError(f"reason must be one of {VALID_REASONS}")

    _init_db()
    answer_preview = answer[:300]
    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO feedback (timestamp, username, role, query, answer_preview, rating, reason, route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), username, role, query, answer_preview, rating, reason, route),
        )
        return cur.lastrowid


def get_recent_feedback(limit: int = 50) -> list[dict[str, Any]]:
    _init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_feedback_summary() -> dict[str, Any]:
    _init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        total = con.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        up = con.execute("SELECT COUNT(*) as c FROM feedback WHERE rating = 'up'").fetchone()["c"]
        down = total - up
        by_reason = con.execute(
            "SELECT reason, COUNT(*) as count FROM feedback WHERE reason IS NOT NULL GROUP BY reason ORDER BY count DESC"
        ).fetchall()
    return {
        "total_feedback": total,
        "thumbs_up": up,
        "thumbs_down": down,
        "satisfaction_pct": round(100 * up / total, 1) if total else 0.0,
        "down_reasons": [dict(r) for r in by_reason],
    }
