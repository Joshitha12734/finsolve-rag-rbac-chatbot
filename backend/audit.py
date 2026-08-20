"""
Audit logging for every /chat request: who asked what, which engine
answered it, whether they actually got data back or were denied/found
nothing, and how long it took.

Backed by SQLite (not DuckDB/vectorstore) so it survives index rebuilds —
`retriever.build()` and reindexing never touch this file.

This directly supports two things a real enterprise deployment would need:
  - Security/compliance: a record of who accessed what, and what was denied.
  - Operability: latency and "most accessed documents" are the kind of
    numbers you'd actually want on a dashboard before shipping this.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
AUDIT_DB_PATH = LOG_DIR / "audit_log.db"


@contextmanager
def _connect():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(AUDIT_DB_PATH))
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                query TEXT NOT NULL,
                route TEXT NOT NULL,
                allowed INTEGER NOT NULL,
                num_sources INTEGER NOT NULL,
                sources TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )


class Timer:
    """Small helper: `with Timer() as t: ...` then `t.elapsed_ms`."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def log_query(
    username: str,
    role: str,
    query: str,
    route: str,
    sources: list[str],
    latency_ms: float,
) -> None:
    init_db()
    allowed = 1 if sources else 0
    with _connect() as con:
        con.execute(
            """
            INSERT INTO audit_log (timestamp, username, role, query, route, allowed, num_sources, sources, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                username,
                role,
                query,
                route,
                allowed,
                len(sources),
                ",".join(sources),
                round(latency_ms, 1),
            ),
        )


def get_recent(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_summary() -> dict[str, Any]:
    """Aggregate stats for the admin analytics dashboard."""
    init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        total = con.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
        denied = con.execute("SELECT COUNT(*) as c FROM audit_log WHERE allowed = 0").fetchone()["c"]
        avg_latency = con.execute("SELECT AVG(latency_ms) as a FROM audit_log").fetchone()["a"]
        by_role = con.execute(
            "SELECT role, COUNT(*) as count FROM audit_log GROUP BY role ORDER BY count DESC"
        ).fetchall()
        by_route = con.execute(
            "SELECT route, COUNT(*) as count FROM audit_log GROUP BY route ORDER BY count DESC"
        ).fetchall()

        # Most-accessed documents: sources is a comma-joined string per row,
        # so tally in Python rather than trying to do this in SQL.
        source_counts: dict[str, int] = {}
        for row in con.execute("SELECT sources FROM audit_log WHERE sources != ''"):
            for src in row[0].split(","):
                if src:
                    source_counts[src] = source_counts.get(src, 0) + 1
        top_sources = sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "total_queries": total,
        "denied_queries": denied,
        "denial_rate_pct": round(100 * denied / total, 1) if total else 0.0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0.0,
        "queries_by_role": [dict(r) for r in by_role],
        "queries_by_route": [dict(r) for r in by_route],
        "most_accessed_documents": [{"source": s, "count": c} for s, c in top_sources],
    }
