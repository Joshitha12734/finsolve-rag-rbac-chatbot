"""
Admin operations: document upload (with role/department tagging) and
triggering re-indexing so newly uploaded documents become searchable
immediately, without restarting the server.

All functions here are called from endpoints gated to the `c-level` role
(see main.py's `require_admin`) — this module itself doesn't check
permissions, by design, the same separation of concerns used in
sql_agent.py.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import DATA_DIR, ALL_DEPARTMENTS

ALLOWED_EXTENSIONS = {".md", ".txt", ".csv"}


def _safe_filename(filename: str) -> str:
    """Strip path components and anything that isn't a safe filename
    character, so an upload can never write outside its department folder."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.md"


def save_uploaded_document(file_bytes: bytes, filename: str, department: str) -> Path:
    if department not in ALL_DEPARTMENTS:
        raise ValueError(f"Unknown department: {department}")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    dept_dir = DATA_DIR / department
    dept_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    dest = dept_dir / safe_name
    dest.write_bytes(file_bytes)
    return dest


def list_documents() -> list[dict]:
    """All documents currently in the corpus, grouped by department —
    used by the admin panel to show what's already there."""
    docs = []
    for dept_dir in sorted(DATA_DIR.iterdir()):
        if not dept_dir.is_dir():
            continue
        for f in sorted(dept_dir.glob("*")):
            if f.is_file():
                docs.append({
                    "department": dept_dir.name,
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return docs


def delete_document(department: str, filename: str) -> bool:
    safe_name = _safe_filename(filename)
    path = DATA_DIR / department / safe_name
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False
