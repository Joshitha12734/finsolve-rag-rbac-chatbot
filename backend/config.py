"""
Central configuration for the FinSolve RBAC RAG chatbot.

- ROLE_PERMISSIONS maps each role to the list of data "departments"
  (i.e. subfolders under app/data/) it is allowed to retrieve from.
- USERS is a small demo user directory. Passwords are stored as
  sha256 hashes (see auth.py). In a production system this would be
  backed by a real identity provider / database.
"""
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vectorstore"

# ---------------------------------------------------------------------------
# Roles -> which data folders they can see
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS = {
    "engineering": {"engineering", "general"},
    "finance": {"finance", "general"},
    "marketing": {"marketing", "general"},
    "hr": {"hr", "general"},
    "c-level": {"engineering", "finance", "marketing", "hr", "general"},
    "employee": {"general"},
}

ALL_DEPARTMENTS = {"engineering", "finance", "marketing", "hr", "general"}

# ---------------------------------------------------------------------------
# Metadata-aware RBAC, beyond department folders.
#
# Every chunk also carries a `classification` tag (see ingest.py, which reads
# data/_classification.json — any file not listed there defaults to
# "internal"). Department membership is still checked first as before; this
# is an *additional* restriction layered on top, so a document can be
# further locked down within a department it belongs to (e.g. a board-only
# quarterly report that most of Finance itself shouldn't see).
#
#   public       - visible to anyone whose role already has department access
#   internal     - default; same as public for now, reserved for future tiers
#   confidential - c-level only, regardless of department
# ---------------------------------------------------------------------------
CLASSIFICATION_LEVELS = ("public", "internal", "confidential")


def role_can_access_classification(role: str, classification: str) -> bool:
    if classification == "confidential":
        return role == "c-level"
    return True


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Demo users. username -> {password_hash, role, full_name}
# Passwords below are intentionally simple ("<role>123") for demo purposes.
# Change these before deploying anywhere real.
# ---------------------------------------------------------------------------
USERS = {
    "peter.pandey": {
        "password_hash": hash_password("engineering123"),
        "role": "engineering",
        "full_name": "Peter Pandey",
    },
    "priya.finance": {
        "password_hash": hash_password("finance123"),
        "role": "finance",
        "full_name": "Priya Sharma",
    },
    "raj.marketing": {
        "password_hash": hash_password("marketing123"),
        "role": "marketing",
        "full_name": "Raj Verma",
    },
    "anita.hr": {
        "password_hash": hash_password("hr123"),
        "role": "hr",
        "full_name": "Anita Desai",
    },
    "tony.sharma": {
        "password_hash": hash_password("clevel123"),
        "role": "c-level",
        "full_name": "Tony Sharma",
    },
    "sam.employee": {
        "password_hash": hash_password("employee123"),
        "role": "employee",
        "full_name": "Sam Employee",
    },
}


def add_user(username: str, password: str, role: str, full_name: str) -> None:
    """Create or overwrite a demo user at runtime (admin panel). Note: this
    is in-memory only — restarting the server resets to the users defined
    above. For anything persistent, back this with a real database."""
    USERS[username] = {
        "password_hash": hash_password(password),
        "role": role,
        "full_name": full_name,
    }


def remove_user(username: str) -> bool:
    return USERS.pop(username, None) is not None


def list_users() -> list[dict]:
    """User directory without password hashes, for the admin panel."""
    return [
        {"username": u, "role": info["role"], "full_name": info["full_name"]}
        for u, info in USERS.items()
    ]
