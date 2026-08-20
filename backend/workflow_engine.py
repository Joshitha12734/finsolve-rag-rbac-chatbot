"""
Workflow Automation Engine — the "Virtual Agent" / "Now Assist" layer.

Beyond answering questions (SQL/RAG), the assistant can *do* things:
submit a leave request, escalate an IT ticket, request a reimbursement.
Each action is a small JSON definition (data/_workflow_actions.json) —
adding a new action is a config change, not a code change, which is the
"low-code" part: an admin defines a name, a natural-language description
of when it applies, which fields it needs, which department can trigger
it, and a confirmation message template. No Python required.

Flow (see main.py's /chat):
  1. detect_action_intent() — an LLM call checks the user's message
     against the *department-filtered* list of available actions (RBAC
     applied before the LLM ever sees an action the user can't use) and
     extracts any field values already present in the message.
  2. If an action matches but required fields are missing, the caller
     asks the user for them (multi-turn — the existing conversation
     history carries this forward).
  3. Once all fields are present, execute_action() records the request
     (SQLite, separate from the audit log and the retrieval index) and
     returns a confirmation message built from the action's template.

This is a real (if intentionally simple) workflow engine: requests are
durably recorded with auto-incrementing IDs, not just a chat reply that
evaporates.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR, BASE_DIR
from . import llm_client

ACTIONS_PATH = DATA_DIR / "_workflow_actions.json"
REQUESTS_DB_PATH = BASE_DIR / "logs" / "workflow_requests.db"


# ---------------------------------------------------------------------------
# Action definitions (low-code: JSON, editable via the Admin panel)
# ---------------------------------------------------------------------------
def load_actions() -> list[dict]:
    if not ACTIONS_PATH.exists():
        return []
    try:
        return json.loads(ACTIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_actions(actions: list[dict]) -> None:
    ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_PATH.write_text(json.dumps(actions, indent=2), encoding="utf-8")


def add_action(action: dict) -> None:
    actions = load_actions()
    actions = [a for a in actions if a["id"] != action["id"]]  # replace if exists
    actions.append(action)
    save_actions(actions)


def delete_action(action_id: str) -> bool:
    actions = load_actions()
    remaining = [a for a in actions if a["id"] != action_id]
    if len(remaining) == len(actions):
        return False
    save_actions(remaining)
    return True


def actions_for_role(allowed_departments: set) -> list[dict]:
    """RBAC applied here, before any action definition (even just its
    name/description) is shown to the LLM for intent matching — same
    filter-before-you-look principle as the retriever and SQL agent."""
    return [a for a in load_actions() if a.get("department", "general") in allowed_departments]


# ---------------------------------------------------------------------------
# Request log (separate SQLite db from the audit trail — different
# lifecycle: these are durable business records, not observability data)
# ---------------------------------------------------------------------------
@contextmanager
def _connect():
    REQUESTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(REQUESTS_DB_PATH))
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                fields TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )


def _record_request(action_id: str, username: str, role: str, fields: dict) -> int:
    _init_db()
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO workflow_requests (action_id, username, role, fields, timestamp) VALUES (?, ?, ?, ?, ?)",
            (action_id, username, role, json.dumps(fields), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_recent_requests(limit: int = 50) -> list[dict]:
    _init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM workflow_requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Intent detection + execution
# ---------------------------------------------------------------------------
class ActionMatch:
    def __init__(self, action: dict | None, fields: dict, missing_fields: list[str]):
        self.action = action
        self.fields = fields
        self.missing_fields = missing_fields

    @property
    def matched(self) -> bool:
        return self.action is not None

    @property
    def ready_to_execute(self) -> bool:
        return self.matched and not self.missing_fields


def detect_action_intent(query: str, allowed_departments: set, history: list[dict] | None = None) -> ActionMatch:
    """Checks the message (plus recent history, so follow-ups like
    "the reason is a family event" resolve against an action already
    proposed) against the department-filtered action list.

    Two strategies, same interface as query_classifier.py:
      - LLM-based (preferred): understands free-form phrasing and extracts
        field values from natural language.
      - Heuristic (fallback, no API key needed): keyword matching against
        each action's `keywords` list, plus simple "field: value" /
        "field is value" pattern extraction. Less flexible, but means the
        Virtual Agent layer is still demoable without any LLM configured —
        it correctly recognizes intent and asks for missing fields instead
        of silently falling through to document search.
    """
    available = actions_for_role(allowed_departments)
    if not available:
        return ActionMatch(None, {}, [])

    if llm_client.api_key_configured():
        match = _detect_action_intent_llm(query, available, history)
        if match.matched:
            return match
        # LLM found nothing — still worth trying the heuristic as a safety
        # net for very explicit keyword matches the LLM might have missed.

    return _detect_action_intent_heuristic(query, available, history)


def _detect_action_intent_heuristic(query: str, available: list[dict], history: list[dict] | None) -> ActionMatch:
    haystack = query.lower()

    if history:
        # If an action was already proposed in the last assistant turn,
        # check whether THIS message actually looks like it's supplying
        # the missing field values — not just blindly assume every
        # following message is still about that action. Without this
        # check, once any workflow is proposed, every subsequent message
        # in the conversation (including totally unrelated questions)
        # gets hijacked into "still missing those fields" forever, which
        # was a real bug caught in testing.
        for turn in reversed(history[-4:]):
            if turn.get("role") == "assistant":
                for action in available:
                    if action["name"].lower() in turn.get("content", "").lower():
                        continuation = _extract_fields_heuristic(query, action)
                        if continuation.fields:
                            # Extracted at least one real field value from
                            # this message — genuinely a continuation.
                            return continuation
                        # Nothing extracted: this message doesn't look like
                        # it's answering the pending action's questions.
                        # Fall through to a fresh keyword check below
                        # instead of forcing the old action to persist.
                break

    for action in available:
        if any(kw in haystack for kw in action.get("keywords", [])):
            return _extract_fields_heuristic(query, action)

    return ActionMatch(None, {}, [])


_FIELD_PATTERN_TEMPLATE = r"{field}\s*(?:is|[:=])\s*([^,;\n]+)"


def _extract_fields_heuristic(query: str, action: dict) -> ActionMatch:
    """Very simple 'field: value' / 'field is value' extraction — no NLP,
    just regex. Accepts either the raw field name ("start_date:") or its
    human-readable form ("start date:"). Good enough to demo the multi-turn
    flow without an LLM; the LLM path handles genuinely free-form phrasing
    much better."""
    fields = {}
    for field in action["required_fields"]:
        variants = {field, field.replace("_", " ")}
        for variant in variants:
            pattern = _FIELD_PATTERN_TEMPLATE.format(field=re.escape(variant))
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                fields[field] = m.group(1).strip()
                break
    missing = [f for f in action["required_fields"] if not fields.get(f)]
    return ActionMatch(action, fields, missing)


def _detect_action_intent_llm(query: str, available: list[dict], history: list[dict] | None) -> ActionMatch:
    if not llm_client.api_key_configured():
        return ActionMatch(None, {}, [])

    actions_desc = "\n".join(
        f"- id: {a['id']} | {a['name']}: {a['description']} | required fields: {a['required_fields']}"
        for a in available
    )
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "Recent conversation:\n" + "\n".join(f"{h['role']}: {h['content']}" for h in recent) + "\n\n"

    prompt = (
        f"Available workflow actions this user can trigger:\n{actions_desc}\n\n"
        f"{history_text}"
        f"Latest message: \"{query}\"\n\n"
        "Does this message (in context of the conversation) intend to trigger one of these actions? "
        "If yes, extract any field values already stated (in this message OR earlier in the conversation). "
        'Respond as JSON only: {"action_id": "..." or null, "fields": {"field_name": "value", ...}}'
    )
    try:
        text = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=300).strip()
        text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
    except Exception:
        return ActionMatch(None, {}, [])

    action_id = parsed.get("action_id")
    if not action_id:
        return ActionMatch(None, {}, [])

    action = next((a for a in available if a["id"] == action_id), None)
    if action is None:
        return ActionMatch(None, {}, [])  # LLM named an action outside the RBAC-filtered set — ignore it

    fields = parsed.get("fields") or {}
    missing = [f for f in action["required_fields"] if not fields.get(f)]
    return ActionMatch(action, fields, missing)


def execute_action(match: ActionMatch, username: str, role: str) -> str:
    if not match.ready_to_execute:
        raise ValueError("Cannot execute an action with missing required fields.")
    request_id = _record_request(match.action["id"], username, role, match.fields)
    return match.action["confirmation_template"].format(request_id=request_id, **match.fields)


_CLARIFICATION_TEMPLATES = [
    "Sure, happy to help with your {action_name} — I just need: {fields}.",
    "Got it — for the {action_name}, I still need: {fields}.",
    "I can take care of your {action_name}. Just need a couple more details: {fields}.",
    "Before I submit your {action_name}, could you share: {fields}?",
]


def clarification_message(match: ActionMatch) -> str:
    import random

    missing_readable = ", ".join(f.replace("_", " ") for f in match.missing_fields)
    template = random.choice(_CLARIFICATION_TEMPLATES)
    return template.format(action_name=match.action["name"].lower(), fields=missing_readable)
