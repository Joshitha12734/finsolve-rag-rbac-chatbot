"""
Tests for the workflow automation engine (the "Virtual Agent" layer):
low-code action definitions, RBAC-filtered visibility, request recording,
and the admin CRUD endpoints.

Intent detection itself needs an LLM call, so it isn't exercised end-to-end
here (this environment has no LLM key) — instead these tests cover the
parts that don't need one: RBAC filtering, execution given a match,
request persistence, and the admin endpoints, using ActionMatch objects
built directly rather than going through detect_action_intent().
"""
import base64

from fastapi.testclient import TestClient

from backend.main import app
from backend import workflow_engine as we

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


ADMIN = ("tony.sharma", "clevel123")
NON_ADMIN = ("peter.pandey", "engineering123")


def test_seed_actions_load():
    actions = we.load_actions()
    ids = {a["id"] for a in actions}
    assert {"leave_request", "it_ticket_escalation", "expense_reimbursement"} <= ids


def test_rbac_filters_department_restricted_actions():
    general_only = we.actions_for_role({"general"})
    ids = {a["id"] for a in general_only}
    assert "expense_reimbursement" not in ids  # finance-only action
    assert "leave_request" in ids  # general action

    finance_role = we.actions_for_role({"finance", "general"})
    assert "expense_reimbursement" in {a["id"] for a in finance_role}


def test_execute_action_records_request_and_formats_confirmation():
    actions = we.load_actions()
    leave_action = next(a for a in actions if a["id"] == "leave_request")
    match = we.ActionMatch(
        leave_action,
        {"start_date": "2026-09-01", "end_date": "2026-09-03", "reason": "personal"},
        [],
    )
    assert match.ready_to_execute
    confirmation = we.execute_action(match, "sam.employee", "employee")
    assert "2026-09-01" in confirmation
    assert "Request ID: #" in confirmation

    recent = we.get_recent_requests(limit=1)
    assert recent[0]["username"] == "sam.employee"
    assert recent[0]["action_id"] == "leave_request"


def test_execute_action_raises_if_fields_missing():
    actions = we.load_actions()
    leave_action = next(a for a in actions if a["id"] == "leave_request")
    match = we.ActionMatch(leave_action, {"start_date": "2026-09-01"}, ["end_date", "reason"])
    assert not match.ready_to_execute
    try:
        we.execute_action(match, "sam.employee", "employee")
        assert False, "should have raised"
    except ValueError:
        pass


def test_clarification_message_lists_missing_fields():
    actions = we.load_actions()
    leave_action = next(a for a in actions if a["id"] == "leave_request")
    match = we.ActionMatch(leave_action, {}, ["start_date", "end_date", "reason"])
    msg = we.clarification_message(match)
    assert "start date" in msg or "leave request" in msg.lower()


def test_non_admin_forbidden_from_workflow_admin_endpoints():
    for method, path in [("get", "/admin/workflows"), ("get", "/admin/workflows/requests")]:
        resp = getattr(client, method)(path, headers=auth_header(*NON_ADMIN))
        assert resp.status_code == 403


def test_admin_can_create_list_and_delete_a_workflow_action():
    resp = client.post(
        "/admin/workflows",
        headers=auth_header(*ADMIN),
        json={
            "id": "test_action_temp",
            "name": "Test Action",
            "description": "A temporary test action.",
            "department": "general",
            "required_fields": ["note"],
            "confirmation_template": "Test recorded: {note}. ID #{request_id}.",
        },
    )
    assert resp.status_code == 200

    resp = client.get("/admin/workflows", headers=auth_header(*ADMIN))
    assert "test_action_temp" in {a["id"] for a in resp.json()["actions"]}

    resp = client.delete("/admin/workflows/test_action_temp", headers=auth_header(*ADMIN))
    assert resp.status_code == 200

    resp = client.get("/admin/workflows", headers=auth_header(*ADMIN))
    assert "test_action_temp" not in {a["id"] for a in resp.json()["actions"]}


def test_admin_workflow_creation_rejects_unknown_department():
    resp = client.post(
        "/admin/workflows",
        headers=auth_header(*ADMIN),
        json={
            "id": "bad_dept_action",
            "name": "Bad",
            "description": "x",
            "department": "not-a-real-department",
            "required_fields": [],
            "confirmation_template": "x {request_id}",
        },
    )
    assert resp.status_code == 400


def test_chat_endpoint_never_errors_without_llm_key_even_for_action_shaped_query():
    # Without an LLM key, detect_action_intent no-ops and falls through to
    # SQL/RAG — the /chat endpoint should never 500 regardless.
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "I want to submit a leave request for next week"},
    )
    assert resp.status_code == 200
    assert resp.json()["route"] in ("action", "sql", "rag")


def test_heuristic_action_detection_works_without_llm_key():
    # Regression test: this environment has no LLM key configured, so this
    # exercises the heuristic keyword-matching fallback specifically (not
    # the LLM path). A keyword-matching message should be recognized as an
    # action intent, not silently fall through to document search.
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "I want to submit a leave request"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "action"
    assert "leave request" in body["answer"].lower()


def test_heuristic_multi_turn_field_collection_and_execution():
    first = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "I want to submit a leave request"},
    )
    history = [
        {"role": "user", "content": "I want to submit a leave request"},
        {"role": "assistant", "content": first.json()["answer"]},
    ]
    second = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={
            "query": "start_date: 2026-08-10, end_date: 2026-08-12, reason: family event",
            "history": history,
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["route"] == "action"
    assert "2026-08-10" in body["answer"]
    assert "Request ID: #" in body["answer"]


def test_heuristic_action_detection_still_respects_rbac():
    # employee role has no finance access, so "reimbursement" should never
    # match the finance-only expense_reimbursement action, heuristically or
    # otherwise — it should fall through to RAG instead.
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "I want a reimbursement for travel expenses"},
    )
    assert resp.status_code == 200
    assert resp.json()["route"] != "action"


def test_unrelated_followup_does_not_get_hijacked_by_pending_action():
    # Regression test for a real bug found via live testing: once an
    # action was proposed (e.g. "I need to escalate an IT issue" ->
    # assistant asks for missing fields), EVERY subsequent message —
    # including completely unrelated questions — was being treated as
    # still trying to complete that same action, because the old
    # continuation check only looked at whether the action's name
    # appeared in the last assistant message, never whether the new
    # message actually supplied any field data. Fixed by only continuing
    # the pending action when field extraction actually succeeds;
    # otherwise falling through to a fresh intent check.
    first = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "I need to escalate an IT issue"},
    )
    assert first.json()["route"] == "action"

    history = [
        {"role": "user", "content": "I need to escalate an IT issue"},
        {"role": "assistant", "content": first.json()["answer"]},
    ]

    unrelated = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "What's our microservices architecture?", "history": history},
    )
    assert unrelated.status_code == 200
    body = unrelated.json()
    assert body["route"] != "action"
    assert any(s.startswith("engineering/") for s in body["sources"])

    # A second, different unrelated question after that should ALSO
    # escape cleanly, not stay stuck.
    history2 = history + [
        {"role": "user", "content": "What's our microservices architecture?"},
        {"role": "assistant", "content": unrelated.json()["answer"]},
    ]
    second_unrelated = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "could you give me info about the finance", "history": history2},
    )
    assert second_unrelated.status_code == 200
    assert second_unrelated.json()["route"] != "action"


def test_genuine_continuation_with_field_values_still_completes_the_action():
    # The fix above must not break real continuations — providing actual
    # field values right after a clarification request should still work.
    first = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "I need to escalate an IT issue"},
    )
    history = [
        {"role": "user", "content": "I need to escalate an IT issue"},
        {"role": "assistant", "content": first.json()["answer"]},
    ]
    second = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "issue_description: VPN not connecting, urgency: high", "history": history},
    )
    assert second.status_code == 200
    body = second.json()
    assert body["route"] == "action"
    assert "Ticket ID: #" in body["answer"]
