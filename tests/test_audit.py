"""
Tests for audit logging and the admin analytics dashboard.
"""
import base64

from fastapi.testclient import TestClient

from backend.main import app
from backend import audit

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


ADMIN = ("tony.sharma", "clevel123")
NON_ADMIN = ("peter.pandey", "engineering123")


def test_non_admin_cannot_view_analytics():
    resp = client.get("/admin/analytics/summary", headers=auth_header(*NON_ADMIN))
    assert resp.status_code == 403
    resp = client.get("/admin/analytics/recent", headers=auth_header(*NON_ADMIN))
    assert resp.status_code == 403


def test_chat_call_is_logged_and_appears_in_recent():
    before = len(audit.get_recent(limit=10000))
    client.post(
        "/chat",
        headers=auth_header(*NON_ADMIN),
        json={"query": "what CI/CD tools do we use?"},
    )
    after = audit.get_recent(limit=10000)
    assert len(after) == before + 1
    latest = after[0]  # most recent first
    assert latest["username"] == "peter.pandey"
    assert latest["role"] == "engineering"
    assert latest["route"] in ("sql", "rag")
    assert latest["latency_ms"] >= 0


def test_hr_row_query_from_employee_role_excludes_hr_sources_but_may_still_answer():
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "Aadhya Patel Sales Manager salary leave balance"},
    )
    assert resp.status_code == 200
    assert not any(s.startswith("hr/") for s in resp.json()["sources"])  # employee has no HR access

    entry = audit.get_recent(limit=1)[0]
    assert entry["username"] == "sam.employee"
    assert not any(s.startswith("hr/") for s in entry["sources"].split(","))


def test_truly_unmatched_query_logged_as_denied():
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "zzqxw blorptastic wibbleflorp"},
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []

    entry = audit.get_recent(limit=1)[0]
    assert entry["allowed"] == 0
    assert entry["num_sources"] == 0


def test_analytics_summary_reflects_logged_queries():
    resp = client.get("/admin/analytics/summary", headers=auth_header(*ADMIN))
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_queries"] >= 1
    assert "avg_latency_ms" in body
    assert "most_accessed_documents" in body
    assert "queries_by_role" in body
    assert "queries_by_route" in body
