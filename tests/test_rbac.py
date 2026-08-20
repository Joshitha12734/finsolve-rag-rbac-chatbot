"""
Automated tests for the FinSolve RBAC RAG chatbot.

Run with:  pytest -v

These tests are the real evidence behind the "Integrated role-based
access control" deliverable: they don't just assert the code compiles,
they assert that a user in one department genuinely cannot retrieve
another department's data, using the real ingested corpus.
"""
import base64
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import USERS, ROLE_PERMISSIONS

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# A (username, plaintext password) pair for every demo role, derived from
# config.py so this stays in sync if demo users change.
DEMO_CREDENTIALS = {
    "engineering": ("peter.pandey", "engineering123"),
    "finance": ("priya.finance", "finance123"),
    "marketing": ("raj.marketing", "marketing123"),
    "hr": ("anita.hr", "hr123"),
    "c-level": ("tony.sharma", "clevel123"),
    "employee": ("sam.employee", "employee123"),
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_rejects_bad_password():
    resp = client.get("/me", headers=auth_header("peter.pandey", "wrong-password"))
    assert resp.status_code == 401


def test_rejects_unknown_user():
    resp = client.get("/me", headers=auth_header("nobody", "whatever"))
    assert resp.status_code == 401


@pytest.mark.parametrize("role,creds", DEMO_CREDENTIALS.items())
def test_me_reports_correct_role_and_permissions(role, creds):
    username, password = creds
    resp = client.get("/me", headers=auth_header(username, password))
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == role
    assert set(body["allowed_departments"]) == ROLE_PERMISSIONS[role]


def test_engineering_role_cannot_retrieve_finance_data():
    username, password = DEMO_CREDENTIALS["engineering"]
    resp = client.post(
        "/chat",
        headers=auth_header(username, password),
        json={"query": "what were vendor services expenses in 2024?"},
    )
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert not any(s.startswith("finance/") for s in sources)


def test_finance_role_can_retrieve_finance_data():
    username, password = DEMO_CREDENTIALS["finance"]
    resp = client.post(
        "/chat",
        headers=auth_header(username, password),
        json={"query": "what were vendor services expenses in 2024?"},
    )
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert any(s.startswith("finance/") for s in sources)


def test_employee_role_blocked_from_hr_row_level_data():
    username, password = DEMO_CREDENTIALS["employee"]
    resp = client.post(
        "/chat",
        headers=auth_header(username, password),
        json={"query": "Aadhya Patel Sales Manager salary leave balance"},
    )
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert not any(s.startswith("hr/") for s in sources)


def test_hr_role_can_see_hr_row_level_data():
    username, password = DEMO_CREDENTIALS["hr"]
    resp = client.post(
        "/chat",
        headers=auth_header(username, password),
        json={"query": "Aadhya Patel Sales Manager salary leave balance"},
    )
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert any(s.startswith("hr/") for s in sources)


def test_c_level_can_access_every_department():
    username, password = DEMO_CREDENTIALS["c-level"]
    queries_and_expected_prefix = [
        ("what were vendor services expenses in 2024?", "finance/"),
        ("what CI/CD tools does engineering use?", "engineering/"),
        ("how did Q2 2024 marketing campaigns perform?", "marketing/"),
    ]
    for query, expected_prefix in queries_and_expected_prefix:
        resp = client.post("/chat", headers=auth_header(username, password), json={"query": query})
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert any(s.startswith(expected_prefix) for s in sources), f"failed for query: {query}"


def test_all_roles_can_access_general_info():
    for role, (username, password) in DEMO_CREDENTIALS.items():
        resp = client.post(
            "/chat",
            headers=auth_header(username, password),
            json={"query": "what is the leave policy?"},
        )
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert any(s.startswith("general/") for s in sources), f"failed for role: {role}"


def test_no_endpoint_allows_unauthenticated_access():
    resp = client.post("/chat", json={"query": "anything"})
    assert resp.status_code == 401
