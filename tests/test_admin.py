"""
Tests for the admin panel: document upload with department tagging,
dynamic re-indexing, and user management — all gated to the c-level role.
"""
import base64

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import USERS

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


ADMIN = ("tony.sharma", "clevel123")
NON_ADMIN = ("peter.pandey", "engineering123")


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Make sure a test user/document doesn't leak between tests."""
    yield
    USERS.pop("temp.test.user", None)
    from backend.admin import delete_document
    delete_document("engineering", "temp_test_doc.md")


def test_non_admin_forbidden_from_all_admin_endpoints():
    endpoints = [
        ("get", "/admin/users"),
        ("get", "/admin/documents"),
        ("post", "/admin/reindex"),
    ]
    for method, path in endpoints:
        resp = getattr(client, method)(path, headers=auth_header(*NON_ADMIN))
        assert resp.status_code == 403, f"{method.upper()} {path} should be 403 for non-admin"


def test_admin_can_list_users():
    resp = client.get("/admin/users", headers=auth_header(*ADMIN))
    assert resp.status_code == 200
    usernames = [u["username"] for u in resp.json()["users"]]
    assert "peter.pandey" in usernames


def test_admin_can_create_and_remove_user():
    resp = client.post(
        "/admin/users",
        headers=auth_header(*ADMIN),
        json={"username": "temp.test.user", "password": "temp123", "role": "marketing", "full_name": "Temp User"},
    )
    assert resp.status_code == 200

    # the new user can immediately authenticate
    resp = client.get("/me", headers=auth_header("temp.test.user", "temp123"))
    assert resp.status_code == 200
    assert resp.json()["role"] == "marketing"

    resp = client.delete("/admin/users/temp.test.user", headers=auth_header(*ADMIN))
    assert resp.status_code == 200

    resp = client.get("/me", headers=auth_header("temp.test.user", "temp123"))
    assert resp.status_code == 401


def test_admin_cannot_create_user_with_invalid_role():
    resp = client.post(
        "/admin/users",
        headers=auth_header(*ADMIN),
        json={"username": "bad.role.user", "password": "x", "role": "not-a-real-role", "full_name": "Bad"},
    )
    assert resp.status_code == 400


def test_admin_cannot_delete_own_account():
    resp = client.delete(f"/admin/users/{ADMIN[0]}", headers=auth_header(*ADMIN))
    assert resp.status_code == 400


def test_upload_document_and_immediate_retrieval():
    resp = client.post(
        "/admin/documents",
        headers=auth_header(*ADMIN),
        data={"department": "engineering"},
        files={"file": ("temp_test_doc.md", b"# Canary Deploys\n\nAll releases require a canary rollout stage.", "text/markdown")},
    )
    assert resp.status_code == 200
    assert resp.json()["reindexed"] is True

    # engineering role should now be able to retrieve it
    resp = client.post(
        "/chat",
        headers=auth_header(*NON_ADMIN),
        json={"query": "what is the canary rollout stage policy?"},
    )
    assert resp.status_code == 200
    assert any("temp_test_doc.md" in s for s in resp.json()["sources"])


def test_non_admin_cannot_upload_documents():
    resp = client.post(
        "/admin/documents",
        headers=auth_header(*NON_ADMIN),
        data={"department": "engineering"},
        files={"file": ("hack.md", b"malicious content", "text/markdown")},
    )
    assert resp.status_code == 403


def test_upload_rejects_unknown_department():
    resp = client.post(
        "/admin/documents",
        headers=auth_header(*ADMIN),
        data={"department": "not-a-department"},
        files={"file": ("x.md", b"content", "text/markdown")},
    )
    assert resp.status_code == 400
