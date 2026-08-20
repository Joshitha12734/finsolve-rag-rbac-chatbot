"""
Tests for the feedback (👍/👎) system.
"""
import base64

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


ADMIN = ("tony.sharma", "clevel123")
NON_ADMIN = ("peter.pandey", "engineering123")


def test_submit_thumbs_up():
    resp = client.post(
        "/feedback",
        headers=auth_header(*NON_ADMIN),
        json={"query": "what CI/CD tools do we use?", "answer": "Jenkins, ArgoCD", "rating": "up"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


def test_submit_thumbs_down_with_reason():
    resp = client.post(
        "/feedback",
        headers=auth_header(*NON_ADMIN),
        json={"query": "vague question", "answer": "I couldn't find that", "rating": "down", "reason": "didnt_answer"},
    )
    assert resp.status_code == 200


def test_rejects_invalid_rating():
    resp = client.post(
        "/feedback",
        headers=auth_header(*NON_ADMIN),
        json={"query": "x", "answer": "y", "rating": "sideways"},
    )
    assert resp.status_code == 400


def test_rejects_invalid_reason():
    resp = client.post(
        "/feedback",
        headers=auth_header(*NON_ADMIN),
        json={"query": "x", "answer": "y", "rating": "down", "reason": "not_a_real_reason"},
    )
    assert resp.status_code == 400


def test_non_admin_cannot_view_feedback_summary():
    resp = client.get("/admin/feedback/summary", headers=auth_header(*NON_ADMIN))
    assert resp.status_code == 403


def test_admin_feedback_summary_reflects_submissions():
    client.post(
        "/feedback",
        headers=auth_header(*NON_ADMIN),
        json={"query": "q1", "answer": "a1", "rating": "up"},
    )
    resp = client.get("/admin/feedback/summary", headers=auth_header(*ADMIN))
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_feedback"] >= 1
    assert "satisfaction_pct" in body


def test_admin_can_see_recent_feedback_entries():
    resp = client.get("/admin/feedback/recent", headers=auth_header(*ADMIN))
    assert resp.status_code == 200
    assert "entries" in resp.json()
