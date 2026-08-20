"""
Tests for the features added on top of the original hybrid engine:
  - BM25 + embedding hybrid retrieval (falls back to BM25-only here, since
    this environment has no Hugging Face access to download embeddings —
    same fallback pattern as before, still fully exercised)
  - Metadata-aware RBAC: a 'confidential'-classified document is
    restricted to c-level even within a department the caller can
    otherwise access
  - Confidence score and retrieved-chunk previews on /chat responses
  - HR analytics endpoint, gated to hr/c-level
"""
import base64

from fastapi.testclient import TestClient

from backend.main import app
from backend.retriever import get_retriever

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_confidential_document_restricted_to_c_level_even_within_department():
    # quarterly_financial_report.md is tagged 'confidential' in
    # data/_classification.json — finance role has the *department*,
    # but not the classification, so it must never see this file.
    resp = client.post(
        "/chat",
        headers=auth_header("priya.finance", "finance123"),
        json={"query": "what is the risk mitigation strategy for Q4 2024?"},
    )
    assert resp.status_code == 200
    assert not any("quarterly_financial_report.md" in s for s in resp.json()["sources"])


def test_confidential_document_visible_to_c_level():
    resp = client.post(
        "/chat",
        headers=auth_header("tony.sharma", "clevel123"),
        json={"query": "what is the risk mitigation strategy for Q4 2024?"},
    )
    assert resp.status_code == 200
    assert any("quarterly_financial_report.md" in s for s in resp.json()["sources"])


def test_cross_department_query_correctly_denied_not_answered_from_weak_match():
    # Regression test for a real bug: "what is the quarterly revenue?" asked
    # by an engineering-role user was answering from an unrelated HR FAQ
    # section (weak/spurious BM25 keyword overlap) instead of correctly
    # saying the real answer isn't in a department this role can access.
    # Root cause was two-fold: no punctuation stripping in tokenization
    # ("revenue?" never matched "revenue"), and no stopword removal (common
    # words dominated scores for text-dense chunks). Both are fixed in
    # retriever._tokenize(); this test guards the end-to-end behavior.
    resp = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "what is the quarterly revenue?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert not any(s.startswith("finance/") for s in body["sources"])
    assert not any(s.startswith("general/") for s in body["sources"])  # no weak-match answer either
    assert body["confidence_pct"] == 0.0
    assert "access" in body["answer"].lower() or "role has access" in body["answer"].lower()

    # The same question from a role that actually has finance access should
    # still get the real answer, confidently.
    resp2 = client.post(
        "/chat",
        headers=auth_header("priya.finance", "finance123"),
        json={"query": "what is the quarterly revenue?"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert any(s.startswith("finance/") for s in body2["sources"])
    assert body2["confidence_pct"] > 0


def test_chat_response_includes_confidence_and_chunk_previews():
    resp = client.post(
        "/chat",
        headers=auth_header("peter.pandey", "engineering123"),
        json={"query": "what CI/CD tools does engineering use?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "confidence_pct" in body
    assert 0.0 <= body["confidence_pct"] <= 100.0
    if body["route"] == "rag" and body["sources"]:
        assert len(body["retrieved_chunks"]) > 0
        chunk = body["retrieved_chunks"][0]
        assert set(chunk.keys()) >= {"source", "department", "score", "preview"}


def test_confidence_is_zero_for_truly_unmatched_query():
    resp = client.post(
        "/chat",
        headers=auth_header("sam.employee", "employee123"),
        json={"query": "zzqxw blorptastic wibbleflorp"},
    )
    assert resp.status_code == 200
    assert resp.json()["confidence_pct"] == 0.0


def test_hybrid_retriever_confidence_is_bounded_0_to_100():
    # Regression test for the bug where raw BM25 scores (unbounded) were
    # clamped straight into a percentage, pinning everything to 100%.
    retriever = get_retriever()
    strong_score = 8.0  # a realistic strong BM25 score for this corpus
    weak_score = 0.3
    assert 0.0 <= retriever.confidence(strong_score) <= 1.0
    assert 0.0 <= retriever.confidence(weak_score) <= 1.0
    assert retriever.confidence(strong_score) > retriever.confidence(weak_score)


def test_hr_analytics_accessible_to_hr_and_c_level():
    for creds in [("anita.hr", "hr123"), ("tony.sharma", "clevel123")]:
        resp = client.get("/hr/analytics", headers=auth_header(*creds))
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["total_employees"] == 100
        assert "avg_performance_rating_overall" in body
        assert "headcount_by_department" in body


def test_hr_analytics_blocked_for_other_roles():
    resp = client.get("/hr/analytics", headers=auth_header("peter.pandey", "engineering123"))
    assert resp.status_code == 403
