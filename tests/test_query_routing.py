"""
Tests for the hybrid SQL/RAG query routing.

Since these run without ANTHROPIC_API_KEY (CI has no LLM key), the SQL
agent's NL-to-SQL step can't actually execute — so these tests verify
the two things that don't require an LLM call:
  1. The classifier correctly identifies structured vs unstructured questions.
  2. RBAC is enforced at the routing layer itself — a role without access
     to `hr` never even attempts the SQL agent against the hr table,
     regardless of what the LLM might have generated.
"""
import base64

from backend.main import app, _try_sql_path
from backend.query_classifier import classify, SQL_ENABLED_DEPARTMENTS
from fastapi.testclient import TestClient

client = TestClient(app)


def auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_classifier_routes_structured_questions_to_sql():
    structured_questions = [
        "What is the average performance rating across the company?",
        "List all employees with performance rating 5 in the Data department",
        "How many employees have attendance below 90%?",
        "Show me employees earning more than 1000000",
    ]
    for q in structured_questions:
        assert classify(q) == "sql", f"expected sql for: {q}"


def test_classifier_routes_narrative_questions_to_rag():
    narrative_questions = [
        "What CI/CD tools does engineering use?",
        "What is the leave policy?",
        "Summarize the Q2 2024 marketing campaign performance.",
    ]
    for q in narrative_questions:
        assert classify(q) == "rag", f"expected rag for: {q}"


def test_sql_agent_never_attempted_for_roles_without_hr_access():
    # Engineering role has no 'hr' in its allowed departments, so the SQL
    # path must be skipped entirely at the routing layer — independent of
    # whether an LLM/API key is even configured.
    allowed_departments = {"engineering", "general"}
    result = _try_sql_path("list all employees with salary over 1000000", allowed_departments)
    assert result is None


def test_sql_enabled_departments_is_hr_only_for_current_dataset():
    # Sanity check that reflects the current data folder (only hr/ has a
    # CSV). If you add more structured datasets under data/<dept>/*.csv,
    # this set should grow automatically — update this test accordingly.
    assert SQL_ENABLED_DEPARTMENTS == {"hr"}


def test_chat_endpoint_includes_route_field():
    resp = client.post(
        "/chat",
        headers=auth_header("anita.hr", "hr123"),
        json={"query": "what is the leave policy?"},
    )
    assert resp.status_code == 200
    assert resp.json()["route"] in ("sql", "rag")
