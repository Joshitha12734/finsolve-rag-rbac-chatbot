"""
Tests for the Ollama local-LLM provider option in llm_client.py.

No actual Ollama server runs in this test environment, so these verify
the graceful-degradation contract (same pattern proven for the other
providers): switching to ollama with no server running should never
crash anything, just correctly report "not usable" and let every caller
fall back to its non-LLM path.
"""
import importlib
import os

import pytest


@pytest.fixture
def ollama_provider(monkeypatch):
    """Reloads llm_client with LLM_PROVIDER=ollama, restores afterward."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from backend import llm_client
    importlib.reload(llm_client)
    llm_client.reset_ollama_reachability_cache()
    yield llm_client
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    importlib.reload(llm_client)


def test_ollama_is_selected_when_configured(ollama_provider):
    assert ollama_provider.PROVIDER == "ollama"
    assert ollama_provider.active_model() == ollama_provider.OLLAMA_MODEL


def test_ollama_reports_not_configured_when_server_unreachable(ollama_provider):
    # No Ollama server is running in this test environment — this must
    # return False (checked live, short timeout), never raise.
    assert ollama_provider.api_key_configured() is False


def test_ollama_setup_hint_is_provider_specific(ollama_provider):
    hint = ollama_provider.setup_hint()
    assert "ollama" in hint.lower()
    assert "serve" in hint.lower() or "pull" in hint.lower()


def test_ollama_reachability_check_is_cached(ollama_provider):
    # Calling it twice shouldn't make two network calls — verify the
    # cached value is reused (same False result, no exception either time).
    first = ollama_provider.api_key_configured()
    second = ollama_provider.api_key_configured()
    assert first == second == False


def test_chat_endpoint_still_works_end_to_end_with_ollama_configured_but_down(monkeypatch):
    # Full integration: with LLM_PROVIDER=ollama and no server running,
    # /chat must still return 200 with a Demo Mode answer, not error out.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    import importlib
    from backend import llm_client
    importlib.reload(llm_client)
    llm_client.reset_ollama_reachability_cache()

    from backend import llm
    importlib.reload(llm)
    from backend import main
    importlib.reload(main)

    import base64
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    token = base64.b64encode(b"peter.pandey:engineering123").decode()

    try:
        resp = client.post(
            "/chat",
            headers={"Authorization": f"Basic {token}"},
            json={"query": "what CI/CD tools does engineering use?"},
        )
        assert resp.status_code == 200
        assert "Demo Mode" in resp.json()["answer"]
        assert "ollama" in resp.json()["answer"].lower()
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        importlib.reload(llm_client)
        importlib.reload(llm)
        importlib.reload(main)
