"""
End-to-end UI tests using Playwright, against the real running Streamlit
app + FastAPI backend.

Unlike tests/test_rbac.py (which hits the API directly), these drive an
actual browser to verify the login flow, role display, and query/answer
round-trip work from a real user's perspective — and catch UI regressions
that pure API tests can't.

Setup (one-time):
    pip install playwright pytest-playwright
    playwright install chromium

Before running, start both servers in separate terminals:
    uvicorn backend.main:app --reload --port 8000
    streamlit run frontend/streamlit_app.py

Run:
    pytest tests/test_ui_playwright.py --headed   # --headed to watch it run
    pytest tests/test_ui_playwright.py             # headless (e.g. in CI)

These are NOT run as part of the default `pytest tests/` / CI invocation
(see pytest.ini) since they need both live servers and a browser install —
run them explicitly as a separate manual/CI step once the app is deployed.
"""
import re

import pytest

pytest.importorskip("playwright")

STREAMLIT_URL = "http://localhost:8501"

pytestmark = pytest.mark.e2e  # excluded from default test runs, see pytest.ini


def test_login_and_role_display(page):
    page.goto(STREAMLIT_URL)
    page.get_by_placeholder("e.g. peter.pandey").fill("peter.pandey")
    page.get_by_label("Password", exact=False).fill("engineering123")
    page.get_by_role("button", name="Sign in").click()

    page.wait_for_selector("text=Role: **engineering**", timeout=15000)
    assert page.get_by_text("engineering").first.is_visible()


def test_engineering_user_cannot_get_finance_data_via_ui(page):
    page.goto(STREAMLIT_URL)
    page.get_by_placeholder("e.g. peter.pandey").fill("peter.pandey")
    page.get_by_label("Password", exact=False).fill("engineering123")
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_selector("text=Role: **engineering**", timeout=15000)

    chat_input = page.get_by_placeholder("Ask about your department's data...")
    chat_input.fill("what were vendor services expenses in 2024?")
    chat_input.press("Enter")

    page.wait_for_selector("text=couldn't find", timeout=20000)
    assert not page.get_by_text(re.compile("finance/", re.IGNORECASE)).count()


def test_logout_returns_to_login_screen(page):
    page.goto(STREAMLIT_URL)
    page.get_by_placeholder("e.g. peter.pandey").fill("sam.employee")
    page.get_by_label("Password", exact=False).fill("employee123")
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_selector("text=Log out", timeout=15000)

    page.get_by_role("button", name="Log out").click()
    page.wait_for_selector("text=Sign in", timeout=15000)
