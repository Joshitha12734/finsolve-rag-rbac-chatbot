"""
Routes an incoming natural-language question to either the SQL agent
(structured, tabular questions) or the RAG agent (everything else).

Two classification strategies, same interface:
  - Heuristic (default, no API calls, instant): keyword/pattern match for
    the shape of a structured query — aggregates, comparisons, "list/show
    me all X where Y", etc.
  - LLM-based (used when the active LLM provider's API key is set and
    the heuristic is unsure): asks the model to pick "sql" or "rag" in
    one word.

Only departments that actually have structured (CSV) data register a SQL
table (currently just HR). The caller is responsible for checking the
user's role has access to that department before running SQL against it.
"""
from __future__ import annotations

import re
from typing import Literal

Route = Literal["sql", "rag"]

# Departments that have a queryable structured table, and roughly what
# their columns mean — used both for routing hints and for prompting the
# SQL agent later.
SQL_ENABLED_DEPARTMENTS = {"hr"}

_AGGREGATE_WORDS = r"(average|avg|sum|total|count|how many|maximum|max|minimum|min|top \d+|highest|lowest)"
_COMPARISON_WORDS = r"(greater than|less than|more than|at least|at most|>=?|<=?|earning over|earning more than)"
_LISTING_WORDS = r"(list all|show me all|list the|show all|give me (a list|the details) of)"

_STRUCTURED_PATTERN = re.compile(
    rf"\b({_AGGREGATE_WORDS}|{_COMPARISON_WORDS}|{_LISTING_WORDS})\b", re.IGNORECASE
)

# Column/field names from hr_data.csv — if these appear alongside a
# structured-shaped clause, that's a strong signal for the SQL path.
_HR_FIELD_HINTS = re.compile(
    r"\b(salary|performance rating|attendance|leave balance|leaves taken|"
    r"department|manager|employee id|date of joining|location)\b",
    re.IGNORECASE,
)


def classify_heuristic(query: str) -> Route:
    has_structured_shape = bool(_STRUCTURED_PATTERN.search(query))
    has_hr_field = bool(_HR_FIELD_HINTS.search(query))
    if has_structured_shape and has_hr_field:
        return "sql"
    # A bare aggregate/comparison question about "employees" is also a
    # reasonable signal even without an exact field name match.
    if has_structured_shape and re.search(r"\bemployees?\b", query, re.IGNORECASE):
        return "sql"
    return "rag"


def classify_with_llm(query: str) -> Route:
    from . import llm_client

    if not llm_client.api_key_configured():
        return classify_heuristic(query)

    prompt = (
        "Classify this employee question as either 'sql' (it asks for filtering, "
        "counting, comparing, or aggregating structured employee records — e.g. "
        "salary, performance rating, attendance, department) or 'rag' (it asks about "
        "policies, reports, architecture, or general narrative content).\n\n"
        f"Question: {query}\n\n"
        "Answer with exactly one word: sql or rag."
    )
    try:
        text = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=5).strip().lower()
        return "sql" if "sql" in text else "rag"
    except Exception:
        # Never let a classifier hiccup break the chat — fall back safely.
        return classify_heuristic(query)


def classify(query: str, use_llm: bool = False) -> Route:
    route = classify_with_llm(query) if use_llm else classify_heuristic(query)
    return route
