"""
SQL Agent: answers structured questions ("employees earning over 100K",
"average performance rating by department") by translating natural
language into SQL and executing it against a DuckDB table built from
the department's CSV data.

Currently only HR has structured (CSV) data, so this agent only ever
registers an `hr` table — but it's written to pick up any future CSV
dropped into a department folder with zero code changes (see
ingest.list_csv_tables below).

RBAC note: the caller (main.py) is responsible for only invoking this
agent when the user's role actually has access to that department. This
module itself has no notion of roles — it just executes SQL against
whatever table it's given, which is why that check must happen before
this is called.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import DATA_DIR, INDEX_DIR
from .query_classifier import SQL_ENABLED_DEPARTMENTS
from . import llm_client

DUCKDB_PATH = INDEX_DIR / "structured_queries.duckdb"

# Only SELECT statements are ever allowed to run — this is a read-only
# assistant, never a means to mutate company data.
_DISALLOWED_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma)\b", re.IGNORECASE)


def list_csv_tables() -> dict[str, Path]:
    """department -> csv path, for every CSV under data/<department>/."""
    tables = {}
    for dept_dir in DATA_DIR.iterdir():
        if not dept_dir.is_dir():
            continue
        for csv_path in dept_dir.glob("*.csv"):
            tables[dept_dir.name] = csv_path  # last one wins if multiple
    return tables


def get_connection() -> duckdb.DuckDBPyConnection:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    for department, csv_path in list_csv_tables().items():
        df = pd.read_csv(csv_path)
        con.register(f"_{department}_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {department} AS SELECT * FROM _{department}_df")
    return con


def get_schema_description(department: str) -> str:
    con = get_connection()
    try:
        cols = con.execute(f"DESCRIBE {department}").fetchdf()
    except duckdb.CatalogException:
        return ""
    lines = [f"Table `{department}` columns:"]
    for _, row in cols.iterrows():
        lines.append(f"  - {row['column_name']} ({row['column_type']})")
    return "\n".join(lines)


def nl_to_sql(question: str, department: str) -> str | None:
    """Translate a natural-language question into a SQL SELECT statement
    against the given department's table, using the active LLM provider.
    Returns None if no API key is configured (caller should fall back to RAG)."""
    if not llm_client.api_key_configured():
        return None

    schema = get_schema_description(department)
    if not schema:
        return None

    prompt = (
        f"{schema}\n\n"
        f"Write a single DuckDB SQL SELECT statement that answers this question:\n"
        f"\"{question}\"\n\n"
        "Rules:\n"
        "- Only a SELECT statement, nothing else.\n"
        "- No markdown code fences, no explanation, no semicolon-separated statements.\n"
        f"- Only query the `{department}` table.\n"
        "- If the question can't be answered with SQL over this table, respond with exactly: NONE"
    )
    sql = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=300).strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()
    if sql.upper() == "NONE" or not sql:
        return None
    return sql


class SqlAgentResult:
    def __init__(self, success: bool, answer: str = "", sql: str = "", rows: int = 0):
        self.success = success
        self.answer = answer
        self.sql = sql
        self.rows = rows


def run_sql_agent(question: str, department: str) -> SqlAgentResult:
    """End-to-end: NL question -> SQL -> execute -> formatted answer.
    Never raises; returns SqlAgentResult(success=False, ...) so the caller
    can cleanly fall back to the RAG path."""
    if department not in SQL_ENABLED_DEPARTMENTS:
        return SqlAgentResult(success=False)

    try:
        sql = nl_to_sql(question, department)
    except Exception as e:
        return SqlAgentResult(success=False, answer=f"SQL generation failed: {e}")

    if not sql:
        return SqlAgentResult(success=False)

    if _DISALLOWED_SQL.search(sql):
        return SqlAgentResult(success=False, answer="Generated SQL used a disallowed statement type.")

    try:
        con = get_connection()
        result_df = con.execute(sql).fetchdf()
    except Exception as e:
        return SqlAgentResult(success=False, sql=sql, answer=f"SQL execution failed: {e}")

    if result_df.empty:
        return SqlAgentResult(success=False, sql=sql, answer="Query ran but returned no rows.")

    # Keep the answer readable: cap at 20 rows in the text response.
    preview = result_df.head(20)
    answer = preview.to_markdown(index=False) if hasattr(preview, "to_markdown") else preview.to_string(index=False)
    if len(result_df) > 20:
        answer += f"\n\n_(showing 20 of {len(result_df)} rows)_"

    return SqlAgentResult(success=True, answer=answer, sql=sql, rows=len(result_df))
