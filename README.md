# FinSolve Internal Assistant — Role-Based RAG Chatbot

A Retrieval-Augmented Generation chatbot for FinSolve Technologies that gives
every employee role-specific, cited answers from internal documents —
Finance, Marketing, HR, Engineering and Executive data — while enforcing
strict role-based access control (RBAC) so nobody sees data outside their
department.


## Why this architecture

The core design decision is: **filter by role (and now, by document
classification) at retrieval time, not at prompt time.** Every document
chunk (and every structured table) is tagged with a `department`, and
optionally a finer-grained `classification` (see `data/_classification.json`
— e.g. a document can be marked `confidential` and become c-level-only even
*within* a department whose role would otherwise grant access). When a user
asks a question, we compute what they're allowed to see *first*, and the
SQL agent, BM25 index, and embedding search only ever touch that allowed
set. The LLM never even sees data outside the user's permission — so
there's no way for a clever prompt to "jailbreak" access to another
department's (or classification tier's) data, because it was never
retrieved in the first place.

The second key decision is a **hybrid query engine at three levels**:
1. **Action vs SQL vs RAG** — before even asking "what does the user want
   to know", the assistant checks "does the user want to *do* something?"
   A message like "I want to submit a leave request" isn't a question to
   answer, it's a workflow to trigger — the "Virtual Agent" / "Now Assist"
   layer (`workflow_engine.py`) checks the message against a
   department-filtered, admin-defined list of actions before anything
   else runs.
2. **SQL vs RAG** — not every remaining question is best answered by
   document retrieval. "What's our CI/CD process?" wants a document.
   "List employees with performance rating 5 in Data" wants a precise,
   filterable answer over a table — something RAG is bad at and SQL is
   exactly built for. A lightweight classifier routes each question,
   with automatic fallback to RAG if the SQL path fails.
3. **BM25 vs embeddings, within RAG** — lexical search (BM25) is strong on
   exact keyword/entity matches; semantic embeddings are strong on
   paraphrases and conceptual matches. Both signals are fused (normalized,
   weighted sum), and the top fused candidates get an optional LLM
   reranking pass for a final relevance check before generation.

```
┌─────────────┐   HTTP Basic Auth    ┌──────────────┐
│  Streamlit  │ ───────────────────► │   FastAPI    │
│   chat UI   │ ◄─────────────────── │   backend    │
└─────────────┘  answer+sources+     └──────┬───────┘
                  confidence+chunks+         │
                  route(sql|rag)     role → allowed departments
                                             │
                                   ┌─────────▼─────────┐
                                   │  Query Classifier   │
                                   │  (structured shape?) │
                                   └─────┬─────────┬─────┘
                                         │sql       │rag
                              ┌──────────▼──┐   ┌───▼───────────────────┐
                              │  SQL Agent   │   │   Hybrid Retriever      │
                              │  NL→SQL(LLM) │   │  BM25 ⊕ Embeddings       │
                              │  → DuckDB     │   │  (dept + classification   │
                              │  (hr table)   │   │   RBAC filter applied)     │
                              └──────┬───────┘   └─────────┬──────────────────┘
                                     │ fail/empty            │ fused top-N
                                     │                        ▼
                                     │              ┌───────────────────┐
                                     │              │  LLM Reranker       │
                                     │              │  (optional, LLM)     │
                                     │              └─────────┬─────────────┘
                                     │                         │ top-k chunks
                                     └──────────►──────────────┤
                                                                ▼
                                                     ┌───────────────────┐
                                                     │  LLM (Groq/Anthropic)│
                                                     │  grounded answer +  │
                                                     │  cited sources +     │
                                                     │  confidence score     │
                                                     └───────────────────┘
```

## Roles & permissions

| Role          | Can access                                             |
|---------------|---------------------------------------------------------|
| `engineering` | Engineering docs + general company info                 |
| `finance`     | Finance docs + general company info*                    |
| `marketing`   | Marketing docs + general company info                   |
| `hr`          | HR docs + general company info                          |
| `c-level`     | Everything, including `confidential`-classified docs     |
| `employee`    | General company info only (policies, FAQs, events)       |

\* Department access is necessary but not always sufficient: a document can
additionally be tagged `confidential` in `data/_classification.json`,
which restricts it to `c-level` regardless of department. Right now
`finance/quarterly_financial_report.md` is tagged this way as a worked
example — finance can see `financial_summary.md` but not the quarterly
report, which is c-level-only. See "Why this architecture" above.

Defined in `backend/config.py` as `ROLE_PERMISSIONS` — add a new role or
department there and the whole pipeline (retrieval, filtering, UI) picks
it up automatically.

## Demo accounts

| Username         | Password        | Role          |
|-------------------|-----------------|---------------|
| peter.pandey       | engineering123  | Engineering   |
| priya.finance       | finance123      | Finance       |
| raj.marketing        | marketing123    | Marketing     |
| anita.hr             | hr123           | HR            |
| tony.sharma          | clevel123       | C-Level       |
| sam.employee          | employee123     | Employee      |

Passwords are demo-only (sha256-hashed in `config.py`). Swap this module
out for a real identity provider before deploying anywhere real.

## Tech stack

- **FastAPI** — backend API, HTTP Basic Auth, request validation (Pydantic)
- **Workflow Automation Engine ("Virtual Agent" layer)** — beyond
  answering questions, the assistant can execute predefined enterprise
  actions (leave requests, IT ticket escalation, expense reimbursement).
  Actions are defined as JSON (name, trigger description, required
  fields, department restriction, confirmation template) — a genuinely
  **low-code** builder: adding a new action is a form submission, not a
  code change. Requests are durably recorded (SQLite) with auto-incrementing
  IDs, not just chat replies that evaporate.
- **Query Classifier** — routes each question to SQL or RAG (heuristic by
  default; LLM-based classification available when a key is set)
- **SQL Agent** — the active LLM (Groq by default) translates natural
  language into SQL, executed
  read-only against **DuckDB** tables built from department CSVs (currently
  `hr_data.csv`); falls back to RAG on failure or empty result
- **Hybrid Retrieval (RAG path)**:
  - **BM25** (`rank_bm25`) — lexical/keyword signal, pure Python, no
    external downloads, always available
  - **Embeddings** — `sentence-transformers` (`all-MiniLM-L6-v2`) in a
    persistent **Chroma** vector store, semantic signal — needs internet on
    first run to download the model
  - Both are fused (normalized, weighted) when embeddings are available;
    falls back to BM25-only otherwise (same graceful degradation pattern)
  - **LLM Reranking** — an optional LLM pass re-scores the fused
    top-N candidates for relevance right before generation
  - **Classification-aware RBAC** — filtering happens on department *and*
    a per-document `classification` tag, both before ranking ever sees a
    chunk's text
- **HR Analytics** — pandas-computed aggregates (headcount, performance,
  salary, attendance) beyond raw SQL rows, gated to `hr`/`c-level`
- **LLM Provider (Groq default, Anthropic or Ollama optional)** —
  `backend/llm_client.py` centralizes every LLM call (answer generation,
  NL-to-SQL, query classification, reranking, evaluation) behind one
  interface. Three providers, one line to switch:
  - **Groq** (default) — `llama-3.3-70b-versatile`, free API tier, fast
  - **Anthropic** — `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`, Claude models
  - **Ollama** — `LLM_PROVIDER=ollama`, fully local, no API key, no
    internet dependency once a model is pulled (`ollama pull llama3.2`).
    "Configured" here means *the local server is actually reachable*
    (checked live, short timeout) rather than *a key is set* — there's no
    key. Falls back to Demo Mode with provider-specific setup instructions
    (e.g. "run `ollama serve`") if the server isn't running, same
    graceful-degradation pattern as every other optional backend in this
    project.

  No other code changes needed to switch — generates the final grounded,
  cited answer from the retrieved, role-filtered context (RAG path) or
  formats the SQL result (SQL path), regardless of which provider is active.
- **Reranking** — two-tier, both graceful-fallback: a real **HF cross-encoder**
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, scores query+passage jointly —
  more accurate than comparing independent vectors) is tried first; if it
  can't load (no internet, no `sentence-transformers`), falls back to the
  existing LLM-based reranking pass; if neither is available, results stay
  in fused-score order. Which method actually ran is reported in every
  `/chat` response (`rerank_method`).
- **Security — prompt-injection defense** (`security.py`) — retrieved
  document content is treated as data, never instructions. Every chunk is
  pattern-scanned before it reaches the LLM context; anything that looks
  like "ignore previous instructions" / "reveal confidential data" /
  "act as an administrator" is stripped out and audit-logged, and the
  system prompt explicitly tells the model to treat document content as
  data even if it contains command-like text.
- **Feedback** (`feedback.py`) — 👍/👎 on every answer, with a reason
  prompt on 👎 (wrong answer, wrong source, outdated, didn't answer,
  access issue). Stored separately from the audit trail; aggregated in
  the Admin tab (satisfaction %, breakdown by reason) — the actual
  "feedback → evaluation → improvement" loop a real product would need.
- **Observability** — every `/chat` response includes a per-stage latency
  breakdown (action detection, classification, retrieval+reranking,
  security scan, LLM generation) and which rerank method ran, viewable
  per-message in the chat UI's "Why this answer?" panel.
- **Streamlit** — chat UI with login, role badges, dark mode, confidence
  indicators, retrieved-passage viewer, source citations, route indicator,
  feedback buttons, explainability panel, HR Insights dashboard, Admin
  dashboard with feedback/audit analytics
- **pytest** — 74 backend tests covering RBAC, routing, classification,
  confidence scoring, security, feedback, and analytics correctness
- **Playwright** — end-to-end browser tests against the live UI (optional,
  separate from the default test run — see Testing below)
- **Docker** — `Dockerfile.backend` + `Dockerfile.frontend` +
  `docker-compose.yml` to run both services together (see Deployment below)

## Project layout

```
app/
├── backend/
│   ├── config.py            # roles, permissions, demo users, classification RBAC
│   ├── auth.py               # HTTP Basic Auth -> user + role
│   ├── ingest.py               # loads & chunks documents from data/, attaches classification
│   ├── query_classifier.py      # routes questions: sql vs rag
│   ├── workflow_engine.py         # low-code action definitions + execution (Virtual Agent layer)
│   ├── sql_agent.py                 # NL -> SQL -> DuckDB (structured data)
│   ├── hr_analytics.py              # pandas aggregates over HR data
│   ├── admin.py                       # document upload/list/delete (role-tagged)
│   ├── audit.py                         # SQLite audit log: who asked what, when, allowed?
│   ├── security.py                        # prompt-injection detection + chunk sanitization
│   ├── feedback.py                          # 👍/👎 storage + aggregation
│   ├── retriever.py                           # BM25 + embeddings hybrid, cross-encoder/LLM reranking, RBAC-filtered
│   ├── llm.py                                   # LLM call: context + question -> answer
│   ├── llm_client.py                              # provider abstraction (Groq default, Anthropic optional)
│   └── main.py                                    # FastAPI app, routing, per-stage timing & endpoints
├── frontend/
│   └── streamlit_app.py       # chat UI: badges, dark mode, confidence, HR/Admin dashboards
├── evaluation/
│   └── evaluate.py              # QA-pair generation + faithfulness/relevance/conciseness scoring
├── tests/
│   ├── test_rbac.py             # RBAC correctness (16 tests)
│   ├── test_query_routing.py     # SQL/RAG classifier + routing RBAC (5 tests)
│   ├── test_admin.py               # admin panel: upload/users, RBAC-gated (8 tests)
│   ├── test_audit.py                 # audit logging + analytics endpoints (5 tests)
│   ├── test_hybrid_and_analytics.py    # classification RBAC, confidence, HR analytics (8 tests)
│   ├── test_workflow_engine.py           # low-code actions, RBAC, heuristic fallback (12 tests)
│   ├── test_security_and_observability.py  # prompt-injection defense, stage timing (6 tests)
│   ├── test_feedback.py                       # 👍/👎 storage + aggregation, RBAC (6 tests)
│   ├── test_ollama_provider.py                  # local LLM provider graceful degradation (5 tests)
│   └── test_ui_playwright.py                     # optional browser e2e tests
├── data/                        # source documents (by department)
│   ├── engineering/
│   ├── finance/
│   ├── general/
│   ├── hr/
│   ├── marketing/
│   ├── _classification.json     # per-document sensitivity tags beyond department
│   └── _workflow_actions.json     # low-code workflow action definitions
├── vectorstore/                  # generated index + DuckDB file (gitignored)
├── logs/                          # SQLite audit trail (gitignored)
├── .github/workflows/ci.yml       # runs pytest on every push
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd app
python -m venv venv && source venv/bin/activate      # or your preferred env tool
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (free — get one at console.groq.com/keys)
# LLM_PROVIDER defaults to "groq"; set it to "anthropic" + ANTHROPIC_API_KEY instead if you prefer Claude
```

### Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The first request will build the retrieval index automatically (a few
seconds for BM25; the embedding backend additionally downloads the
`all-MiniLM-L6-v2` model on first run — needs internet once — after which
retrieval runs as the full BM25+embeddings hybrid).

### Run the frontend

In a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Open the URL Streamlit prints, log in with one of the demo accounts above,
and start asking questions.


## Deployment with Docker

```bash
cp .env.example .env   # set GROQ_API_KEY as usual
docker compose up --build
```

This builds and runs both services (`Dockerfile.backend`,
`Dockerfile.frontend`, orchestrated by `docker-compose.yml`):
- Backend on `http://localhost:8000`
- Frontend on `http://localhost:8501`, pointed at the backend automatically
  via the `BACKEND_URL` environment variable

`vectorstore/` and `logs/` are Docker volumes so the search index and audit
trail survive container restarts; `./data` is bind-mounted so documents
uploaded through the Admin panel land on your host filesystem too, not just
inside the container.

## Audit logging & analytics (c-level only)

Every `/chat` call is logged to `logs/audit_log.db` (SQLite, separate from
the search index so rebuilding it never touches the audit trail): who
asked, their role, the query, which engine answered (SQL vs RAG), whether
they actually got data back or nothing matched/was permitted, and latency.

The **Analytics** section at the top of the Admin tab surfaces this as:
- Total queries, denied/no-match count and rate, average latency, SQL vs RAG split
- Queries by role and most-accessed documents (bar charts)
- A recent query log (timestamp, user, role, route, query, latency)

Backed by `GET /admin/analytics/summary` and `GET /admin/analytics/recent`
(both `require_admin`-gated), covered by `tests/test_audit.py`.

## Admin panel (c-level only)

Log in as `tony.sharma` (c-level) to see an **Admin** tab alongside Chat:

- **Document upload with role tagging** — pick a department, upload a
  `.md`/`.txt`/`.csv` file, and it's immediately searchable — no restart,
  no manual re-index step. Documents can also be deleted from the same view.
- **User management** — create new users with a username/password/role/full
  name, view the current user directory, and remove users (you can't
  remove your own account).

Both are backed by real endpoints (not just UI mockups): `POST
/admin/documents`, `DELETE /admin/documents/{department}/{filename}`,
`POST /admin/users`, `DELETE /admin/users/{username}`, all gated behind a
`require_admin` dependency that 403s anyone who isn't `c-level` —
covered by `tests/test_admin.py`.

Note: uploaded documents and created users are **in-memory / on-disk within
this run** — user accounts reset on server restart (see `config.py`),
though uploaded documents persist as real files under `data/<department>/`
since they're written straight to disk.

## Workflow Automation — the Virtual Agent layer (c-level builds, all roles use)

Beyond answering questions, the assistant can **execute** predefined
enterprise actions — this is the part of the system that maps most
directly to "Now Assist" / "Virtual Agent" / "Workflow Automation" /
"Low-Code Application Development":

- **Trigger an action just by asking**: "I want to submit a leave request
  for next week" or "I need to escalate an IT issue" — the assistant
  detects the intent, asks for any missing required fields (multi-turn),
  then durably records the request and confirms it with a generated
  request ID.
- **Low-code builder**: c-level users define new actions from the Admin
  tab — name, a plain-language trigger description, trigger keywords
  (comma-separated, used for heuristic matching without an LLM), required
  fields, which department can use it, and a confirmation message
  template with `{field}` placeholders. No Python required; the action is
  triggerable in chat immediately.
- **RBAC applies here too**: actions are department-restricted exactly
  like documents — `expense_reimbursement` is finance-only, so it's
  filtered out of what the LLM even sees when a `general`-role user's
  message is checked for intent (`tests/test_workflow_engine.py`
  verifies this).
- **Durable, auditable records**: submitted requests go into
  `logs/workflow_requests.db` (separate SQLite file from the audit
  trail — different lifecycle: these are business records, not
  observability data), viewable from the Admin tab.

Seed actions ship in `data/_workflow_actions.json`: `leave_request`,
`it_ticket_escalation`, `expense_reimbursement` — edit that file directly,
or use the Admin tab's form.

## HR Insights (hr and c-level)

Log in as `anita.hr` or `tony.sharma` to see an **HR Insights** tab:
headcount by department, average performance rating overall and by
department, salary distribution (min/median/mean/max) and by department,
attendance and leave averages, and headcount by location — all computed
with pandas over the full HR dataset (`backend/hr_analytics.py`), not just
whatever rows a single SQL query happens to return.

Backed by `GET /hr/analytics`, gated to `hr`/`c-level` via
`require_hr_or_admin`, covered by `tests/test_hybrid_and_analytics.py`.

## Usage examples

**Workflow/action path (the Virtual Agent layer) — try these:**
- "I want to submit a leave request from Aug 10 to Aug 12 for a family event" → executes immediately, all fields present
- "I need to escalate an IT issue" → assistant asks for the missing fields (issue description, urgency) before executing
- "I want an expense reimbursement" as `sam.employee` → not offered — this action is finance-only, filtered out before the LLM ever sees it

**RAG path (unstructured documents, BM25+embeddings hybrid + reranking):**
- **Engineering:** "What's our microservices architecture look like?" → `engineering_master_doc.md`
- **Finance:** "What drove the increase in vendor costs in 2024?" → `financial_summary.md`
- **Marketing:** "How did Q2 2024 campaigns perform?" → `marketing_report_q2_2024.md`
- **Employee:** "What's the leave policy?" → `employee_handbook.md` (general, available to everyone)
- **C-level only:** "What is the risk mitigation strategy for Q4 2024?" → `quarterly_financial_report.md` (confidential — try this as `priya.finance` too, and note it correctly comes back empty/denied despite Finance department access)

**SQL path (structured HR data) — HR and C-level only:**
- "List all employees with performance rating 5 in the Data department" → generates and runs SQL against the `hr` DuckDB table, returns a formatted table
- "What is the average performance rating across the company?"
- "How many employees have attendance below 90%?"
- "Show me employees earning more than 1,000,000"

Try the same structured questions as `peter.pandey` (engineering) — the SQL
agent is never even attempted, because `engineering` isn't in
`SQL_ENABLED_DEPARTMENTS ∩` the user's allowed departments; it goes straight
to RAG and (correctly) can't find HR salary data.

**HR analytics dashboard** — log in as `anita.hr` or `tony.sharma`, open
the **HR Insights** tab: headcount by department, salary distribution,
performance ratings, attendance and leave averages, all computed live from
`hr_data.csv`, no chat query needed.

## RAG evaluation framework

`evaluation/evaluate.py` automates quality measurement instead of relying on
spot-checking:

```bash
python -m evaluation.evaluate
```

1. **Generates QA pairs** — samples a few chunks per department, asks the
   LLM to write a realistic employee question + reference answer for each,
   saved to `evaluation/qa_pairs.csv`.
2. **Runs each question through the real pipeline** (same retriever + LLM
   call path as `/chat`), then asks the LLM to score the actual answer
   against the reference on **faithfulness** (grounded, not hallucinated),
   **relevance** (addresses the question), and **conciseness** (1-5 each).
3. Saves per-question scores to `evaluation/evaluation_results.csv` and
   prints department-level averages — useful for comparing prompt/retrieval
   changes over time, or for justifying design choices in your presentation.

## Testing

RBAC correctness isn't just asserted, it's tested — `tests/test_rbac.py`,
`tests/test_query_routing.py`, `tests/test_admin.py`, `tests/test_audit.py`,
and `tests/test_hybrid_and_analytics.py`, and `tests/test_workflow_engine.py`
run real queries through the real
ingested corpus for every demo role:

```bash
pip install pytest   # already in requirements.txt
pytest -v tests/
```

74 tests, including: engineering can't retrieve finance data (and vice
versa), employee role is blocked from row-level HR records that the HR role
can see, c-level can reach every department, a `confidential`-classified
document is blocked from finance despite department access but visible to
c-level, the SQL agent is never even attempted for roles without HR access
(regardless of what an LLM might generate), the query classifier correctly
separates structured vs narrative questions, confidence scores are
correctly bounded 0-100% for both strong and weak/no matches, workflow
actions are RBAC-filtered before the LLM ever sees them and correctly
recorded with auto-incrementing request IDs on execution,
unauthenticated requests are rejected, non-admins get a 403 from every
admin endpoint while admins can upload a document and query it
immediately, and HR analytics is reachable by `hr`/`c-level` only. A
GitHub Actions workflow (`.github/workflows/ci.yml`) runs this suite on
every push/PR.

### Browser end-to-end tests (Playwright)

`tests/test_ui_playwright.py` drives a real browser against the live
Streamlit + FastAPI app to catch UI-level regressions (login flow, role
display, RBAC as experienced by an actual user) that API tests can't. These
are excluded from the default `pytest tests/` run (see `pytest.ini`) since
they need both servers running and a browser installed:

```bash
pip install playwright pytest-playwright
playwright install chromium

# in separate terminals:
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py

# then:
pytest tests/test_ui_playwright.py --headed
```

## Extensibility notes

- **New department/role:** add a folder under `data/`, add an entry to
  `ROLE_PERMISSIONS` in `config.py`. No other code changes needed.
- **Bigger corpus:** the hybrid retriever already blends BM25 + embeddings
  when both are available; BM25-only is a reasonable fallback for a few
  hundred chunks but embeddings matter more as the corpus grows.
- **Real auth:** swap `backend/auth.py` for OAuth/SSO/JWT against your
  identity provider; `get_current_user()` is the only integration point
  the rest of the app depends on.
- **More structured datasets:** drop another CSV into any `data/<dept>/`
  folder and `sql_agent.list_csv_tables()` picks it up automatically on
  next connection — add the department to `SQL_ENABLED_DEPARTMENTS` in
  `query_classifier.py` to route questions to it.
- **Finer-grained classification:** `data/_classification.json` currently
  only distinguishes `confidential` from the default; add more tiers in
  `config.CLASSIFICATION_LEVELS` and extend `role_can_access_classification()`
  as access policies get more complex (e.g. per-quarter, per-region).

## Innovation / notable design choices

- **Agentic workflow automation (Virtual Agent layer)**: the assistant
  doesn't just answer questions, it executes predefined enterprise
  actions (leave requests, IT escalation, reimbursement) with multi-turn
  field collection and durable request recording. Actions are defined
  via a genuinely low-code JSON/form interface — no code changes needed
  to add a new one — and RBAC-filtered before the LLM ever evaluates
  intent against them.
- **Hybrid SQL + RAG query engine with fallback**: structured questions
  ("employees earning over 100K", "average performance rating by
  department") are answered precisely via generated SQL over DuckDB,
  instead of forcing everything through document retrieval, which is bad
  at aggregation and filtering. Falls back to RAG automatically if the SQL
  path fails, returns nothing, or isn't permitted for the user's role.
- **Hybrid BM25 + embedding retrieval with LLM reranking**: lexical (BM25)
  and semantic (embeddings) signals are fused with normalized weighted
  scoring rather than picking one, and an optional LLM reranking pass
  re-scores the fused top candidates for relevance right before
  generation — closer to a production retrieval stack than a single
  similarity search.
- **Metadata-aware RBAC beyond department folders**: documents carry an
  optional `classification` tag (`data/_classification.json`) that can
  restrict access further within a department a role already has — e.g.
  a board-only quarterly report that most of Finance itself can't see.
  This is layered on top of, not instead of, department-level RBAC.
- **Calibrated confidence scoring**: retrieval scores aren't on a
  consistent 0-1 scale across backends (BM25 is unbounded, cosine
  similarity roughly is), so each retriever implements its own
  `confidence()` normalization — this was actually a real bug caught and
  fixed during development (BM25 scores were briefly pinning every answer
  to "100% confident" until the saturating transform was added).
- **Retrieval transparency**: every RAG answer comes back with the actual
  retrieved passages (source, department, score, preview) and a response
  latency, viewable in the UI — not just a final answer to trust blindly.
- **HR analytics beyond SQL rows**: `hr_analytics.py` computes real
  aggregates (department averages, salary distribution, headcount by
  location) with pandas — the kind of thing you'd actually want on a
  people-analytics dashboard, not just filtered row dumps.
- **Live document ingestion with role tagging**: the Admin tab lets a
  c-level user upload a new document, tag it to a department, and have it
  searchable *in the same request* — no server restart, no separate
  reindex step to remember.
- **Audit logging + analytics dashboard**: every query is logged (who,
  role, engine used, allowed/denied, latency) to a SQLite trail that
  survives index rebuilds; the Admin tab surfaces denial rate, latency,
  most-accessed documents, and a recent query log — real observability,
  not just a claim of RBAC working.
- **Retrieval-time RBAC, not prompt-time**: see "Why this architecture"
  above — the LLM is architecturally incapable of leaking cross-department
  data, since it never receives it, whether via the RAG or SQL path.
- **RBAC enforced at the routing layer, independent of the LLM**: which
  departments' SQL tables are even *attempted* is computed from
  `SQL_ENABLED_DEPARTMENTS ∩ allowed_departments` before any SQL is
  generated — a role without HR access never triggers an LLM call against
  the HR table at all, so there's no reliance on the LLM "behaving" itself.
- **Confidence-aware "I don't know"**: when nothing relevant is retrieved,
  the app computes a *department-agnostic* best-match score to tell the
  difference between "this genuinely isn't in any of our docs" (→ asks the
  user to rephrase) and "this exists, but not for your role" (→ says so
  plainly, rather than either hallucinating or a generic failure message).
- **Vague-query handling**: the system prompt instructs the model to ask
  one clarifying question rather than guess when a query is ambiguous.
- **Multi-turn conversation**: the last few turns are sent with each
  request so follow-ups like "what about Q3?" resolve against prior context.
- **Automated RAG evaluation**: `evaluation/evaluate.py` generates QA pairs
  and LLM-judges faithfulness/relevance/conciseness — quality is measured,
  not assumed.
- **Graceful degradation without an API key or without internet**: if
  the active provider's API key isn't set (e.g. `GROQ_API_KEY`), `/chat`
  still returns the raw retrieved, RBAC-filtered context; if the
  embedding model can't be downloaded, retrieval falls back to
  BM25-only — both verified in this repo's own development environment,
  which had neither.
- **Containerized for deployment**: `docker-compose up --build` runs the
  full stack (see Deployment below) — not required for the challenge, but
  signals this was built to actually run somewhere, not just on one laptop.

## How this maps to the evaluation criteria

| Criterion | Where it's addressed |
|---|---|
| Functionality | `/chat` returns role-filtered, cited answers via action execution, SQL, or hybrid RAG depending on intent; verified end-to-end across 74 tests |
| Code Quality | Modular files, docstrings/comments throughout, Pydantic validation, type hints, read-only SQL guardrails, filename sanitization on upload |
| Innovation | Agentic workflow automation (low-code, Virtual Agent layer), hybrid SQL+RAG routing, BM25+embedding fusion with LLM reranking, metadata-aware classification RBAC, calibrated confidence scoring, live document ingestion, audit logging + analytics, HR analytics, Docker packaging |
| Presentation | This README + suggested demo flow below |
| NLP Query Understanding | Query classifier separates structured vs. narrative intent; clarifying-question behavior for vague queries; natural-language CSV row conversion for the RAG path; hybrid lexical+semantic matching handles both keyword and paraphrase queries |
| User Experience | Streamlit chat UI with role badges, dark mode, confidence indicators, retrieved-passage viewer, per-role suggested questions (including workflow-triggering examples), source citations, route indicator (action/SQL/RAG), HR Insights and Admin dashboards with a low-code workflow builder, login/logout flow |
| Modularity | `auth.py` / `ingest.py` / `query_classifier.py` / `workflow_engine.py` / `sql_agent.py` / `hr_analytics.py` / `admin.py` / `audit.py` / `retriever.py` / `llm.py` / `llm_client.py` / `main.py` each own one concern; frontend fully decoupled via HTTP |
| Well Documented README | This file: setup, tech stack, roles, usage examples, architecture, admin panel, HR insights, evaluation, testing, deployment |
| Scalability & Extensibility | New role/department = one config entry; new structured dataset = drop a CSV; new document = upload via Admin tab; new classification tier = extend `config.py`; BM25 → hybrid embeddings scales retrieval without touching the rest of the app; Docker Compose for horizontal deployment |


## Production-hardening: reranking, security, feedback, observability

Four additions aimed at closing the gap between "RAG demo" and something
that would survive contact with 5,000 real users:

**Cross-encoder reranking.** `retrieval_and_reranking`'s final step tries
a real Hugging Face cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
before falling back to LLM-based reranking, before falling back to no
reranking at all — three tiers, each degrading gracefully. A cross-encoder
scores the query and passage *together* through one small transformer,
which is a meaningfully different (and generally more accurate) signal
than comparing two independently-computed vectors. Every `/chat` response
reports `rerank_method` so you can see which tier actually ran.

**Prompt-injection defense.** `security.py` pattern-scans every retrieved
chunk before it reaches the LLM's context — text like "ignore previous
instructions", "reveal confidential data", or "act as an administrator"
gets the chunk excluded entirely (not just flagged) and audit-logged. The
system prompt separately and explicitly tells the model that document
content is data to report on, never instructions to obey, so there are
two independent layers here, not one. Verified with a real malicious
document upload in testing (`tests/test_security_and_observability.py`).

**Feedback loop.** 👍/👎 on every answer (with a reason prompt on 👎:
wrong answer, wrong source, outdated, didn't answer, access issue),
stored separately from the audit trail and aggregated in the Admin tab
(satisfaction %, breakdown by reason). This is the actual mechanism a
real team would use to know what to fix — not just a claim that feedback
"could" inform future work.

**Per-stage observability.** Every `/chat` response includes a
`stage_latencies_ms` breakdown (action detection, classification,
retrieval+reranking, security scan, LLM generation) — viewable per-message
in the chat UI's "Why this answer?" panel, so you can see exactly where
time is going on any given request, not just a single total latency number.


## 🧪 Testing & Engineering Validation

The system was tested across RAG retrieval, RBAC enforcement, workflow execution,
LLM provider switching, query routing, and security-related scenarios.

### Key validation scenarios

- **RBAC-aware retrieval:** Users only receive information from departments
  they are authorized to access.
- **Cross-department protection:** The system detects when a stronger match
  exists outside the user's permissions and avoids returning a weaker,
  potentially misleading answer.
- **Hybrid retrieval:** BM25 + semantic embeddings can be combined for improved
  retrieval quality, with BM25 fallback support when embeddings are unavailable.
- **Workflow isolation:** Pending multi-turn workflows do not hijack unrelated
  follow-up questions.
- **LLM provider abstraction:** Supports Groq, Anthropic, and fully local Ollama
  inference through a unified `llm_client.py`.
- **Graceful degradation:** The application remains usable without an LLM API
  key through retrieval-only/demo mode and heuristic routing.
- **Security and observability:** Authentication, RBAC, audit logging,
  feedback handling, and security-related test cases are covered.

### Retrieval fallback

The application supports a BM25-only fallback when embedding models are
unavailable. With normal internet access, the embedding retriever can download
`all-MiniLM-L6-v2` and automatically enable hybrid BM25 + embedding retrieval.

### Known considerations

- Local Ollama inference can be slower than hosted APIs, particularly on CPU.
- The first embedding-based run may download the embedding model.
- Docker deployment should be validated with `docker compose up --build`
  before production use.
