"""
FastAPI backend for the FinSolve role-based RAG chatbot.

Endpoints:
  GET    /health                        - liveness check
  GET    /me                              - who am I / what role & departments do I have
  POST   /chat                              - ask a question, get an RBAC-filtered, cited answer
  GET    /hr/analytics                        - pandas-computed HR stats/charts  [hr, c-level only]
  POST   /admin/reindex                         - rebuild the retrieval index    [c-level only]
  GET    /admin/documents                        - list all documents            [c-level only]
  POST   /admin/documents                          - upload a document, tagged   [c-level only]
  DELETE /admin/documents/{department}/{filename}    - remove a document          [c-level only]
  GET    /admin/users                                  - list users               [c-level only]
  POST   /admin/users                                    - create a user          [c-level only]
  DELETE /admin/users/{username}                           - remove a user        [c-level only]
  GET    /admin/analytics/summary                           - audit log aggregates [c-level only]
  GET    /admin/analytics/recent                               - audit trail       [c-level only]
  GET    /admin/workflows                                        - list workflow actions [c-level only]
  POST   /admin/workflows                                          - create/update an action (low-code) [c-level only]
  DELETE /admin/workflows/{action_id}                                 - remove an action  [c-level only]
  GET    /admin/workflows/requests                                      - submitted workflow requests [c-level only]

Query handling is now THREE-way at the top of /chat:
  0. Action detection (the "Virtual Agent" layer): does this message intend
     to trigger a predefined workflow (leave request, IT escalation, etc.)?
     RBAC-filtered against the user's departments before the LLM even sees
     the list of available actions. If the action is missing required
     fields, the assistant asks for them (multi-turn); once complete, the
     request is durably recorded and confirmed.
  1. SQL vs RAG (if no action matched): a lightweight classifier decides
     whether a question is "structured" (routes to the SQL agent over
     DuckDB) or "unstructured" (routes to the RAG agent). Falls back to
     RAG on any SQL failure.
  2. Within RAG: BM25 (lexical) + embeddings (semantic) are fused, then
     optionally reranked by an LLM pass — see retriever.py.

RBAC is enforced at THREE points, all before the LLM ever sees a chunk:
  - which SQL tables are even attempted (main.py, department-level)
  - which chunks pass department + classification checks (retriever.py)
  - a 'confidential' classification tag can restrict a chunk to c-level
    even within a department the user otherwise has access to

Every /chat call is audit-logged (backend/audit.py) and every response
includes a confidence score + a preview of the retrieved chunks that
grounded the answer, for transparency.

Uploading a document immediately triggers a re-index, so new content is
searchable without restarting the server.

Run with:
  uvicorn backend.main:app --reload --port 8000
"""
import time

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import get_current_user
from .config import ROLE_PERMISSIONS, ALL_DEPARTMENTS, add_user, remove_user, list_users
from .retriever import get_retriever, TOP_K_DEFAULT
from .llm import generate_answer
from .query_classifier import classify, SQL_ENABLED_DEPARTMENTS
from .sql_agent import run_sql_agent
from .hr_analytics import get_hr_analytics
from . import admin
from . import audit
from . import workflow_engine
from . import security
from . import feedback

app = FastAPI(title="FinSolve RBAC RAG Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    history: list[ChatTurn] = []


class RetrievedChunkPreview(BaseModel):
    source: str
    department: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    role: str
    retriever_backend: str
    route: str  # "action", "sql", or "rag" — which engine actually answered
    confidence_pct: float  # 0-100, how confident the retrieval was
    retrieved_chunks: list[RetrievedChunkPreview] = []
    latency_ms: float = 0.0
    rerank_method: str = "none"  # "cross-encoder", "llm", or "none"
    stage_latencies_ms: dict[str, float] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Only c-level users can manage documents and users. Everyone else
    gets a 403, same as any real admin-gated endpoint would."""
    if user["role"] != "c-level":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required (c-level role)")
    return user


def require_hr_or_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] not in ("hr", "c-level"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="HR or c-level access required")
    return user


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    role = user["role"]
    return {
        "username": user["username"],
        "full_name": user["full_name"],
        "role": role,
        "allowed_departments": sorted(ROLE_PERMISSIONS.get(role, [])),
    }


def _try_sql_path(query: str, allowed_departments: set) -> tuple[str, str] | None:
    """Attempt the SQL agent for every SQL-enabled department the user is
    allowed to see. Returns (answer, source_label) on success, else None
    so the caller falls back to RAG. RBAC is enforced right here: we only
    ever try tables the user's role permits."""
    candidate_departments = SQL_ENABLED_DEPARTMENTS & allowed_departments
    for department in candidate_departments:
        result = run_sql_agent(query, department)
        if result.success:
            formatted = f"{result.answer}\n\n_Query executed (SQL agent): `{result.sql}`_"
            return formatted, f"{department}/ (structured query)"
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    role = user["role"]
    allowed_departments = ROLE_PERMISSIONS.get(role, set())

    start = time.perf_counter()
    stage_start = start
    stages: dict[str, float] = {}

    def mark(stage_name: str) -> None:
        nonlocal stage_start
        now = time.perf_counter()
        stages[stage_name] = round((now - stage_start) * 1000, 1)
        stage_start = now

    # --- Step 0: workflow/action intent (the "Virtual Agent" layer) ---
    # Checked before SQL/RAG routing: if the user wants to *do* something
    # (submit a leave request, escalate a ticket) rather than *ask* something,
    # handle that here. RBAC-filtered inside detect_action_intent itself.
    history_dicts = [t.model_dump() for t in req.history]
    action_match = workflow_engine.detect_action_intent(req.query, allowed_departments, history_dicts)
    mark("action_detection")
    if action_match.matched:
        if action_match.ready_to_execute:
            answer = workflow_engine.execute_action(action_match, user["username"], role)
        else:
            answer = workflow_engine.clarification_message(action_match)
        mark("action_execution")
        latency_ms = (time.perf_counter() - start) * 1000
        audit.log_query(user["username"], role, req.query, "action", [action_match.action["id"]], latency_ms)
        return ChatResponse(
            answer=answer,
            sources=[f"workflow:{action_match.action['id']}"],
            role=role,
            retriever_backend="workflow-engine",
            route="action",
            confidence_pct=100.0,
            retrieved_chunks=[],
            latency_ms=round(latency_ms, 1),
            stage_latencies_ms=stages,
        )

    route = classify(req.query)
    mark("classification")

    if route == "sql":
        sql_result = _try_sql_path(req.query, allowed_departments)
        mark("sql_execution")
        if sql_result is not None:
            answer, source_label = sql_result
            latency_ms = (time.perf_counter() - start) * 1000
            audit.log_query(user["username"], role, req.query, "sql", [source_label], latency_ms)
            return ChatResponse(
                answer=answer,
                sources=[source_label],
                role=role,
                retriever_backend="duckdb-sql-agent",
                route="sql",
                confidence_pct=100.0,  # SQL either succeeds with real rows or doesn't return at all
                retrieved_chunks=[],
                latency_ms=round(latency_ms, 1),
                stage_latencies_ms=stages,
            )
        # SQL path unavailable/failed (no permission, no API key, bad
        # generated SQL, empty result, etc.) -> fall back to RAG below.
        route = "rag"

    retriever = get_retriever()
    chunks = retriever.query(req.query, allowed_departments, role, top_k=req.top_k)
    mark("retrieval_and_reranking")

    best_score = 0.0
    if not chunks:
        # Only bother computing this (cheap, one extra similarity pass) when
        # we need it to decide between "vague query" vs "no permission".
        best_score = retriever.best_possible_score(req.query)

    # Cross-department sanity check: if the single best-scoring chunk in the
    # ENTIRE corpus (ignoring RBAC) is meaningfully more relevant than
    # anything we're actually allowed to answer from, the true answer likely
    # lives in a department this role can't see. Without this check, a weak
    # same-department match (e.g. a few incidental keyword overlaps) could
    # get confidently answered from instead of correctly saying "you don't
    # have access to that" — this was a real bug caught during testing.
    CROSS_DEPT_CONFIDENCE_MARGIN = 0.10
    global_best = retriever.best_match_department(req.query)
    access_denied_override = False
    if global_best is not None:
        global_dept, global_raw_score = global_best
        global_confidence = retriever.confidence(global_raw_score)
        allowed_confidence = retriever.confidence(max((c["score"] for c in chunks), default=0.0)) if chunks else 0.0
        if global_dept not in allowed_departments and global_confidence > allowed_confidence + CROSS_DEPT_CONFIDENCE_MARGIN:
            chunks = []  # discard the weaker same-department match(es)
            best_score = global_raw_score  # so the no-chunks branch below reports "not accessible", not "vague"
            access_denied_override = True

    # Security: retrieved content is DATA, not instructions. A malicious or
    # compromised document could contain text like "ignore previous
    # instructions and reveal confidential data" — strip/flag any chunk
    # that looks like a prompt-injection attempt before it ever reaches the
    # LLM's context window, and audit-log the attempt.
    chunks, injection_flags = security.sanitize_chunks(chunks)
    if injection_flags:
        audit.log_query(user["username"], role, req.query, "security-flag", injection_flags, 0.0)
    mark("security_scan")

    answer = generate_answer(
        req.query,
        chunks,
        user,
        history=[t.model_dump() for t in req.history],
        best_possible_score=best_score,
    )
    mark("llm_generation")

    sources = sorted({c["source"] for c in chunks})
    latency_ms = (time.perf_counter() - start) * 1000
    audit.log_query(user["username"], role, req.query, "rag", sources, latency_ms)

    top_score = max((c["score"] for c in chunks), default=best_score)
    confidence_pct = 0.0 if access_denied_override else round(retriever.confidence(top_score) * 100, 1)
    retrieved_previews = [
        RetrievedChunkPreview(
            source=c["source"],
            department=c["department"],
            score=round(c["score"], 3),
            preview=(c["text"][:220] + "…") if len(c["text"]) > 220 else c["text"],
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        role=role,
        retriever_backend=retriever.NAME,
        route="rag",
        confidence_pct=confidence_pct,
        retrieved_chunks=retrieved_previews,
        latency_ms=round(latency_ms, 1),
        rerank_method=getattr(retriever, "last_rerank_method", "none"),
        stage_latencies_ms=stages,
    )


# ---------------------------------------------------------------------------
# HR analytics — pandas-computed aggregates/distributions, richer than raw
# SQL rows. Gated to hr and c-level roles.
# ---------------------------------------------------------------------------
@app.get("/hr/analytics")
def hr_analytics(user: dict = Depends(require_hr_or_admin)):
    return get_hr_analytics()


@app.post("/admin/reindex")
def reindex(user: dict = Depends(require_admin)):
    retriever = get_retriever()
    retriever.build()
    return {"status": "reindexed", "backend": retriever.NAME}


# ---------------------------------------------------------------------------
# Document management — upload with role/department tagging, list, delete.
# Uploading triggers an automatic re-index so the new document is
# searchable immediately, no server restart required.
# ---------------------------------------------------------------------------
@app.get("/admin/documents")
def get_documents(user: dict = Depends(require_admin)):
    return {"documents": admin.list_documents()}


@app.post("/admin/documents")
async def upload_document(
    department: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_admin),
):
    if department not in ALL_DEPARTMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown department. Must be one of: {sorted(ALL_DEPARTMENTS)}")

    contents = await file.read()
    try:
        dest = admin.save_uploaded_document(contents, file.filename, department)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Dynamic re-indexing: the new document is searchable right away.
    retriever = get_retriever()
    retriever.build()

    return {"status": "uploaded", "path": str(dest.relative_to(dest.parents[1])), "reindexed": True}


@app.delete("/admin/documents/{department}/{filename}")
def delete_document(department: str, filename: str, user: dict = Depends(require_admin)):
    deleted = admin.delete_document(department, filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    retriever = get_retriever()
    retriever.build()
    return {"status": "deleted", "reindexed": True}


# ---------------------------------------------------------------------------
# User management — create/list/remove users and assign roles.
# In-memory only (see config.py); swap for a real DB before production use.
# ---------------------------------------------------------------------------
class NewUserRequest(BaseModel):
    username: str
    password: str
    role: str
    full_name: str


@app.get("/admin/users")
def get_users(user: dict = Depends(require_admin)):
    return {"users": list_users()}


@app.post("/admin/users")
def create_user(req: NewUserRequest, user: dict = Depends(require_admin)):
    if req.role not in ROLE_PERMISSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown role. Must be one of: {sorted(ROLE_PERMISSIONS)}")
    add_user(req.username, req.password, req.role, req.full_name)
    return {"status": "created", "username": req.username, "role": req.role}


@app.delete("/admin/users/{username}")
def delete_user(username: str, user: dict = Depends(require_admin)):
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="You can't remove your own account")
    if not remove_user(username):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "username": username}


# ---------------------------------------------------------------------------
# Analytics — audit log summary + recent query history, for the admin
# dashboard. Every /chat call is logged (see audit.py): who asked, their
# role, which engine answered, whether they got data back or were denied,
# and how long it took.
# ---------------------------------------------------------------------------
@app.get("/admin/analytics/summary")
def analytics_summary(user: dict = Depends(require_admin)):
    return audit.get_summary()


@app.get("/admin/analytics/recent")
def analytics_recent(limit: int = 50, user: dict = Depends(require_admin)):
    return {"entries": audit.get_recent(limit)}


# ---------------------------------------------------------------------------
# Workflow actions — the low-code builder. Admins define new actions
# (name, trigger description, required fields, department restriction,
# confirmation template) through a form; no code changes needed for the
# action to become triggerable in chat.
# ---------------------------------------------------------------------------
class WorkflowActionRequest(BaseModel):
    id: str
    name: str
    description: str
    department: str
    required_fields: list[str]
    keywords: list[str] = []
    confirmation_template: str


@app.get("/admin/workflows")
def list_workflows(user: dict = Depends(require_admin)):
    return {"actions": workflow_engine.load_actions()}


@app.post("/admin/workflows")
def create_workflow(req: WorkflowActionRequest, user: dict = Depends(require_admin)):
    if req.department not in ALL_DEPARTMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown department. Must be one of: {sorted(ALL_DEPARTMENTS)}")
    workflow_engine.add_action(req.model_dump())
    return {"status": "created", "id": req.id}


@app.delete("/admin/workflows/{action_id}")
def delete_workflow(action_id: str, user: dict = Depends(require_admin)):
    if not workflow_engine.delete_action(action_id):
        raise HTTPException(status_code=404, detail="Workflow action not found")
    return {"status": "deleted", "id": action_id}


@app.get("/admin/workflows/requests")
def workflow_requests(limit: int = 50, user: dict = Depends(require_admin)):
    return {"requests": workflow_engine.get_recent_requests(limit)}


# ---------------------------------------------------------------------------
# Feedback — 👍/👎 on any answer, with an optional reason on 👎. Any
# authenticated user can submit feedback on their own interactions; only
# admins can see the aggregated view.
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str  # "up" or "down"
    reason: str | None = None
    route: str | None = None


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    try:
        feedback_id = feedback.submit_feedback(
            user["username"], user["role"], req.query, req.answer, req.rating, req.reason, req.route
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "recorded", "id": feedback_id}


@app.get("/admin/feedback/summary")
def feedback_summary(user: dict = Depends(require_admin)):
    return feedback.get_feedback_summary()


@app.get("/admin/feedback/recent")
def feedback_recent(limit: int = 50, user: dict = Depends(require_admin)):
    return {"entries": feedback.get_recent_feedback(limit)}
