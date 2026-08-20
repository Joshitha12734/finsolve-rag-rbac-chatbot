"""
Streamlit chat UI for the FinSolve RBAC RAG chatbot.

Run with:
    streamlit run frontend/streamlit_app.py

Expects the FastAPI backend to be running (default http://localhost:8000).
C-level users additionally see an Admin tab (document upload/role tagging,
user management, audit analytics). HR and c-level users also see an HR
Insights tab (pandas-computed workforce analytics).
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ALL_DEPARTMENTS = ["engineering", "finance", "marketing", "hr", "general"]
ALL_ROLES = ["engineering", "finance", "marketing", "hr", "c-level", "employee"]

ROLE_BADGE_COLORS = {
    "engineering": "#3B82F6",
    "finance": "#10B981",
    "marketing": "#F59E0B",
    "hr": "#EC4899",
    "c-level": "#8B5CF6",
    "employee": "#6B7280",
}

st.set_page_config(page_title="FinSolve Assistant", page_icon="💬", layout="centered")

if "auth" not in st.session_state:
    st.session_state.auth = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def inject_theme_css():
    """Streamlit's theme is normally set via config.toml at launch, not
    at runtime — this injects an override so the in-app toggle actually
    works without restarting the server."""
    if not st.session_state.dark_mode:
        return
    st.markdown(
        """
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        [data-testid="stSidebar"] { background-color: #161B22; }
        .stChatMessage { background-color: #1C2128; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def role_badge(role: str) -> str:
    color = ROLE_BADGE_COLORS.get(role, "#6B7280")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600;">{role}</span>'
    )


def confidence_badge(pct: float) -> str:
    if pct >= 70:
        color, label = "#10B981", "High"
    elif pct >= 40:
        color, label = "#F59E0B", "Medium"
    else:
        color, label = "#EF4444", "Low"
    return (
        f'<span style="color:{color};font-weight:600;">● {label} confidence ({pct}%)</span>'
    )


def try_login(username: str, password: str):
    try:
        resp = requests.get(f"{BACKEND_URL}/me", auth=(username, password), timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}. Is `uvicorn backend.main:app` running?")
        return None
    if resp.status_code == 200:
        return resp.json()
    st.error("Invalid username or password.")
    return None


def login_screen():
    inject_theme_css()
    st.title("💬 FinSolve Internal Assistant")
    st.caption("Sign in with your company credentials to get role-specific answers.")
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="e.g. peter.pandey")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        profile = try_login(username, password)
        if profile:
            st.session_state.auth = (username, password)
            st.session_state.profile = profile
            st.rerun()

    with st.expander("Demo accounts"):
        st.markdown(
            "| Username | Password | Role |\n"
            "|---|---|---|\n"
            "| peter.pandey | engineering123 | Engineering |\n"
            "| priya.finance | finance123 | Finance |\n"
            "| raj.marketing | marketing123 | Marketing |\n"
            "| anita.hr | hr123 | HR (also gets HR Insights) |\n"
            "| tony.sharma | clevel123 | C-Level (Admin + HR Insights) |\n"
            "| sam.employee | employee123 | Employee (general only) |"
        )


SUGGESTED_QUESTIONS = {
    "engineering": ["What's our microservices architecture?", "What CI/CD tools do we use?", "I need to escalate an IT issue"],
    "finance": ["What drove the increase in vendor costs?", "How did gross margin change in 2024?", "I want to submit an expense reimbursement"],
    "marketing": ["How did Q2 2024 campaigns perform?", "What's our customer acquisition cost?", "I want to request leave"],
    "hr": ["What's the average performance rating?", "List employees with performance rating 5 in the Data department", "I want to request leave"],
    "c-level": ["Give me a cross-department summary of 2024 performance.", "What is the risk mitigation strategy for Q4 2024?"],
    "employee": ["What's the leave policy?", "I want to request leave for next week", "I need to escalate an IT issue"],
}


def render_assistant_message(msg: dict, key_prefix: str):
    st.markdown(msg["content"])
    if msg.get("meta"):
        st.markdown(msg["meta"], unsafe_allow_html=True)
    if msg.get("sources"):
        st.caption("📎 Sources: " + ", ".join(msg["sources"]))
    if msg.get("retrieved_chunks"):
        with st.expander(f"🔍 View {len(msg['retrieved_chunks'])} retrieved passages"):
            for c in msg["retrieved_chunks"]:
                st.caption(f"**{c['source']}** (score: {c['score']}) — _{c['department']}_")
                st.text(c["preview"])
                st.divider()

    stages = msg.get("stage_latencies_ms")
    if stages:
        with st.expander("🧭 Why this answer? (explainability)"):
            st.caption(f"Route: **{msg.get('route', 'rag')}** — Rerank method: **{msg.get('rerank_method', 'none')}**")
            st.caption("Stage latency breakdown:")
            st.bar_chart(stages)

    # Feedback — only offer it once per message, and only for messages that
    # actually came from the assistant with a real query/answer to attach it to.
    if msg.get("query") is not None:
        fb_key = f"fb_{key_prefix}"
        if st.session_state.get(fb_key):
            st.caption(f"Thanks for the feedback! ({st.session_state[fb_key]})")
        else:
            fcol1, fcol2, fcol3 = st.columns([1, 1, 8])
            if fcol1.button("👍", key=f"{fb_key}_up"):
                _submit_feedback(msg, "up", None)
                st.session_state[fb_key] = "👍"
                st.rerun()
            if fcol2.button("👎", key=f"{fb_key}_down"):
                st.session_state[f"{fb_key}_show_reason"] = True
            if st.session_state.get(f"{fb_key}_show_reason"):
                reason = fcol3.selectbox(
                    "What went wrong?",
                    ["wrong_answer", "wrong_source", "outdated", "didnt_answer", "access_issue", "other"],
                    key=f"{fb_key}_reason",
                )
                if fcol3.button("Submit", key=f"{fb_key}_submit"):
                    _submit_feedback(msg, "down", reason)
                    st.session_state[fb_key] = "👎"
                    st.rerun()


def _submit_feedback(msg: dict, rating: str, reason: str | None):
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            auth=st.session_state.auth,
            json={
                "query": msg.get("query", ""),
                "answer": msg.get("content", ""),
                "rating": rating,
                "reason": reason,
                "route": msg.get("route"),
            },
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        pass  # feedback is best-effort; don't block the UI on it


def render_chat_tab(profile):
    with st.sidebar:
        st.markdown(f"**{profile['full_name']}**", unsafe_allow_html=True)
        st.markdown(role_badge(profile["role"]), unsafe_allow_html=True)
        st.caption("Access to: " + ", ".join(profile["allowed_departments"]))
        st.divider()
        st.checkbox("🌙 Dark mode", key="dark_mode")
        if st.button("Log out"):
            st.session_state.auth = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Try asking:")
        for q in SUGGESTED_QUESTIONS.get(profile["role"], []):
            if st.button(q, key=f"suggest_{q}", use_container_width=True):
                st.session_state.pending_question = q

    inject_theme_css()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg, key_prefix=str(i))
            else:
                st.markdown(msg["content"])

    typed_question = st.chat_input("Ask about your department's data...")
    question = st.session_state.pop("pending_question", None) or typed_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]  # exclude the question just added
                        if m["role"] in ("user", "assistant")
                    ]
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        auth=st.session_state.auth,
                        json={"query": question, "history": history},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()

                        route_labels = {
                            "action": "⚙️ Workflow Action",
                            "sql": "🗄️ Structured (SQL)",
                            "rag": "📚 Document search (RAG)",
                        }
                        route_label = route_labels.get(data.get("route"), "📚 Document search (RAG)")
                        meta = (
                            f'<span style="font-size:0.85em;">{route_label} &nbsp;|&nbsp; '
                            f'{confidence_badge(data.get("confidence_pct", 0))} &nbsp;|&nbsp; '
                            f'⏱️ {data.get("latency_ms", 0)} ms</span>'
                        )

                        new_msg = {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get("sources", []),
                            "meta": meta,
                            "retrieved_chunks": data.get("retrieved_chunks", []),
                            "query": question,
                            "route": data.get("route"),
                            "rerank_method": data.get("rerank_method"),
                            "stage_latencies_ms": data.get("stage_latencies_ms"),
                        }
                        render_assistant_message(new_msg, key_prefix=str(len(st.session_state.messages)))
                        st.session_state.messages.append(new_msg)
                    else:
                        st.error(f"Backend error: {resp.status_code} {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Can't reach backend at {BACKEND_URL}.")


def render_hr_insights_tab():
    inject_theme_css()
    st.subheader("📈 HR Insights")
    st.caption("Pandas-computed workforce analytics over the full HR dataset.")
    try:
        resp = requests.get(f"{BACKEND_URL}/hr/analytics", auth=st.session_state.auth, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")
        return

    if resp.status_code != 200:
        st.error(f"Couldn't load HR analytics: {resp.status_code}")
        return

    d = resp.json()
    if not d.get("available"):
        st.info("No HR dataset found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total employees", d["total_employees"])
    c2.metric("Avg performance rating", d.get("avg_performance_rating_overall", "N/A"))
    c3.metric("Avg attendance", f"{d.get('avg_attendance_pct', 'N/A')}%")
    c4.metric("Avg leaves taken", d.get("avg_leaves_taken", "N/A"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Headcount by department")
        if d.get("headcount_by_department"):
            st.bar_chart(d["headcount_by_department"])
    with col_b:
        st.caption("Avg performance rating by department")
        if d.get("avg_performance_rating_by_department"):
            st.bar_chart(d["avg_performance_rating_by_department"])

    col_c, col_d = st.columns(2)
    with col_c:
        st.caption("Avg salary by department")
        if d.get("avg_salary_by_department"):
            st.bar_chart(d["avg_salary_by_department"])
    with col_d:
        st.caption("Headcount by location")
        if d.get("headcount_by_location"):
            st.bar_chart(d["headcount_by_location"])

    if d.get("salary_stats"):
        st.caption("Salary distribution")
        s = d["salary_stats"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Min", f"{s['min']:,.0f}")
        sc2.metric("Median", f"{s['median']:,.0f}")
        sc3.metric("Mean", f"{s['mean']:,.0f}")
        sc4.metric("Max", f"{s['max']:,.0f}")


def render_admin_tab():
    inject_theme_css()

    st.subheader("💬 User Feedback")
    try:
        resp = requests.get(f"{BACKEND_URL}/admin/feedback/summary", auth=st.session_state.auth, timeout=10)
        if resp.status_code == 200:
            f = resp.json()
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Total feedback", f["total_feedback"])
            fc2.metric("Satisfaction", f"{f['satisfaction_pct']}%")
            fc3.metric("👎 count", f["thumbs_down"])
            if f["down_reasons"]:
                st.caption("Reasons for 👎")
                st.bar_chart({r["reason"]: r["count"] for r in f["down_reasons"]})
            with st.expander("Recent feedback"):
                recent_resp = requests.get(f"{BACKEND_URL}/admin/feedback/recent", auth=st.session_state.auth, timeout=10)
                if recent_resp.status_code == 200:
                    for e in recent_resp.json()["entries"]:
                        icon = "👍" if e["rating"] == "up" else "👎"
                        reason_txt = f" ({e['reason']})" if e.get("reason") else ""
                        st.caption(f"{icon}{reason_txt} `{e['timestamp'][:19]}` **{e['username']}** — \"{e['query'][:60]}\"")
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")

    st.divider()
    st.subheader("⚙️ Workflow Actions (Low-Code Builder)")
    st.caption("Define new actions the assistant can execute — no code required. Once created, users can trigger it just by asking in chat.")

    with st.form("new_workflow_form", clear_on_submit=True):
        wc1, wc2 = st.columns(2)
        w_id = wc1.text_input("Action ID (unique, no spaces)", placeholder="e.g. password_reset")
        w_name = wc2.text_input("Display name", placeholder="e.g. Password Reset Request")
        w_desc = st.text_area("Description (when should this trigger?)", placeholder="Request a password reset for a company system.")
        wc3, wc4 = st.columns(2)
        w_dept = wc3.selectbox("Who can trigger it", ALL_DEPARTMENTS, key="workflow_dept")
        w_fields = wc4.text_input("Required fields (comma-separated)", placeholder="system_name")
        w_keywords = st.text_input(
            "Trigger keywords (comma-separated — used to recognize intent even without an LLM connected)",
            placeholder="password reset, reset password, forgot password",
        )
        w_template = st.text_input(
            "Confirmation message (use {field_name} and {request_id})",
            placeholder="Your request for {system_name} has been submitted. ID: #{request_id}.",
        )
        workflow_submitted = st.form_submit_button("Create workflow action")

    if workflow_submitted:
        if not (w_id and w_name and w_desc and w_template):
            st.warning("Fill in all fields.")
        else:
            resp = requests.post(
                f"{BACKEND_URL}/admin/workflows",
                auth=st.session_state.auth,
                json={
                    "id": w_id.strip().replace(" ", "_"),
                    "name": w_name,
                    "description": w_desc,
                    "department": w_dept,
                    "required_fields": [f.strip() for f in w_fields.split(",") if f.strip()],
                    "keywords": [k.strip().lower() for k in w_keywords.split(",") if k.strip()],
                    "confirmation_template": w_template,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                st.success(f"Created workflow action '{w_name}' — try asking about it in the Chat tab.")
                st.rerun()
            else:
                st.error(f"Failed: {resp.status_code} {resp.text}")

    st.caption("Current workflow actions")
    try:
        resp = requests.get(f"{BACKEND_URL}/admin/workflows", auth=st.session_state.auth, timeout=10)
        if resp.status_code == 200:
            for a in resp.json()["actions"]:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(f"**{a['name']}** (`{a['id']}`) — {a['department']}")
                col2.caption(", ".join(a["required_fields"]) or "no fields")
                if col3.button("🗑️", key=f"delwf_{a['id']}"):
                    del_resp = requests.delete(f"{BACKEND_URL}/admin/workflows/{a['id']}", auth=st.session_state.auth, timeout=10)
                    if del_resp.status_code == 200:
                        st.rerun()

        with st.expander("Recent workflow requests submitted by users"):
            req_resp = requests.get(f"{BACKEND_URL}/admin/workflows/requests", auth=st.session_state.auth, timeout=10)
            if req_resp.status_code == 200:
                for r in req_resp.json()["requests"]:
                    st.caption(f"`{r['timestamp'][:19]}` **{r['username']}** ({r['role']}) submitted **{r['action_id']}** — {r['fields']}")
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")

    st.divider()
    st.subheader("📊 Analytics")
    try:
        resp = requests.get(f"{BACKEND_URL}/admin/analytics/summary", auth=st.session_state.auth, timeout=10)
        if resp.status_code == 200:
            s = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total queries", s["total_queries"])
            c2.metric("Denied / no match", s["denied_queries"], f"{s['denial_rate_pct']}%")
            c3.metric("Avg latency", f"{s['avg_latency_ms']} ms")
            sql_count = next((r["count"] for r in s["queries_by_route"] if r["route"] == "sql"), 0)
            c4.metric("SQL-routed queries", sql_count)

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Queries by role")
                if s["queries_by_role"]:
                    st.bar_chart({r["role"]: r["count"] for r in s["queries_by_role"]})
                else:
                    st.caption("No data yet.")
            with col_b:
                st.caption("Most accessed documents")
                if s["most_accessed_documents"]:
                    st.bar_chart({d["source"]: d["count"] for d in s["most_accessed_documents"]})
                else:
                    st.caption("No data yet.")

            with st.expander("Recent query log"):
                recent_resp = requests.get(f"{BACKEND_URL}/admin/analytics/recent", auth=st.session_state.auth, timeout=10)
                if recent_resp.status_code == 200:
                    for e in recent_resp.json()["entries"]:
                        status_icon = "✅" if e["allowed"] else "🚫"
                        st.caption(
                            f"{status_icon} `{e['timestamp'][:19]}` **{e['username']}** ({e['role']}) "
                            f"[{e['route']}] — \"{e['query'][:60]}\" — {e['latency_ms']} ms"
                        )
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")

    st.divider()
    st.subheader("📁 Document management")
    st.caption("Upload a document and tag it to a department — it's searchable immediately, no restart needed.")

    with st.form("upload_form", clear_on_submit=True):
        department = st.selectbox("Department (role tag)", ALL_DEPARTMENTS)
        uploaded_file = st.file_uploader("Document (.md, .txt, or .csv)", type=["md", "txt", "csv"])
        upload_submitted = st.form_submit_button("Upload & index")

    if upload_submitted:
        if not uploaded_file:
            st.warning("Choose a file first.")
        else:
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/admin/documents",
                    auth=st.session_state.auth,
                    data={"department": department},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=60,
                )
                if resp.status_code == 200:
                    st.success(f"Uploaded to `{department}/` and re-indexed — try asking about it in the Chat tab.")
                else:
                    st.error(f"Upload failed: {resp.status_code} {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach backend at {BACKEND_URL}.")

    st.divider()
    st.caption("Current documents")
    try:
        resp = requests.get(f"{BACKEND_URL}/admin/documents", auth=st.session_state.auth, timeout=10)
        if resp.status_code == 200:
            docs = resp.json()["documents"]
            for d in docs:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(f"**{d['department']}**/{d['filename']}")
                col2.write(f"{d['size_kb']} KB")
                if col3.button("🗑️", key=f"del_{d['department']}_{d['filename']}"):
                    del_resp = requests.delete(
                        f"{BACKEND_URL}/admin/documents/{d['department']}/{d['filename']}",
                        auth=st.session_state.auth,
                        timeout=30,
                    )
                    if del_resp.status_code == 200:
                        st.rerun()
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")

    st.divider()
    st.subheader("👤 User management")

    with st.form("new_user_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        new_username = c1.text_input("Username")
        new_full_name = c2.text_input("Full name")
        c3, c4 = st.columns(2)
        new_password = c3.text_input("Password", type="password")
        new_role = c4.selectbox("Role", ALL_ROLES)
        create_submitted = st.form_submit_button("Create user")

    if create_submitted:
        if not (new_username and new_password and new_full_name):
            st.warning("Fill in all fields.")
        else:
            resp = requests.post(
                f"{BACKEND_URL}/admin/users",
                auth=st.session_state.auth,
                json={"username": new_username, "password": new_password, "role": new_role, "full_name": new_full_name},
                timeout=10,
            )
            if resp.status_code == 200:
                st.success(f"Created {new_username} ({new_role})")
                st.rerun()
            else:
                st.error(f"Failed: {resp.status_code} {resp.text}")

    st.caption("Current users")
    try:
        resp = requests.get(f"{BACKEND_URL}/admin/users", auth=st.session_state.auth, timeout=10)
        if resp.status_code == 200:
            for u in resp.json()["users"]:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(f"**{u['full_name']}** ({u['username']})")
                col2.write(u["role"])
                if u["username"] != st.session_state.auth[0]:
                    if col3.button("🗑️", key=f"deluser_{u['username']}"):
                        del_resp = requests.delete(
                            f"{BACKEND_URL}/admin/users/{u['username']}",
                            auth=st.session_state.auth,
                            timeout=10,
                        )
                        if del_resp.status_code == 200:
                            st.rerun()
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach backend at {BACKEND_URL}.")


def chat_screen():
    profile = st.session_state.profile
    st.title("💬 FinSolve Internal Assistant")

    role = profile["role"]
    tabs_config = [("💬 Chat", render_chat_tab, True)]
    if role in ("hr", "c-level"):
        tabs_config.append(("📈 HR Insights", render_hr_insights_tab, False))
    if role == "c-level":
        tabs_config.append(("🛠️ Admin", render_admin_tab, False))

    if len(tabs_config) == 1:
        render_chat_tab(profile)
    else:
        tabs = st.tabs([label for label, _, _ in tabs_config])
        for tab, (label, render_fn, needs_profile) in zip(tabs, tabs_config):
            with tab:
                render_fn(profile) if needs_profile else render_fn()


if st.session_state.auth is None:
    login_screen()
else:
    chat_screen()
