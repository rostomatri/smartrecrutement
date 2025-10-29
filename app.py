import os
import random
import textwrap
from typing import TypedDict, List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== LangGraph (lightweight orchestration) =====
from langgraph.graph import StateGraph, END

# =========================
# Branding / Page Config
# =========================
st.set_page_config(
    page_title="SmartRecruiter — Candidate Feedback Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_TITLE = "SmartRecruiter"
FALLBACK_PATH = "resume_screening_dataset_train.csv"
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# 🔑 Load Groq API Key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Corporate CSS
# =========================
CORP_CSS = """
<style>
:root {
  --bg: #f6f8fb; --card:#fff; --text:#0f172a; --muted:#64748b;
  --accent:#2563eb; --accent-weak:#eff6ff; --shadow:0 6px 24px rgba(15,23,42,.06); --radius:14px;
}
html, body [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.block-container{ padding-top:1rem; padding-bottom:2rem; }

/* Header */
.brand-bar{display:flex;align-items:center;gap:12px;margin:0 0 18px 0;padding:14px 18px;background:linear-gradient(135deg,#fff,#f3f7ff);border:1px solid #e6eefc;border-radius:var(--radius);box-shadow:var(--shadow);}
.brand-logo{width:42px;height:42px;border-radius:10px;background:linear-gradient(135deg,#2563eb,#60a5fa);display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:700;}
.brand-title{font-size:20px;font-weight:700;color:#0f172a;}
.brand-sub{font-size:13px;color:var(--muted);}

/* Job card */
.job-card{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);padding:18px;box-shadow:var(--shadow);}
.job-title{font-weight:700;font-size:18px;margin-bottom:6px;color:#0f172a;}
.job-chip{display:inline-block;font-size:12px;color:#1e40af;background:var(--accent-weak);border:1px solid #c7dcff;padding:4px 8px;border-radius:999px;margin-right:8px;}
.job-desc{color:#0f172a;line-height:1.5;}
.job-reason{color:#334155;font-size:13px;background:#eef2ff;padding:6px 10px;border-radius:10px;display:inline-block;}

/* Input panel */
.panel{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow);}

/* Chat */
.chat-wrap{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);box-shadow:var(--shadow);padding:14px;height:520px;overflow-y:auto;}
.msg{max-width:78%;padding:10px 14px;border-radius:16px;margin:8px 0;line-height:1.45;}
.msg-user{background:#e7f0ff;color:#0f172a;border:1px solid #cfe3ff;margin-left:auto;border-bottom-right-radius:6px;}
.msg-bot{background:#f8fafc;border:1px solid #e2e8f0;color:#0f172a;margin-right:auto;border-bottom-left-radius:6px;}
.meta{font-size:12px;color:var(--muted);margin-top:2px;}
.footer-note{color:var(--muted);font-size:12px;margin-top:6px;}
.send-btn button{width:100%;}
</style>
"""
st.markdown(CORP_CSS, unsafe_allow_html=True)

# =========================
# Utils
# =========================
def ensure_groq():
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_your_real"):
        st.error("Groq API key is not set. Set GROQ_API_KEY env var or edit it in app.py.")
        st.stop()

def call_groq(prompt: str) -> str:
    # proxy-safe Groq client
    import groq
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        if hasattr(client, "_client") and hasattr(client._client, "_client_args"):
            client._client._client_args.pop("proxies", None)
    except TypeError:
        client = groq.Groq(api_key=GROQ_API_KEY)

    chat = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an empathetic HR assistant who gives concrete, constructive feedback."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return chat.choices[0].message.content

def extract_text_from_pdf(uploaded_pdf) -> str:
    text = ""
    reader = PdfReader(uploaded_pdf)
    for page in reader.pages:
        page_txt = page.extract_text() or ""
        text += page_txt + "\n"
    return text.strip()

# =========================
# Data loading & indexing
# =========================
@st.cache_resource(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(FALLBACK_PATH):
        st.error(f"❌ Dataset not found at {FALLBACK_PATH}. Place it next to app.py.")
        st.stop()
    df = pd.read_csv(FALLBACK_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "job_title" not in df.columns:
        if "role" in df.columns:
            df = df.rename(columns={"role":"job_title"})
        else:
            for c in df.columns:
                if "title" in c: df = df.rename(columns={c:"job_title"}); break
            if "job_title" not in df.columns:
                df["job_title"] = [f"Job {i+1}" for i in range(len(df))]
    if "job_description" not in df.columns:
        for c in df.columns:
            if "description" in c: df = df.rename(columns={c:"job_description"}); break
        if "job_description" not in df.columns:
            text_cols = [c for c in df.columns if df[c].dtype=="object"]
            df["job_description"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    if "reason_for_decision" not in df.columns:
        df["reason_for_decision"] = "Not specified"

    df["job_title"] = df["job_title"].fillna("").astype(str)
    df["job_description"] = df["job_description"].fillna("").astype(str)
    df["reason_for_decision"] = df["reason_for_decision"].fillna("Not specified").astype(str)
    df = df[df["job_description"].str.strip()!=""].reset_index(drop=True)
    return df

df = load_dataset()

@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame):
    embedder = SentenceTransformer(EMBED_MODEL)
    corpus = (
        df["job_title"].astype(str) + " | " +
        df["job_description"].astype(str) + " | Reason: " +
        df["reason_for_decision"].astype(str)
    ).tolist()
    vectors = embedder.encode(corpus, convert_to_numpy=True)
    nn = NearestNeighbors(n_neighbors=min(TOP_K, len(vectors)), metric="cosine")
    nn.fit(vectors)
    return {"embedder": embedder, "vectors": vectors, "nn": nn}

index = build_index(df)

def retrieve_similar(query: str, index, top_k=TOP_K):
    qv = index["embedder"].encode([query], convert_to_numpy=True)
    dist, idx = index["nn"].kneighbors(qv, n_neighbors=min(top_k, len(df)))
    return [(int(i), 1 - float(d)) for i, d in zip(idx[0], dist[0])]

# =========================
# LangGraph state & nodes
# =========================
class SRState(TypedDict):
    cv_text: str
    user_question: str
    selected_job_idx: int
    retrieved: List[Tuple[int, float]]  # (row_index, similarity)
    feedback: str
    coaching: str
    matches: str
    final_answer: str

def node_retrieve(state: SRState) -> Dict[str, Any]:
    """RAG retrieval node: compute top-k similar job rows from CV + question."""
    query = (state["cv_text"] or "") + "\n\n" + (state["user_question"] or "")
    hits = retrieve_similar(query, index, TOP_K)
    return {"retrieved": hits}

def _context_block(hits: List[Tuple[int,float]]) -> str:
    chunks = []
    for idx, sim in hits:
        r = df.iloc[idx]
        chunks.append(
            f"- {r['job_title']} (sim {sim:.2f})\n  Desc: {r['job_description'][:420]}...\n  Reason: {r['reason_for_decision']}"
        )
    return "\n".join(chunks)

def node_feedback(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    prompt = f"""
The candidate appears to be evaluated against the position: **{job['job_title']}**.
A common decision reason in similar cases: "{job['reason_for_decision']}".
Candidate CV:
{state['cv_text'][:1500]}
Top matching job descriptions with real rejection reasons:
{_context_block(state['retrieved'])}
Candidate’s question:
{state['user_question']}
TASK: As a Feedback Agent, kindly explain the most likely reasons for rejection grounded in the context.
Limit to ~6-8 lines, specific, respectful, and actionable.
"""
    return {"feedback": call_groq(prompt)}

def node_coaching(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    prompt = f"""
Context:
- Target job: {job['job_title']}
- Typical decision reasons: {job['reason_for_decision']}
- Retrieved matches:
{_context_block(state['retrieved'])}
- Candidate CV (truncated): {state['cv_text'][:800]}
TASK: As a Career Coach Agent, propose 3–5 concrete steps (skills, courses, project ideas, certifications) to close gaps.
Keep it realistic for the next 6–12 weeks.
"""
    return {"coaching": call_groq(prompt)}

def node_match(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    prompt = f"""
We compared the candidate to: {job['job_title']}.
Using the retrieved context:
{_context_block(state['retrieved'])}
TASK: As a Matcher Agent, suggest 2–3 alternative job roles or tracks that fit the candidate's current profile better.
For each role, give a one-sentence rationale. Keep it concise and practical.
"""
    return {"matches": call_groq(prompt)}

def node_synthesize(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    prompt = f"""
You are the Lead HR Assistant at SmartRecruiter.
Merge the sections into one cohesive, professional, and empathetic message (10–14 lines).
Include headings: Feedback, Coaching, Alternative Roles. Avoid repetition.

JOB: {job['job_title']}
RETRIEVED CONTEXT:
{_context_block(state['retrieved'])}

FEEDBACK:
{state['feedback']}

COACHING:
{state['coaching']}

ALTERNATIVE ROLES:
{state['matches']}
"""
    return {"final_answer": call_groq(prompt)}

# Build the LangGraph
graph = StateGraph(SRState)
graph.add_node("retrieve", node_retrieve)
graph.add_node("feedback", node_feedback)
graph.add_node("coaching", node_coaching)
graph.add_node("matches", node_match)
graph.add_node("synthesize", node_synthesize)

# Edges: start -> retrieve -> (feedback, coaching, matches) -> synthesize -> END
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "feedback")
graph.add_edge("retrieve", "coaching")
graph.add_edge("retrieve", "matches")
graph.add_edge("feedback", "synthesize")
graph.add_edge("coaching", "synthesize")
graph.add_edge("matches", "synthesize")
graph.add_edge("synthesize", END)

compiled_graph = graph.compile()

# =========================
# Session state (chat)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role":"user"/"assistant","content":str}
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

# =========================
# Header
# =========================
st.markdown(
    f"""
<div class="brand-bar">
  <div class="brand-logo">SR</div>
  <div>
    <div class="brand-title">{APP_TITLE}</div>
    <div class="brand-sub">LangGraph Orchestration • RAG • Groq • Analytics</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Random job on reload
random_row = df.sample(1, random_state=None).iloc[0]
selected_idx = int(random_row.name)

# =========================
# Tabs: Chat | Analytics
# =========================
tab_chat, tab_dash = st.tabs(["💬 Chat", "📊 Analytics"])

with tab_chat:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown(
            f"""
<div class="job-card">
  <div class="job-title">{random_row['job_title']}</div>
  <div style="margin-bottom: 8px;">
    <span class="job-chip">Selected automatically</span>
    <span class="job-chip">Randomized on reload</span>
  </div>
  <div class="job-desc">{textwrap.shorten(random_row['job_description'], width=550, placeholder="…")}</div>
  <div style="margin-top:10px;">
    <span class="job-reason"><b>Dataset Decision Reason:</b> {random_row['reason_for_decision']}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Upload your CV (PDF)**")
        uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            st.session_state.cv_text = extract_text_from_pdf(uploaded_pdf)
            st.success("CV loaded successfully.")

        st.markdown("---")
        user_input = st.text_input("Ask a question", placeholder="Why was I rejected? What should I improve?")
        send = st.button("Send", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="chat-wrap" id="chat-window">', unsafe_allow_html=True)
        # Render history
        for m in st.session_state.messages:
            role_class = "msg-user" if m["role"] == "user" else "msg-bot"
            st.markdown(f'<div class="msg {role_class}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="footer-note">Conversation is grounded on the selected job plus top-matched roles from your dataset (via RAG + LangGraph).</div>', unsafe_allow_html=True)

    # Send action
    if send:
        ensure_groq()
        if not st.session_state.cv_text:
            st.warning("Please upload your CV (PDF) first for personalization.")
        elif not user_input.strip():
            st.warning("Please type a question.")
        else:
            # Append user message
            st.session_state.messages.append({"role":"user","content":user_input})

            # Run the LangGraph pipeline
            init_state: SRState = {
                "cv_text": st.session_state.cv_text,
                "user_question": user_input,
                "selected_job_idx": selected_idx,
                "retrieved": [],
                "feedback": "",
                "coaching": "",
                "matches": "",
                "final_answer": "",
            }
            result: SRState = compiled_graph.invoke(init_state)
            answer = result.get("final_answer","(no answer)")

            # Append assistant message
            st.session_state.messages.append({"role":"assistant","content":answer})

with tab_dash:
    st.subheader("Candidate Analytics Dashboard")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Rejection Reasons (Top 10)**")
        reason_counts = df["reason_for_decision"].value_counts().head(10).reset_index()
        reason_counts.columns = ["reason_for_decision", "count"]
        st.bar_chart(reason_counts.set_index("reason_for_decision"))

    with colB:
        st.markdown("**Most Common Roles (Top 10)**")
        role_counts = df["job_title"].value_counts().head(10).reset_index()
        role_counts.columns = ["job_title", "count"]
        st.bar_chart(role_counts.set_index("job_title"))

    st.markdown("---")
    st.markdown("**Description Length Distribution**")
    desc_len = df["job_description"].str.len().describe()[["mean","50%","min","max"]].to_frame("value")
    st.table(desc_len)

    st.markdown("---")
    st.markdown("**Sample of Dataset (Top 10)**")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
st.caption("© SmartRecruiter — LangGraph Orchestration • RAG • Groq • Streamlit Analytics")