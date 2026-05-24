import os
import streamlit as st
from rag_chain import ingest_pdf, ask_question, build_rag_chain
from vector_store import load_vector_store

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(14,165,233,0.18);
}
.header-banner h1 {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 6px 0;
}
.header-banner p {
    color: #bfdbfe;
    font-size: 0.95rem;
    margin: 0;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    border: 1px solid #2563eb44;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin-top: 40px;
}
.welcome-card h2 { color: #93c5fd; font-size: 1.5rem; margin-bottom: 12px; }
.welcome-card p  { color: #94a3b8; font-size: 1rem; margin-bottom: 20px; }

/* Step cards row */
.steps-row {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 24px;
}
.step-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    width: 140px;
}
.step-card .step-num {
    font-size: 1.6rem;
    margin-bottom: 6px;
}
.step-card .step-text {
    color: #94a3b8;
    font-size: 0.78rem;
    line-height: 1.4;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.stat-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 6px 16px;
    color: #93c5fd;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Chat messages */
.user-bubble {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    color: white;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}
.assistant-bubble {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    color: #e2e8f0;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Source passage card */
.source-card {
    background: #0f172a;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    color: #94a3b8;
    font-size: 0.85rem;
    font-family: monospace;
    line-height: 1.5;
}
.source-label {
    color: #60a5fa;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 6px;
    font-family: sans-serif;
}

/* Sidebar section labels */
.sidebar-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 16px 0 8px 0;
}

/* Active file badge */
.active-badge {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 8px 12px;
    color: #6ee7b7;
    font-size: 0.82rem;
    margin-top: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #1e293b;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("rag_chain", None),
    ("pdf_processed", False),
    ("chat_history", []),
    ("current_pdf", ""),
    ("question_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.2rem;'>📄</div>
        <div style='color:#93c5fd; font-weight:700; font-size:1rem; margin-top:4px;'>Research Assistant</div>
        <div style='color:#475569; font-size:0.75rem;'>Powered by Groq · LangChain · FAISS</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sidebar-label'>📂 Upload Document</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any research paper, report, or document",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        os.makedirs("data/uploads", exist_ok=True)
        save_path = f"data/uploads/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name != st.session_state.current_pdf:
            with st.spinner("Processing document..."):
                st.session_state.rag_chain = ingest_pdf(save_path)
                st.session_state.pdf_processed = True
                st.session_state.current_pdf = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.question_count = 0
            st.success("✅ Ready! Start asking questions.")

    if st.session_state.pdf_processed:
        st.markdown(f"""
        <div class='active-badge'>
            📖 &nbsp;<b>Active:</b> {st.session_state.current_pdf}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sidebar-label'>⚙️ How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("📥", "Upload PDF"),
        ("✂️", "Split into chunks"),
        ("🔢", "Convert to vectors"),
        ("🔍", "Retrieve top matches"),
        ("🤖", "LLM generates answer"),
    ]
    for icon, text in steps:
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.82rem; padding:3px 0;'>{icon} &nbsp;{text}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("""
    <div style='text-align:center; padding: 8px 0;'>
        <div style='color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;'>Developed by</div>
        <div style='color:#93c5fd; font-weight:700; font-size:0.9rem; margin-top:4px;'>👨‍💻 Nithin Datta Desu</div>
        <div style='color:#475569; font-size:0.72rem; margin-top:2px;'>© 2025 All rights reserved</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-banner'>
    <h1>📄 AI Research Paper Assistant</h1>
    <p>Upload any research paper and get instant, cited answers in plain English &nbsp;·&nbsp; Built by <b>Nithin Datta Desu</b></p>
</div>
""", unsafe_allow_html=True)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class='welcome-card'>
        <h2>👋 Welcome! Upload a paper to get started</h2>
        <p>This tool reads your research PDF and answers any question you have about it — with source citations.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        ("📥", "Upload PDF"),
        ("✂️", "Chunk text"),
        ("🔢", "Embed vectors"),
        ("🔍", "Retrieve"),
        ("🤖", "LLM answers"),
    ]
    for col, (icon, text) in zip([col1, col2, col3, col4, col5], cards):
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-num'>{icon}</div>
                <div class='step-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:28px; color:#475569; font-size:0.85rem;'>
        🔒 &nbsp;Completely free · No paid APIs · Runs locally
    </div>
    """, unsafe_allow_html=True)

# ── Chat interface ────────────────────────────────────────────────────────────
else:
    # Stats bar
    st.markdown(f"""
    <div class='stats-bar'>
        <div class='stat-chip'>📄 &nbsp;{st.session_state.current_pdf}</div>
        <div class='stat-chip'>💬 &nbsp;{st.session_state.question_count} question{"s" if st.session_state.question_count != 1 else ""} asked</div>
        <div class='stat-chip'>🤖 &nbsp;Llama 3.3 70B</div>
        <div class='stat-chip'>🔍 &nbsp;FAISS · top-4 chunks</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View {len(message['sources'])} source passages"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class='source-label'>Passage {i+1} &nbsp;·&nbsp; Chunk #{src['chunk_index']} &nbsp;·&nbsp; {src['source_file']}</div>
                    <div class='source-card'>{src['preview']}</div>
                    """, unsafe_allow_html=True)

    # Input
    question = st.chat_input("Ask anything about the paper...")

    if question:
        st.session_state.question_count += 1
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching paper and generating answer..."):
                result = ask_question(st.session_state.rag_chain, question)
                answer = result["answer"]
                sources = result["sources"]
            st.markdown(answer)
            with st.expander(f"📚 View {len(sources)} source passages used"):
                for i, src in enumerate(sources):
                    st.markdown(f"""
                    <div class='source-label'>Passage {i+1} &nbsp;·&nbsp; Chunk #{src['chunk_index']} &nbsp;·&nbsp; {src['source_file']}</div>
                    <div class='source-card'>{src['preview']}</div>
                    """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    AI Research Paper Assistant &nbsp;·&nbsp; Developed by <b>Nithin Datta Desu</b> &nbsp;·&nbsp; © 2025
</div>
""", unsafe_allow_html=True)
