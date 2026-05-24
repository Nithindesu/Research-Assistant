import os
import streamlit as st
from rag_chain import ingest_pdf, ask_question, build_rag_chain
from vector_store import load_vector_store

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Research Paper Assistant")
st.caption("Upload a research paper and ask questions in plain English. | Built by **Nithin Datta Desu**")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = ""

with st.sidebar:
    st.header("📂 Upload Research Paper")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any research paper, report, or document"
    )

    if uploaded_file is not None:
        os.makedirs("data/uploads", exist_ok=True)
        save_path = f"data/uploads/{uploaded_file.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name != st.session_state.current_pdf:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                st.session_state.rag_chain = ingest_pdf(save_path)
                st.session_state.pdf_processed = True
                st.session_state.current_pdf = uploaded_file.name
                st.session_state.chat_history = []
            st.success(f"✅ Ready! Ask questions about {uploaded_file.name}")

    if st.session_state.pdf_processed:
        st.info(f"📖 Active: {st.session_state.current_pdf}")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload PDF → text extracted")
    st.markdown("2. Text split into chunks")
    st.markdown("3. Chunks converted to embeddings")
    st.markdown("4. Your question retrieves top chunks")
    st.markdown("5. LLM answers from those chunks")

    st.divider()
    st.markdown("**Developed by**")
    st.markdown("👨‍💻 Nithin Datta Desu")
    st.markdown("© 2025 All rights reserved.")

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View {len(message['sources'])} source passages"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Passage {i+1}** (Chunk #{src['chunk_index']})")
                    st.text(src["preview"])
                    st.divider()

    question = st.chat_input("Ask a question about the paper...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
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
                    st.markdown(f"**Passage {i+1}** (Chunk #{src['chunk_index']})")
                    st.text(src["preview"])
                    st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
