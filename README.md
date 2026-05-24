# AI Research Paper Assistant

A free, intelligent document Q&A system that lets you upload any research paper and ask natural-language questions about it. Built with RAG (Retrieval-Augmented Generation), LangChain, FAISS, and Groq's free LLM API.

> Developed by **Nithin Datta Desu** · © 2025

---

## What It Does

Upload a research PDF → Ask any question → Get a precise, cited answer pulled directly from the document. No hallucination. No paid APIs.

---

## Features

- Upload any research paper in PDF format
- Ask questions in plain English and get cited answers
- Semantic search using local HuggingFace embeddings (no API needed)
- Powered by Groq's free Llama 3.3 70B LLM (14,400 requests/day free)
- Interactive chat interface built with Streamlit
- REST API built with FastAPI for programmatic access
- Docker support for easy deployment
- Completely free — no paid subscriptions required

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — Llama 3.3 70B |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| PDF Parsing | pypdf |

---

## Project Structure

```
research_assistant/
├── app.py              # Streamlit chat interface
├── api.py              # FastAPI REST backend
├── rag_chain.py        # RAG pipeline (core logic)
├── vector_store.py     # FAISS vector store management
├── pdf_processor.py    # PDF text extraction and chunking
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── .env                # API keys (not committed to git)
├── .gitignore
└── docs/
    ├── project_documentation.md
    └── langchain_rag_guide.md
```

---

## Prerequisites

- Python 3.9 or above
- A free Groq API key — sign up at [console.groq.com](https://console.groq.com) (no credit card needed)
- Internet connection for first-time embedding model download (~90 MB)

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/research-assistant.git
cd research-assistant
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install langchain-groq sentence-transformers
```

**4. Create your `.env` file**
```bash
touch .env
```
Add the following line inside `.env`:
```
GROQ_API_KEY=gsk_<your_key_here>
```

---

## Running the App

### Option 1 — Streamlit UI (Recommended)
```bash
streamlit run app.py --server.port=8501
```
Open your browser → `http://localhost:8501`

### Option 2 — FastAPI REST Backend
```bash
uvicorn api:app --reload --port 8000
```
Open your browser → `http://localhost:8000/docs`

### Option 3 — Docker
```bash
docker build -t research-assistant .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_<your_key> research-assistant
```

---

## How to Use

1. Open `http://localhost:8501` in your browser
2. Click **Browse files** in the sidebar and upload a research PDF
3. Wait for the **Ready!** confirmation (first upload downloads the embedding model — one time only)
4. Type any question in the chat box and press Enter
5. Read the cited answer and expand **View source passages** to see which parts of the paper were used

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload a PDF for indexing |
| POST | `/ask` | Ask a question about an uploaded PDF |
| GET | `/documents` | List all loaded documents |

**Example request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"pdf_name": "paper.pdf", "question": "What is the main contribution?"}'
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com | Yes |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AuthenticationError 401` | Check your `GROQ_API_KEY` in `.env` |
| `RateLimitError 429` | Groq free limit reached — wait until midnight UTC |
| Port 8501 already in use | Run `pkill -9 -f "streamlit"` then restart |
| First upload is slow | Normal — embedding model downloading once (~90 MB) |
| PDF shows no text | Use a text-based PDF (not a scanned image) |

---

## License

This project is for educational and research purposes.  
Developed by **Nithin Datta Desu** · © 2025 All Rights Reserved.
