import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import ingest_pdf, ask_question

app = FastAPI(
    title="AI Research Paper Assistant API",
    description="Upload research PDFs and ask natural-language questions via REST. Developed by Nithin Datta Desu.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

active_chains = {}


class QuestionRequest(BaseModel):
    pdf_name: str
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    pdf_name: str
    question: str


@app.get("/")
def root():
    return {"message": "AI Research Paper Assistant API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    os.makedirs("data/uploads", exist_ok=True)
    save_path = f"data/uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chain = ingest_pdf(save_path)
        active_chains[file.filename] = chain
        return {
            "status": "success",
            "filename": file.filename,
            "message": "PDF processed and ready for questions."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if request.pdf_name not in active_chains:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{request.pdf_name}' not found. Upload it first via /upload"
        )
    chain = active_chains[request.pdf_name]
    result = ask_question(chain, request.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        pdf_name=request.pdf_name,
        question=request.question
    )


@app.get("/documents")
def list_documents():
    return {"loaded_documents": list(active_chains.keys())}
