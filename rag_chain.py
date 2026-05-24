import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from pdf_processor import process_pdf
from vector_store import create_vector_store, save_vector_store, load_vector_store

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert research assistant. Use ONLY the context below to answer
the question. If the answer is not in the context, say:
"I could not find this information in the uploaded document."

Always cite which part of the document supports your answer.

Context from the research paper:
{context}

Question: {question}

Answer (be specific, cite relevant sections):
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=1000
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


def ingest_pdf(pdf_path: str, store_path: str = "faiss_index") -> RetrievalQA:
    pdf_name = os.path.basename(pdf_path)
    chunks = process_pdf(pdf_path)
    vectorstore = create_vector_store(chunks, pdf_name)
    save_vector_store(vectorstore, store_path)
    chain = build_rag_chain(vectorstore)
    print(f"PDF ingested. Ready to answer questions about: {pdf_name}")
    return chain


def ask_question(chain: RetrievalQA, question: str) -> dict:
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    source_texts = []
    for doc in sources:
        source_texts.append({
            "chunk_index": doc.metadata.get("chunk_index"),
            "source_file": doc.metadata.get("source"),
            "preview": doc.page_content[:200] + "..."
        })
    return {
        "answer": answer,
        "sources": source_texts
    }


if __name__ == "__main__":
    chain = ingest_pdf("sample_paper.pdf")
    questions = [
        "What is the main contribution of this paper?",
        "What dataset was used in the experiments?",
        "What were the key results and metrics reported?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = ask_question(chain, q)
        print(f"A: {result['answer']}")
        print(f"Sources used: {len(result['sources'])} chunks")
