import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    return _embeddings


def create_vector_store(chunks: list, pdf_name: str) -> FAISS:
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": pdf_name, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    vectorstore = FAISS.from_documents(documents, get_embeddings())
    print(f"Vector store created with {len(documents)} document chunks")
    return vectorstore


def save_vector_store(vectorstore: FAISS, save_path: str):
    vectorstore.save_local(save_path)
    print(f"Vector store saved to {save_path}")


def load_vector_store(save_path: str) -> FAISS:
    vectorstore = FAISS.load_local(
        save_path, get_embeddings(),
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from {save_path}")
    return vectorstore


def search_similar_chunks(vectorstore: FAISS, query: str, k: int = 4) -> list:
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\nTop {k} chunks for query: '{query[:60]}...'")
    for i, (doc, score) in enumerate(results):
        print(f"  Chunk {i+1} | Score: {score:.4f} | "
              f"Preview: {doc.page_content[:80]}...")
    return results


if __name__ == "__main__":
    from pdf_processor import process_pdf
    chunks = process_pdf("sample_paper.pdf")
    vectorstore = create_vector_store(chunks, "sample_paper.pdf")
    results = search_similar_chunks(
        vectorstore,
        "What methodology did the authors use?",
        k=3
    )
