import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += page_text
    print(f"Extracted {len(reader.pages)} pages, {len(full_text)} characters")
    return full_text


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks


def process_pdf(pdf_path: str) -> list:
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    return chunks


if __name__ == "__main__":
    chunks = process_pdf("sample_paper.pdf")
    print(f"\nFirst chunk preview:")
    print(chunks[0][:300])
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")
