# scripts/ingest.py — Chargement et découpage des PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def split_documents(pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
    print(f"Chargement de {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  -> {len(docs)} page(s) chargée(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"  -> {len(chunks)} chunks créés")
    return chunks