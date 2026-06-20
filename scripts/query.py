# scripts/query.py — Recherche par similarité cosinus
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def search(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results