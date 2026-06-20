# scripts/embed.py — Vectorisation HuggingFace + stockage Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def embed_and_store(chunks):
    print("Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("Indexation dans Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"  -> {len(chunks)} chunks indexés dans {CHROMA_DIR}")
    return vectorstore