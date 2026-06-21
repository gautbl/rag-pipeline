# # scripts/embed.py — Vectorisation HuggingFace + stockage Chroma
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import os

# CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
# MODEL_NAME = os.getenv(
    # "HF_MODEL_NAME",
    # "./models/paraphrase-multilingual-MiniLM-L12-v2"
    # #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" Si accès au site hf.co
# )

# def embed_and_store(chunks):
    # print("Chargement du modèle d'embeddings...")
    # embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # print("Indexation dans Chroma...")
    # vectorstore = Chroma.from_documents(
        # documents=chunks,
        # embedding=embeddings,
        # persist_directory=CHROMA_DIR
    # )
    # print(f"  -> {len(chunks)} chunks indexés dans {CHROMA_DIR}")
    # return vectorstore
    
    
    
# scripts/embed.py — Vectorisation HuggingFace + stockage DuckDB+VSS
import duckdb
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "./data/rag.duckdb")
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
)

def embed_and_store(chunks):
    print("Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    con = duckdb.connect(DB_PATH)
    con.execute("LOAD vss;")
    con.execute("SET hnsw_enable_experimental_persistence = true;")
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("""
        CREATE TABLE chunks (
            id        INTEGER,
            content   VARCHAR,
            source    VARCHAR,
            embedding FLOAT[384]
        )
    """)

    print("Vectorisation et stockage dans DuckDB...")
    for i, chunk in enumerate(chunks):
        # Conversion explicite float64 -> float32
        vector = np.array(
            embeddings.embed_query(chunk.page_content), dtype=np.float32
        ).tolist()
        con.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            [i, chunk.page_content, chunk.metadata.get("source", ""), vector]
        )

    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_hnsw ON chunks USING HNSW (embedding)"
    )
    print(f"  -> {len(chunks)} chunks indexés dans DuckDB")
    con.close()