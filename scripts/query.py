# # scripts/query.py — Recherche par similarité cosinus
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import os

# CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
# MODEL_NAME = os.getenv(
    # "HF_MODEL_NAME",
    # "./models/paraphrase-multilingual-MiniLM-L12-v2"
    # #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" Si accès au site hf.co
# )

# def search(query: str, k: int = 3):
    # embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    # vectorstore = Chroma(
        # persist_directory=CHROMA_DIR,
        # embedding_function=embeddings
    # )
    # results = vectorstore.similarity_search_with_score(query, k=k)
    # return results
    
    
    
# scripts/query.py — Recherche par similarité cosinus via DuckDB+VSS
import duckdb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "./data/rag.duckdb")
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
																														
)

def search(query: str, k: int = 3):
    # 1. Vectoriser la requête
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    query_vector = embeddings.embed_query(query)

    # 2. Recherche par similarité cosinus dans DuckDB
    con = duckdb.connect(DB_PATH)
    con.execute("LOAD vss;")

    results = con.execute("""
        SELECT
            content,
            source,
            array_cosine_similarity(embedding, ?::FLOAT[384]) AS score
        FROM chunks
        ORDER BY score DESC
        LIMIT ?
    """, [query_vector, k]).fetchall()

    con.close()

    # 3. Retourner au même format que Chroma : liste de (Document, score)
    return [
        (Document(page_content=row[0], metadata={"source": row[1]}), row[2])
        for row in results
    ]