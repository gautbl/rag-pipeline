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
    """Vectorise les chunks et les stocke dans DuckDB+VSS.

    Args:
        chunks: Liste de documents LangChain à vectoriser.

    Raises:
        RuntimeError: Si le modèle d'embeddings ne peut pas être chargé.
        duckdb.IOException: Si la base DuckDB est inaccessible ou corrompue.
    """
    print("Chargement du modèle d'embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"   ❌ Erreur chargement modèle {MODEL_NAME}")
        print(f"   {e}")
        raise RuntimeError(
            f"Impossible de charger le modèle d'embeddings. "
            f"Vérifiez que le chemin '{MODEL_NAME}' est valide ou "
            f"définissez HF_MODEL_NAME dans .env pour utiliser un autre modèle."
        )

    con = None
    try:
        con = duckdb.connect(DB_PATH)
        con.execute("INSTALL vss;")
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

    except duckdb.IOException as e:
        print(f"   ❌ Erreur d'accès à la base DuckDB : {DB_PATH}")
        print(f"   {e}")
        raise
    except Exception as e:
        print(f"   ❌ Erreur lors de l'écriture dans DuckDB : {e}")
        raise
    finally:
        if con is not None:
            con.close()