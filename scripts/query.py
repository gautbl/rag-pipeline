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
    """Recherche les k chunks les plus similaires à la requête.

    Args:
        query: Texte de la requête en langage naturel.
        k: Nombre de résultats à retourner (défaut: 3).

    Returns:
        Liste de tuples (Document, score) triés par score décroissant.
        Retourne une liste vide si la base est vide ou inaccessible.

    Raises:
        RuntimeError: Si le modèle d'embeddings ne peut pas être chargé.
    """
    # 1. Vectoriser la requête
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"   ❌ Erreur chargement modèle {MODEL_NAME}")
        print(f"   {e}")
        raise RuntimeError(
            f"Impossible de charger le modèle d'embeddings. "
            f"Vérifiez que le chemin '{MODEL_NAME}' est valide ou "
            f"définissez HF_MODEL_NAME dans .env."
        )

    query_vector = embeddings.embed_query(query)

    # 2. Recherche par similarité cosinus dans DuckDB
    con = None
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
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

    except duckdb.CatalogException:
        # Table 'chunks' n'existe pas — base vide ou non initialisée
        return []
    except duckdb.IOException as e:
        print(f"   ⚠️ Base DuckDB inaccessible : {e}")
        return []
    except Exception as e:
        print(f"   ⚠️ Erreur lors de la recherche : {e}")
        return []
    finally:
        if con is not None:
            con.close()

    # 3. Retourner au format (Document, score)
    return [
        (Document(page_content=row[0], metadata={"source": row[1]}), row[2])
        for row in results
    ]