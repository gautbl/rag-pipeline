"""
main.py — API REST du pipeline RAG
Endpoints :
  GET  /health  → statut de l'API et de la base vectorielle
  POST /query   → recherche sémantique sur le corpus
"""

import os
import numpy as np
import duckdb
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from api.models import QueryRequest, QueryResponse, ChunkResult, HealthResponse

# ── Configuration ────────────────────────────────────────────────
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/rag.duckdb")
MODEL_PATH  = os.getenv("MODEL_PATH",  "./models/paraphrase-multilingual-MiniLM-L12-v2")

# ── État global de l'application ─────────────────────────────────
# Le modèle est chargé une seule fois au démarrage via le lifespan
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle d'embeddings au démarrage de l'API."""
    print("⏳ Chargement du modèle d'embeddings...")
    app_state["model"] = SentenceTransformer(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")
    yield
    # Nettoyage à l'arrêt
    app_state.clear()


# ── Initialisation FastAPI ───────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "API REST du pipeline de recherche sémantique sur corpus PDF.\n\n"
        "Basée sur **DuckDB+VSS** (recherche vectorielle) et "
        "**HuggingFace sentence-transformers** (embeddings multilingues FR/EN).\n\n"
        "Permet d'interroger un corpus de documents en langage naturel "
        "et de retourner les passages les plus pertinents avec un score "
        "de similarité cosinus."
    ),
    version="0.1.0",
    lifespan=lifespan,
    contact={
        "name": "Gautier Blondel",
        "url": "https://github.com/gautbl/rag-pipeline",
    },
    license_info={"name": "MIT"},
)


# ── Utilitaires ──────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Vectorise un texte avec le modèle chargé en mémoire."""
    model = app_state.get("model")
    if model is None:
        raise RuntimeError("Modèle d'embeddings non initialisé.")
    return model.encode(text, normalize_embeddings=True).astype(np.float32)


def search_chunks(question: str, top_k: int) -> list[ChunkResult]:
    """Recherche les top-K chunks par similarité cosinus dans DuckDB."""
    query_vector = get_embedding(question)

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT content, source,
                   array_cosine_similarity(embedding, $1::FLOAT[384]) AS score
            FROM chunks
            ORDER BY score DESC
            LIMIT $2
            """,
            [query_vector.tolist(), top_k]
        ).fetchall()
    finally:
        conn.close()

    return [
        ChunkResult(content=row[0], source=row[1], score=round(float(row[2]), 4))
        for row in rows
    ]


# ── Endpoints ────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    description="Vérifie que l'API et la base vectorielle DuckDB sont opérationnelles.",
    tags=["Monitoring"],
    responses={
        200: {"description": "API opérationnelle (dégradée ou non)."},
    },
)
def health_check() -> HealthResponse:
    """Vérifie que l'API est opérationnelle et que la base DuckDB est accessible."""
    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        conn.close()
        return HealthResponse(status="ok", db_reachable=True, chunk_count=count)
    except Exception as e:
        return HealthResponse(status=f"degraded: {e}", db_reachable=False, chunk_count=0)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Recherche sémantique",
    description=(
        "Reçoit une question en langage naturel et retourne les `top_k` "
        "chunks les plus pertinents du corpus, triés par score de "
        "similarité cosinus décroissant."
    ),
    tags=["RAG"],
    responses={
        200: {"description": "Résultats retournés avec succès."},
        404: {"description": "Aucun chunk trouvé — corpus non ingéré."},
        500: {"description": "Erreur interne (modèle ou base inaccessible)."},
    },
)
def query(request: QueryRequest) -> QueryResponse:
    """
    Reçoit une question en langage naturel et retourne les chunks
    les plus pertinents du corpus avec leurs scores de similarité cosinus.
    """
    try:
        results = search_chunks(request.question, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Aucun chunk trouvé. Vérifiez que le corpus a été ingéré."
        )

    return QueryResponse(
        question=request.question,
        top_k=request.top_k,
        results=results,
    )
