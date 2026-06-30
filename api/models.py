"""
models.py — Schémas Pydantic pour l'API RAG
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Corps de la requête POST /query."""
    question: str = Field(
        ...,
        min_length=3,
        description="Question en langage naturel à soumettre au pipeline RAG.",
        examples=["Quelle est la formation de Gautier Blondel ?"]
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de chunks à retourner (entre 1 et 10)."
    )


class ChunkResult(BaseModel):
    """Un chunk retourné avec son score de similarité."""
    content: str  = Field(description="Texte du chunk récupéré.")
    source:  str  = Field(description="Document source du chunk.")
    score:   float = Field(description="Score de similarité cosinus (0 à 1).")


class QueryResponse(BaseModel):
    """Réponse du endpoint POST /query."""
    question: str               = Field(description="Question soumise.")
    top_k:    int               = Field(description="Nombre de chunks demandés.")
    results:  list[ChunkResult] = Field(description="Chunks les plus pertinents.")


class HealthResponse(BaseModel):
    """Réponse du endpoint GET /health."""
    status:      str  = Field(description="Statut de l'API.")
    db_reachable: bool = Field(description="Base DuckDB accessible.")
    chunk_count:  int  = Field(description="Nombre de chunks en base.")