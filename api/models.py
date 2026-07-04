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
        , examples=[3]
    )


class ChunkResult(BaseModel):
    """Un chunk retourné avec son score de similarité."""
    content: str  = Field(
        description="Texte du chunk récupéré.",
        examples=["École Centrale de Lille — Cursus Ingénieur, Informatique et systèmes."]
    )
    source:  str  = Field(
        description="Document source du chunk.",
        examples=["./data/pdfs/CV Gautier Blondel 06 2026 2.pdf"]
    )
    score:   float = Field(
        ge=0.0,
        le=1.0,
        description="Score de similarité cosinus (0 à 1).",
        examples=[0.87]
    )


class QueryResponse(BaseModel):
    """Réponse du endpoint POST /query."""
    question: str               = Field(
        description="Question soumise.",
        examples=["Quelle est la formation de Gautier Blondel ?"]
    )
    top_k:    int               = Field(
        description="Nombre de chunks demandés.",
        examples=[3]
    )
    results:  list[ChunkResult] = Field(description="Chunks les plus pertinents.")


class HealthResponse(BaseModel):
    """Réponse du endpoint GET /health."""
    status:      str  = Field(
        description="Statut de l'API.",
        examples=["ok"]
    )
    db_reachable: bool = Field(
        description="Base DuckDB accessible.",
        examples=[True]
    )
    chunk_count:  int  = Field(
        description="Nombre de chunks en base.",
        examples=[20]
    )
