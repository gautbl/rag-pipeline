# tests/test_similarity.py — Validation de la recherche par similarité
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dotenv import load_dotenv
from scripts.ingest import split_documents
from scripts.embed import embed_and_store
from scripts.query import search

load_dotenv()

PDF_PATH = "tests/fixtures/sample.pdf"
DB_PATH  = "./data/test_rag.duckdb"

@pytest.fixture(scope="module", autouse=True)
def setup_pipeline():
    """Fixture : prépare la base de test avant les tests de similarité."""
    os.environ["DUCKDB_PATH"] = DB_PATH
    chunks = split_documents(PDF_PATH)
    embed_and_store(chunks)
    yield
    # Nettoyage après les tests
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_returns_k_results():
    """La recherche doit retourner exactement k résultats."""
    results = search("formation académique", k=3)
    assert len(results) == 3

def test_results_have_content():
    """Chaque résultat doit avoir un contenu non vide."""
    results = search("expérience professionnelle", k=2)
    for doc, score in results:
        assert len(doc.page_content.strip()) > 0

def test_score_range():
    """Le score cosinus doit être compris entre 0 et 1."""
    results = search("bases de données", k=3)
    for doc, score in results:
        assert 0.0 <= score <= 1.0

def test_results_are_ordered():
    """Les résultats doivent être triés par score décroissant."""
    results = search("outils ETL", k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

def test_different_queries_different_results():
    """Deux requêtes différentes ne doivent pas retourner exactement les mêmes chunks."""
    results_1 = search("formation académique", k=2)
    results_2 = search("bases de données Oracle", k=2)
    contents_1 = {doc.page_content for doc, _ in results_1}
    contents_2 = {doc.page_content for doc, _ in results_2}
    assert contents_1 != contents_2