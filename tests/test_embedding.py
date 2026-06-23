# tests/test_embedding.py — Validation de la vectorisation DuckDB+VSS
import sys
import os

# Définir DB_PATH AVANT tout import de scripts
DB_PATH = os.path.abspath("./data/test_rag.duckdb")
os.environ["DUCKDB_PATH"] = DB_PATH

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import pytest
from dotenv import load_dotenv
from scripts.ingest import split_documents
from scripts.embed import embed_and_store

load_dotenv()

PDF_PATH = "tests/fixtures/sample.pdf"

@pytest.fixture(scope="module", autouse=True)
def embedded_chunks():
    """Fixture : ingestion + vectorisation une seule fois pour tous les tests."""
    print(f"\nDB_PATH : {DB_PATH}")

    # Nettoyage avant les tests
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    chunks = split_documents(PDF_PATH)
    embed_and_store(chunks)
    yield chunks

    # Nettoyage après les tests
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_table_exists(embedded_chunks):
    """La table chunks doit exister dans DuckDB."""
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = [t for t in con.execute("SHOW TABLES").fetchall()[0]]
    con.close()
    assert "chunks" in tables

def test_nb_rows_matches_chunks(embedded_chunks):
    """Le nombre de lignes dans DuckDB doit correspondre au nombre de chunks."""
    con = duckdb.connect(DB_PATH, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    assert count == len(embedded_chunks)

def test_embedding_dimension(embedded_chunks):
    """Chaque vecteur doit avoir 384 dimensions (modèle MiniLM)."""
    con = duckdb.connect(DB_PATH, read_only=True)
    row = con.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()[0]
    con.close()
    assert len(row) == 384

def test_no_empty_content(embedded_chunks):
    """Aucun chunk stocké ne doit avoir un contenu vide."""
    con = duckdb.connect(DB_PATH, read_only=True)
    empty = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE TRIM(content) = ''"
    ).fetchone()[0]
    con.close()
    assert empty == 0