# tests/test_chunking.py — Validation du découpage en chunks
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.ingest import split_documents

PDF_PATH = "tests/fixtures/sample.pdf"

def test_chunks_not_empty():
    """Le découpage doit produire au moins un chunk."""
    chunks = split_documents(PDF_PATH)
    assert len(chunks) > 0

def test_chunk_size():
    """Chaque chunk ne doit pas dépasser chunk_size=300 caractères."""
    chunks = split_documents(PDF_PATH)
    for chunk in chunks:
        assert len(chunk.page_content) <= 300

def test_chunks_have_content():
    """Aucun chunk ne doit être vide ou contenir uniquement des espaces."""
    chunks = split_documents(PDF_PATH)
    for chunk in chunks:
        assert chunk.page_content.strip() != ""

def test_chunks_have_metadata():
    """Chaque chunk doit avoir une metadata source."""
    chunks = split_documents(PDF_PATH)
    for chunk in chunks:
        assert "source" in chunk.metadata