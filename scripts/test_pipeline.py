# scripts/test_pipeline.py — Validation du pipeline RAG end-to-end (DuckDB+VSS)
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from scripts.ingest import split_documents
from scripts.embed import embed_and_store
from scripts.query import search

# Vérifier les dépendances critiques avant de commencer
PDF_PATH = os.getenv("TEST_PDF_PATH", "./data/pdfs/")
MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "./models/paraphrase-multilingual-MiniLM-L12-v2"
)

if not os.path.isdir(PDF_PATH):
    print(f"⚠️ Répertoire PDF introuvable : {PDF_PATH}")
    print(f"   Créez-le ou définissez TEST_PDF_PATH dans .env")
    sys.exit(1)

if not os.path.isdir(MODEL_NAME):
    print(f"⚠️ Modèle introuvable : {MODEL_NAME}")
    print(f"   Téléchargez-le ou définissez HF_MODEL_NAME dans .env")
    sys.exit(1)

# --- 1. Ingest ---
print("\n=== 1. INGEST ===")
start = time.time()
chunks = split_documents(PDF_PATH)
end = time.time()

if len(chunks) == 0:
    print("⚠️ Aucun chunk généré. Vérifiez les PDFs dans", PDF_PATH)
    sys.exit(1)

print(f"{len(chunks)} chunks générés")
print(f"Temps d'ingestion : {end - start:.2f} sec")
print(f"   Exemple : {chunks[0].page_content[:100]}...")

# --- 2. Embed & Store ---
print("\n=== 2. EMBED & STORE (DuckDB+VSS) ===")
embed_and_store(chunks)
print("Chunks vectorisés et indexés dans DuckDB")

# --- 3. Query ---
print("\n=== 3. QUERY (similarité cosinus SQL) ===")
queries = [
    "Quelles bases de données a-t-il utilisées ?",
    "Quelle est sa formation académique ?",
    "A-t-il travaillé avec SAP ?",
    "Quels outils ETL maîtrise-t-il ?"
]

for q in queries:
    print(f"\nQuestion : {q}")
    print("-" * 50)
    results = search(q, k=2)

    if len(results) == 0:
        print("  ⚠️ Aucun résultat trouvé")
        continue

    for i, (doc, score) in enumerate(results):
        assert len(doc.page_content) > 0, "Contenu vide"
        assert 0.0 <= score <= 1.0, f"Score hors intervalle [0,1] : {score}"
        print(f"  Résultat {i+1} — score cosinus : {score:.4f}")
        print(f"  Source : {doc.metadata.get('source', 'N/A')}")
        print(f"  Contenu : {doc.page_content.strip()[:120]}...")

print("\nPipeline RAG end-to-end fonctionnel (DuckDB+VSS)")