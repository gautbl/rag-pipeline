# # scripts/test_pipeline.py — Validation du pipeline RAG end-to-end
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from scripts.ingest import split_documents
# from scripts.embed import embed_and_store
# from scripts.query import search

# PDF_PATH = "./data/pdfs/Resume_Gautier_Blondel_04_2026.pdf"

# # --- 1. Ingest ---
# print("\n=== 1. INGEST ===")
# chunks = split_documents(PDF_PATH)
# assert len(chunks) > 0, "Aucun chunk généré"
# print(f"{len(chunks)} chunks générés")
# print(f"   Exemple : {chunks[0].page_content[:100]}...")

# # --- 2. Embed & Store ---
# print("\n=== 2. EMBED & STORE ===")
# vectorstore = embed_and_store(chunks)
# print("Chunks indexés dans Chroma")

# # --- 3. Query ---
# print("\n=== 3. QUERY ===")
# queries = [
    # "Quelles bases de données a-t-il utilisées ?",
    # "Quelle est sa formation académique ?",
# ]
# for q in queries:
    # print(f"\nQuestion : {q}")
    # results = search(q, k=2)
    # assert len(results) == 2, f"Attendu 2 résultats, obtenu {len(results)}"
    # for i, (doc, score) in enumerate(results):
        # print(f"  Résultat {i+1} — score : {score:.4f}")
        # print(f"  Contenu : {doc.page_content.strip()[:120]}...")

# print("\nPipeline RAG end-to-end fonctionnel")


# scripts/test_pipeline.py — Validation du pipeline RAG end-to-end (DuckDB+VSS)
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from scripts.ingest import split_documents
from scripts.embed import embed_and_store
from scripts.query import search

PDF_PATH = "./data/pdfs/Resume_Gautier_Blondel_04_2026.pdf"

# --- 1. Ingest ---
print("\n=== 1. INGEST ===")
chunks = split_documents(PDF_PATH)
assert len(chunks) > 0, "Aucun chunk généré"
print(f"{len(chunks)} chunks générés")
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

    assert len(results) == 2, f"Attendu 2 résultats, obtenu {len(results)}"

    for i, (doc, score) in enumerate(results):
        assert len(doc.page_content) > 0, "Contenu vide"
        assert 0.0 <= score <= 1.0, f"Score hors intervalle [0,1] : {score}"
        print(f"  Résultat {i+1} — score cosinus : {score:.4f}")
        print(f"  Source : {doc.metadata.get('source', 'N/A')}")
        print(f"  Contenu : {doc.page_content.strip()[:120]}...")

print("\nPipeline RAG end-to-end fonctionnel (DuckDB+VSS)")