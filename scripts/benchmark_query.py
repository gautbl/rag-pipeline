# scripts/benchmark_query.py — Benchmark de latence des requêtes vectorielles
import time
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import duckdb

DUCKDB_PATH = "./data/rag.duckdb"
MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"

test_queries = [
    "Quelle est la formation de Gautier Blondel ?",
    "Quels outils ETL maîtrise-t-il ?",
    "Quelle est son expérience chez Sopra Steria ?",
    "Quel projet en IA a-t-il réalisé ?",
    "Quels langages de programmation utilise-t-il ?",
]

# --- Vérifications préalables ---
if not os.path.isfile(DUCKDB_PATH):
    print(f"⚠️ Base DuckDB introuvable : {DUCKDB_PATH}")
    print(f"   Exécutez d'abord : python scripts/test_pipeline.py")
    sys.exit(1)

# Vérifier que la base contient la table chunks
try:
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
except duckdb.CatalogException:
    print(f"⚠️ La table 'chunks' n'existe pas dans {DUCKDB_PATH}")
    print(f"   Exécutez d'abord : python scripts/test_pipeline.py")
    sys.exit(1)
except duckdb.IOException as e:
    print(f"⚠️ Base DuckDB inaccessible : {e}")
    sys.exit(1)

if row_count == 0:
    print(f"⚠️ La table 'chunks' est vide dans {DUCKDB_PATH}")
    print(f"   Exécutez d'abord : python scripts/test_pipeline.py")
    sys.exit(1)

# --- Chargement du modèle ---
if not os.path.isdir(MODEL_PATH):
    print(f"⚠️ Modèle introuvable : {MODEL_PATH}")
    print(f"   Téléchargez-le ou vérifiez le chemin.")
    sys.exit(1)

try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"❌ Erreur chargement du modèle : {e}")
    sys.exit(1)

# --- Benchmark ---
times = []
conn = duckdb.connect(DUCKDB_PATH, read_only=True)

for q in test_queries:
    query_vec = model.encode(q, normalize_embeddings=True).astype(np.float32).tolist()

    start = time.perf_counter()
    conn.execute(
        """
        SELECT content, array_cosine_similarity(embedding, $1::FLOAT[384]) AS score
        FROM chunks
        ORDER BY score DESC
        LIMIT 3
        """,
        [query_vec]
    ).fetchall()
    end = time.perf_counter()

    times.append((end - start) * 1000)  # en millisecondes

conn.close()

print(f"Temps de requête moyen : {np.mean(times):.2f} ms")
print(f"Min : {np.min(times):.2f} ms | Max : {np.max(times):.2f} ms")