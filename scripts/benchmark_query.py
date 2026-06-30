# scripts/benchmark_query.py
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import duckdb

DUCKDB_PATH = "./data/rag.duckdb"
MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_PATH)

test_queries = [
    "Quelle est la formation de Gautier Blondel ?",
    "Quels outils ETL maîtrise-t-il ?",
    "Quelle est son expérience chez Sopra Steria ?",
    "Quel projet en IA a-t-il réalisé ?",
    "Quels langages de programmation utilise-t-il ?",
]

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