# tests/test_api.py
"""
Tests des endpoints FastAPI du pipeline RAG.
Couvre :
  - GET  /health  → statut nominal, statut dégradé (DB absente)
  - POST /query   → résultats valides, validation Pydantic, DB vide
  - Authentification  → clé manquante, clé invalide, clé valide
  - Unicode/caractères spéciaux dans les questions
  - Questions très longues (limites de validation)
  - Requêtes concurrentes
  - DB inaccessible → 503
  - Modèle non chargé → 503
  - API_KEY non configurée → 503
"""

import os
import sys
import pytest
import duckdb
import numpy as np

# ── Résolution du path racine ────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ── DB de test isolée — AVANT import de l'app ───────────────────
TEST_DB_PATH    = "./data/test_rag_api.duckdb"
TEST_MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"

os.environ["DUCKDB_PATH"] = TEST_DB_PATH
os.environ["MODEL_PATH"]  = TEST_MODEL_PATH
os.environ["API_KEY"]     = "test-api-key-2026"

# ── Import app APRÈS injection des variables d'environnement ─────
from fastapi.testclient import TestClient
from api.main import app
import api.main as main_module 


# On garde une référence pour les tests qui manipulent app_state
MODEL_HELD = main_module.app_state.get("model")


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

API_KEY_VALUE = "test-api-key-2026"


def _query(client, json_data, api_key=API_KEY_VALUE):
    """POST /query avec l'en-tête X-API-Key."""
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    return client.post("/query", json=json_data, headers=headers)


def _create_test_db(db_path: str, with_data: bool = True) -> None:
    """
    Crée une base DuckDB minimale pour les tests.
    - with_data=True  → 3 chunks réalistes (bail social, charges, loyer)
    - with_data=False → table vide (simule corpus non ingéré)
    """
    from sentence_transformers import SentenceTransformer

    conn = duckdb.connect(db_path)
    conn.execute("INSTALL vss; LOAD vss;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            content   TEXT,
            source    TEXT,
            embedding FLOAT[384]
        )
        """
    )

    if with_data:
        model = SentenceTransformer(TEST_MODEL_PATH)
        sample_texts = [
            "Le bail social est un contrat de location soumis à des plafonds de ressources.",
            "Les charges locatives comprennent l'entretien des parties communes.",
            "Le loyer est révisé annuellement selon l'indice de référence des loyers.",
        ]
        for i, text in enumerate(sample_texts):
            # np.float32 passé directement → DuckDB reçoit FLOAT sans conversion
            embedding = model.encode(text, normalize_embeddings=True).astype(np.float32)
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [text, f"doc_{i}.pdf", embedding]
            )

    conn.close()


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def client():
    """
    Fixture principale (scope=module) :
    DB avec 3 chunks, modèle chargé une seule fois pour tout le module.
    """
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    _create_test_db(TEST_DB_PATH, with_data=True)

    with TestClient(app) as c:
        yield c

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="function")
def client_empty_db(tmp_path):
    """
    DB vide dans un répertoire temporaire pytest.
    On surcharge la dépendance DB de l'app plutôt que l'env var,
    pour éviter que l'app réutilise la connexion déjà ouverte.
    """
    empty_db_path = str(tmp_path / "empty.duckdb")
    _create_test_db(empty_db_path, with_data=False)

    original = main_module.DUCKDB_PATH
    main_module.DUCKDB_PATH = empty_db_path

    yield TestClient(app)

    main_module.DUCKDB_PATH = original


@pytest.fixture(scope="function")                                 
def client_no_db(tmp_path):
    """
    DB inaccessible : sous-répertoire inexistant → DuckDB lève une IOError.
																			
    """
    bad_path = str(tmp_path / "nonexistent_subdir" / "db.duckdb")
										  

    original = main_module.DUCKDB_PATH
    main_module.DUCKDB_PATH = bad_path
			   

    yield TestClient(app)
			   

    main_module.DUCKDB_PATH = original


@pytest.fixture(scope="function")
def client_no_model(tmp_path):
    """
    API sans modèle chargé : vide app_state["model"] pendant le test.
    Le modèle est remis en place après le test via MODEL_HELD.
    """
    original_model = main_module.app_state.pop("model", None)
    yield TestClient(app)
    if original_model is not None:
        main_module.app_state["model"] = original_model


@pytest.fixture(scope="function")
def client_no_api_key(tmp_path):
    """
    API sans API_KEY configurée : on vide la variable d'env
    et la constante du module.
    """
    original_env   = os.environ.pop("API_KEY", None)
    original_constant = main_module.API_KEY
    main_module.API_KEY = None

    yield TestClient(app)

    main_module.API_KEY = original_constant
    if original_env is not None:
        os.environ["API_KEY"] = original_env


# ════════════════════════════════════════════════════════════════
# TESTS — GET /health
# ════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Vérifie le comportement du endpoint GET /health."""

    def test_health_returns_200(self, client):
        """Doit toujours retourner HTTP 200, même en mode dégradé."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """La réponse doit contenir les 3 champs du HealthResponse."""
        data = client.get("/health").json()
        assert "status"       in data
        assert "db_reachable" in data
        assert "chunk_count"  in data

    def test_health_nominal(self, client):
        """Avec une DB valide, status=ok et db_reachable=True."""
        data = client.get("/health").json()
        assert data["status"]       == "ok"
        assert data["db_reachable"] is True
        assert data["chunk_count"]  == 3

    def test_health_degraded_when_no_db(self, client_no_db):
        """
        Sans DB accessible, status contient 'degraded' et db_reachable=False.
        Utilise la fixture dédiée — ne recrée pas TestClient inline.
        """
        data = client_no_db.get("/health").json()
        assert "degraded"           in data["status"]
        assert data["db_reachable"] is False
        assert data["chunk_count"]  == 0


# ════════════════════════════════════════════════════════════════
# TESTS — POST /query
# ════════════════════════════════════════════════════════════════

class TestQueryEndpoint:
    """Vérifie le comportement du endpoint POST /query."""

    # ── Cas nominaux ─────────────────────────────────────────────

    def test_query_returns_200(self, client):
        """Une question valide doit retourner HTTP 200."""
        response = _query(client, {"question": "bail social", "top_k": 2})
        assert response.status_code == 200

    def test_query_response_structure(self, client):
        """La réponse doit contenir question, top_k et results."""
        data = _query(client, {"question": "charges locatives", "top_k": 2}).json()
        assert "question" in data
        assert "top_k"    in data
        assert "results"  in data

    def test_query_results_count_respects_top_k(self, client):
        """Le nombre de résultats retournés doit être <= top_k."""
        top_k = 2
        data  = _query(client, {"question": "loyer", "top_k": top_k}).json()
        assert len(data["results"]) <= top_k

    def test_query_result_fields(self, client):
        """Chaque chunk retourné doit avoir content, source et score."""
        data  = _query(client, {"question": "bail social", "top_k": 1}).json()
        chunk = data["results"][0]
        assert "content" in chunk
        assert "source"  in chunk
        assert "score"   in chunk

    def test_query_scores_are_ordered_desc(self, client):
        """Les scores doivent être triés par ordre décroissant (invariant métier)."""
        data   = _query(client, {"question": "loyer révision", "top_k": 3}).json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_query_score_in_range(self, client):
        """
        Avec normalize_embeddings=True (main.py), le score cosinus
        est dans [0.0, 1.0] — on resserre par rapport à [-1, 1].
        """
        data = _query(client, {"question": "charges", "top_k": 3}).json()
        for result in data["results"]:
            assert 0.0 <= result["score"] <= 1.0

    def test_query_echoes_question(self, client):
        """La réponse doit renvoyer la question posée telle quelle."""
        question = "indice de référence des loyers"
        data     = _query(client, {"question": question, "top_k": 1}).json()
        assert data["question"] == question

    def test_query_echoes_top_k(self, client):
        """La réponse doit renvoyer le top_k demandé tel quel."""
        data = _query(client, {"question": "bail social", "top_k": 2}).json()
        assert data["top_k"] == 2

    # ── Cas limites ──────────────────────────────────────────────

    def test_query_top_k_1_returns_single_result(self, client):
        """top_k=1 doit retourner exactement 1 résultat."""
        data = _query(client, {"question": "bail", "top_k": 1}).json()
        assert len(data["results"]) == 1

    def test_query_top_k_gt_corpus_returns_all(self, client):
        """
        top_k > nombre de chunks en base (3) doit retourner
        tous les chunks disponibles, pas lever d'erreur.
        """
        data = _query(client, {"question": "bail", "top_k": 10}).json()
        assert len(data["results"]) == 3  # corpus de test = 3 chunks

    def test_query_empty_db_returns_404(self, client_empty_db):
        """Sans chunks en base, l'API doit retourner 404."""
        response = _query(client_empty_db, {"question": "bail social", "top_k": 2})
        assert response.status_code == 404

    # ── Authentification ─────────────────────────────────────────

    def test_query_missing_key_returns_403(self, client):
        """Appel /query sans en-tête X-API-Key → 403."""
        response = client.post("/query", json={"question": "bail", "top_k": 1})
        assert response.status_code == 403

    def test_query_wrong_key_returns_403(self, client):
        """Appel /query avec une clé invalide → 403."""
        response = _query(client, {"question": "bail", "top_k": 1}, api_key="wrong-key")
        assert response.status_code == 403

    def test_query_valid_key_returns_200(self, client):
        """Appel /query avec la bonne clé → 200."""
        response = _query(client, {"question": "bail", "top_k": 1})
        assert response.status_code == 200

    # ── Unicode / caractères spéciaux ────────────────────────────

    @pytest.mark.parametrize("question", [
        "charges locatives",                    # français standard
        "qué tal",                              # accents
        "ありがとう",                              # japonais
        "liberté, égalité, fraternité",         # caractères spéciaux
        "‹›€£¥†‡°•…–—№",                        # symboles typographiques
        "<script>alert('xss')</script>",        # séquence dangereuse
        "\n\t\r",                              # whitespace pur
    ])
    def test_query_unicode_and_special_chars(self, client, question):
        """
        Une question avec caractères spéciaux doit retourner 200
        ou 404 (pas de correspondance sémantique — pour bruit).
        """
        response = _query(client, {"question": question, "top_k": 1})
        # Le score cosinus est clampé à >= 0, donc pas de 500
        assert response.status_code in (200, 404), f"Got {response.status_code}: {response.text[:200]}"

    # ── Question très longue ─────────────────────────────────────

    def test_query_long_question_returns_422(self, client):
        """Question > 5000 caractères → violation max_length → 422."""
        long_q = "a" * 5001
        response = _query(client, {"question": long_q, "top_k": 1})
        assert response.status_code == 422

    def test_query_max_length_question_returns_200(self, client):
        """
        Question de 5000 caractères (limite exacte).
        Doit passer la validation (pas de 422).
        """
        long_q = "a" * 5000
        response = _query(client, {"question": long_q, "top_k": 1})
        # Le score cosinus est clampé à >= 0, donc pas de 500
        assert response.status_code in (200, 404), f"Got {response.status_code}: {response.text[:200]}"

    # ── Requêtes concurrentes ────────────────────────────────────

    def test_concurrent_queries(self, client):
        """Plusieurs requêtes simultanées ne doivent pas échouer."""
        import concurrent.futures

        questions = [
            "bail social",
            "charges locatives",
            "loyer révision",
            "indice de référence",
        ]

        def post(q):
            return _query(client, {"question": q, "top_k": 2})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(post, questions))

        # Tous les statuts doivent être 200 (ou 404 pour requêtes sans résultat)
        for resp in results:
            assert resp.status_code in (200, 404), f"{resp.status_code}: {resp.json()}"

    # ── DB inaccessible sur /query ───────────────────────────────

    def test_query_no_db_returns_503(self, client_no_db):
        """/query sans DB accessible → 503 Service Unavailable."""
        response = _query(client_no_db, {"question": "bail social", "top_k": 2})
        assert response.status_code == 503

    # ── Validation Pydantic ──────────────────────────────────────

    def test_query_missing_question_returns_422(self, client):
        """Payload sans champ 'question' → erreur de validation 422."""
        response = _query(client, {"top_k": 2})
        assert response.status_code == 422

    def test_query_missing_top_k_uses_default(self, client):
        """
        Sans top_k, FastAPI utilise default=3 (models.py).
        La réponse doit être 200 avec top_k=3.
        """
        response = _query(client, {"question": "bail social"})
        assert response.status_code == 200
        assert response.json()["top_k"] == 3  # default=3 dans QueryRequest

    def test_query_empty_payload_returns_422(self, client):
        """Payload vide → erreur de validation 422."""
        response = _query(client, {})
        assert response.status_code == 422

    def test_query_question_too_short_returns_422(self, client):
        """
        question < 3 caractères → violation de min_length=3 (models.py)
        → erreur de validation 422.
        """
        response = _query(client, {"question": "ab", "top_k": 2})
        assert response.status_code == 422

    def test_query_top_k_zero_returns_422(self, client):
        """
        top_k=0 → violation de ge=1 (models.py)
        → erreur de validation 422.
        """
        response = _query(client, {"question": "bail social", "top_k": 0})
        assert response.status_code == 422

    def test_query_top_k_above_max_returns_422(self, client):
        """
        top_k=11 → violation de le=10 (models.py)
        → erreur de validation 422.
        """
        response = _query(client, {"question": "bail social", "top_k": 11})
        assert response.status_code == 422


# ════════════════════════════════════════════════════════════════
# TESTS — Erreurs internes
# ════════════════════════════════════════════════════════════════

class TestInternalErrors:
    """Vérifie les erreurs internes (modèle, config)."""

    def test_query_no_model_returns_503(self, client_no_model):
        """Sans modèle chargé, /query doit retourner 503."""
        response = _query(client_no_model, {"question": "bail social", "top_k": 2})
        assert response.status_code == 503

    def test_query_no_api_key_returns_503(self, client_no_api_key):
        """Sans API_KEY configurée côté serveur, /query doit retourner 503."""
        response = client_no_api_key.post(
            "/query",
            json={"question": "bail", "top_k": 1},
            headers={"X-API-Key": "some-key"}
        )
        assert response.status_code == 503

    def test_health_still_works_without_model(self, client_no_model):
        """GET /health ne dépend pas du modèle → doit rester 200."""
        response = client_no_model.get("/health")
        assert response.status_code == 200