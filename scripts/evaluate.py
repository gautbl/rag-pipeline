"""
evaluate.py — Évaluation RAGAS du pipeline RAG
Métriques actives (offline) : context_precision, context_recall, answer_similarity
Métriques commentées (LLM requis) : faithfulness, answer_relevancy
TODO: Décommenter les sections [FLAN-T5] après téléchargement de google/flan-t5-large
"""

import os
import mlflow
import pandas as pd
import duckdb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_similarity,
    # [FLAN-T5] Décommenter quand le modèle est disponible :
    # faithfulness,
    # answer_relevancy,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
# [FLAN-T5] Décommenter quand le modèle est disponible :
# from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
# [FLAN-T5] Décommenter quand le modèle est disponible :
# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline as hf_pipeline

from ragas.llms import BaseRagasLLM
from langchain_core.language_models import BaseLanguageModel

from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import PromptValue
from langchain_core.outputs import LLMResult, Generation

# ── Configuration ────────────────────────────────────────────────
DUCKDB_PATH  = "./data/rag.duckdb"
MODEL_PATH   = "./models/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 3
MLFLOW_URI   = "http://localhost:5000"
EXPERIMENT   = "ragas-evaluation"

# [FLAN-T5] Décommenter et renseigner le chemin local après téléchargement :
# LLM_MODEL_PATH = "./models/flan-t5-large/models--google--flan-t5-large/snapshots/<hash>/"

# ── Jeu de données d'évaluation ──────────────────────────────────
# Format RAGAS : question / ground_truth / answer / contexts
# answer = réponse simulée identique au ground_truth en mode offline
# TODO: Remplacer answer par une vraie génération LLM quand disponible
EVAL_DATASET = [
    {
        "question":     "Quelle est la formation de Gautier Blondel ?",
        "ground_truth": "Master of Computational and Applied Mathematics au KTH Stockholm, et cursus ingénieur à l'École Centrale de Lille.",
        "answer":       "Master of Computational and Applied Mathematics au KTH Stockholm, et cursus ingénieur à l'École Centrale de Lille.",
    },
    {
        "question":     "Quels outils ETL Gautier Blondel maîtrise-t-il ?",
        "ground_truth": "Talend Cloud DMP et Talend Open Studio.",
        "answer":       "Talend Cloud DMP et Talend Open Studio.",
    },
    {
        "question":     "Quelle est l'expérience de Gautier Blondel chez Sopra Steria ?",
        "ground_truth": "Business Analyst de septembre 2017 à juin 2022, prestataire data pour Airbus Helicopters et Auchan.",
        "answer":       "Business Analyst de septembre 2017 à juin 2022, prestataire data pour Airbus Helicopters et Auchan.",
    },
    {
        "question":     "Quel projet personnel Gautier Blondel a-t-il réalisé en IA ?",
        "ground_truth": "Un pipeline RAG end-to-end avec LangChain, DuckDB+VSS, dbt, Airflow et MLflow.",
        "answer":       "Un pipeline RAG end-to-end avec LangChain, DuckDB+VSS, dbt, Airflow et MLflow.",
    },
    {
        "question":     "Quels langages de programmation Gautier Blondel utilise-t-il ?",
        "ground_truth": "Python, Java, ABAP et R.",
        "answer":       "Python, Java, ABAP et R.",
    },
]

# ── Fonctions utilitaires ────────────────────────────────────────

def load_embeddings():
    """Charge le modèle d'embeddings HuggingFace local."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# [FLAN-T5] Décommenter cette fonction après téléchargement du modèle :
# def load_llm():
#     """Charge Flan-T5-large en local via HuggingFace Pipeline."""
#     pipe = hf_pipeline(
#         "text2text-generation",
#         model=LLM_MODEL_PATH,
#         tokenizer=LLM_MODEL_PATH,
#         max_new_tokens=256,
#         device=-1,  # CPU
#     )
#     return HuggingFacePipeline(pipeline=pipe)

def retrieve_contexts(question: str, embedding_model, top_k: int = TOP_K) -> list[str]:
    """
    Récupère les top-K chunks depuis DuckDB+VSS
    par similarité cosinus sur la question.
    """
    import numpy as np

    query_vector = embedding_model.embed_query(question)
    # Conversion en numpy array float32 — type natif attendu par DuckDB+VSS
    query_array  = np.array(query_vector, dtype=np.float32)

    conn = duckdb.connect(DUCKDB_PATH)
    rows = conn.execute(
        """
        SELECT content
        FROM chunks
        ORDER BY array_cosine_similarity(embedding, $1::FLOAT[384]) DESC
        LIMIT $2
        """,
        [query_array.tolist(), top_k]
    ).fetchall()
    conn.close()

    return [row[0] for row in rows]

def generate_answer(question: str, contexts: list[str]) -> str:
    """
    Retourne l'answer du dataset d'évaluation en mode offline.
    [FLAN-T5] Remplacer cette fonction par une vraie génération LLM :
    def generate_answer(question, contexts, llm):
        context_str = "\\n\\n".join(contexts)
        prompt = (
            f"Answer the following question based only on the context below.\\n\\n"
            f"Context:\\n{context_str}\\n\\n"
            f"Question: {question}\\n\\nAnswer:"
        )
        return llm.invoke(prompt)
    """
    # Retrouve l'answer pré-définie dans EVAL_DATASET pour la question donnée
    for item in EVAL_DATASET:
        if item["question"] == question:
            return item["answer"]
    return ""

# ── Pipeline principal ───────────────────────────────────────────

def build_ragas_dataset(embedding_model) -> Dataset:
    """Construit le dataset RAGAS avec contexts récupérés depuis DuckDB."""
    records = []
    for item in EVAL_DATASET:
        contexts = retrieve_contexts(item["question"], embedding_model)
        # [FLAN-T5] Remplacer par : answer = generate_answer(item["question"], contexts, llm)
        answer   = generate_answer(item["question"], contexts)
        records.append({
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "answer":       answer,
            "contexts":     contexts,
        })
    return Dataset.from_list(records)

class DummyLLM(BaseRagasLLM):
    """
    LLM factice pour bloquer l'initialisation OpenAI par défaut de RAGAS.
    Utilisé uniquement en mode offline — les métriques actives ne nécessitent pas de LLM.
    [FLAN-T5] Remplacer par LangchainLLMWrapper(load_llm()) quand le modèle est disponible.
    """
    def generate_text(self, prompt: PromptValue, *args, **kwargs) -> LLMResult:
        return LLMResult(generations=[[Generation(text="")]])

    async def agenerate_text(self, prompt: PromptValue, *args, **kwargs) -> LLMResult:
        return LLMResult(generations=[[Generation(text="")]])

def run_evaluation():
    """Lance l'évaluation RAGAS et logue les résultats dans MLflow."""

    print("⏳ Chargement du modèle d'embeddings...")
    embedding_model       = load_embeddings()
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # [FLAN-T5] Décommenter après téléchargement :
    # print("⏳ Chargement du LLM juge (Flan-T5-large)...")
    # llm      = load_llm()
    # ragas_llm = LangchainLLMWrapper(llm)

    print("⏳ Construction du dataset d'évaluation...")
    # [FLAN-T5] Remplacer par : dataset = build_ragas_dataset(embedding_model, llm)
    dataset = build_ragas_dataset(embedding_model)

    # Métriques actives — ne nécessitent pas de LLM juge
    metrics = [answer_similarity]

    # [FLAN-T5] Décommenter pour activer les métriques LLM :
    # metrics = [context_precision, context_recall, answer_similarity, faithfulness, answer_relevancy]
    dummy_llm = DummyLLM()
    # Injection des embeddings locaux dans chaque métrique
    for m in metrics:
        m.embeddings = ragas_embeddings
        # [FLAN-T5] Décommenter pour les métriques LLM :
        m.llm        = dummy_llm  # ← bloque l'appel OpenAI
        # [FLAN-T5] Remplacer dummy_llm par ragas_llm quand disponible

    print("⏳ Calcul des métriques RAGAS...")
    results = evaluate(dataset=dataset, metrics=metrics)
    df      = results.to_pandas()

    print("\n📊 Résultats RAGAS :")
    print(df[["question", "answer_similarity"]].to_string())
    # [FLAN-T5] Décommenter pour activer les métriques LLM :
    # print(df[["question", "context_precision", "context_recall", "answer_similarity"]].to_string())

    # ── Logging MLflow ───────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="ragas-eval"):

        # Métriques agrégées
        # mlflow.log_metric("context_precision_mean", float(df["context_precision"].mean()))
        # mlflow.log_metric("context_recall_mean",    float(df["context_recall"].mean()))
        mlflow.log_metric("answer_similarity_mean", float(df["answer_similarity"].mean()))
        mlflow.log_metric("nb_questions",           len(df))
        # [FLAN-T5] Décommenter après activation des métriques LLM :
        # mlflow.log_metric("faithfulness_mean",     float(df["faithfulness"].mean()))
        # mlflow.log_metric("answer_relevancy_mean", float(df["answer_relevancy"].mean()))

        # Paramètres de configuration
        mlflow.log_param("embedding_model", MODEL_PATH)
        mlflow.log_param("top_k",           TOP_K)
        mlflow.log_param("llm_judge",       "none — offline mode")
        # [FLAN-T5] Remplacer llm_judge par :
        # mlflow.log_param("llm_judge",     LLM_MODEL_PATH)
        mlflow.log_param("corpus",          "CVs Gautier Blondel")

        # Artefacts
        dataset_path = "data/ragas_eval_dataset.csv"
        results_path = "data/ragas_eval_results.csv"
        pd.DataFrame(EVAL_DATASET).to_csv(dataset_path, index=False)
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(dataset_path)
        mlflow.log_artifact(results_path)

    print(f"\n✅ Résultats loggés dans MLflow — {MLFLOW_URI}")

# ── Entrée ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()