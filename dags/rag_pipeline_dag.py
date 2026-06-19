# dags/rag_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow

def extract_and_chunk():
    chunks = [
        "Le pipeline RAG permet une recherche sémantique efficace.",
        "LangChain facilite l'intégration des LLMs.",
        "Chroma est une base vectorielle légère et locale.",
    ]
    print(f"{len(chunks)} chunks extraits.")
    return chunks

def vectorize_and_store():
    print("Vectorisation et stockage Chroma terminés.")

def log_to_mlflow():
    # localhost fonctionne directement, plus besoin de host.docker.internal
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name="airflow_rag_run"):
        mlflow.log_param("chunk_size", 512)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_metric("nb_chunks", 3)
        print("Métriques loguées dans MLflow.")

with DAG(
    dag_id="rag_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rag", "genai"],
) as dag:

    t1 = PythonOperator(task_id="extract_and_chunk",   python_callable=extract_and_chunk)
    t2 = PythonOperator(task_id="vectorize_and_store", python_callable=vectorize_and_store)
    t3 = PythonOperator(task_id="log_to_mlflow",       python_callable=log_to_mlflow)

    t1 >> t2 >> t3