# dags/rag_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.ingest import split_documents
from scripts.embed import embed_and_store

DATA_DIR = os.getenv("DATA_DIR", "./data/pdfs")

def extract_and_chunk(**context):
    start = time.time()
    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR) if f.endswith(".pdf")
    ]
    all_chunks = []
    for pdf in pdf_files:
        all_chunks.extend(split_documents(pdf))

    duration = round(time.time() - start, 2)
    # Passe les métriques à la tâche suivante via XCom
    context["ti"].xcom_push(key="nb_chunks", value=len(all_chunks))
    context["ti"].xcom_push(key="ingest_duration", value=duration)
    print(f"  -> {len(all_chunks)} chunks en {duration}s")
    return all_chunks

def vectorize_and_store(**context):
    chunks = context["ti"].xcom_pull(task_ids="extract_and_chunk")
    embed_and_store(chunks)

def log_to_mlflow(**context):
    nb_chunks = context["ti"].xcom_pull(
        task_ids="extract_and_chunk", key="nb_chunks"
    )
    duration = context["ti"].xcom_pull(
        task_ids="extract_and_chunk", key="ingest_duration"
    )
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(run_name="airflow_rag_run"):
        mlflow.log_param("embedding_model", os.getenv(
            "HF_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ))
        mlflow.log_param("chunk_size", 300)
        mlflow.log_param("chunk_overlap", 50)
        mlflow.log_metric("nb_chunks", nb_chunks)
        mlflow.log_metric("ingest_duration_sec", duration)
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