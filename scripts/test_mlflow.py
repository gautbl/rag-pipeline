# scripts/test_mlflow.py
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")

with mlflow.start_run(run_name="rag_test_run"):
    mlflow.log_param("chunk_size", 512)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_metric("nb_chunks", 42)
    mlflow.log_metric("nb_documents", 3)
    print("Run logué dans MLflow.")