# start_mlflow.ps1 — à placer à la racine du projet
cd D:\projets\rag-pipeline
.\.venv\Scripts\Activate.ps1
Stop-Process -FilePath ".\.venv\Scripts\mlflow.exe" `
  -ArgumentList "ui --backend-store-uri sqlite:///mlflow_data/mlflow.db --port 5000" `
  -WindowStyle Hidden
Write-Host "MLflow UI arrêté sur http://localhost:5000"