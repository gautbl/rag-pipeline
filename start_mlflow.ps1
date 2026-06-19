cd D:\projets\rag-pipeline
.\.venv-mlflow\Scripts\Activate.ps1

$existing = netstat -ano | findstr ":5000" | ForEach-Object {
    ($_ -split '\s+')[-1]
} | Sort-Object -Unique

if ($existing) {
    $existing | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Write-Host "Instances précédentes nettoyées."
}

Start-Process -FilePath ".\.venv-mlflow\Scripts\mlflow.exe" `
  -ArgumentList "ui --backend-store-uri sqlite:///D:/projets/rag-pipeline/mlflow_data/mlflow.db --port 5000" `
  -WindowStyle Hidden

Write-Host "MLflow UI démarré sur http://localhost:5000"