cd D:\projets\rag-pipeline
$env:AIRFLOW_HOME = "D:\projets\rag-pipeline\airflow"

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd D:\projets\rag-pipeline; .\.venv-airflow\Scripts\Activate.ps1; `$env:AIRFLOW_HOME='D:\projets\rag-pipeline\airflow'; airflow webserver --port 8080"

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd D:\projets\rag-pipeline; .\.venv-airflow\Scripts\Activate.ps1; `$env:AIRFLOW_HOME='D:\projets\rag-pipeline\airflow'; airflow scheduler"

Write-Host "Airflow démarré sur http://localhost:8080"