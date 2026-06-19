# Structure du projet RAG Pipeline

```
D:\projets\rag-pipeline\
│
├── .venv\                          # Venv principal (LangChain, Chroma, HuggingFace)
├── .venv-mlflow\                   # Venv MLflow
├── .venv-dbt\                      # Venv dbt + DuckDB
├── .venv-airflow\                  # Venv Airflow (installation tentée, bloquée sur Windows natif)
│
├── dags\                           # DAGs Airflow
│   └── rag_pipeline_dag.py         # DAG principal : extract → vectorize → mlflow log
│
├── scripts\                        # Scripts utilitaires Python
│   ├── seed_duckdb.py              # Génère les données fictives dans DuckDB (chunks_raw)
│   └── test_mlflow.py              # Test de logging MLflow
│
├── mlflow_data\                    # Stockage MLflow
│   └── mlflow.db                   # Base SQLite MLflow (tracking des runs)
│
├── dbt_project\                    # Projet dbt
│   ├── dev.duckdb                  # Base DuckDB dev (chunks_raw + chunks_clean)
│   ├── prod.duckdb                 # Base DuckDB prod
│   └── mon_rag_dbt\                # Projet dbt généré par dbt init
│       ├── dbt_project.yml         # Config du projet dbt (nom, profil, chemins)
│       ├── models\
│       │   ├── chunks_clean.sql    # Modèle dbt : nettoyage des chunks bruts
│       │   ├── sources.yml         # Déclaration de la source raw.chunks_raw
│       │   └── example\            # Modèles d'exemple générés par dbt init
│       │       ├── my_first_dbt_model.sql
│       │       └── my_second_dbt_model.sql
│       ├── analyses\
│       ├── tests\
│       ├── seeds\
│       ├── macros\
│       ├── snapshots\
│       └── target\                 # Artefacts générés par dbt run (ignoré par git)
│
├── airflow\                        # Répertoire AIRFLOW_HOME
│   ├── airflow.cfg                 # Configuration Airflow
│   ├── airflow.db                  # Base SQLite Airflow
│   └── webserver_config.py        # Config du webserver FAB
│
├── constraints-3.11.txt            # Contraintes pip pour Airflow
├── docker-compose.yaml             # docker-compose Airflow (téléchargé, non utilisé)
├── .env                            # Variables d'environnement (AIRFLOW_UID, FERNET_KEY)
├── start_mlflow.ps1                # Script de démarrage MLflow
└── start_airflow.ps1               # Script de démarrage Airflow (WSL2 requis)
```

---

## Environnements virtuels

| Venv | Outils installés | Activation |
|---|---|---|
| `.venv` | LangChain, Chroma, HuggingFace | `.\.venv\Scripts\Activate.ps1` |
| `.venv-mlflow` | MLflow | `.\.venv-mlflow\Scripts\Activate.ps1` |
| `.venv-dbt` | dbt-duckdb, duckdb | `.\.venv-dbt\Scripts\Activate.ps1` |
| `.venv-airflow` | apache-airflow | `.\.venv-airflow\Scripts\Activate.ps1` |

---

## URLs locales

| Service | URL | Démarrage |
|---|---|---|
| MLflow UI | `http://localhost:5000` | `.\start_mlflow.ps1` |
| dbt docs | `http://localhost:8081` | `dbt docs serve --port 8081` |
| Airflow UI | `http://localhost:8080` | WSL2 requis |

---

## État d'avancement

| Outil | Statut |
|---|---|
| MLflow | ✅ Fonctionnel |
| dbt + DuckDB | ✅ Fonctionnel |
| Airflow | ⚠️ Bloqué — nécessite WSL2 ou Docker |
```