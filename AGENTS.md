# RAG Pipeline — Agent orientation

## Project overview

A semantic search RAG pipeline on PDF documents — LangChain, HuggingFace, DuckDB+VSS, MLflow, dbt, Airflow. Context: social housing provider with large volumes of unstructured internal documents (regulatory notices, lease contracts, technical specifications).

This is a **portfolio showcase** — clarity and correctness matter more than new features.

## How agents should orient themselves

- **Start here.** Read this file first.
- **Source code is the source of truth** for implementation details.
- **Domain glossary** at `docs/agents/domain.md`.
- **Wayfinding map** at `.scratch/<effort>/map.md` — check this when given a task, before starting work.

## Known constraints

- **Python 3.11** required throughout
- **6 venvs** — see README.md for which tools run in which venv:
  - `.venv` — LangChain, HuggingFace, DuckDB
  - `.venv-api` — FastAPI, Uvicorn
  - `.venv-mlflow` — MLflow
  - `.venv-dbt` — dbt-duckdb
  - `.venv-airflow` — Airflow (WSL2 only)
  - `.venv-ragas` — RAGAS, datasets
- **Windows-first** — PowerShell activation, paths use backslashes in README
- **WSL2 required for Airflow** — the DAG (`dags/rag_pipeline_dag.py`) is WSL2-only
- **DuckDB+VSS** is the vector store (not Chroma — old Chroma code is commented out in places)

## Issue tracker

See `docs/agents/issue-tracker.md` for local Markdown issue conventions.
See `docs/agents/triage-labels.md` for triage label conventions.