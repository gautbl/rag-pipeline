# RAG Pipeline — Domain model

## Core concepts

- **chunk**: A fixed-size text segment (300 chars with 50 overlap) extracted from a PDF page via RecursiveCharacterTextSplitter
- **embedding**: 384-dim vector from paraphrase-multilingual-MiniLM-L12-v2, stored as FLOAT in DuckDB+VSS
- **HNSW index**: Hierarchical Navigable Small World index in DuckDB+VSS for approximate nearest-neighbor search
- **rag.duckdb**: The main vector database file at `data/rag.duckdb`
- **dev.duckdb**: dbt's separate DuckDB database for transformation examples

## Evaluation

- **answer_similarity**: RAGAS metric comparing generated answer to ground truth via embedding similarity (score: 0.41 with Flan-T5)
- **context_precision / context_recall**: RAGAS metrics requiring structured JSON generation — blocked by Flan-T5 CPU speed

## Infrastructure

- **6 venvs**: Isolation strategy — each tool chain gets its own virtual environment to avoid dependency conflicts (pydantic v1/v2, sqlalchemy, protobuf)
- **Airflow DAG**: `dags/rag_pipeline_dag.py` — WSL2-only, currently has broken imports

## MCP (archived)

Model Context Protocol integration was explored but archived. FastAPI `/query` endpoint was envisioned as an MCP Server resource. See README.md "Archived vision" section.

## Acronyms

| Term | Expansion |
|------|-----------|
| VSS | Vector Similarity Search (DuckDB extension) |
| MCP | Model Context Protocol |
| RAG | Retrieval-Augmented Generation |
| RAGAS | RAG Assessment (evaluation framework) |
| HNSW | Hierarchical Navigable Small World |
| dbt | data build tool |