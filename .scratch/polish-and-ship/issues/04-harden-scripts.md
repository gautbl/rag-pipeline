## Question

What error handling improvements are needed in the Python scripts?

Based on audit (ticket 01), apply fixes to `scripts/`:
- `embed.py`: handle DuckDB write failures, model loading failures
- `query.py`: handle connection failures, empty results gracefully
- `ingest.py`: handle corrupt PDFs gracefully (skip, don't crash)
- `test_pipeline.py`: handle missing env vars
- `benchmark_query.py`: handle empty DB

Type: task
Blocked by: 01 (01-code-audit)
Status: resolved