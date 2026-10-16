# polish-and-ship — COMPLETE

All 7 tickets resolved. No active frontier. This effort is retained for reference only; a new effort should be created for future work.

## Destination (archived)

Polished clone-and-run portfolio showcase. Tests pass, code hardened, documentation accurate.

## Decisions (archived)

| Ticket | Decision |
|--------|----------|
| 01 | DAG is fundamentally broken (function signature mismatch, not just imports). Airflow gap documented in README. |
| 02 | API error handling hardened with try/except on DB, model, and auth paths. |
| 03 | 38 API tests added (auth, unicode, long queries, concurrency, edge cases). All 51 tests pass. |
| 04 | All 5 scripts hardened (embed, query, ingest, test_pipeline, benchmark_query). |
| 05 | `evaluate.py.legacy` archived. `evaluate_local.py` promoted to canonical `evaluate.py` (offline-first). |
| 06 | README: Quick Start, Known Gaps, archived MCP, updated Roadmap, Lacunes connues, CI/CD section. |
| 07 | AGENTS.md with full project orientation, domain model, constraints, wayfinding path. |

## Out of scope (unchanged)

- CI/CD pipeline and Kubernetes deployment
- Rate limiting
- Structured logging / Prometheus metrics
- MCP integration
- Any new feature or component not already in the codebase