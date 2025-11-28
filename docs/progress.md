---
description: "Current project state - updated after fixes"
alwaysApply: true
---

# Current Progress

## Recent Fixes (keep max 5)

| Date | Issue | Fix | File |
|------|-------|-----|------|
| 2025-11-28 | GPT-5.1 Responses API | Fixed reasoning param (effort: low default), removed unsupported temperature | `api/core/providers/cloud/llm.py` |
| 2025-11-28 | Duplicate search results | Added query dedup + content/page dedup in search tools | `api/services/tools/search_tools.py`, `api/services/environment.py` |
| 2025-11-28 | Agent query rewriting | DSPy DecisionSignature now tracks previous_queries, enforces NEW queries | `api/prompts/signatures/decision.py`, `api/prompts/chain_of_thought.py` |
| 2025-11-28 | Benchmark capture | Captured model_answer/sources/tools with dedup + typing | `api/services/benchmark.py`, `api/services/tools/base.py` |
| 2025-11-28 | Agentic UI | Clickable sources, visual thumbnails/scores | `frontend/app/page.tsx`, `frontend/lib/hooks/useAgenticSearch.ts` |

## Known Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **Benchmark validation pending** | `logs/benchmarks/` | Need fresh run after capture fixes |
| **Weaviate/protobuf drift** | `weaviate-client` deps | Keep protobuf pinned <6 to avoid grpc issues |

## Active Migration

**Cloud Migration Phase 8 Complete (needs validation run)** - See `TODO.md`
- ✅ Phase 5: Cloud Implementation (done)
- ✅ Phase 6: Tool & Agent Refactor (done)
- ✅ Phase 7: Cloud Ingestion (complete - 260 text chunks, 260 visual pages)
- ✅ Phase 8: Benchmarking System (capture fixes + dashboard shipped)
  - Backend: Judge + capture working; rerun benchmark to confirm metrics
  - Frontend: Dashboard + agentic UX updated

## Last Benchmark

```
Date: 2025-11-28 (pre-fix)
Status: Capture issues produced 0.0 scores; rerun needed after fixes
Evidence: logs/benchmarks/20251128-085417-cloud.json shows empty model_answer/sources
```

---

**Note to Coding Agent:** After successful fix, update this file:
1. Add new fix to table (remove oldest if > 5)
2. Update "Last Benchmark" if you ran one
3. Add to "Known Issues" if something is still broken
