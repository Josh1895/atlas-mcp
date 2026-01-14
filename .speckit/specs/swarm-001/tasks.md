# Tasks: SWARM-001

**Plan:** .speckit/specs/swarm-001/plan.md
**Created:** 2025-01-13

## Task Format
`[ID] [P?] [Story] Description`
- P = Parallelizable (no file conflicts)
- Story = User story reference

---

## Phase 1: Setup

Project initialization and basic structure - DB tables, feature flags, package structure.

- [T-101] [-] [-] Copy Agent-MCP Python source into unified codebase
  - Files: agent_mcp/
  - Depends on: (none)
  - Acceptance: agent_mcp/ directory exists with all Python modules; No import errors when loading agent_mcp package

- [T-102] [-] [US-007] Add swarm feature flags to agent_mcp/core/config.py
  - Files: agent_mcp/core/config.py
  - Depends on: T-101
  - Acceptance: SWARM_ENABLED, SWARM_PATCH_MODE_ENABLED, SWARM_ANSWER_MODE_ENABLED, SWARM_ENABLE_WEB, SWARM_ENABLE_CONTEXT7, SWARM_ENABLE_TEST_VERIFICATION flags exist; All flags default to False

- [T-103] [P] [US-001] Add swarm_runs table to agent_mcp/db/schema.py
  - Files: agent_mcp/db/schema.py
  - Depends on: T-101
  - Acceptance: swarm_runs table created with all columns from spec; CREATE IF NOT EXISTS

- [T-104] [P] [US-001] Add swarm_agents table to agent_mcp/db/schema.py
  - Files: agent_mcp/db/schema.py
  - Depends on: T-101
  - Acceptance: swarm_agents table with composite PK (run_id, swarm_agent_id)

- [T-105] [P] [US-001] Add swarm_outputs table to agent_mcp/db/schema.py
  - Files: agent_mcp/db/schema.py
  - Depends on: T-101
  - Acceptance: swarm_outputs table with composite PK

- [T-106] [P] [US-005] Add tool_cache table to agent_mcp/db/schema.py
  - Files: agent_mcp/db/schema.py
  - Depends on: T-101
  - Acceptance: tool_cache table with cache_key PK, expires_at, hit_count

- [T-107] [P] [US-005] Add tool_calls table to agent_mcp/db/schema.py
  - Files: agent_mcp/db/schema.py
  - Depends on: T-101
  - Acceptance: tool_calls table with indexes on (run_id), (tool_name, created_at)

- [T-108] [-] [-] Create agent_mcp/features/swarm/ package structure
  - Files: agent_mcp/features/swarm/__init__.py, agent_mcp/features/swarm/schemas.py
  - Depends on: T-101
  - Acceptance: __init__.py exports main classes; schemas.py defines SwarmRequest, SwarmResult, SwarmMode

---

## Phase 2: Foundation

Core infrastructure that MUST be complete before ANY user story implementation.

- [T-201] [-] [US-001] Implement persistence.py with DB access functions for swarm tables
  - Files: agent_mcp/features/swarm/persistence.py
  - Depends on: T-103, T-104, T-105, T-106, T-107
  - Acceptance: save_swarm_run(), save_swarm_agent(), save_swarm_output(), get_swarm_run(), update_swarm_run_status() functions; All use execute_db_write

- [T-202] [P] [US-006] Implement budget_manager.py for resource limit enforcement
  - Files: agent_mcp/features/swarm/budget_manager.py
  - Depends on: T-108
  - Acceptance: BudgetManager tracks cost, tokens, tool_calls, elapsed_time; check_budget(), record_cost(), is_exceeded, get_remaining()

- [T-203] [P] [US-001] Implement repo_resolver.py for local/remote repo resolution
  - Files: agent_mcp/features/swarm/repo_resolver.py
  - Depends on: T-108
  - Acceptance: RepoResolver.resolve() returns resolved path; Local mode uses MCP_PROJECT_DIR; Remote mode clones to temp; Worktree support

- [T-204] [-] [US-005] Implement tool_router.py with caching and rate limiting
  - Files: agent_mcp/features/swarm/tool_router.py
  - Depends on: T-106, T-107, T-201
  - Acceptance: Permission check, cache lookup/write, rate limiter, circuit breaker, tool_calls logging

- [T-205] [P] [US-002] Implement context_builder.py for building swarm context from task/memory
  - Files: agent_mcp/features/swarm/context_builder.py
  - Depends on: T-108
  - Acceptance: ContextBuilder.build() assembles task + parent chain + project_context + RAG hits; Dedup; Token budget

- [T-206] [P] [US-001] Implement memory_writer.py for writing results back to Agent-MCP
  - Files: agent_mcp/features/swarm/memory_writer.py
  - Depends on: T-108, T-201
  - Acceptance: write_to_task(), write_to_project_context(), index_to_rag(); Respects memory policy flags

---

## Phase 3: User Stories

Individual feature implementations - the core swarm functionality.

- [T-301] [-] [US-001] Implement atlas_adapter.py wrapping Atlas AgentPoolManager
  - Files: agent_mcp/features/swarm/atlas_adapter.py
  - Depends on: T-108
  - Acceptance: AtlasAdapter imports Atlas classes; generate_candidates_patch_mode(), generate_candidates_answer_mode(); Agent failure handling; Cost/token tracking

- [T-302] [-] [US-001] Implement consensus_engine.py for clustering and voting
  - Files: agent_mcp/features/swarm/consensus_engine.py
  - Depends on: T-301
  - Acceptance: ConsensusEngine.cluster_and_vote(); Uses SimilarityClustering for patch, embedding clustering for answer; First-to-ahead-by-K; Tie-breakers

- [T-303] [P] [US-003] Implement answer_clustering.py for answer mode embedding clustering
  - Files: agent_mcp/features/swarm/answer_clustering.py
  - Depends on: T-108
  - Acceptance: Embeddings via Agent-MCP model; K-means with cosine; Medoid selection; Citation detection

- [T-304] [-] [US-001] Implement swarm_manager.py orchestrating full swarm workflow
  - Files: agent_mcp/features/swarm/swarm_manager.py
  - Depends on: T-201, T-202, T-203, T-204, T-205, T-206, T-301, T-302
  - Acceptance: SwarmManager.run() orchestrates full flow; Timeout handling; Budget handling; Returns SwarmResult

- [T-305] [-] [US-001] Implement swarm_tools.py with run_swarm_consensus MCP tool
  - Files: agent_mcp/tools/swarm_tools.py
  - Depends on: T-102, T-304
  - Acceptance: Tool registered; Input schema matches spec; Auth via verify_token; Feature flag check; Calls SwarmManager.run()

- [T-306] [-] [US-001] Register swarm_tools in agent_mcp/tools/__init__.py
  - Files: agent_mcp/tools/__init__.py
  - Depends on: T-305
  - Acceptance: Import swarm_tools; Tool visible when SWARM_ENABLED=true; Hidden when false

---

## Phase 4: Integration

Cross-cutting concerns - test verification, external tools, circuit breakers.

- [T-401] [-] [US-004] Integrate Atlas patch verification (PatchApplier, TestRunner)
  - Files: agent_mcp/features/swarm/atlas_adapter.py
  - Depends on: T-301
  - Acceptance: validate_and_score_patch_candidates() applies patches; Runs tests; Returns test results; Uses QualityScorer

- [T-402] [-] [US-004] Implement behavioral clustering when tests available
  - Files: agent_mcp/features/swarm/consensus_engine.py
  - Depends on: T-302, T-401
  - Acceptance: cluster_by_test_outcomes(); Passing clusters preferred; Fallback to similarity

- [T-403] [P] [US-003] Integrate web search via ToolRouter
  - Files: agent_mcp/features/swarm/tool_router.py
  - Depends on: T-204
  - Acceptance: web_search() calls Atlas WebSearchClient; 7-day TTL cache; Domain lists; 2 RPS limit

- [T-404] [P] [US-003] Integrate Context7 via ToolRouter
  - Files: agent_mcp/features/swarm/tool_router.py
  - Depends on: T-204
  - Acceptance: context7() calls Atlas Context7Client; 7-day TTL cache; 1 RPS limit

- [T-405] [P] [US-002] Integrate Agent-MCP RAG via ToolRouter
  - Files: agent_mcp/features/swarm/tool_router.py
  - Depends on: T-204
  - Acceptance: agent_mcp_rag() queries local RAG; 10 min TTL; Query limits

- [T-406] [-] [US-005] Implement circuit breaker for tool providers
  - Files: agent_mcp/features/swarm/tool_router.py
  - Depends on: T-204
  - Acceptance: CircuitBreaker with open/closed/half-open; Opens after N failures; Cooldown; Fast-fail when open

---

## Phase 5: Polish

Final improvements - logging, metrics, testing, evaluation.

- [T-501] [P] [-] Add structured logging with run_id correlation
  - Files: agent_mcp/features/swarm/swarm_manager.py
  - Depends on: T-304
  - Acceptance: All logs include run_id; Structured format; Proper log levels

- [T-502] [P] [-] Implement metrics collection (latency, cost, consensus rate)
  - Files: agent_mcp/features/swarm/swarm_manager.py
  - Depends on: T-304
  - Acceptance: Metrics in metrics_json; duration_ms, cost_usd, tokens_total, tool_calls_count

- [T-503] [P] [-] Write unit tests for SwarmManager
  - Files: tests/features/swarm/test_swarm_manager.py
  - Depends on: T-304
  - Acceptance: Test happy path, timeout, budget, partial failure

- [T-504] [P] [-] Write unit tests for ConsensusEngine
  - Files: tests/features/swarm/test_consensus_engine.py
  - Depends on: T-302
  - Acceptance: Test consensus rule, tie-breakers, behavioral preference

- [T-505] [P] [-] Write unit tests for ToolRouter
  - Files: tests/features/swarm/test_tool_router.py
  - Depends on: T-204
  - Acceptance: Test cache, rate limiting, circuit breaker, permissions

- [T-506] [-] [US-001] Write integration test for end-to-end swarm run
  - Files: tests/integration/test_swarm_e2e.py
  - Depends on: T-305, T-503, T-504, T-505
  - Acceptance: Test run_swarm_consensus; Verify DB records; Verify task notes; Mock LLM

- [T-507] [-] [-] Implement offline replay harness for golden tests
  - Files: tests/golden/test_consensus_determinism.py
  - Depends on: T-302
  - Acceptance: Load stored outputs; Re-run clustering/voting; Assert deterministic; 3+ scenarios

- [T-508] [-] [US-007] Test backward compatibility with SWARM_ENABLED=false
  - Files: tests/integration/test_backward_compat.py
  - Depends on: T-306
  - Acceptance: Existing tests pass; run_swarm_consensus returns error; No swarm_runs records

---

## Summary

| Phase | Tasks | Parallelizable |
|-------|-------|----------------|
| 1. Setup | 8 | 5 |
| 2. Foundation | 6 | 4 |
| 3. User Stories | 6 | 1 |
| 4. Integration | 6 | 3 |
| 5. Polish | 8 | 5 |
| **Total** | **34** | **18** |
