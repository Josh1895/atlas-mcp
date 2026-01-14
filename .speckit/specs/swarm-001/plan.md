# Implementation Plan: SWARM-001

**Specification:** .speckit/specs/swarm-001/spec.md
**Created:** 2025-01-13

## Summary

Integrate Atlas swarm capabilities into Agent-MCP by: (1) adding new DB tables for swarm persistence, (2) creating a swarm_tools.py module with run_swarm_consensus tool, (3) implementing SwarmManager that wraps Atlas components, (4) adding ToolRouter for managed tool access, (5) implementing BudgetManager for cost/time limits, and (6) building consensus engine that orchestrates clustering and voting.

---

## Technical Context

| Aspect | Value |
|--------|-------|
| Language | Python |
| Version | 3.11+ |
| Storage | SQLite (Agent-MCP's existing mcp_state.db) |
| Testing Framework | pytest with pytest-asyncio |
| Target Platforms | Linux, macOS, Windows (WSL) |

### Dependencies
- starlette (existing Agent-MCP server)
- mcp (MCP protocol library)
- sqlite3 (persistence)
- sqlite-vec (vector embeddings)
- httpx (async HTTP for tool calls)
- tiktoken (token counting)
- google-generativeai (Gemini API for swarm agents)
- numpy (clustering calculations)
- scikit-learn (k-means for answer mode clustering)

### Performance Targets
- **p50_latency**: 90 seconds per swarm run
- **p95_latency**: 900 seconds (configurable timeout)
- **max_concurrent_agents**: 10 per swarm run
- **cache_hit_ratio**: >50% for repeated tool queries

### Constraints
- Must use Agent-MCP's DatabaseWriteQueue for all DB writes
- Cannot modify existing tool schemas
- Must work with SWARM_ENABLED=false (no-op mode)

---

## Architecture Decisions

1. **AD-001**: Agent-MCP remains the only server; Atlas is embedded as a library, not a separate FastMCP server. This avoids port conflicts and simplifies deployment.

2. **AD-002**: SwarmManager is the single orchestration class that coordinates AtlasAdapter, ConsensusEngine, ToolRouter, BudgetManager, and Persistence. Clean separation of concerns.

3. **AD-003**: AtlasAdapter wraps Atlas classes (AgentPoolManager, VotingManager, SimilarityClustering) to isolate Agent-MCP from Atlas internals. Enables future Atlas upgrades without breaking integration.

4. **AD-004**: ToolRouter implements policy enforcement, caching, rate limiting, and circuit breakers for all external tool calls. Single point of control.

5. **AD-005**: RepoResolver supports both local (MCP_PROJECT_DIR) and remote (git clone) modes. Local mode is default for Agent-MCP compatibility.

6. **AD-006**: Answer mode uses Agent-MCP's existing embedding infrastructure (via openai_service.py) for clustering, avoiding duplicate embedding providers.

7. **AD-007**: All swarm DB operations go through execute_db_write to maintain serialized write semantics and avoid SQLite concurrency issues.

8. **AD-008**: Structured logging uses run_id as correlation ID across all swarm operations for traceability.

---

## Project Structure

### Source Files
```
agent_mcp/features/swarm/__init__.py
agent_mcp/features/swarm/swarm_manager.py
agent_mcp/features/swarm/atlas_adapter.py
agent_mcp/features/swarm/consensus_engine.py
agent_mcp/features/swarm/tool_router.py
agent_mcp/features/swarm/repo_resolver.py
agent_mcp/features/swarm/budget_manager.py
agent_mcp/features/swarm/persistence.py
agent_mcp/features/swarm/schemas.py
agent_mcp/features/swarm/answer_clustering.py
agent_mcp/features/swarm/context_builder.py
agent_mcp/features/swarm/memory_writer.py
agent_mcp/tools/swarm_tools.py
agent_mcp/db/schema.py (modify - add new tables)
agent_mcp/core/config.py (modify - add feature flags)
agent_mcp/tools/__init__.py (modify - import swarm_tools)
```

### Test Files
```
tests/features/swarm/test_swarm_manager.py
tests/features/swarm/test_atlas_adapter.py
tests/features/swarm/test_consensus_engine.py
tests/features/swarm/test_tool_router.py
tests/features/swarm/test_budget_manager.py
tests/features/swarm/test_persistence.py
tests/features/swarm/test_answer_clustering.py
tests/tools/test_swarm_tools.py
tests/integration/test_swarm_e2e.py
tests/golden/test_consensus_determinism.py
```

---

## API Contracts

### run_swarm_consensus Input

```json
{
  "type": "object",
  "properties": {
    "token": {"type": "string"},
    "task_id": {"type": "string"},
    "description": {"type": "string"},
    "mode": {"type": "string", "enum": ["patch", "answer"]},
    "repo": {"type": "object"},
    "swarm": {"type": "object"},
    "budgets": {"type": "object"},
    "tools": {"type": "object"},
    "memory": {"type": "object"}
  },
  "required": ["token"]
}
```

### SwarmResult Output

```json
{
  "type": "object",
  "properties": {
    "run_id": {"type": "string"},
    "task_id": {"type": "string"},
    "mode": {"type": "string"},
    "consensus_reached": {"type": "boolean"},
    "selected_output": {"type": "string"},
    "confidence_score": {"type": "number"},
    "vote_counts": {"type": "object"},
    "clusters": {"type": "array"},
    "metrics": {"type": "object"},
    "errors": {"type": "array"},
    "warnings": {"type": "array"}
  },
  "required": ["run_id", "mode", "consensus_reached", "selected_output"]
}
```

---

## Complexity Notes

- **Concurrency**: SwarmManager must handle parallel agent execution with proper cancellation on timeout/budget
- **State Management**: Must track in-flight swarm runs for graceful shutdown
- **Error Handling**: Partial failures (some agents fail) should not fail entire run
- **Caching**: Tool cache must handle concurrent reads/writes from multiple agents
- **Testing**: Golden tests require deterministic replay, need to mock LLM responses

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_swarm_consensus Tool                     │
│                    (agent_mcp/tools/swarm_tools.py)              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SwarmManager                              │
│                 (swarm/swarm_manager.py)                         │
│  Orchestrates: resolve → context → generate → cluster → vote    │
└──────┬──────────────┬──────────────┬──────────────┬─────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ RepoResolver │ │ContextBuilder│ │ AtlasAdapter │ │ Consensus    │
│              │ │              │ │              │ │ Engine       │
└──────────────┘ └──────────────┘ └──────┬───────┘ └──────────────┘
                                         │
                      ┌──────────────────┼──────────────────┐
                      ▼                  ▼                  ▼
               ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
               │AgentPool     │  │ Voting       │  │ Similarity   │
               │Manager       │  │ Manager      │  │ Clustering   │
               │(Atlas)       │  │(Atlas)       │  │(Atlas)       │
               └──────────────┘  └──────────────┘  └──────────────┘
                                         │
                      ┌──────────────────┼──────────────────┐
                      ▼                  ▼                  ▼
               ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
               │ ToolRouter   │  │BudgetManager │  │ Persistence  │
               │ (cache,rate) │  │(cost,time)   │  │ (DB writes)  │
               └──────────────┘  └──────────────┘  └──────────────┘
```
