# Atlas + Agent-MCP Integration PRD

## 1. Executive Summary

### What we're building

A unified orchestration system that **keeps Agent-MCP as the primary "conductor"** (task decomposition, dispatch, persistent memory, and result composition), while adding an **optional Atlas-style Swarm Execution Mode** that can be invoked **at any step** (root task or any subtask).

Concretely, we will:

* **Embed Atlas Swarm capabilities into the Agent-MCP runtime** as a first-class subsystem (not a separate product).
* Add a **single entrypoint** where MCP Agent can call a swarm solver and receive a **consensus-selected output** (with traceability, metrics, and persistence).
* Ensure both systems share:
  * **Persistent memory** (Agent-MCP SQLite + vector store)
  * **Task graph** (Agent-MCP tasks + dependencies + parent-child)
  * **Tooling** (repo search, MCP context, DB/RAG search, web research, Context7)
  * **Production-grade controls** (budgets, timeouts, retries, circuit breakers, audit logs)
  * **Observability & evaluation** (structured events, traces, replay harness, golden tests)

---

### Why we're building it

* **Agent-MCP** is strong at:
  * Persistent project memory (SQLite + RAG)
  * Task tracking and coordination (task graph, agent assignment)
  * Operating as a tool hub for multiple agents

* **Atlas MCP** is strong at:
  * **Parallel redundancy** (swarm agents solve the *same* problem)
  * **Robust selection** (clustering + voting consensus + verification)
  * Tool-using "agentic" micro-agents (Context7 + web + repo search)

The unified system produces higher correctness and robustness without losing Agent-MCP's multi-task orchestration and memory persistence.

---

### Success metrics (measurable)

#### Functional success (MVP acceptance criteria)

1. **Swarm invocation works end-to-end**
   * Given an Agent-MCP `task_id`, invoking `run_swarm_consensus` returns a `SwarmResult` containing:
     * `selected_output` (patch or answer)
     * `consensus_reached` boolean
     * `vote_counts` / cluster summary
     * `duration_ms`, `total_cost_usd`, `tool_calls_count`
   * Must complete within configured `timeout_seconds`.

2. **Persistence**
   * Every swarm run creates a durable record in SQLite:
     * `swarm_runs` row + `swarm_agents` rows + `swarm_outputs` rows
   * Restarting the server does not lose swarm history.

3. **Optionality**
   * Existing Agent-MCP flows work **unchanged** when swarm mode is disabled.
   * Feature-flagged rollout: `SWARM_ENABLED=false` yields old behavior.

4. **Task-graph integration**
   * Swarm can be invoked on:
     * (a) a "root task" created via `assign_task` (admin)
     * (b) a "child task" created via `create_self_task` (worker)
   * Swarm results can be automatically attached to:
     * task notes / task metadata
     * project context entries
     * RAG chunks (optional)

5. **Tool gating**
   * Swarm agents can only call tools explicitly enabled by config:
     * web search, Context7, repo search/open, DB/RAG retrieval
   * Any attempt to call a disabled tool is denied and logged.

#### Reliability / performance success (SLOs)

* **Latency SLO (configurable per task)**
  * P50 swarm run ≤ 90s
  * P95 swarm run ≤ configured `timeout_seconds` (default 900s)
* **Consensus rate**
  * For a representative eval set, `consensus_reached=true` ≥ 70% (baseline target; adjust after harness)
* **Cost control**
  * ≥ 99% of runs remain under configured `max_cost_usd` (hard stop with graceful best-effort result)

#### Quality success

* For "patch" mode tasks:
  * If tests are configured, the selected patch must be from a **passing** behavioral cluster whenever any passing cluster exists.
* For "answer" mode tasks:
  * Selected answer must include citations when web research is enabled; otherwise mark `citations_present=false`.

---

## 2. Glossary

* **Task**: A unit of work tracked in Agent-MCP (`tasks` table / `g.tasks` cache), created via tools like `assign_task` or `create_self_task`.
* **Subtask**: A task with `parent_task_id` set (Agent-MCP supports parent-child).
* **Agent (Agent-MCP)**: A managed worker session (often tmux-backed) created via admin tools, with an auth token and working directory.
* **Swarm agent (Atlas)**: A stateless micro-agent (e.g., `MicroAgent` or `AgenticMicroAgent`) instantiated for a single swarm run, configured with a specific `PromptStyle`.
* **Swarm**: A set of N swarm agents solving the *same* input in parallel, producing candidate outputs.
* **Consensus**: A selection decision from clustering + voting (Atlas `VotingManager` / `IncrementalVoter`) producing a winning cluster/solution when threshold is met.
* **Clustering**:
  * Patch mode: similarity clustering over diffs (`SimilarityClustering`)
  * Behavioral mode: cluster by test outcomes (`cluster_by_test_outcomes`)
  * Answer mode (new): embedding-based clustering (k-means / hierarchical; specified below)
* **Tool-call**: A structured call by an agent to an external capability (web search, Context7 docs, repo search, DB/RAG query). Must be permissioned, metered, cached, and logged.
* **Memory**:
  * **Short-term**: in-run state, cached context, active task info
  * **Task memory**: tasks, notes, agent actions, swarm run artifacts
  * **Long-term**: project context (`project_context`) + RAG index (`rag_chunks`, `rag_embeddings`)
  * **Episodic**: event-like history (audit log, agent actions, swarm trace)
* **Budget**: Limits for time, cost, tokens, tool calls, and concurrency applied per task and per swarm run.
* **Circuit breaker**: Protective mechanism that disables a failing tool/provider temporarily to avoid cascading failures.

---

## 3. Current State Analysis

### 3.1 Agent-MCP: architecture summary

#### Runtime shape

* **Server**: Starlette app (`agent_mcp/app/main_app.py`) exposing:
  * MCP endpoints (SSE + messages)
  * Dashboard/static UI endpoints (`agent_mcp/app/routes.py`)
* **Lifecycle**: `agent_mcp/app/server_lifecycle.py` initializes:
  * `.agent/` directory
  * SQLite database (`mcp_state.db`)
  * background tasks:
    * RAG indexing worker
    * Claude session monitor
* **Tool system**:
  * Tool registration via `agent_mcp/tools/registry.py`
  * Tools imported in `agent_mcp/tools/__init__.py`
  * Tool implementations are async and typically return `List[mcp.types.TextContent]`

#### Key functional subsystems

1. **Auth / identities**
   * Admin token + agent tokens
   * Functions: `verify_token`, `get_agent_id` (`agent_mcp/core/auth.py`)

2. **Task system**
   * Tools: `assign_task`, `create_self_task`, `update_task_status`, `view_tasks`, `search_tasks` (`agent_mcp/tools/task_tools.py`)
   * Supports:
     * parent-child tasks
     * dependency lists
     * assignment to agents
     * notes and coordination metadata (stored in DB)

3. **Persistent memory**
   * `project_context` table accessed by tools in `agent_mcp/tools/project_context_tools.py`
   * RAG system:
     * SQLite + sqlite-vec vectors
     * tables: `rag_chunks`, `rag_embeddings`, etc. (`agent_mcp/db/schema.py`)
     * querying: `agent_mcp/features/rag/query.py`

4. **Agent runtime**
   * Agents can be created and run as tmux sessions (see `agent_mcp/tools/admin_tools.py`, `agent_mcp/tools/agent_tools.py`, `agent_mcp/tools/task_tools.py`)
   * File status locking is in-memory (`g.file_map`) via tools in `agent_mcp/tools/file_management_tools.py`
   * Optional Git worktrees for isolation (`agent_mcp/features/worktree_integration.py`)

5. **DB concurrency**
   * Serialized writes through `DatabaseWriteQueue` (`agent_mcp/db/write_queue.py`)

#### What Agent-MCP *does not* currently provide natively

* No built-in **swarm consensus** engine
* No unified **web research** tool layer (outside of local RAG)
* No standardized **structured swarm result artifacts** and selection trace
* No single "call swarm here" interface

---

### 3.2 Atlas MCP: architecture summary

#### Runtime shape

* **Server**: FastMCP-based server (`atlas/server.py`) exposing MCP tools like `solve_issue`.
* **Core orchestrator**: `atlas/core/orchestrator.py`
  * Clones repo → analyzes context → runs swarm generation → validates patches → clusters → votes → returns result

#### Swarm engine

* **Agent pool**: `atlas/agents/agent_pool.py`
  * `AgentPoolManager.create_swarm(num_agents, use_agentic)`
  * uses diverse prompt styles (`atlas/agents/prompt_styles.py`)
* **Agents**:
  * `MicroAgent`: generates patch from issue+context
  * `AgenticMicroAgent`: uses tool-calling via `AgenticGeminiClient`
* **Diversity**:
  * Prompt styles: `SENIOR_ENGINEER`, `SECURITY_FOCUSED`, `PERFORMANCE_EXPERT`, `SYSTEMS_ARCHITECT`, `CODE_REVIEWER` (`ALL_STYLES`)

#### Consensus logic

* **Clustering**:
  * Patch similarity clustering: `atlas/verification/clustering.py` (`SimilarityClustering`)
  * Behavioral clustering by test outcome: `atlas/verification/behavioral_clustering.py`
* **Voting**:
  * `VotingManager(k)` + `IncrementalVoter` (`atlas/voting/consensus.py`)
  * "First-to-ahead-by-K" consensus; early stop when margin ≥ k

#### Tooling

* Web search: `atlas/rag/web_search.py` (DDG / SerpAPI optional)
* Context7: `atlas/rag/context7.py` (HTTP API fallback, caching in-memory)
* Repo indexing/search: `atlas/rag/codebase.py`, `atlas/scout/repo_tools.py`
* Patch apply + tests:
  * `atlas/verification/patch_applier.py`
  * `atlas/verification/test_runner.py`
* Quality scoring: `atlas/quality/quality_scorer.py`

#### What Atlas MCP *does not* provide natively

* Persistent project memory across sessions (beyond local run artifacts)
* Agent-MCP's task graph & coordination model
* A stable "conductor" orchestrating multiple *different* tasks concurrently (Atlas is per-task swarm)

---

### 3.3 Gaps / overlaps / conflicts

#### Overlaps

* Both expose MCP servers/tools (but different server frameworks)
* Both have "task" concepts (but different data models and persistence)
* Both have some form of context retrieval (Agent-MCP RAG vs Atlas repo analyzer)

#### Gaps that must be closed

1. **Single entrypoint**: Agent-MCP needs a first-class way to call a swarm run.
2. **Shared persistence**: Atlas outputs must be persisted in Agent-MCP's DB (and optionally indexed into RAG).
3. **Unified tool policy**: Tool permissions, rate limits, and caching must be consistent for swarm agents.
4. **Operational controls**: Timeouts, retries, budgets, circuit breakers must be production-grade and consistent.

#### Conflicts

* **Python version**:
  * Agent-MCP: `>=3.10`
  * Atlas: `>=3.11`
  * **Resolution**: unify on **Python 3.11** as platform minimum (MVI requirement).
* **MCP server framework mismatch**:
  * Agent-MCP uses Starlette + low-level MCP server
  * Atlas uses FastMCP
  * **Resolution**: keep Agent-MCP as the only server; integrate Atlas as a library subsystem (no second server in-process).
* **Repo model**:
  * Agent-MCP operates on a local project dir (`MCP_PROJECT_DIR`)
  * Atlas clones remote repos per task
  * **Resolution**: support both via `RepoResolver`:
    * local repo mode (default)
    * remote clone mode (optional)

---

## 4. Target Architecture

### 4.1 Unified mental model

Think of the unified system as:

* **Agent-MCP is the "Workflow Orchestrator"**:
  * maintains the task DAG (root tasks + subtasks)
  * assigns work to specialized agents
  * composes final result
  * persists memory and context

* **Atlas Swarm is a "Solver Strategy"**:
  * can be invoked for any task node
  * produces consensus-selected output with trace artifacts
  * optionally verifies correctness with tests/static checks
  * writes results back into Agent-MCP memory and task notes

A task node can be solved by either:

* **Single-agent execution** (existing path): tmux worker agent completes work.
* **Swarm execution** (new path): N agents run in parallel → cluster → vote → select winner → persist.

---

### 4.2 Component diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              Agent-MCP Server                               │
│ (Starlette + mcp.server.lowlevel.Server; tools registered via registry.py)   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────┐        ┌──────────────────────────────┐ │
│  │ Task Orchestrator (existing)  │        │   Swarm Orchestrator (new)   │ │
│  │ - tasks (task_tools.py)       │        │ - SwarmManager               │ │
│  │ - assignment, deps, notes     │        │ - ConsensusEngine            │ │
│  │ - agent sessions (tmux)       │        │ - ToolRouter + Budgets       │ │
│  └───────────────┬──────────────┘        └───────────────┬──────────────┘ │
│                  │                                       │                 │
│                  │ calls tool                             │ calls tools    │
│                  ▼                                       ▼                 │
│  ┌──────────────────────────────┐        ┌──────────────────────────────┐ │
│  │ Tool Registry (registry.py)   │        │ Atlas Library (embedded)     │ │
│  │ - list_tools/call_tool        │        │ - AgentPoolManager           │ │
│  │ - auth + audit hooks          │        │ - VotingManager/Incremental  │ │
│  └───────────────┬──────────────┘        │ - Clustering, TestRunner      │ │
│                  │                       └───────────────┬──────────────┘ │
│                  │ writes/reads DB                        │                │
│                  ▼                                       ▼                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Persistence Layer (SQLite)                      │ │
│  │  Existing: tasks, agents, project_context, rag_*                        │ │
│  │  New: swarm_runs, swarm_agents, swarm_outputs, tool_cache               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 Key interfaces

#### 4.3.1 New modules and boundaries (Agent-MCP)

We add a new package:

* `agent_mcp/features/swarm/`
  * `swarm_manager.py`
  * `atlas_adapter.py`
  * `consensus_engine.py`
  * `tool_router.py`
  * `repo_resolver.py`
  * `budget_manager.py`
  * `persistence.py`
  * `schemas.py`

And a new tool module:

* `agent_mcp/tools/swarm_tools.py` (registered in `agent_mcp/tools/__init__.py`)

---

#### 4.3.2 SwarmManager public interface

```python
# agent_mcp/features/swarm/swarm_manager.py

from dataclasses import dataclass
from typing import Literal, Optional

SwarmMode = Literal["patch", "answer"]

@dataclass
class SwarmRequest:
    task_id: Optional[str]
    description: str
    mode: SwarmMode
    repo: dict  # RepoSpec
    budgets: dict  # BudgetSpec
    swarm: dict  # SwarmSpec
    tools: dict  # ToolPolicy
    memory: dict  # MemoryPolicy
    caller: dict  # CallerContext (agent_id, role, token hash)

@dataclass
class SwarmResult:
    run_id: str
    task_id: Optional[str]
    mode: SwarmMode
    consensus_reached: bool
    selected_output: str              # patch or answer
    selected_variant_id: str          # agent_id or cluster rep id
    confidence_score: float
    clusters: list[dict]              # summarized
    vote_counts: dict[str, int]
    metrics: dict                     # latency, cost, tokens, tool_calls
    trace_ref: str                    # pointer to stored trace artifacts
    errors: list[str]
    warnings: list[str]

class SwarmManager:
    async def run(self, req: SwarmRequest) -> SwarmResult:
        ...
```

---

#### 4.3.3 MCP tool: `run_swarm_consensus`

**Tool name**: `run_swarm_consensus`
**Location**: `agent_mcp/tools/swarm_tools.py`

**Input schema**

```json
{
  "type": "object",
  "properties": {
    "token": { "type": "string", "description": "Admin or agent token" },
    "task_id": { "type": "string", "description": "Optional: bind swarm run to an existing Agent-MCP task" },
    "description": { "type": "string", "description": "Task description if task_id not provided" },
    "mode": { "type": "string", "enum": ["patch", "answer"], "default": "patch" },
    "repo": {
      "type": "object",
      "properties": {
        "type": { "type": "string", "enum": ["local", "remote"], "default": "local" },
        "path": { "type": "string" },
        "url": { "type": "string" },
        "branch": { "type": "string", "default": "main" },
        "use_worktree": { "type": "boolean", "default": true }
      }
    },
    "swarm": {
      "type": "object",
      "properties": {
        "size": { "type": "integer", "minimum": 1, "maximum": 50, "default": 10 },
        "use_agentic": { "type": "boolean", "default": true },
        "consensus_k": { "type": "integer", "default": 3 },
        "similarity_threshold": { "type": "number", "default": 0.8 }
      }
    },
    "budgets": {
      "type": "object",
      "properties": {
        "timeout_seconds": { "type": "integer", "default": 900 },
        "max_cost_usd": { "type": "number", "default": 2.0 },
        "max_tool_calls": { "type": "integer", "default": 100 }
      }
    },
    "tools": {
      "type": "object",
      "properties": {
        "enable_web_search": { "type": "boolean", "default": true },
        "enable_context7": { "type": "boolean", "default": true },
        "enable_repo_search": { "type": "boolean", "default": true },
        "enable_agent_mcp_rag": { "type": "boolean", "default": true }
      }
    },
    "memory": {
      "type": "object",
      "properties": {
        "write_back_to_task": { "type": "boolean", "default": true },
        "write_back_to_project_context": { "type": "boolean", "default": true },
        "index_into_rag": { "type": "boolean", "default": false }
      }
    }
  },
  "required": ["token"]
}
```

**Output contract**

```json
{
  "run_id": "swarm_...",
  "task_id": "optional",
  "mode": "patch|answer",
  "consensus_reached": true,
  "confidence_score": 0.8,
  "selected_output": "....",
  "vote_counts": {"cluster_0": 5, "cluster_1": 3},
  "clusters": [
    {"id":"cluster_0","size":5,"is_valid":true,"rep_agent":"agent_3"},
    {"id":"cluster_1","size":3,"is_valid":true,"rep_agent":"agent_7"}
  ],
  "metrics": {"duration_ms": 81234, "cost_usd": 1.42, "tokens": 123456, "tool_calls": 18},
  "errors": [],
  "warnings": []
}
```

---

## 5. Core Workflows

### Workflow A: Simple task, no swarm

```
User/Client
  -> Agent-MCP MCP Server (call_tool: assign_task)
      -> task_tools.assign_task_tool_impl
          -> DB: insert task
          -> (optional) assign to worker agent
      <- task_id returned

Worker Agent (tmux)
  -> Agent-MCP tools (ask_project_rag, view_project_context, file ops...)
  -> completes work outside swarm

Worker Agent
  -> Agent-MCP (call_tool: update_task_status {completed, notes})
      -> DB update task + notes
      <- ok

Admin/Orchestrator
  -> Agent-MCP (view_tasks/search_tasks)
      <- final status
```

### Workflow B: Root task uses swarm

```
User/Client
  -> Agent-MCP (assign_task root task)
      <- task_id

Admin/Orchestrator
  -> Agent-MCP (run_swarm_consensus {task_id, mode})
      -> SwarmTools.run_swarm_consensus_tool_impl
          -> Auth verify_token/get_agent_id
          -> SwarmManager.run(SwarmRequest)
              -> RepoResolver.resolve(local repo path or clone)
              -> ContextBuilder.build_context(task + memory + repo)
              -> AtlasAdapter.generate_candidates(...)
                  -> AgentPoolManager.create_swarm(N, use_agentic)
                  -> each agent executes in parallel (bounded)
              -> ConsensusEngine.cluster_and_vote(...)
              -> Persistence.save_run(...)
              -> MemoryWriter.write_back(task_notes/project_context/RAG optional)
          <- SwarmResult JSON

Admin/Orchestrator
  -> optionally creates follow-up tasks (apply patch / review / merge)
```

### Workflow C: Subtask uses swarm

```
Worker Agent (working on parent task)
  -> Agent-MCP (create_self_task)  # creates child task
      <- child_task_id

Worker Agent or Admin
  -> Agent-MCP (run_swarm_consensus {task_id=child_task_id})
      -> SwarmManager resolves context using:
           - task description
           - parent task notes (via DB join)
           - project_context entries
           - RAG hits
      <- SwarmResult

Worker Agent
  -> uses selected_output to implement or validate
  -> update_task_status(child completed)
```

---

## 6. Data & Memory Design

### Storage backends + schemas

#### Backend: SQLite (Agent-MCP `.agent/mcp_state.db`)

##### Table: `swarm_runs`

| Column | Type | Notes |
|--------|------|-------|
| run_id | TEXT PK | `swarm_<uuid>` |
| task_id | TEXT NULL | FK to tasks |
| mode | TEXT | `patch` or `answer` |
| status | TEXT | `running/completed/failed/timeout/budget_exceeded` |
| config_json | TEXT | full SwarmRequest (redacted secrets) |
| started_at | TEXT | ISO |
| completed_at | TEXT | ISO |
| consensus_reached | INTEGER | 0/1 |
| confidence_score | REAL | 0..1 |
| selected_output | TEXT | patch or answer |
| selected_variant_id | TEXT | representative agent or cluster |
| vote_counts_json | TEXT | map cluster_id→count |
| metrics_json | TEXT | cost/tokens/tool_calls/latency |
| errors_json | TEXT | list |
| warnings_json | TEXT | list |

##### Table: `swarm_agents`

| Column | Type | Notes |
|--------|------|-------|
| run_id | TEXT | |
| swarm_agent_id | TEXT | e.g. `agent_3` |
| prompt_style | TEXT | from Atlas `PromptStyleName` |
| model | TEXT | provider model name |
| temperature | REAL | |
| status | TEXT | success/fail/timeout |
| started_at | TEXT | |
| completed_at | TEXT | |
| tokens_used | INTEGER | |
| cost_usd | REAL | |
| error | TEXT | |

PK: `(run_id, swarm_agent_id)`

##### Table: `swarm_outputs`

| Column | Type | Notes |
|--------|------|-------|
| run_id | TEXT | |
| swarm_agent_id | TEXT | |
| output_type | TEXT | `patch` or `answer` |
| output_text | TEXT | raw output |
| explanation | TEXT | optional |
| is_valid | INTEGER | |
| validation_errors_json | TEXT | |
| cluster_id | TEXT | assigned by clustering |
| test_result_json | TEXT | if patch mode verification |
| quality_score_json | TEXT | from `QualityScorer` |

PK: `(run_id, swarm_agent_id)`

##### Table: `tool_cache`

| Column | Type | Notes |
|--------|------|-------|
| cache_key | TEXT PK | stable hash of (tool, args, policy) |
| tool_name | TEXT | |
| value_json | TEXT | |
| created_at | TEXT | |
| expires_at | TEXT | |
| hit_count | INTEGER | |

##### Table: `tool_calls`

| Column | Type |
|--------|------|
| run_id | TEXT |
| swarm_agent_id | TEXT |
| tool_name | TEXT |
| args_json | TEXT |
| result_meta_json | TEXT |
| status | TEXT |
| duration_ms | INTEGER |
| error | TEXT |
| created_at | TEXT |

---

## 7. Swarm Design

### 7.1 Diversity generation

Reuse Atlas prompt styles from `atlas/agents/prompt_styles.py`:

* `SENIOR_ENGINEER`
* `SECURITY_FOCUSED`
* `PERFORMANCE_EXPERT`
* `SYSTEMS_ARCHITECT`
* `CODE_REVIEWER`

#### Diversity strategies

1. **prompt_styles** (default) - Round-robin styles across agents
2. **prompt_styles+temperature** - Variants with small temperature deltas
3. **prompt_styles+models** (full vision) - Mix model providers

### 7.2 Clustering

#### Patch mode (reuse Atlas)

* Use `SimilarityClustering(similarity_threshold)` from `atlas/verification/clustering.py`
* If tests run: use behavioral clustering (`cluster_by_test_outcomes`)

#### Answer mode (new)

* Compute embeddings for each candidate answer
* Run k-means clustering with cosine distance
* Choose representative per cluster (medoid)

### 7.3 Consensus rules

#### Default rule (matches Atlas)

Use **first-to-ahead-by-K**:
* Compute clusters
* `margin = leader.size - runner_up.size`
* Consensus when `margin >= consensus_k` AND leader is valid

#### Tie-breakers

1. Prefer **passing behavioral cluster** (patch mode)
2. Prefer **highest QualityScore.overall_score**
3. Prefer **largest cluster size**
4. Prefer **lowest cost**
5. Deterministic: smallest `swarm_agent_id`

---

## 8. Tooling & Integrations

### Web research layer

* Reuse Atlas `WebSearchClient` with **persistent caching** in `tool_cache`
* Add **domain allow/deny lists**
* Add **source ranking**

### DB search layer

* For structured DB reads: tasks, project_context, swarm runs
* For semantic retrieval: call RAG query

### Tool-call sandboxing + rate limits

* Web + Context7: HTTP via `httpx.AsyncClient` with timeout
* Tests / patch application: run in Git worktree or temp copy

---

## 9. Non-Functional Requirements

### Performance budgets

* `timeout_seconds`: 900 (15 min)
* `max_cost_usd`: 2.0
* `max_tool_calls`: 100
* `max_tokens_total`: 500k
* `max_concurrency_agents`: 10

### Reliability targets

* Tool-call success rate ≥ 98%
* Swarm result persisted ≥ 99.9%

### Cost controls

* Hard stops when `cost_usd >= max_cost_usd`
* Prefer early consensus via `IncrementalVoter`

---

## 10. Implementation Plan

### Epics

#### Epic 1: Persistence & schemas
* Add `swarm_runs`, `swarm_agents`, `swarm_outputs`, `tool_cache`, `tool_calls` tables
* Add DB access layer using `execute_db_write`

#### Epic 2: Swarm entrypoint tool
* Implement `run_swarm_consensus` tool with schema and auth checks
* Bind swarm run to `task_id` and write back notes

#### Epic 3: Atlas embedding (library integration)
* Vendor or add dependency for `atlas` package
* Implement `AtlasAdapter` to call AgentPoolManager, VotingManager, etc.

#### Epic 4: ToolRouter + policy enforcement
* Implement tool permission matrix
* Implement caching + TTL
* Implement rate limits + circuit breakers

#### Epic 5: Observability + eval
* Structured logs + run_id correlation
* Offline replay harness
* Golden tests suite

---

## 11. Feature Flags

* `SWARM_ENABLED` (global)
* `SWARM_PATCH_MODE_ENABLED`
* `SWARM_ANSWER_MODE_ENABLED`
* `SWARM_ENABLE_WEB`
* `SWARM_ENABLE_CONTEXT7`
* `SWARM_ENABLE_TEST_VERIFICATION`
* `SWARM_WRITEBACK_TASK_NOTES`
* `SWARM_WRITEBACK_PROJECT_CONTEXT`

---

## 12. Migration & Rollout

### MVI steps

**Week 1: Foundations**
* Add new DB tables + migrations
* Add feature flags

**Week 2: Swarm tool + minimal adapter**
* Implement `run_swarm_consensus` tool
* Implement `RepoResolver` local-only
* Implement `AtlasAdapter` patch-mode generation only

**Week 3: Consensus + verification**
* Integrate Atlas `VotingManager` and similarity clustering
* Add patch parsing + apply-to-sandbox verification

**Week 4: ToolRouter + caching**
* Integrate Context7 + web search through ToolRouter
* Add tool_cache + tool_calls
* Enforce budgets, rate limits, circuit breakers

**Week 5: Answer mode + eval harness**
* Add answer-mode prompt + embedding clustering
* Add replay harness + golden tests
* Add observability metrics/tracing stubs

### Backwards compatibility

* No existing tool schemas change
* Swarm is additive: new tool only, new DB tables only

### Rollback strategy

* Disable feature flags
* Keep DB tables; they're inert when swarm is off
