# SWARM-001: Atlas Swarm Integration into Agent-MCP

**Branch:** `swarm-001-feature`
**Created:** 2025-01-13
**Status:** Draft

## Overview

Embed Atlas Swarm capabilities into Agent-MCP as a first-class subsystem, providing a single entrypoint (run_swarm_consensus tool) that invokes parallel multi-agent generation, clustering, and voting consensus to solve tasks with higher correctness and robustness.

---

## User Scenarios & Testing

### US-001: Admin invokes swarm on root task
**Priority:** P1

An admin user creates a task and invokes swarm consensus to generate a patch solution with multi-agent verification.

**Acceptance Criteria:**
```gherkin
Given:
  - Agent-MCP server is running with SWARM_ENABLED=true
  - A task exists in the tasks table with task_id
  - Admin has a valid admin token

When:
  - Admin calls run_swarm_consensus with task_id and mode=patch
  - Swarm generates N candidate patches using diverse prompt styles
  - Candidates are clustered by similarity/behavior
  - Voting determines winning cluster

Then:
  - SwarmResult is returned with selected_output containing the patch
  - consensus_reached indicates if threshold was met
  - Run is persisted in swarm_runs table
  - Task notes are updated with swarm summary
```

**Edge Cases:**
- No consensus reached within timeout - return best-effort
- All candidates invalid - return error with validation details
- Budget exceeded mid-run - stop and return partial result

### US-002: Worker agent uses swarm for subtask
**Priority:** P1

A worker agent creates a child task and invokes swarm to solve a complex sub-problem while working on a larger task.

**Acceptance Criteria:**
```gherkin
Given:
  - Worker agent is assigned to parent task
  - Worker has created a child task via create_self_task
  - Worker has a valid agent token

When:
  - Worker calls run_swarm_consensus with child task_id
  - Swarm builds context from task description + parent notes + project context

Then:
  - SwarmResult includes parent task context in reasoning
  - Child task notes updated with swarm result
  - Parent task can reference child's swarm output
```

**Edge Cases:**
- Parent task has no notes - use only child description
- Deep nesting (>3 levels) - cap context depth

### US-003: Answer mode for research questions
**Priority:** P2

Admin uses swarm in answer mode to get consensus on a research question with web search and citations.

**Acceptance Criteria:**
```gherkin
Given:
  - SWARM_ANSWER_MODE_ENABLED=true
  - enable_web_search=true in tool config

When:
  - Admin calls run_swarm_consensus with mode=answer and a description (question)
  - Swarm agents perform web research
  - Answers are clustered by embedding similarity
  - Voting selects consensus answer

Then:
  - Selected answer includes citations from web sources
  - citations_present field indicates if citations were found
  - Answer is stored in swarm_outputs with source URLs
```

**Edge Cases:**
- Web search disabled - answer without citations
- Conflicting sources - flag in warnings
- No consensus on factual claim - lower confidence score

### US-004: Swarm with test verification
**Priority:** P1

Swarm generates patch candidates, runs tests against each, and uses behavioral clustering to prefer passing solutions.

**Acceptance Criteria:**
```gherkin
Given:
  - SWARM_ENABLE_TEST_VERIFICATION=true
  - test_command is provided in task or discovered from repo

When:
  - Swarm generates patch candidates
  - Each candidate is applied in isolated worktree
  - Tests are run against each candidate
  - Candidates clustered by test outcome

Then:
  - Passing clusters preferred over failing clusters
  - test_result_json stored for each output
  - If any cluster passes, consensus only from passing clusters
```

**Edge Cases:**
- All candidates fail tests - select by similarity consensus
- Tests timeout - mark as failed, not invalid
- Test infrastructure broken - skip verification, log warning

### US-005: Tool caching across swarm agents
**Priority:** P2

Expensive tool calls (web search, Context7) are cached and shared across agents in the same run.

**Acceptance Criteria:**
```gherkin
Given:
  - Multiple swarm agents query the same documentation or search terms

When:
  - First agent calls web_search with query Q
  - Result is cached in tool_cache with TTL
  - Second agent calls web_search with same query Q

Then:
  - Second call returns cached result
  - hit_count incremented
  - Total tool_calls_count reflects unique calls, not cached hits
```

**Edge Cases:**
- Cache expired mid-run - re-fetch
- Cache corrupted - invalidate and re-fetch

### US-006: Budget enforcement stops runaway costs
**Priority:** P1

Swarm run is stopped gracefully when budget limits are reached.

**Acceptance Criteria:**
```gherkin
Given:
  - max_cost_usd=1.0 configured
  - Current run has spent $0.95

When:
  - Next agent generation would exceed budget
  - BudgetManager detects overage

Then:
  - No more agents spawned
  - Consensus computed from existing candidates
  - status=budget_exceeded in result
  - Warning included in response
```

**Edge Cases:**
- Budget exactly hit - allow completion of current agent
- Budget check race condition - soft overage allowed

### US-007: Swarm disabled maintains backward compatibility
**Priority:** P1

When SWARM_ENABLED=false, all existing Agent-MCP functionality works unchanged.

**Acceptance Criteria:**
```gherkin
Given:
  - SWARM_ENABLED=false in environment

When:
  - User calls run_swarm_consensus tool

Then:
  - Tool returns error: Swarm feature is disabled
  - No swarm_runs record created
  - All other tools (assign_task, update_task_status, etc.) work normally
```

**Edge Cases:**
- Flag changed mid-session - check at tool invocation time

---

## Functional Requirements

- **FR-001**: MUST Implement run_swarm_consensus MCP tool that accepts task_id or description, mode (patch/answer), and configuration options
- **FR-002**: MUST Persist all swarm runs in swarm_runs table with full metadata, config, and results
- **FR-003**: MUST Persist individual agent outputs in swarm_outputs table with validation status and cluster assignment
- **FR-004**: MUST Integrate Atlas AgentPoolManager for swarm generation with diverse prompt styles
- **FR-005**: MUST Integrate Atlas SimilarityClustering for patch mode clustering
- **FR-006**: MUST Integrate Atlas VotingManager for first-to-ahead-by-K consensus
- **FR-007**: MUST Implement embedding-based clustering for answer mode using Agent-MCP's embedding model
- **FR-008**: MUST Implement BudgetManager to enforce timeout, cost, token, and tool call limits
- **FR-009**: MUST Implement ToolRouter with permission matrix for enabling/disabling web_search, context7, repo_search, agent_mcp_rag
- **FR-010**: MUST Implement tool_cache table for caching expensive tool outputs with TTL
- **FR-011**: SHOULD Implement circuit breaker for tool providers to prevent cascading failures
- **FR-012**: MUST Implement RepoResolver to support both local project dir and remote clone modes
- **FR-013**: SHOULD Integrate Atlas PatchApplier and TestRunner for patch verification
- **FR-014**: SHOULD Implement behavioral clustering (cluster_by_test_outcomes) when tests are available
- **FR-015**: MUST Write swarm results back to task notes when write_back_to_task=true
- **FR-016**: SHOULD Write swarm summaries to project_context when write_back_to_project_context=true
- **FR-017**: MAY Index selected outputs into RAG when index_into_rag=true
- **FR-018**: MUST Gate all swarm functionality behind SWARM_ENABLED feature flag
- **FR-019**: MUST Implement structured logging with run_id correlation for all swarm operations
- **FR-020**: SHOULD Implement offline replay harness for deterministic testing of consensus selection

---

## Data Entities

### SwarmRun

A single execution of the swarm consensus process for a task

**Attributes:**
- `run_id`: TEXT PK - unique identifier (swarm_<uuid>)
- `task_id`: TEXT NULL - FK to tasks table
- `mode`: TEXT - 'patch' or 'answer'
- `status`: TEXT - running/completed/failed/timeout/budget_exceeded
- `config_json`: TEXT - serialized SwarmRequest
- `started_at`: TEXT - ISO timestamp
- `completed_at`: TEXT - ISO timestamp
- `consensus_reached`: INTEGER - 0/1
- `confidence_score`: REAL - 0.0 to 1.0
- `selected_output`: TEXT - winning patch or answer
- `selected_variant_id`: TEXT - winning agent or cluster ID
- `vote_counts_json`: TEXT - cluster_id to vote count map
- `metrics_json`: TEXT - duration, cost, tokens, tool_calls
- `errors_json`: TEXT - list of error messages
- `warnings_json`: TEXT - list of warnings

**Relationships:**
- has_many SwarmAgent
- has_many SwarmOutput
- belongs_to Task (optional)

### SwarmAgent

A single micro-agent instance within a swarm run

**Attributes:**
- `run_id`: TEXT - FK to swarm_runs
- `swarm_agent_id`: TEXT - agent identifier (agent_0, agent_1, ...)
- `prompt_style`: TEXT - SENIOR_ENGINEER, SECURITY_FOCUSED, etc.
- `model`: TEXT - model name used
- `temperature`: REAL - temperature setting
- `status`: TEXT - success/fail/timeout
- `started_at`: TEXT - ISO timestamp
- `completed_at`: TEXT - ISO timestamp
- `tokens_used`: INTEGER
- `cost_usd`: REAL
- `error`: TEXT - error message if failed

**Relationships:**
- belongs_to SwarmRun
- has_one SwarmOutput

### SwarmOutput

The output produced by a single swarm agent

**Attributes:**
- `run_id`: TEXT - FK to swarm_runs
- `swarm_agent_id`: TEXT - FK to swarm_agents
- `output_type`: TEXT - 'patch' or 'answer'
- `output_text`: TEXT - raw output content
- `explanation`: TEXT - optional reasoning
- `is_valid`: INTEGER - 0/1 validation result
- `validation_errors_json`: TEXT - list of validation errors
- `cluster_id`: TEXT - assigned cluster
- `test_result_json`: TEXT - test execution results
- `quality_score_json`: TEXT - QualityScorer output

**Relationships:**
- belongs_to SwarmRun
- belongs_to SwarmAgent

### ToolCache

Cached results from expensive tool calls

**Attributes:**
- `cache_key`: TEXT PK - hash of (tool, args)
- `tool_name`: TEXT - web_search, context7, etc.
- `value_json`: TEXT - cached response
- `created_at`: TEXT - ISO timestamp
- `expires_at`: TEXT - ISO timestamp
- `hit_count`: INTEGER - number of cache hits

### ToolCall

Audit record of a tool invocation by a swarm agent

**Attributes:**
- `run_id`: TEXT - FK to swarm_runs
- `swarm_agent_id`: TEXT
- `tool_name`: TEXT
- `args_json`: TEXT - tool arguments
- `result_meta_json`: TEXT - result metadata
- `status`: TEXT - success/fail/cached
- `duration_ms`: INTEGER
- `error`: TEXT
- `created_at`: TEXT - ISO timestamp

**Relationships:**
- belongs_to SwarmRun
- belongs_to SwarmAgent

---

## Success Criteria

- **SC-001**: Swarm invocation end-to-end success rate
  - Target: >= 95% of run_swarm_consensus calls return SwarmResult without internal errors
  - Measurement: Count errors_json empty / total runs

- **SC-002**: Consensus rate on representative eval set
  - Target: >= 70% consensus_reached=true
  - Measurement: Run golden test suite, count consensus_reached

- **SC-003**: P50 latency for swarm runs
  - Target: <= 90 seconds
  - Measurement: Median of metrics.duration_ms across runs

- **SC-004**: Budget compliance rate
  - Target: >= 99% of runs stay under max_cost_usd
  - Measurement: Count metrics.cost_usd <= config.max_cost_usd / total runs

- **SC-005**: Persistence reliability
  - Target: >= 99.9% of runs have complete DB records
  - Measurement: Audit swarm_runs vs swarm_agents vs swarm_outputs counts

- **SC-006**: Backward compatibility
  - Target: 100% of existing Agent-MCP tests pass with SWARM_ENABLED=false
  - Measurement: Run existing test suite with swarm disabled

---

## Assumptions

- Python 3.11 is acceptable as minimum version for unified codebase
- SQLite is sufficient for persistence (no need for PostgreSQL in MVP)
- Atlas library can be imported directly without running a separate server
- Gemini API is available for swarm agent LLM calls
- Agent-MCP's existing auth (admin token + agent tokens) is sufficient for swarm access control

---

## Constraints

- Must not break existing Agent-MCP tool schemas or behavior
- Must use Agent-MCP's existing DatabaseWriteQueue for DB writes
- Must be fully functional with web/context7/tests disabled (minimal mode)
- Total implementation should not exceed 5 weeks for MVP

---

## Open Questions

- Should swarm runs be allowed to create their own child tasks automatically?
- What is the retention policy for swarm_outputs (full history vs. only winning)?
- Should answer mode support multi-turn clarification?
- How to handle model provider outages (retry vs. fallback to different provider)?
