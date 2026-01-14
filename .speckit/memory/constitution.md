# Project Constitution: Atlas-AgentMCP Unified System

**Version:** 1.0
**Created:** 2025-01-13
**Updated:** 2025-01-13

---

## Core Principles

### P-001: Agent-MCP as Primary Orchestrator

Agent-MCP remains the primary conductor for task decomposition, dispatch, persistent memory, and result composition. Atlas Swarm is embedded as a solver strategy, not a separate product.

**Rationale:** Maintains backward compatibility and leverages Agent-MCP's existing strengths in coordination while adding swarm capabilities.

**Examples:**
- All swarm runs are tracked in Agent-MCP's SQLite database
- Swarm results are automatically written back to task notes
- Existing Agent-MCP tools continue to work unchanged

### P-002: Tests as Primary Oracle

In patch mode, tests are the primary correctness oracle. Patches are validated by running actual tests, not just by similarity matching. Behavioral clustering takes precedence over syntactic similarity when tests are available.

**Rationale:** Behavioral correctness is more important than syntactic similarity. A passing test proves functionality.

**Examples:**
- Prefer passing behavioral cluster over larger syntactic cluster
- Always run test_command when provided
- Mark consensus_reached=true only for valid patches

### P-003: Multi-Agent Consensus Over Single-Agent

Use diverse AI agents with different perspectives (prompt styles) to generate solutions, then use clustering + voting to select the best. Consensus improves robustness.

**Rationale:** Diverse approaches reduce single-point-of-failure and improve solution quality through ensemble effects.

**Examples:**
- Use at least 5 agents with different prompt styles
- Wait for consensus (K votes ahead) before selecting
- Include SENIOR_ENGINEER, SECURITY_FOCUSED, PERFORMANCE_EXPERT styles

### P-004: Feature-Flagged Rollout

All swarm functionality is gated behind feature flags. Existing Agent-MCP flows work unchanged when swarm mode is disabled.

**Rationale:** Enables safe rollout, A/B testing, and immediate rollback if issues arise.

**Examples:**
- SWARM_ENABLED=false disables all swarm features
- Individual features (web, context7) can be toggled independently
- DB tables exist but are inert when swarm is disabled

### P-005: Budget Enforcement

Every swarm run has hard limits on time, cost, tokens, and tool calls. Budget overruns trigger graceful best-effort results, never runaway costs.

**Rationale:** Production systems must have predictable costs and latency. No runaway API calls.

**Examples:**
- Hard stop at max_cost_usd with best-effort result
- Timeout returns best consensus so far
- Early consensus via IncrementalVoter saves costs

### P-006: Persistent Traceability

Every swarm run creates durable records: run metadata, agent outputs, cluster assignments, vote counts, tool calls. Full replay capability for evaluation.

**Rationale:** Debugging, evaluation, and audit require complete history. Reproducibility enables golden tests.

**Examples:**
- swarm_runs table stores complete run metadata
- swarm_outputs stores every agent's raw output
- tool_calls logs every external tool invocation

---

## Coding Standards

- Use type hints for all function signatures
- Follow PEP 8 style guidelines
- Use dataclasses for data structures
- Keep functions under 50 lines where possible
- All async functions must handle cancellation gracefully
- Use structured logging with run_id correlation

---

## Testing Requirements

- All new features must have unit tests
- Integration tests for cross-module functionality
- Golden tests for consensus selection determinism
- Test coverage should not decrease
- Mock external APIs (LLM, web search) in tests

---

## Documentation Standards

- All public functions must have docstrings
- Complex algorithms must have inline comments
- API contracts documented with JSON schema
- Architecture decisions recorded in ADR format

---

## Patterns

### Required Patterns
- `execute_db_write` for all database writes
- `verify_token` for all tool authentication
- `log_audit` for security-relevant actions

### Forbidden Patterns
- `eval\s*\(`
- `exec\s*\(`
- `__import__\s*\(`
- `subprocess\.Popen.*shell=True`
- hardcoded API keys or tokens
