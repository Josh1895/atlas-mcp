# Code Review Fixes Specification

## Status: In Progress

## Overview
This spec addresses confirmed issues from an external code review of the ATLAS MCP Server codebase.

---

## P0 Issues (Critical)

### Issue 1: `repo_url` vs `repository_url` Bug
**File**: `agent_mcp/features/swarm/atlas_adapter.py`
**Lines**: 176, 220

**Problem**: `TaskSubmission` dataclass expects `repository_url` but code uses `repo_url`, causing silent failures or TypeErrors.

**Evidence**:
```python
# atlas_adapter.py:176 - WRONG
task = TaskSubmission(
    repo_url="local",  # SHOULD BE repository_url
)

# src/atlas/core/task.py - TaskSubmission definition
@dataclass
class TaskSubmission:
    repository_url: str = ""  # CORRECT FIELD NAME
```

---

### Issue 2: `.agent/` Not in `.gitignore`
**File**: `.gitignore`

**Problem**: `.agent/` directory contains local state (`.agent/mcp_state.db`) but is not gitignored. Could leak local development state or SQLite databases.

**Evidence**:
- `.gitignore` contains `.atlas/` but NOT `.agent/`
- `.agent/mcp_state.db` exists in repository

---

## P1 Issues (Important)

### Issue 3: Deprecated `asyncio.get_event_loop()`
**Files** (7 total):
- `agent_mcp/tui/actions.py`
- `src/atlas/core/dag_orchestrator.py`
- `src/atlas/core/orchestrator.py`
- `src/atlas/agents/agentic_client.py`
- `src/atlas/rag/web_search.py`
- `src/atlas/agents/gemini_client.py`

**Problem**: `asyncio.get_event_loop()` is deprecated in Python 3.10+ and raises `DeprecationWarning`. In Python 3.12+, it will raise `RuntimeError` if no event loop is running.

**Correct Pattern**:
```python
# DEPRECATED:
loop = asyncio.get_event_loop()
result = loop.run_until_complete(coro)

# CORRECT (Python 3.7+):
result = asyncio.run(coro)

# Or inside async context:
loop = asyncio.get_running_loop()
```

---

### Issue 4: Model Naming Inconsistency
**Problem**: Multiple different model names scattered across the codebase with no single source of truth.

**Evidence**:
| File | Model Name |
|------|------------|
| `.env.example` | `gemini-2.0-flash` |
| `agent_mcp/core/config.py` | `gemini-3.0-flash` |
| `agent_mcp/external/gemini_service.py` | `gemini-3.0-flash` |
| `agent_mcp/features/swarm/schemas.py` | `gemini-3.0-flash` |
| `agent_mcp/tools/swarm_tools.py` | `gemini-2.0-flash` |
| `src/atlas/core/config.py` | `gemini-3-flash-preview` |
| `src/atlas/quality/pr_reviewer.py` | `gemini-2.5-flash` |
| `tests/test_orchestrator.py` | `gemini-2.5-flash` |

**Impact**: Inconsistent behavior, confusion about which model is actually used, potential API errors if model names don't exist.

---

## Out of Scope (Deferred)

### Sandboxing
Per MCP specification, sandboxing is "SHOULD" not "MUST". For local development MCP server, sandboxing is overkill. Deferred to future security hardening phase.

### Concurrency (Semaphores)
Valid theoretical concern but no evidence of actual race conditions in local single-user MCP context. Deferred.

---

## Success Criteria

1. [ ] No Python TypeErrors from incorrect field names
2. [ ] `.agent/` directory properly gitignored
3. [ ] No deprecation warnings from asyncio
4. [ ] Single source of truth for model configuration
5. [ ] All tests pass after changes
