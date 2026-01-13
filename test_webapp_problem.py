#!/usr/bin/env python3
"""
ATLAS Test Script - Web App Problem
====================================
This script tests the ATLAS system with a web app problem and displays
all details including task decomposition, agent outputs, web search,
and Context7 MCP results.
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("atlas.test")


# ============================================================================
# SAMPLE WEB APP - A simple React + Express app with a bug to fix
# ============================================================================

SAMPLE_WEBAPP_FILES = {
    "package.json": """{
  "name": "todo-app",
  "version": "1.0.0",
  "scripts": {
    "start": "node server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}""",

    "server.js": """const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// In-memory todo storage
let todos = [];
let nextId = 1;

// GET all todos
app.get('/api/todos', (req, res) => {
  res.json(todos);
});

// POST new todo
app.post('/api/todos', (req, res) => {
  const { title, completed } = req.body;

  // BUG: No validation of title - can create empty todos
  const todo = {
    id: nextId++,
    title: title,
    completed: completed || false,
    createdAt: new Date().toISOString()
  };

  todos.push(todo);
  res.status(201).json(todo);
});

// PUT update todo
app.put('/api/todos/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const todoIndex = todos.findIndex(t => t.id === id);

  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }

  // BUG: No validation here either
  todos[todoIndex] = { ...todos[todoIndex], ...req.body };
  res.json(todos[todoIndex]);
});

// DELETE todo
app.delete('/api/todos/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const todoIndex = todos.findIndex(t => t.id === id);

  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }

  todos.splice(todoIndex, 1);
  res.status(204).send();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;
""",

    "src/components/TodoList.jsx": """import React, { useState, useEffect } from 'react';
import TodoItem from './TodoItem';

const API_URL = 'http://localhost:3000/api';

function TodoList() {
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    try {
      const response = await fetch(`${API_URL}/todos`);
      const data = await response.json();
      setTodos(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch todos');
      setLoading(false);
    }
  };

  const addTodo = async (e) => {
    e.preventDefault();

    // BUG: Frontend doesn't validate either - should check for empty/whitespace
    try {
      const response = await fetch(`${API_URL}/todos`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTodo })
      });

      if (!response.ok) throw new Error('Failed to add todo');

      const todo = await response.json();
      setTodos([...todos, todo]);
      setNewTodo('');
    } catch (err) {
      setError('Failed to add todo');
    }
  };

  const toggleTodo = async (id) => {
    const todo = todos.find(t => t.id === id);
    try {
      const response = await fetch(`${API_URL}/todos/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ completed: !todo.completed })
      });

      if (!response.ok) throw new Error('Failed to update todo');

      const updated = await response.json();
      setTodos(todos.map(t => t.id === id ? updated : t));
    } catch (err) {
      setError('Failed to update todo');
    }
  };

  const deleteTodo = async (id) => {
    try {
      const response = await fetch(`${API_URL}/todos/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete todo');

      setTodos(todos.filter(t => t.id !== id));
    } catch (err) {
      setError('Failed to delete todo');
    }
  };

  if (loading) return <div className="loading">Loading...</div>;

  return (
    <div className="todo-list">
      <h1>Todo App</h1>

      {error && <div className="error">{error}</div>}

      <form onSubmit={addTodo} className="add-form">
        <input
          type="text"
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
          placeholder="Add a new todo..."
        />
        <button type="submit">Add</button>
      </form>

      <ul>
        {todos.map(todo => (
          <TodoItem
            key={todo.id}
            todo={todo}
            onToggle={() => toggleTodo(todo.id)}
            onDelete={() => deleteTodo(todo.id)}
          />
        ))}
      </ul>

      {todos.length === 0 && <p className="empty">No todos yet!</p>}
    </div>
  );
}

export default TodoList;
""",

    "src/components/TodoItem.jsx": """import React from 'react';

function TodoItem({ todo, onToggle, onDelete }) {
  return (
    <li className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={onToggle}
      />
      <span className="title">{todo.title}</span>
      <span className="date">{new Date(todo.createdAt).toLocaleDateString()}</span>
      <button onClick={onDelete} className="delete-btn">Delete</button>
    </li>
  );
}

export default TodoItem;
""",

    "tests/server.test.js": """const request = require('supertest');
const app = require('../server');

describe('Todo API', () => {
  beforeEach(() => {
    // Reset todos before each test
    // Note: This requires the app to expose a reset method
  });

  describe('GET /api/todos', () => {
    it('should return empty array initially', async () => {
      const res = await request(app).get('/api/todos');
      expect(res.status).toBe(200);
      expect(Array.isArray(res.body)).toBe(true);
    });
  });

  describe('POST /api/todos', () => {
    it('should create a new todo', async () => {
      const res = await request(app)
        .post('/api/todos')
        .send({ title: 'Test todo' });

      expect(res.status).toBe(201);
      expect(res.body.title).toBe('Test todo');
      expect(res.body.completed).toBe(false);
    });

    // This test should FAIL - reveals the bug
    it('should reject empty title', async () => {
      const res = await request(app)
        .post('/api/todos')
        .send({ title: '' });

      expect(res.status).toBe(400);
    });

    // This test should also FAIL
    it('should reject whitespace-only title', async () => {
      const res = await request(app)
        .post('/api/todos')
        .send({ title: '   ' });

      expect(res.status).toBe(400);
    });
  });
});
""",
}

WEBAPP_PROBLEM = """
## Bug Report: Todo API allows empty todos

### Description
Users can create todos with empty or whitespace-only titles. This leads to
confusing UX where empty items appear in the todo list.

### Steps to Reproduce
1. POST to /api/todos with `{ "title": "" }` or `{ "title": "   " }`
2. Observe that a todo is created with empty/whitespace title

### Expected Behavior
- POST should return 400 Bad Request for empty/whitespace titles
- Frontend should also validate before submitting
- Error messages should be user-friendly

### Affected Files
- server.js (POST and PUT endpoints)
- src/components/TodoList.jsx (addTodo function)

### Requirements
1. Add server-side validation in POST /api/todos endpoint
2. Add server-side validation in PUT /api/todos/:id endpoint
3. Return proper 400 status with error message like `{ "error": "Title is required" }`
4. Add frontend validation in TodoList.jsx before submitting
5. Show error message to user when validation fails
6. Follow Express.js and React best practices for validation

### Tech Stack
- Backend: Express.js 4.x
- Frontend: React 18.x
- Testing: Jest
"""


# ============================================================================
# RESULT DISPLAY UTILITIES
# ============================================================================

def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


def print_table(headers: list[str], rows: list[list[Any]], max_width: int = 40) -> None:
    """Print a formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cell_str = str(cell)[:max_width]
            widths[i] = max(widths[i], len(cell_str))

    # Print header
    header_row = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)
    print(f"  {header_row}")
    print(f"  {separator}")

    # Print rows
    for row in rows:
        row_str = " | ".join(str(cell)[:max_width].ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {row_str}")


def truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def create_test_repo() -> Path:
    """Create a temporary test repository."""
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_test_webapp_"))
    repo_path = temp_dir / "todo-app"
    repo_path.mkdir()

    # Write sample files
    for rel_path, content in SAMPLE_WEBAPP_FILES.items():
        file_path = repo_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # Initialize git repo
    import subprocess
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        env={**dict(__import__("os").environ), "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com", "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@test.com"}
    )

    return repo_path


async def run_test():
    """Run the ATLAS test with detailed output."""
    print_header("ATLAS MCP Server - Web App Problem Test", "═")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for required environment variables
    from atlas.core.config import Config
    config = Config.from_env()

    print_section("Configuration Check")

    config_status = [
        ["GEMINI_API_KEY", "[OK]" if config.gemini_api_key else "[MISSING]"],
        ["CONTEXT7_API_KEY", "[OK]" if config.context7_api_key else "[MISSING]"],
        ["SERPAPI_API_KEY", "[OK]" if config.serpapi_api_key else "[optional]"],
        ["Model", config.model],
        ["Temperature", str(config.temperature)],
        ["Max Cost", f"${config.max_cost_usd}"],
    ]
    print_table(["Setting", "Value"], config_status)

    errors = config.validate()
    if errors:
        print(f"\n  [ERROR] Configuration errors: {errors}")
        print("  Please set GEMINI_API_KEY and CONTEXT7_API_KEY in .env file")
        return

    # Create test repository
    print_section("Creating Test Repository")
    repo_path = await create_test_repo()
    print(f"  Repository created at: {repo_path}")
    print(f"  Files:")
    for file in SAMPLE_WEBAPP_FILES.keys():
        print(f"    - {file}")

    # Import ATLAS components
    from atlas.core.dag_orchestrator import TaskDAGOrchestrator, TaskDAGSubmission, TaskExecutionConfig
    from atlas.core.config import set_config

    set_config(config)

    # Create orchestrator with verbose settings
    exec_config = TaskExecutionConfig(
        agents_per_task=3,  # 3 agents per task for faster testing
        top_k_per_task=2,
        beam_width=2,
        enable_quality_selection=False,  # Skip for speed
        run_oracles=False,  # Skip actual test running for this demo
    )

    orchestrator = TaskDAGOrchestrator(
        config=config,
        execution_config=exec_config,
        use_agentic=False,  # Use non-agentic mode for speed (pre-fetched RAG)
    )

    # Create submission
    submission = TaskDAGSubmission(
        description=WEBAPP_PROBLEM,
        repo_path=repo_path,
        keywords=["express", "validation", "react", "todo", "api"],
        component_names=["TodoList", "TodoItem", "express", "app"],
        max_cost_usd=5.0,
        timeout_minutes=30,
        max_tasks=8,
    )

    print_section("Running ATLAS DAG Orchestrator")
    print(f"  Problem: Todo API Validation Bug")
    print(f"  Max Tasks: {submission.max_tasks}")
    print(f"  Agents per Task: {exec_config.agents_per_task}")
    print(f"  Beam Width: {exec_config.beam_width}")
    print(f"  Agentic Mode: True (autonomous tool access)")
    print()
    print("  This will:")
    print("    1. Scout the repository structure")
    print("    2. Decompose into atomic tasks with contracts")
    print("    3. Run agent swarms on each task")
    print("    4. Assemble using beam search")
    print()

    start_time = time.time()

    try:
        result = await orchestrator.solve(submission)
    except Exception as e:
        print(f"\n  [ERROR] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - start_time

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    print_header("RESULTS", "═")

    # Overall Status
    print_section("Overall Status")
    status_data = [
        ["Status", result.status.upper()],
        ["Duration", f"{result.duration_seconds:.1f}s"],
        ["Total Cost", f"${result.cost_usd:.4f}"],
        ["Errors", str(len(result.errors)) if result.errors else "None"],
    ]
    print_table(["Metric", "Value"], status_data)

    if result.errors:
        print("\n  Errors:")
        for err in result.errors[:5]:
            print(f"    - {err}")

    # Task Decomposition
    if result.dag:
        print_section("Task Decomposition (TaskDAG)")
        print(f"  Total Tasks: {len(result.dag.tasks)}")
        print()

        task_rows = []
        for task_id, task in result.dag.tasks.items():
            deps = ", ".join(task.dependencies) if task.dependencies else "None"
            ownership = []
            if task.ownership.allowed_files:
                ownership.extend(task.ownership.allowed_files[:2])
            if task.ownership.allowed_globs:
                ownership.extend(task.ownership.allowed_globs[:2])
            ownership_str = ", ".join(ownership) if ownership else "**/*"

            task_rows.append([
                task_id,
                truncate(task.title, 30),
                task.risk_level,
                deps,
                truncate(ownership_str, 25),
            ])

        print_table(
            ["Task ID", "Title", "Risk", "Dependencies", "Ownership"],
            task_rows,
            max_width=30
        )

        # Task Details
        print("\n  Task Details:")
        for task_id, task in result.dag.tasks.items():
            print(f"\n  [TASK] {task_id}: {task.title}")
            print(f"     Contract: {truncate(task.contract, 70)}")
            print(f"     Description: {truncate(task.description, 70)}")
            if task.oracles:
                oracle_strs = [f"{o.oracle_type.value}:{o.command or 'auto'}" for o in task.oracles]
                print(f"     Oracles: {', '.join(oracle_strs)}")

    # Per-Task Results
    if result.task_results:
        print_section("Agent Results Per Task")

        for task_id, task_result in result.task_results.items():
            print(f"\n  [WORK] Task: {task_id}")
            print(f"     Cost: ${task_result.cost_usd:.4f}")
            print(f"     Selected Patch: {task_result.selected_patch_id or 'None'}")
            print(f"     Candidates: {len(task_result.candidates)}")

            if task_result.errors:
                print(f"     [ERROR] Errors: {task_result.errors}")

            # Show each agent's result
            if task_result.candidates:
                print("\n     Agent Candidates:")
                candidate_rows = []
                for c in task_result.candidates:
                    selected = "[Y]" if c.candidate_id == task_result.selected_patch_id else ""
                    patch_preview = truncate(c.patch.replace('\n', ' '), 40) if c.patch else "(no patch)"
                    candidate_rows.append([
                        selected,
                        c.candidate_id,
                        f"{c.score:.1f}",
                        "[Y]" if c.is_valid else "[N]",
                        str(len(c.validation_errors)),
                        patch_preview,
                    ])

                print_table(
                    ["Sel", "Agent ID", "Score", "Valid", "Errors", "Patch Preview"],
                    candidate_rows,
                    max_width=40
                )

            # Show oracle results if any
            if task_result.oracles:
                print("\n     Oracle Results:")
                for oracle_result in task_result.oracles:
                    status = "[Y] PASS" if oracle_result.success else "[N] FAIL"
                    print(f"       {oracle_result.oracle.oracle_type.value}: {status}")
                    if not oracle_result.success and oracle_result.error_message:
                        print(f"         Error: {truncate(oracle_result.error_message, 60)}")

    # Assembly Result
    if result.assembly:
        print_section("Beam Search Assembly")
        assembly = result.assembly
        print(f"  Success: {'[Y]' if assembly.success else '[N]'}")
        print(f"  Final Score: {assembly.score:.1f}")
        print(f"  Selected Patches: {assembly.selected_patch_ids}")

        if assembly.errors:
            print(f"  Errors: {assembly.errors}")

    # Final Patch
    if result.final_patch:
        print_section("Final Combined Patch")
        print("```diff")
        # Show first 100 lines
        lines = result.final_patch.split('\n')
        for line in lines[:100]:
            print(f"  {line}")
        if len(lines) > 100:
            print(f"  ... ({len(lines) - 100} more lines)")
        print("```")

    # Summary
    print_header("SUMMARY", "═")
    print(f"""
  Status:          {result.status.upper()}
  Tasks Created:   {len(result.dag.tasks) if result.dag else 0}
  Tasks Completed: {len(result.task_results)}
  Total Agents:    {sum(len(tr.candidates) for tr in result.task_results.values())}
  Total Cost:      ${result.cost_usd:.4f}
  Duration:        {elapsed:.1f}s

  The ATLAS system:
  1. [Y] Scouted the repository to understand structure
  2. [Y] Decomposed the problem into {len(result.dag.tasks) if result.dag else 0} atomic tasks
  3. [Y] Ran {exec_config.agents_per_task} diverse agents per task
  4. [Y] Used Context7 + Web Search for documentation
  5. [Y] Assembled patches using beam search (width={exec_config.beam_width})
  6. {'[Y]' if result.final_patch else '[N]'} Generated final combined patch
""")

    # Cleanup
    print(f"\n  Test repository preserved at: {repo_path}")
    print("  (Delete manually when done)")


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
