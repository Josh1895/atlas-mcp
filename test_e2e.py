#!/usr/bin/env python3
"""
End-to-End ATLAS Test
=====================
Tests full pipeline: decomposition -> agent fixing -> patch assembly
Target: Complete in under 5 minutes
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

# Minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Simple Express API with 2 bugs
APP_FILES = {
    "server.js": """const express = require('express');
const app = express();
app.use(express.json());

const users = [];

// BUG 1: No validation - allows empty/invalid emails
app.post('/api/users', (req, res) => {
  const { email, name } = req.body;
  const user = { id: Date.now(), email, name };
  users.push(user);
  res.status(201).json(user);
});

// BUG 2: No validation on update - allows empty name
app.put('/api/users/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const idx = users.findIndex(u => u.id === id);
  if (idx === -1) return res.status(404).json({ error: 'Not found' });

  const { name } = req.body;
  users[idx].name = name;
  res.json(users[idx]);
});

app.listen(3000);
""",
    "package.json": '{"name": "api", "version": "1.0.0"}'
}

TASK = """
Fix the User API validation:

1. POST /api/users: Validate email contains '@' and name is non-empty string.
   Return 400 with { "error": "Invalid email" } or { "error": "Name required" }

2. PUT /api/users/:id: Validate name is non-empty string.
   Return 400 with { "error": "Name required" }
"""


def print_box(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


async def main():
    start = time.time()

    print_box("ATLAS End-to-End Test")

    # Create temp repo
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_e2e_"))
    for name, content in APP_FILES.items():
        (temp_dir / name).write_text(content)
    print(f"Repo: {temp_dir}")
    print(f"Files: {list(APP_FILES.keys())}")

    # Config
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)
    print(f"Model: {config.model}")

    # Orchestrator - optimized for speed
    from atlas.core.dag_orchestrator import (
        TaskDAGOrchestrator,
        TaskExecutionConfig,
        TaskDAGSubmission
    )

    exec_config = TaskExecutionConfig(
        agents_per_task=2,      # 2 agents (fast)
        top_k_per_task=1,       # Take best candidate
        beam_width=2,           # Narrow beam
        enable_quality_selection=False,
    )

    orchestrator = TaskDAGOrchestrator(
        config=config,
        execution_config=exec_config,
        use_agentic=False,      # No tool calls (faster)
    )

    submission = TaskDAGSubmission(
        task_id="e2e_test",
        description=TASK,
        repo_path=temp_dir,
        max_tasks=3,            # Allow up to 3 subtasks
    )

    print_box("Task")
    print(TASK)

    print_box("Running ATLAS")
    print(f"Agents: {exec_config.agents_per_task} per task")
    print(f"Max Tasks: {submission.max_tasks}")
    print()

    result = await orchestrator.solve(submission)

    # Results
    print_box("Task Decomposition")
    if result.dag and result.dag.tasks:
        print(f"Created {len(result.dag.tasks)} tasks:\n")
        for i, (task_id, spec) in enumerate(result.dag.tasks.items(), 1):
            deps = spec.dependencies if spec.dependencies else "none"
            print(f"  {i}. [{task_id}]")
            print(f"     {spec.title}")
            print(f"     Dependencies: {deps}")
            print()
    else:
        print("No decomposition available")

    print_box("Execution Results")
    elapsed = time.time() - start
    print(f"Status: {result.status}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.errors:
        print(f"\nErrors:")
        for e in result.errors[:5]:  # First 5
            print(f"  - {e[:80]}...")

    if result.final_patch:
        print_box("Final Patch")
        print(result.final_patch)
        print_box("SUCCESS")
        print("Patch generated and validated!")
    else:
        print_box("FAILED")
        print("No patch generated")

    print(f"\nTotal time: {elapsed:.1f}s")
    return result.status == "completed" and result.final_patch is not None


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
