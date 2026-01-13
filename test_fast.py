#!/usr/bin/env python3
"""Fast ATLAS test - no agentic mode, 2 agents, skip Context7."""

import asyncio
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TEST_FILES = {
    "server.js": """const express = require('express');
const app = express();
app.use(express.json());

let todos = [];

// BUG: No validation - allows empty titles
app.post('/api/todos', (req, res) => {
  const { title } = req.body;
  const todo = { id: Date.now(), title, completed: false };
  todos.push(todo);
  res.status(201).json(todo);
});

app.listen(3000);
""",
    "package.json": '{"name": "todo-api", "version": "1.0.0"}',
}

PROBLEM = """
Fix the POST /api/todos endpoint to validate that title is not empty or whitespace.
Return 400 status with { "error": "Title is required" } if validation fails.
"""


async def main():
    print("=" * 60)
    print("  ATLAS Fast Test (No Agentic Mode)")
    print("=" * 60)

    # Create temp repo
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_fast_"))
    for name, content in TEST_FILES.items():
        (temp_dir / name).write_text(content)
    print(f"Repo: {temp_dir}")

    # Load config
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)
    print(f"Model: {config.model}")

    # Create orchestrator - NO agentic mode, 2 agents
    from atlas.core.dag_orchestrator import TaskDAGOrchestrator, TaskExecutionConfig, TaskDAGSubmission

    exec_config = TaskExecutionConfig(
        agents_per_task=2,  # Just 2 agents (faster)
        top_k_per_task=1,
        beam_width=2,
        enable_quality_selection=False,  # Skip quality scoring
    )

    orchestrator = TaskDAGOrchestrator(
        config=config,
        execution_config=exec_config,
        use_agentic=False,  # NO autonomous tools - much faster
    )

    submission = TaskDAGSubmission(
        task_id="fast_test",
        description=PROBLEM,
        repo_path=temp_dir,
        max_tasks=2,
    )

    print(f"\nSettings:")
    print(f"  Agents: {exec_config.agents_per_task}")
    print(f"  Agentic Mode: OFF (no tool calls)")
    print(f"  Quality Selection: OFF")
    print()

    # Run
    print("-" * 60)
    print("Running...")
    result = await orchestrator.solve(submission)
    print("-" * 60)

    # Results
    print(f"\nStatus: {result.status}")
    print(f"Tasks: {len(result.dag.tasks) if result.dag else 0}")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Time: {result.duration_seconds:.1f}s")

    if result.dag and result.dag.tasks:
        print("\nTask Breakdown:")
        for task_spec in result.dag.tasks.values() if isinstance(result.dag.tasks, dict) else result.dag.tasks:
            if hasattr(task_spec, 'task_id'):
                print(f"  - {task_spec.task_id}: {task_spec.description[:60]}...")
            else:
                print(f"  - Task: {str(task_spec)[:60]}...")

    if result.final_patch:
        print("\n" + "=" * 60)
        print("FINAL PATCH:")
        print("=" * 60)
        print(result.final_patch)
        print("=" * 60)
        print("\n[SUCCESS] Patch generated!")
    else:
        print("\n[FAILED] No patch generated")
        if result.errors:
            print("Errors:")
            for e in result.errors:
                print(f"  - {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
