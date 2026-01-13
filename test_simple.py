#!/usr/bin/env python3
"""
Simple ATLAS Test - Direct agent testing without full orchestration.
"""

import asyncio
import logging
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test")

# Simple test repo
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
}

SIMPLE_PROBLEM = """
Fix the POST /api/todos endpoint to validate that title is not empty or whitespace.
Return 400 status with { "error": "Title is required" } if invalid.
"""


async def main():
    print("=" * 60)
    print("  ATLAS Simple Test")
    print("=" * 60)

    # Create temp repo
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_simple_"))
    for name, content in TEST_FILES.items():
        (temp_dir / name).write_text(content)
    print(f"\nRepo: {temp_dir}")

    # Load config
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)

    print(f"Model: {config.model}")
    print(f"Gemini Key: {'[OK]' if config.gemini_api_key else '[MISSING]'}")

    if not config.gemini_api_key:
        print("ERROR: No Gemini API key")
        return

    # Test 1: Direct LLM call
    print("\n" + "-" * 60)
    print("  Test 1: Direct LLM Generation")
    print("-" * 60)

    from atlas.agents.gemini_client import GeminiClient
    client = GeminiClient(config)

    prompt = f"""
You are a senior engineer. Fix this bug:

{SIMPLE_PROBLEM}

Current code:
```javascript
{TEST_FILES['server.js']}
```

Output ONLY a unified diff patch:
```diff
--- a/server.js
+++ b/server.js
@@ ... @@
...
```
"""

    try:
        result = await client.generate_with_retry(
            prompt=prompt,
            system_prompt="You are an expert Node.js developer. Output only a diff patch.",
            temperature=0.3,
        )
        print(f"\nTokens: {result.input_tokens + result.output_tokens}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"\nResponse preview:\n{result.text[:1000]}...")

        # Try to extract patch
        import re
        diff_match = re.search(r"```(?:diff)?\s*\n([\s\S]*?)```", result.text)
        if diff_match:
            patch = diff_match.group(1)
            print(f"\n[OK] Extracted patch ({len(patch)} chars)")
            print("-" * 40)
            print(patch[:500])
        else:
            print("\n[WARN] No diff block found in response")

    except Exception as e:
        print(f"\n[ERROR] LLM call failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Task Decomposition
    print("\n" + "-" * 60)
    print("  Test 2: Task Decomposition")
    print("-" * 60)

    from atlas.core.task_decomposer import TaskDecomposer, TaskDecomposerConfig
    from atlas.scout.repo_scout import RepoScout

    scout = RepoScout()
    report = await scout.scan(temp_dir, keywords=["validation", "express"])

    print(f"Languages: {report.languages}")
    print(f"Entrypoints: {report.entrypoints}")

    decomposer = TaskDecomposer(
        llm_client=client,
        config=TaskDecomposerConfig(max_tasks=4),
    )

    try:
        dag = await decomposer.decompose(SIMPLE_PROBLEM, report)
        print(f"\nTasks created: {len(dag.tasks)}")
        for tid, task in dag.tasks.items():
            print(f"  - {tid}: {task.title}")
            print(f"    Contract: {task.contract[:80]}...")
            print(f"    Ownership: files={task.ownership.allowed_files}, globs={task.ownership.allowed_globs}")
    except Exception as e:
        print(f"\n[ERROR] Decomposition failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Single Agent
    print("\n" + "-" * 60)
    print("  Test 3: Single MicroAgent")
    print("-" * 60)

    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import get_style_by_name, PromptStyleName
    from atlas.core.task import TaskSubmission

    agent = MicroAgent(
        agent_id="test_agent",
        prompt_style=get_style_by_name(PromptStyleName.SENIOR_ENGINEER),
        config=config,
    )

    task_sub = TaskSubmission(
        description=SIMPLE_PROBLEM,
        repository_url=str(temp_dir),
        relevant_files=["server.js"],
    )

    context = AgentContext(
        task=task_sub,
        repository_content=f"=== server.js ===\n{TEST_FILES['server.js']}",
    )

    try:
        solution = await agent.generate(context)
        print(f"\nAgent: {solution.agent_id}")
        print(f"Style: {solution.prompt_style}")
        print(f"Tokens: {solution.tokens_used}")
        print(f"Cost: ${solution.cost:.4f}")
        print(f"Valid: {solution.is_valid}")

        if solution.patch:
            print(f"\n[OK] Patch generated ({len(solution.patch)} chars)")
            print("-" * 40)
            print(solution.patch[:600])
        else:
            print(f"\n[WARN] No patch: {solution.validation_errors}")

    except Exception as e:
        print(f"\n[ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
