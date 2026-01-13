#!/usr/bin/env python3
"""Debug test to see raw LLM outputs."""

import asyncio
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

TEST_CODE = """const express = require('express');
const app = express();
app.use(express.json());

let todos = [];

// BUG: No validation
app.post('/api/todos', (req, res) => {
  const { title } = req.body;
  todos.push({ id: Date.now(), title, completed: false });
  res.status(201).json(todos[todos.length - 1]);
});

app.listen(3000);
"""

PROBLEM = "Fix POST /api/todos to validate title is not empty. Return 400 with {error: 'Title required'} if invalid."


async def main():
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)

    print("=" * 60)
    print("DEBUG TEST")
    print("=" * 60)

    # Create temp repo
    temp = Path(tempfile.mkdtemp())
    (temp / "server.js").write_text(TEST_CODE)
    (temp / "package.json").write_text('{"name": "test", "scripts": {"test": "jest"}}')

    # Test 1: Task Decomposition - show raw output
    print("\n[1] TASK DECOMPOSITION - Raw LLM Output")
    print("-" * 60)

    from atlas.agents.gemini_client import GeminiClient
    from atlas.core.task_decomposer import TaskDecomposer, TaskDecomposerConfig
    from atlas.scout.repo_scout import RepoScout

    client = GeminiClient(config)
    scout = RepoScout()
    report = await scout.scan(temp)

    decomposer = TaskDecomposer(client, TaskDecomposerConfig(max_tasks=3))

    # Get raw prompt and response
    prompt = decomposer._build_prompt(PROBLEM, report)
    print(f"Prompt length: {len(prompt)} chars")

    result = await client.generate_with_retry(
        prompt=prompt,
        system_prompt="You are a senior staff engineer decomposing work into verified tasks.",
        temperature=0.2,
    )

    print(f"\nRaw LLM Response ({len(result.text)} chars):")
    print("-" * 40)
    print(result.text[:2000])
    print("-" * 40)

    # Try to parse it
    parsed = decomposer._parse_json(result.text)
    if parsed:
        print(f"\n[OK] Parsed JSON successfully!")
        print(f"Tasks: {len(parsed.get('tasks', []))}")
        for t in parsed.get("tasks", []):
            print(f"  - {t.get('id')}: {t.get('title')}")
    else:
        print("\n[FAIL] Could not parse JSON")

        # Debug: show what we're trying to parse
        import re
        fenced = re.search(r"```json\s*(\{[\s\S]*\})\s*```", result.text)
        if fenced:
            print(f"\nFound fenced JSON block ({len(fenced.group(1))} chars)")
            print(fenced.group(1)[:500])
        else:
            print("\nNo ```json block found")
            start = result.text.find("{")
            if start != -1:
                print(f"First {{ at position {start}")
                print(result.text[start:start+500])

    # Test 2: Single Agent Patch Generation
    print("\n\n[2] SINGLE AGENT - Patch Generation")
    print("-" * 60)

    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import get_style_by_name, PromptStyleName
    from atlas.core.task import TaskSubmission

    agent = MicroAgent(
        agent_id="debug_agent",
        prompt_style=get_style_by_name(PromptStyleName.SENIOR_ENGINEER),
        config=config,
    )

    task = TaskSubmission(description=PROBLEM, repository_url=str(temp))
    context = AgentContext(task=task, repository_content=TEST_CODE)

    solution = await agent.generate(context)

    print(f"Agent: {solution.agent_id}")
    print(f"Valid: {solution.is_valid}")
    print(f"Errors: {solution.validation_errors}")

    if solution.patch:
        print(f"\nPatch ({len(solution.patch)} chars):")
        print("-" * 40)
        print(solution.patch)
        print("-" * 40)

        # Test validation
        from atlas.verification.patch_validator import PatchValidator
        validator = PatchValidator()
        result = validator.validate(solution.patch)
        print(f"\nPatch Validation:")
        print(f"  is_valid: {result.is_valid}")
        print(f"  can_apply: {result.can_apply}")
        print(f"  errors: {result.errors}")
        print(f"  warnings: {result.warnings}")
    else:
        print("\n[FAIL] No patch generated")
        print(f"Explanation: {solution.explanation[:500]}")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
