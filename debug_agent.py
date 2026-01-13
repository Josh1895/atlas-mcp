#!/usr/bin/env python3
"""Debug MicroAgent to see raw output."""

import asyncio
import logging
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("debug")

TEST_CODE = """const express = require('express');
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
"""

PROBLEM = """
Fix the POST /api/todos endpoint to validate that title is not empty or whitespace.
Return 400 status with { "error": "Title is required" } if invalid.
"""


async def main():
    print("=" * 60)
    print("  Debug MicroAgent Raw Output")
    print("=" * 60)

    # Create temp repo
    temp_dir = Path(tempfile.mkdtemp(prefix="debug_agent_"))
    (temp_dir / "server.js").write_text(TEST_CODE)
    print(f"Repo: {temp_dir}")

    # Load config
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)

    # Create agent
    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import get_style_by_name, PromptStyleName
    from atlas.core.task import TaskSubmission

    agent = MicroAgent(
        agent_id="debug_agent",
        prompt_style=get_style_by_name(PromptStyleName.SENIOR_ENGINEER),
        config=config,
    )

    task = TaskSubmission(
        description=PROBLEM,
        repository_url=str(temp_dir),
    )

    context = AgentContext(
        task=task,
        repository_content=f"=== server.js ===\n{TEST_CODE}",
    )

    # Build and print prompt
    prompt = agent._build_prompt(context)
    system_prompt = agent._build_system_prompt()

    print("\n" + "-" * 60)
    print("SYSTEM PROMPT (first 500 chars):")
    print("-" * 60)
    print(system_prompt[:500] + "...")

    print("\n" + "-" * 60)
    print("USER PROMPT:")
    print("-" * 60)
    print(prompt)

    print("\n" + "-" * 60)
    print("GENERATING RESPONSE...")
    print("-" * 60)

    # Generate directly to see raw output
    from atlas.agents.gemini_client import GeminiClient
    client = GeminiClient(config)

    result = await client.generate_with_retry(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.7,
    )

    print("\n" + "-" * 60)
    print("RAW RESPONSE:")
    print("-" * 60)
    print(result.text)
    print("-" * 60)
    print(f"Tokens: {result.input_tokens + result.output_tokens}")

    # Try extraction
    patch = agent._extract_patch(result.text)
    print("\n" + "-" * 60)
    print("EXTRACTED PATCH:")
    print("-" * 60)
    if patch:
        print(patch)
    else:
        print("[NONE - Extraction failed]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
