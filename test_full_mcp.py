#!/usr/bin/env python3
"""
ATLAS Full MCP Test - 10 Agents with Quality Selection
=======================================================
Tests the complete pipeline as if called via MCP:
- All 10 agents (2 per style × 5 styles)
- Full quality selection/voting
- Shows which agent won and why
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

# Logging - show important events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Simple API with 2 bugs to fix
APP_CODE = """const express = require('express');
const app = express();
app.use(express.json());

const items = [];

// BUG 1: No validation on POST - allows empty name
app.post('/api/items', (req, res) => {
  const { name, price } = req.body;
  const item = { id: Date.now(), name, price };
  items.push(item);
  res.status(201).json(item);
});

// BUG 2: No validation on PUT - allows negative price
app.put('/api/items/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const idx = items.findIndex(i => i.id === id);
  if (idx === -1) return res.status(404).json({ error: 'Not found' });

  const { name, price } = req.body;
  items[idx] = { ...items[idx], name, price };
  res.json(items[idx]);
});

app.listen(3000);
"""

TASK_DESCRIPTION = """
Fix validation bugs in the Items API:

1. POST /api/items: Validate that 'name' is a non-empty string.
   Return 400 with { "error": "Name is required" } if invalid.

2. PUT /api/items/:id: Validate that 'price' is a positive number (> 0).
   Return 400 with { "error": "Price must be positive" } if invalid.
"""


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


async def main():
    start = time.time()
    print_header("ATLAS FULL MCP TEST - 10 Agents with Quality Selection")

    # Create temp repo with git init
    import subprocess
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_mcp_"))
    (temp_dir / "server.js").write_text(APP_CODE)
    (temp_dir / "package.json").write_text('{"name": "items-api", "version": "1.0.0"}')

    # Initialize git repo with main branch
    subprocess.run(["git", "init", "-b", "main"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=temp_dir, capture_output=True)

    print(f"Repository: {temp_dir}")

    # Initialize config
    from atlas.core.config import Config, set_config
    config = Config.from_env()

    # Override for full test - 10 agents
    config.initial_samples = 10  # Start with 10 agents
    config.max_samples = 10      # Max 10 agents
    config.voting_k = 3          # Top 3 for voting

    set_config(config)

    print(f"Model: {config.model}")
    print(f"Agents: {config.initial_samples} (2 per style × 5 styles)")
    print(f"Voting K: {config.voting_k}")

    print_section("Task Description")
    print(TASK_DESCRIPTION)

    # Use the full ATLASOrchestrator
    from atlas.core.orchestrator import ATLASOrchestrator
    from atlas.core.task import TaskSubmission

    orchestrator = ATLASOrchestrator(
        config=config,
        enable_quality_selection=True,   # FULL quality selection
        enable_test_execution=False,     # No test command provided
        use_agentic=False,               # Pre-fetch RAG (faster for demo)
    )

    task = TaskSubmission(
        description=TASK_DESCRIPTION,
        repository_url=str(temp_dir),
        relevant_files=["server.js"],
        max_cost_usd=5.0,
        timeout_minutes=10,
        voting_k=config.voting_k,
        initial_samples=config.initial_samples,
        max_samples=config.max_samples,
    )

    print_header("Running ATLAS Pipeline")
    print("  Phase 1: Repository Analysis")
    print("  Phase 2: Deploying 10 Diverse Agents")
    print("  Phase 3: Patch Generation")
    print("  Phase 4: Validation & Clustering")
    print("  Phase 5: Quality Selection (Voting)")
    print("  Phase 6: Final Patch Assembly")
    print()

    # Enable debug logging for validation
    import logging
    logging.getLogger("atlas.core.orchestrator").setLevel(logging.DEBUG)
    logging.getLogger("atlas.verification").setLevel(logging.DEBUG)

    # Run the full pipeline
    result = await orchestrator.solve(task)

    elapsed = time.time() - start

    # Display Results
    print_header("Results")
    print(f"Status: {result.status.value}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total Cost: ${result.cost_usd:.4f}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Samples Generated: {result.samples_generated}")
    print(f"Votes Cast: {result.votes_cast}")

    # Agent Results
    print_section("Agent Performance")

    if result.execution_trace and result.execution_trace.agent_outputs:
        agents = result.execution_trace.agent_outputs
        print(f"\nTotal Agents Run: {len(agents)}")
        print(f"{'Agent ID':<20} {'Style':<25} {'Valid':<8} {'Tokens':<10}")
        print("-" * 70)

        valid_count = 0
        for output in agents:
            agent_id = output.get('agent_id', 'unknown')
            style = output.get('prompt_style', 'unknown')
            valid = output.get('is_valid', False)
            tokens = output.get('tokens', 0)
            errors = output.get('errors', [])
            valid_str = "YES" if valid else "NO"
            print(f"{agent_id:<20} {style:<25} {valid_str:<8} {tokens:<10}")
            if errors:
                print(f"    Errors: {errors[0][:60] if errors else 'none'}...")
            if valid:
                valid_count += 1

        print(f"\nValid Patches: {valid_count}/{len(agents)}")

    # Voting Results
    print_section("Voting Results")

    if result.execution_trace and result.execution_trace.voting_rounds:
        for i, vote_round in enumerate(result.execution_trace.voting_rounds, 1):
            print(f"\nRound {i}:")
            if 'winner' in vote_round:
                print(f"  Winner: {vote_round['winner']}")
            if 'scores' in vote_round:
                print(f"  Top Scores:")
                sorted_scores = sorted(vote_round['scores'].items(), key=lambda x: x[1], reverse=True)
                for agent, score in sorted_scores[:5]:
                    print(f"    {agent}: {score:.3f}")
    else:
        print("No voting data available")

    # Quality Selection
    print_section("Quality Selection")

    print(f"Consensus Reached: {result.consensus_reached}")
    if result.confidence_score > 0:
        print(f"Confidence Score: {result.confidence_score:.2%}")

    # Final Patch
    if result.patch:
        print_header("FINAL PATCH")
        print(result.patch)
        print_header("SUCCESS")
        print("Patch generated via multi-agent consensus!")
    else:
        print_header("FAILED - No Patch Generated")
        if result.execution_trace and result.execution_trace.errors:
            print("Errors:")
            for err in result.execution_trace.errors[:5]:
                print(f"  - {err}")

    print(f"\nTotal Time: {elapsed:.1f}s")
    return result.status.value == "completed" and result.patch is not None


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
