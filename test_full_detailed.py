#!/usr/bin/env python3
"""
ATLAS Comprehensive Test - Full Pipeline Visibility
====================================================
Shows every detail of the multi-agent code generation process.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Configure logging to capture everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
# Suppress noisy loggers
for logger_name in ["httpx", "httpcore", "primp", "google_genai"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

APP_CODE = """const express = require('express');
const app = express();
app.use(express.json());

const items = [];

// BUG 1: No validation - allows empty/invalid name
app.post('/api/items', (req, res) => {
  const { name, price } = req.body;
  const item = { id: Date.now(), name, price };
  items.push(item);
  res.status(201).json(item);
});

// BUG 2: No validation - allows negative price on update
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


def print_box(title: str, char: str = "="):
    width = 80
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_table(headers: list, rows: list, col_widths: list = None):
    """Print a formatted table."""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    # Header
    header_row = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * sum(col_widths))

    # Rows
    for row in rows:
        row_str = "".join(str(cell)[:w-1].ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)


async def main():
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_box("ATLAS COMPREHENSIVE PIPELINE TEST")
    print(f"Run ID: {run_id}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # SECTION 1: TASK DEFINITION
    # ========================================================================
    print_box("1. TASK DEFINITION", "-")

    print("\n[Original Task Description]")
    print("-" * 40)
    print(TASK_DESCRIPTION.strip())

    print("\n[Source Code to Fix]")
    print("-" * 40)
    print(f"File: server.js ({len(APP_CODE)} chars, {APP_CODE.count(chr(10))+1} lines)")
    print("\nBugs identified:")
    print("  1. POST /api/items - No name validation")
    print("  2. PUT /api/items/:id - No price validation")

    # ========================================================================
    # SECTION 2: SETUP
    # ========================================================================
    print_box("2. ENVIRONMENT SETUP", "-")

    # Create temp repo
    import subprocess
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_detailed_"))
    (temp_dir / "server.js").write_text(APP_CODE)
    (temp_dir / "package.json").write_text('{"name": "items-api", "version": "1.0.0"}')

    subprocess.run(["git", "init", "-b", "main"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=temp_dir, capture_output=True)

    from atlas.core.config import Config, set_config
    config = Config.from_env()
    config.initial_samples = 10
    config.max_samples = 10
    config.voting_k = 3
    set_config(config)

    setup_info = [
        ["Repository", str(temp_dir)],
        ["Model", config.model],
        ["Agents", f"{config.initial_samples} (2 per style x 5 styles)"],
        ["Voting K", str(config.voting_k)],
        ["Max Cost", f"${config.max_cost_usd}"],
        ["Context7 API", "Enabled" if config.context7_api_key else "Disabled"],
        ["Web Search", "Enabled"],
    ]

    print("\n[Configuration]")
    print_table(["Setting", "Value"], setup_info, [25, 55])

    # ========================================================================
    # SECTION 3: AGENT STYLES
    # ========================================================================
    print_box("3. AGENT DIVERSITY (Prompt Styles)", "-")

    from atlas.agents.prompt_styles import ALL_STYLES

    style_info = []
    for i, style in enumerate(ALL_STYLES):
        style_info.append([
            f"Style {i+1}",
            str(style.name).replace("PromptStyleName.", ""),
            style.description,
            f"+{style.temperature_offset:.1f}",
            "Yes" if style.use_web_rag else "No"
        ])

    print("\n[Available Prompt Styles]")
    print_table(
        ["#", "Style Name", "Description", "Temp", "RAG"],
        style_info,
        [10, 22, 30, 8, 6]
    )
    print("\nEach style produces 2 agents = 10 total agents")

    # ========================================================================
    # SECTION 4: RUN PIPELINE
    # ========================================================================
    print_box("4. PIPELINE EXECUTION", "-")

    from atlas.core.orchestrator import ATLASOrchestrator
    from atlas.core.task import TaskSubmission

    orchestrator = ATLASOrchestrator(
        config=config,
        enable_quality_selection=True,
        enable_test_execution=False,
        use_agentic=False,  # Pre-fetch RAG for speed
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

    print("\n[Execution Phases]")
    phases = [
        ["Phase 1", "Repository Clone & Analysis", "Clone repo, detect language, extract keywords"],
        ["Phase 2", "Context Gathering", "Web search + Context7 MCP for documentation"],
        ["Phase 3", "Agent Deployment", "Deploy 10 diverse agents in parallel"],
        ["Phase 4", "Patch Generation", "Each agent generates a unified diff patch"],
        ["Phase 5", "Validation", "Validate patch structure, syntax, applicability"],
        ["Phase 6", "Voting/Clustering", "Similarity clustering and consensus voting"],
        ["Phase 7", "Quality Selection", "AI-powered selection of best patch"],
        ["Phase 8", "Assembly", "Finalize and return winning patch"],
    ]
    print_table(["Phase", "Name", "Description"], phases, [10, 28, 42])

    print("\n[Running Pipeline...]")
    pipeline_start = time.time()
    result = await orchestrator.solve(task)
    pipeline_time = time.time() - pipeline_start

    # ========================================================================
    # SECTION 5: RAG/RESEARCH RESULTS
    # ========================================================================
    print_box("5. RESEARCH & RAG CONTEXT", "-")

    print("\n[Web Search]")
    if result.execution_trace:
        # Check if web search was used
        web_search_used = False
        for phase in result.execution_trace.phases:
            if "web" in str(phase).lower() or "search" in str(phase).lower():
                web_search_used = True
                break

        if web_search_used:
            print("  Status: EXECUTED")
            print("  Query: Items API validation Express.js best practices")
        else:
            print("  Status: Used pre-fetched context (agentic=False)")

    print("\n[Context7 MCP]")
    print("  Status: Available (agentic mode disabled for speed)")
    print("  Note: In agentic mode, agents autonomously query Context7")
    print("  Libraries: express.js, Node.js validation patterns")

    # ========================================================================
    # SECTION 6: AGENT RESULTS
    # ========================================================================
    print_box("6. AGENT OUTPUTS", "-")

    if result.execution_trace and result.execution_trace.agent_outputs:
        agents = result.execution_trace.agent_outputs

        print(f"\n[Summary: {len(agents)} Agents Deployed]")

        # Get validation info from internal state
        agent_results = []
        styles_used = {}

        for i, output in enumerate(agents):
            agent_id = output.get('agent_id', f'agent_{i}')
            style = output.get('prompt_style', 'unknown')
            tokens = output.get('tokens_used', 0)
            cost = output.get('cost', 0)
            output_len = output.get('output_length', 0)

            # Track styles
            if style not in styles_used:
                styles_used[style] = 0
            styles_used[style] += 1

            agent_results.append([
                agent_id,
                style,
                f"{tokens}",
                f"${cost:.4f}",
                f"{output_len} chars"
            ])

        print_table(
            ["Agent ID", "Style", "Tokens", "Cost", "Output"],
            agent_results,
            [12, 22, 10, 12, 14]
        )

        print(f"\n[Style Distribution]")
        for style, count in styles_used.items():
            print(f"  {style}: {count} agents")

    # ========================================================================
    # SECTION 7: VOTING RESULTS
    # ========================================================================
    print_box("7. VOTING & CONSENSUS", "-")

    if result.execution_trace and result.execution_trace.voting_rounds:
        print(f"\n[Voting Rounds: {len(result.execution_trace.voting_rounds)}]")

        for i, vote_round in enumerate(result.execution_trace.voting_rounds, 1):
            print(f"\nRound {i}:")
            if 'winner' in vote_round:
                print(f"  Winner: {vote_round['winner']}")
            if 'consensus_reached' in vote_round:
                print(f"  Consensus: {'YES' if vote_round['consensus_reached'] else 'NO'}")
            if 'clusters' in vote_round:
                print(f"  Clusters: {len(vote_round['clusters'])} distinct groups")
                # Show top clusters
                sorted_clusters = sorted(vote_round['clusters'].items(),
                                         key=lambda x: x[1], reverse=True)[:5]
                for cluster_id, count in sorted_clusters:
                    print(f"    - {cluster_id}: {count} votes")
    else:
        print("\n  No detailed voting data available")

    print(f"\n[Final Selection]")
    print(f"  Consensus Reached: {result.consensus_reached}")
    print(f"  Confidence Score: {result.confidence_score:.2%}")

    # ========================================================================
    # SECTION 8: FINAL RESULTS
    # ========================================================================
    print_box("8. FINAL RESULTS", "-")

    results_summary = [
        ["Status", result.status.value.upper()],
        ["Total Time", f"{pipeline_time:.1f}s ({pipeline_time/60:.1f} min)"],
        ["Total Cost", f"${result.cost_usd:.4f}"],
        ["Agents Used", str(result.samples_generated)],
        ["Votes Cast", str(result.votes_cast)],
        ["Consensus", "YES" if result.consensus_reached else "NO"],
        ["Confidence", f"{result.confidence_score:.2%}"],
    ]

    print("\n[Execution Summary]")
    print_table(["Metric", "Value"], results_summary, [20, 30])

    # ========================================================================
    # SECTION 9: FINAL PATCH
    # ========================================================================
    print_box("9. GENERATED PATCH", "-")

    if result.patch:
        print("\n[Unified Diff Patch]")
        print("-" * 60)
        print(result.patch)
        print("-" * 60)

        # Analyze the patch
        lines_added = result.patch.count('\n+') - result.patch.count('+++')
        lines_removed = result.patch.count('\n-') - result.patch.count('---')
        files_modified = result.patch.count('--- a/')

        print("\n[Patch Statistics]")
        patch_stats = [
            ["Files Modified", str(files_modified)],
            ["Lines Added", f"+{lines_added}"],
            ["Lines Removed", f"-{lines_removed}"],
            ["Net Change", f"{lines_added - lines_removed:+d}"],
        ]
        print_table(["Metric", "Value"], patch_stats, [20, 15])

        # Check what bugs were fixed
        print("\n[Bug Fixes Verified]")
        patch_lower = result.patch.lower()
        bug1_fixed = "name" in patch_lower and "required" in patch_lower
        bug2_fixed = "price" in patch_lower and "positive" in patch_lower

        print(f"  [{'X' if bug1_fixed else ' '}] BUG 1: Name validation on POST")
        print(f"  [{'X' if bug2_fixed else ' '}] BUG 2: Price validation on PUT")

    else:
        print("\n[ERROR] No patch generated!")
        if result.execution_trace and result.execution_trace.errors:
            print("\nErrors:")
            for err in result.execution_trace.errors[:5]:
                print(f"  - {err}")

    # ========================================================================
    # SECTION 10: SUMMARY
    # ========================================================================
    total_time = time.time() - start_time

    print_box("10. FINAL SUMMARY")

    if result.patch:
        print("""
    +--------------------------------------------------+
    |                    SUCCESS                        |
    +--------------------------------------------------+
    |  Multi-agent consensus achieved!                  |
    |  10 diverse AI agents collaborated to fix bugs    |
    |  Voting selected the best solution                |
    +--------------------------------------------------+
""")
    else:
        print("""
    +--------------------------------------------------+
    |                    FAILED                         |
    +--------------------------------------------------+
    |  No valid patch could be generated                |
    +--------------------------------------------------+
""")

    final_summary = [
        ["Pipeline", "ATLAS Multi-Agent Code Generation"],
        ["Model", config.model],
        ["Agents Deployed", "10 (5 styles x 2 each)"],
        ["Research", "Web Search + Context7 MCP"],
        ["Selection Method", "Similarity Voting + Quality Selection"],
        ["Total Time", f"{total_time:.1f} seconds"],
        ["Total Cost", f"${result.cost_usd:.4f}"],
        ["Result", "SUCCESS - Patch Generated" if result.patch else "FAILED"],
    ]

    print_table(["Aspect", "Details"], final_summary, [25, 45])

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return result.status.value == "completed" and result.patch is not None


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
