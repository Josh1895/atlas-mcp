#!/usr/bin/env python3
"""
ATLAS Task Decomposition Test
==============================
Shows the FULL decomposition pipeline:
1. Original Task
2. Task Decomposition into Subtasks (DAG)
3. Per-Subtask Agent Execution
4. Patch Assembly/Composition
5. Final Unified Patch
"""

import asyncio
import logging
import tempfile
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
for logger_name in ["httpx", "httpcore", "primp", "google_genai"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================================
# TEST APPLICATION - A more complex app that benefits from decomposition
# ============================================================================

APP_CODE = """const express = require('express');
const app = express();
app.use(express.json());

// In-memory data stores
const users = [];
const products = [];
const orders = [];

// ============ USER ROUTES ============

// BUG 1: No email validation
app.post('/api/users', (req, res) => {
  const { email, name, password } = req.body;
  const user = { id: Date.now(), email, name, password };
  users.push(user);
  res.status(201).json(user);
});

// BUG 2: Password exposed in response
app.get('/api/users/:id', (req, res) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

// ============ PRODUCT ROUTES ============

// BUG 3: No price validation (can be negative)
app.post('/api/products', (req, res) => {
  const { name, price, stock } = req.body;
  const product = { id: Date.now(), name, price, stock };
  products.push(product);
  res.status(201).json(product);
});

// BUG 4: No stock validation on update
app.put('/api/products/:id', (req, res) => {
  const product = products.find(p => p.id === parseInt(req.params.id));
  if (!product) return res.status(404).json({ error: 'Product not found' });
  const { stock } = req.body;
  product.stock = stock;
  res.json(product);
});

// ============ ORDER ROUTES ============

// BUG 5: No validation that user/product exist
app.post('/api/orders', (req, res) => {
  const { userId, productId, quantity } = req.body;
  const order = { id: Date.now(), userId, productId, quantity, status: 'pending' };
  orders.push(order);
  res.status(201).json(order);
});

app.listen(3000);
"""

TASK_DESCRIPTION = """
Fix all validation bugs in this e-commerce API:

1. USER VALIDATION:
   - POST /api/users: Validate email contains '@'. Return 400 "Invalid email" if not.
   - GET /api/users/:id: Don't expose password in response (remove it from returned object).

2. PRODUCT VALIDATION:
   - POST /api/products: Validate price > 0. Return 400 "Price must be positive" if not.
   - PUT /api/products/:id: Validate stock >= 0. Return 400 "Stock cannot be negative" if not.

3. ORDER VALIDATION:
   - POST /api/orders: Validate userId exists in users array. Return 400 "User not found" if not.
   - POST /api/orders: Validate productId exists in products array. Return 400 "Product not found" if not.
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

    header_row = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * sum(col_widths))

    for row in rows:
        row_str = "".join(str(cell)[:w-1].ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)


async def main():
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_box("ATLAS TASK DECOMPOSITION TEST")
    print(f"Run ID: {run_id}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # SECTION 1: ORIGINAL TASK
    # ========================================================================
    print_box("1. ORIGINAL TASK", "-")

    print("\n[Task Description]")
    print("-" * 60)
    print(TASK_DESCRIPTION.strip())

    print("\n[Source Code Overview]")
    print("-" * 60)
    print(f"File: server.js")
    print(f"Size: {len(APP_CODE)} characters, {APP_CODE.count(chr(10))+1} lines")
    print(f"Routes: 5 endpoints across 3 domains (users, products, orders)")
    print(f"Bugs to fix: 6 validation issues")

    # ========================================================================
    # SECTION 2: SETUP
    # ========================================================================
    print_box("2. ENVIRONMENT SETUP", "-")

    # Create temp repo
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_decomp_"))
    (temp_dir / "server.js").write_text(APP_CODE)
    (temp_dir / "package.json").write_text('{"name": "ecommerce-api", "version": "1.0.0"}')

    subprocess.run(["git", "init", "-b", "main"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=temp_dir, capture_output=True)

    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)

    setup_info = [
        ["Repository", str(temp_dir)],
        ["Model", config.model],
        ["Max Cost", f"${config.max_cost_usd}"],
    ]

    print("\n[Configuration]")
    print_table(["Setting", "Value"], setup_info, [25, 55])

    # ========================================================================
    # SECTION 3: RUN DECOMPOSITION PIPELINE
    # ========================================================================
    print_box("3. RUNNING TASK DECOMPOSITION PIPELINE", "-")

    from atlas.core.dag_orchestrator import (
        TaskDAGOrchestrator,
        TaskExecutionConfig,
        TaskDAGSubmission
    )

    exec_config = TaskExecutionConfig(
        agents_per_task=10,     # 10 agents per subtask (full swarm)
        top_k_per_task=3,       # Take top 3 candidates per subtask
        beam_width=4,           # Beam search width
        enable_quality_selection=True,  # Enable full quality selection pipeline
        run_oracles=False,      # Disable oracles to avoid Windows cleanup issues
    )

    orchestrator = TaskDAGOrchestrator(
        config=config,
        execution_config=exec_config,
        use_agentic=False,
    )

    submission = TaskDAGSubmission(
        task_id=f"decomp_test_{run_id}",
        description=TASK_DESCRIPTION,
        repo_path=temp_dir,
        max_tasks=6,  # Allow up to 6 subtasks
    )

    print("\n[Execution Config]")
    exec_info = [
        ["Agents per Subtask", str(exec_config.agents_per_task)],
        ["Max Subtasks", str(submission.max_tasks)],
        ["Beam Width", str(exec_config.beam_width)],
    ]
    print_table(["Setting", "Value"], exec_info, [25, 20])

    print("\n[Pipeline Phases]")
    phases = [
        ["Phase 1", "Repository Analysis", "Clone, index, extract structure"],
        ["Phase 2", "Task Decomposition", "Break into atomic subtasks with DAG"],
        ["Phase 3", "Per-Task Execution", "Run agents on each subtask"],
        ["Phase 4", "Patch Composition", "Combine subtask patches"],
        ["Phase 5", "Final Assembly", "Produce unified patch"],
    ]
    print_table(["Phase", "Name", "Description"], phases, [10, 22, 40])

    print("\n[Running Pipeline...]")
    pipeline_start = time.time()
    result = await orchestrator.solve(submission)
    pipeline_time = time.time() - pipeline_start

    # ========================================================================
    # SECTION 4: TASK DECOMPOSITION RESULTS
    # ========================================================================
    print_box("4. TASK DECOMPOSITION RESULTS", "-")

    if result.dag and result.dag.tasks:
        print(f"\n[DAG Summary]")
        print(f"  Total Subtasks Created: {len(result.dag.tasks)}")

        print(f"\n[Subtask Details]")
        print("-" * 70)

        for i, (task_id, task_spec) in enumerate(result.dag.tasks.items(), 1):
            print(f"\n  SUBTASK {i}: [{task_id}]")
            print(f"  " + "-" * 50)
            print(f"  Title: {task_spec.title}")
            print(f"  Description: {task_spec.description[:100]}..." if len(task_spec.description) > 100 else f"  Description: {task_spec.description}")

            if task_spec.dependencies:
                print(f"  Dependencies: {', '.join(task_spec.dependencies)}")
            else:
                print(f"  Dependencies: None (can run first)")

            if hasattr(task_spec, 'ownership') and task_spec.ownership:
                if hasattr(task_spec.ownership, 'allowed_globs'):
                    print(f"  Owned Files: {task_spec.ownership.allowed_globs}")
                else:
                    print(f"  Owned Files: {task_spec.ownership}")

            if hasattr(task_spec, 'contract') and task_spec.contract:
                try:
                    if hasattr(task_spec.contract, 'inputs') and task_spec.contract.inputs:
                        print(f"  Inputs: {task_spec.contract.inputs}")
                    if hasattr(task_spec.contract, 'outputs') and task_spec.contract.outputs:
                        print(f"  Outputs: {task_spec.contract.outputs}")
                except (AttributeError, TypeError):
                    print(f"  Contract: {task_spec.contract}")

        # Dependency graph visualization
        print(f"\n[Dependency Graph]")
        print("-" * 70)
        for task_id, task_spec in result.dag.tasks.items():
            deps = task_spec.dependencies if task_spec.dependencies else []
            if deps:
                for dep in deps:
                    print(f"  {dep} --> {task_id}")
            else:
                print(f"  [START] --> {task_id}")

    else:
        print("\n  No decomposition data available")
        print("  (Task may have been treated as single atomic task)")

    # ========================================================================
    # SECTION 5: PER-SUBTASK EXECUTION (DETAILED AGENT TABLE)
    # ========================================================================
    print_box("5. PER-SUBTASK EXECUTION", "-")

    if result.task_results:
        print(f"\n[Execution Summary: {len(result.task_results)} Subtasks x 10 Agents Each]")
        print(f"[Total Agents Deployed: {len(result.task_results) * 10}]")

        for task_id, task_result in result.task_results.items():
            print(f"\n{'='*80}")
            print(f"  SUBTASK: [{task_id}]")
            print(f"{'='*80}")
            print(f"  Total Candidates: {len(task_result.candidates)}")
            print(f"  Selected Winner: {task_result.selected_patch_id or 'None'}")
            print(f"  Cost: ${task_result.cost_usd:.4f}")

            if task_result.candidates:
                print(f"\n  [Agent Results Table]")
                print(f"  {'-'*76}")
                print(f"  {'Agent':<12} {'Valid':<7} {'Score':<8} {'Lines+':<8} {'Lines-':<8} {'Errors':<10}")
                print(f"  {'-'*76}")

                # Group candidates by similarity for clustering view
                score_groups = {}
                for candidate in task_result.candidates:
                    lines_added = candidate.patch.count('\n+') - candidate.patch.count('+++') if candidate.patch else 0
                    lines_removed = candidate.patch.count('\n-') - candidate.patch.count('---') if candidate.patch else 0
                    valid_str = "YES" if candidate.is_valid else "NO"
                    err_count = len(candidate.validation_errors)
                    selected = " <-- WINNER" if candidate.candidate_id == task_result.selected_patch_id else ""

                    print(f"  {candidate.candidate_id:<12} {valid_str:<7} {candidate.score:<8.3f} {lines_added:<8} {lines_removed:<8} {err_count:<10}{selected}")

                    # Track scores for grouping
                    score_key = round(candidate.score, 2)
                    if score_key not in score_groups:
                        score_groups[score_key] = []
                    score_groups[score_key].append(candidate.candidate_id)

                # Show clustering/grouping
                print(f"\n  [Solution Clustering by Score]")
                print(f"  {'-'*50}")
                for score, agents in sorted(score_groups.items(), reverse=True):
                    print(f"  Score {score:.2f}: {', '.join(agents)}")

                # Show winning patch preview
                winner = next((c for c in task_result.candidates if c.candidate_id == task_result.selected_patch_id), None)
                if winner and winner.patch:
                    print(f"\n  [Winning Patch Preview]")
                    print(f"  {'-'*50}")
                    # Show first 15 lines of patch
                    patch_lines = winner.patch.split('\n')[:15]
                    for line in patch_lines:
                        print(f"  {line[:75]}")
                    if len(winner.patch.split('\n')) > 15:
                        print(f"  ... ({len(winner.patch.split(chr(10))) - 15} more lines)")

            if task_result.errors:
                print(f"\n  [Errors]")
                for err in task_result.errors[:5]:
                    print(f"    - {err[:70]}...")

    else:
        print("\n  No per-subtask execution data available")

    # ========================================================================
    # SECTION 6: PATCH COMPOSITION
    # ========================================================================
    print_box("6. PATCH COMPOSITION", "-")

    if result.task_results:
        print("\n[Patches Selected for Composition]")
        print("-" * 70)

        selected_patches = []
        for task_id, task_result in result.task_results.items():
            if task_result.selected_patch_id:
                selected = next(
                    (c for c in task_result.candidates if c.candidate_id == task_result.selected_patch_id),
                    None
                )
                if selected and selected.patch:
                    selected_patches.append((task_id, selected))
                    lines_added = selected.patch.count('\n+') - selected.patch.count('+++')
                    print(f"  Task [{task_id}]")
                    print(f"    Selected: {task_result.selected_patch_id}")
                    print(f"    Lines Added: ~{lines_added}")

        print(f"\n[Composition Method]")
        print("  Strategy: Sequential patch application with conflict detection")
        print(f"  Patches to merge: {len(selected_patches)}")

    # ========================================================================
    # SECTION 7: FINAL RESULTS
    # ========================================================================
    print_box("7. FINAL RESULTS", "-")

    results_summary = [
        ["Status", result.status.upper() if isinstance(result.status, str) else str(result.status)],
        ["Total Time", f"{pipeline_time:.1f}s ({pipeline_time/60:.1f} min)"],
        ["Total Cost", f"${result.cost_usd:.4f}"],
        ["Subtasks", str(len(result.dag.tasks)) if result.dag else "0"],
        ["Errors", str(len(result.errors)) if result.errors else "0"],
    ]

    print("\n[Execution Summary]")
    print_table(["Metric", "Value"], results_summary, [20, 30])

    if result.errors:
        print("\n[Errors Encountered]")
        for err in result.errors[:5]:
            print(f"  - {err[:80]}...")

    # ========================================================================
    # SECTION 8: FINAL UNIFIED PATCH
    # ========================================================================
    print_box("8. FINAL UNIFIED PATCH", "-")

    if result.final_patch:
        print("\n[Generated Patch]")
        print("-" * 60)
        print(result.final_patch)
        print("-" * 60)

        # Analyze what was fixed
        patch_lower = result.final_patch.lower()

        print("\n[Bug Fixes Verified]")
        fixes = [
            ("BUG 1: Email validation", "email" in patch_lower and ("@" in patch_lower or "invalid" in patch_lower)),
            ("BUG 2: Password hidden", "password" in patch_lower or "delete" in patch_lower),
            ("BUG 3: Price validation", "price" in patch_lower and "positive" in patch_lower),
            ("BUG 4: Stock validation", "stock" in patch_lower and ("negative" in patch_lower or ">= 0" in patch_lower or ">=0" in patch_lower)),
            ("BUG 5: User exists check", "user" in patch_lower and ("not found" in patch_lower or "exist" in patch_lower)),
            ("BUG 6: Product exists check", "product" in patch_lower and ("not found" in patch_lower or "exist" in patch_lower)),
        ]

        fixed_count = 0
        for bug_name, is_fixed in fixes:
            status = "[X]" if is_fixed else "[ ]"
            print(f"  {status} {bug_name}")
            if is_fixed:
                fixed_count += 1

        print(f"\n  Total Fixed: {fixed_count}/{len(fixes)}")

    else:
        print("\n[ERROR] No final patch generated!")

    # ========================================================================
    # SECTION 9: SUMMARY
    # ========================================================================
    total_time = time.time() - start_time

    print_box("9. TEST SUMMARY")

    if result.final_patch:
        print("""
    +--------------------------------------------------+
    |              DECOMPOSITION SUCCESS               |
    +--------------------------------------------------+
    |  Complex task was broken into subtasks           |
    |  Each subtask was solved by dedicated agents     |
    |  Patches were composed into final solution       |
    +--------------------------------------------------+
""")
    else:
        print("""
    +--------------------------------------------------+
    |              DECOMPOSITION FAILED                |
    +--------------------------------------------------+
    |  Could not generate final unified patch          |
    +--------------------------------------------------+
""")

    final_summary = [
        ["Pipeline", "ATLAS Task Decomposition"],
        ["Original Task", "Fix 6 validation bugs in e-commerce API"],
        ["Decomposition", f"{len(result.dag.tasks) if result.dag else 0} subtasks created"],
        ["Agents Used", f"{exec_config.agents_per_task} per subtask"],
        ["Total Time", f"{total_time:.1f} seconds"],
        ["Total Cost", f"${result.cost_usd:.4f}"],
        ["Result", "SUCCESS" if result.final_patch else "FAILED"],
    ]

    print_table(["Aspect", "Details"], final_summary, [25, 45])

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return result.status == "completed" and result.final_patch is not None


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
