#!/usr/bin/env python3
"""Test the full ATLAS MCP pipeline.

This test simulates what happens when an AI assistant calls the solve_issue tool.
It uses the real orchestrator with agentic agents that autonomously search
Context7 and the web.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.core.task import TaskSubmission


BUGGY_CODE = '''import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG 1: Never initialized as asyncio.Lock()

    async def acquire(self):
        # BUG 2: Race condition - no lock protection!
        now = time.time()
        self.tokens += (now - self.last_update) * self.rate
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def wait_for_token(self):
        while not await self.acquire():
            time.sleep(0.01)  # BUG 3: Blocking call in async!
        return True
'''


def create_test_repo(temp_dir: Path) -> tuple[str, str]:
    """Create a temporary git repository with the buggy code.

    Returns:
        Tuple of (repo_path, branch_name)
    """
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Write the buggy code
    code_file = repo_dir / "rate_limiter.py"
    code_file.write_text(BUGGY_CODE)

    # Initialize git repo with main branch
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

    # Get the actual branch name
    result = subprocess.run(["git", "branch", "--show-current"], cwd=repo_dir, capture_output=True, text=True)
    branch = result.stdout.strip() or "main"

    return str(repo_dir), branch


async def main():
    print("=" * 100)
    print(" ATLAS MCP PIPELINE TEST")
    print(" Full End-to-End Test: Agentic Agents -> Clustering -> Voting -> Consensus")
    print("=" * 100)

    config = Config.from_env()

    # Validate config
    errors = config.validate()
    if errors:
        print(f"\nConfiguration errors: {errors}")
        return

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Voting K: {config.voting_k}")
    print(f"  Max Samples: {config.max_samples}")
    print(f"  Has Gemini Key: {bool(config.gemini_api_key)}")

    # Create a temporary directory for the test repo
    temp_dir = Path(tempfile.mkdtemp())
    try:
        print(f"\nCreating test repository in: {temp_dir}")
        repo_path, branch = create_test_repo(temp_dir)
        print(f"Test repo created at: {repo_path}")
        print(f"Branch: {branch}")

        # Create the task submission
        task = TaskSubmission(
            description="""Fix the rate_limiter.py file which has 3 bugs:

1. self._lock = None is never initialized as an actual asyncio.Lock()
2. Race condition - multiple coroutines can access self.tokens simultaneously without synchronization
3. time.sleep(0.01) blocks the event loop - should use asyncio.sleep()

Generate a unified diff patch to fix all 3 bugs. Make sure to:
- Initialize self._lock as asyncio.Lock() in __init__
- Use async with self._lock to protect the token access
- Replace time.sleep with asyncio.sleep""",
            repository_url=repo_path,  # Local git repo
            branch=branch,
            relevant_files=["rate_limiter.py"],
            max_cost_usd=1.0,
            timeout_minutes=10,
            initial_samples=5,  # Start with 5 agents
            max_samples=10,     # Max 10 total
            voting_k=3,         # Consensus when lead by 3
        )

        print(f"\nTask ID: {task.task_id}")
        print(f"Initial samples: {task.initial_samples}")
        print(f"Max samples: {task.max_samples}")
        print(f"Voting K: {task.voting_k}")

        # Create orchestrator with agentic mode
        print("\n" + "=" * 100)
        print(" STARTING ORCHESTRATOR")
        print(" Agents will autonomously search Context7 AND the web")
        print("=" * 100)

        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=True,  # Enable autonomous tool access
        )

        print("\nRunning solve()...")
        print("(This will take a few minutes as agents research and generate patches)\n")

        try:
            result = await asyncio.wait_for(
                orchestrator.solve(task),
                timeout=task.timeout_minutes * 60,
            )

            # Display results
            print("\n" + "=" * 100)
            print(" RESULTS")
            print("=" * 100)

            print(f"\nStatus: {result.status.value}")
            print(f"Consensus Reached: {result.consensus_reached}")
            print(f"Confidence Score: {result.confidence_score:.2%}")
            print(f"Samples Generated: {result.samples_generated}")
            print(f"Cost: ${result.cost_usd:.4f}")
            print(f"Duration: {result.duration_seconds:.1f}s")

            if result.patch:
                print(f"\nWinning Patch ({len(result.patch)} chars):")
                print("-" * 80)
                print(result.patch)
                print("-" * 80)

                # Check patterns
                patch_lower = result.patch.lower()
                patterns = {
                    "asyncio.Lock": "asyncio.lock" in patch_lower,
                    "asyncio.sleep": "asyncio.sleep" in patch_lower,
                    "async with": "async with" in patch_lower,
                }
                print("\nPatterns in patch:")
                for pattern, found in patterns.items():
                    status = "YES" if found else "NO"
                    print(f"  {pattern}: {status}")
            else:
                print("\nNo patch generated!")
                if result.error_message:
                    print(f"Error: {result.error_message}")

            # Show execution trace
            if result.execution_trace:
                trace = result.execution_trace
                print(f"\nExecution Trace:")
                print(f"  Phases: {[p['phase'] for p in trace.phases]}")
                print(f"  Agent outputs: {len(trace.agent_outputs)}")
                print(f"  Voting rounds: {len(trace.voting_rounds)}")

                if trace.voting_rounds:
                    print(f"\n  Last voting round:")
                    last_round = trace.voting_rounds[-1]
                    print(f"    Clusters: {last_round.get('clusters', {})}")
                    print(f"    Winner: {last_round.get('winner')}")
                    print(f"    Consensus: {last_round.get('consensus_reached')}")

                if trace.errors:
                    print(f"\n  Errors: {trace.errors}")

        except asyncio.TimeoutError:
            print(f"\nTask timed out after {task.timeout_minutes} minutes")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 100)
    print(" TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
