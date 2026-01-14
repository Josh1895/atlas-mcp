#!/usr/bin/env python3
"""End-to-end test without Context7 RAG to avoid rate limits."""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Disable Context7 for this test
os.environ["CONTEXT7_API_KEY"] = ""

from atlas.core.config import Config
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.core.task import TaskSubmission


BUGGY_CODE = '''import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG: Never initialized as asyncio.Lock()

    async def acquire(self):
        # BUG: No lock protection - race condition!
        now = time.time()
        self.tokens += (now - self.last_update) * self.rate
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def wait_for_token(self):
        while not await self.acquire():
            time.sleep(0.01)  # BUG: Blocking call in async!
        return True
'''


def create_test_repo(temp_dir: Path) -> tuple[str, str]:
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    code_file = repo_dir / "rate_limiter.py"
    code_file.write_text(BUGGY_CODE)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_dir, capture_output=True)
    result = subprocess.run(["git", "branch", "--show-current"], cwd=repo_dir, capture_output=True, text=True)
    return str(repo_dir), result.stdout.strip() or "main"


async def main():
    print("=" * 70)
    print(" ATLAS E2E TEST - No Context7 RAG")
    print(" 3 agents, direct generation, voting consensus")
    print("=" * 70)

    config = Config.from_env()

    # Force disable Context7
    config.context7_api_key = ""

    errors = config.validate()
    if errors:
        print(f"Config errors (expected Context7 warning): {errors}")

    print(f"\nConfig:")
    print(f"  Model: {config.model}")
    print(f"  Voting K: {config.voting_k}")
    print(f"  Context7 disabled: {not config.context7_api_key}")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        repo_path, branch = create_test_repo(temp_dir)
        print(f"  Repo: {repo_path}")
        print(f"  Branch: {branch}")

        task = TaskSubmission(
            description="""Fix rate_limiter.py - 3 bugs:
1. self._lock = None should be asyncio.Lock()
2. Race condition - need async with self._lock in acquire()
3. time.sleep blocks - use asyncio.sleep instead

Generate a unified diff patch that fixes all three bugs.""",
            repository_url=repo_path,
            branch=branch,
            relevant_files=["rate_limiter.py"],
            max_cost_usd=0.50,
            timeout_minutes=5,
            initial_samples=3,
            max_samples=3,
            voting_k=2,
        )

        print(f"\nTask: {task.task_id}")
        print(f"Agents: {task.initial_samples}")

        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=False,  # No tool calls - direct generation
        )

        print("\n" + "=" * 70)
        print(" RUNNING ORCHESTRATOR")
        print("=" * 70)
        print("\nGenerating patches from 3 diverse agents...")

        result = await asyncio.wait_for(
            orchestrator.solve(task),
            timeout=300,
        )

        print("\n" + "=" * 70)
        print(" RESULTS")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Consensus: {result.consensus_reached}")
        print(f"Confidence: {result.confidence_score:.0%}")
        print(f"Samples: {result.samples_generated}")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        if result.patch:
            print(f"\n{'='*70}")
            print(" WINNING PATCH")
            print("=" * 70)
            print(result.patch)
            print("=" * 70)

            # Check key fixes
            checks = {
                "asyncio.Lock": "asyncio.Lock" in result.patch,
                "asyncio.sleep": "asyncio.sleep" in result.patch,
                "async with": "async with" in result.patch,
            }
            print("\nFix patterns found:")
            all_found = True
            for k, v in checks.items():
                status = "YES" if v else "NO"
                print(f"  {k}: {status}")
                if not v:
                    all_found = False

            if all_found:
                print("\n ALL BUGS FIXED!")
            else:
                print("\n Some bugs not addressed in winning patch")
        else:
            print(f"\nNo patch generated!")
            if result.error_message:
                print(f"Error: {result.error_message}")

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
