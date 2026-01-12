#!/usr/bin/env python3
"""Fast test - 3 agents, no agentic tool calls."""

import asyncio
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
    print("=" * 60)
    print(" FAST TEST - 3 agents, NO agentic tool calls")
    print("=" * 60)

    config = Config.from_env()
    if config.validate():
        print(f"Config errors: {config.validate()}")
        return

    temp_dir = Path(tempfile.mkdtemp())
    try:
        repo_path, branch = create_test_repo(temp_dir)
        print(f"Repo: {repo_path}, Branch: {branch}")

        task = TaskSubmission(
            description="""Fix rate_limiter.py - 3 bugs:
1. self._lock = None should be asyncio.Lock()
2. Race condition - need async with self._lock
3. time.sleep blocks - use asyncio.sleep""",
            repository_url=repo_path,
            branch=branch,
            relevant_files=["rate_limiter.py"],
            max_cost_usd=0.50,
            timeout_minutes=5,
            initial_samples=3,  # Only 3 agents
            max_samples=3,
            voting_k=2,
        )

        print(f"\nTask: {task.task_id}")
        print(f"Agents: {task.initial_samples}")

        # NON-AGENTIC mode - direct generation, no tool calls
        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=False,  # FAST - no tool calls
        )

        print("\nRunning (should take ~30-60 seconds)...\n")

        result = await asyncio.wait_for(
            orchestrator.solve(task),
            timeout=300,
        )

        print("\n" + "=" * 60)
        print(" RESULTS")
        print("=" * 60)
        print(f"Status: {result.status.value}")
        print(f"Consensus: {result.consensus_reached}")
        print(f"Confidence: {result.confidence_score:.0%}")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        if result.patch:
            print(f"\nPatch ({len(result.patch)} chars):")
            print("-" * 40)
            print(result.patch[:1500])
            if len(result.patch) > 1500:
                print("... (truncated)")
            print("-" * 40)

            # Check key fixes
            checks = {
                "asyncio.Lock": "asyncio.Lock" in result.patch,
                "asyncio.sleep": "asyncio.sleep" in result.patch,
                "async with": "async with" in result.patch,
            }
            print("\nFix patterns found:")
            for k, v in checks.items():
                print(f"  {k}: {'YES' if v else 'NO'}")
        else:
            print(f"\nNo patch! Error: {result.error_message}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
