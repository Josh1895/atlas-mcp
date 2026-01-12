#!/usr/bin/env python3
"""Full test with detailed logging of what each agent searches and receives."""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Monkey-patch to capture tool calls per agent
AGENT_TOOL_LOG = {}  # agent_id -> list of {tool, args, result_preview}

from atlas.agents import agentic_client
original_execute_tool = agentic_client.AgenticGeminiClient._execute_tool

async def logged_execute_tool(self, name: str, args: dict) -> str:
    """Wrapper that logs tool calls."""
    result = await original_execute_tool(self, name, args)

    # Get current agent from call stack (hacky but works for logging)
    import inspect
    agent_id = "unknown"
    for frame_info in inspect.stack():
        if 'agent' in frame_info.frame.f_locals:
            agent = frame_info.frame.f_locals['agent']
            if hasattr(agent, 'agent_id'):
                agent_id = agent.agent_id
                break
        if 'self' in frame_info.frame.f_locals:
            obj = frame_info.frame.f_locals['self']
            if hasattr(obj, 'agent_id'):
                agent_id = obj.agent_id
                break

    if agent_id not in AGENT_TOOL_LOG:
        AGENT_TOOL_LOG[agent_id] = []

    AGENT_TOOL_LOG[agent_id].append({
        'tool': name,
        'args': args,
        'result_len': len(result) if result else 0,
        'result_preview': result[:200] if result else "None"
    })

    return result

# Apply the monkey patch
agentic_client.AgenticGeminiClient._execute_tool = logged_execute_tool

from atlas.core.config import Config
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.core.task import TaskSubmission


BUGGY_CODE = '''import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG: Never initialized

    async def acquire(self):
        # BUG: No lock - race condition
        now = time.time()
        self.tokens += (now - self.last_update) * self.rate
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def wait_for_token(self):
        while not await self.acquire():
            time.sleep(0.01)  # BUG: Blocking in async
        return True
'''


def create_test_repo(temp_dir: Path) -> tuple[str, str]:
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "rate_limiter.py").write_text(BUGGY_CODE)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_dir, capture_output=True)
    result = subprocess.run(["git", "branch", "--show-current"], cwd=repo_dir, capture_output=True, text=True)
    return str(repo_dir), result.stdout.strip() or "main"


async def main():
    print("=" * 100)
    print(" ATLAS FULL TEST WITH DETAILED AGENT LOGGING")
    print(" Shows what each agent searches via Context7 and Web")
    print("=" * 100)

    config = Config.from_env()
    if config.validate():
        print(f"Config errors: {config.validate()}")
        return

    temp_dir = Path(tempfile.mkdtemp())
    try:
        repo_path, branch = create_test_repo(temp_dir)
        print(f"\nRepo: {repo_path}")

        task = TaskSubmission(
            description="""Fix rate_limiter.py with 3 bugs:
1. self._lock = None should be asyncio.Lock()
2. Race condition - need async with self._lock
3. time.sleep blocks - use asyncio.sleep""",
            repository_url=repo_path,
            branch=branch,
            relevant_files=["rate_limiter.py"],
            max_cost_usd=1.0,
            timeout_minutes=5,
            initial_samples=8,  # 8 agents as requested
            max_samples=8,
            voting_k=3,
        )

        print(f"Task: {task.task_id}")
        print(f"Agents: {task.initial_samples}")
        print("\nStarting orchestrator with AGENTIC mode (agents search autonomously)...\n")

        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=True,
        )

        result = await asyncio.wait_for(
            orchestrator.solve(task),
            timeout=300,
        )

        # Print results
        print("\n" + "=" * 100)
        print(" RESULTS")
        print("=" * 100)
        print(f"Status: {result.status.value}")
        print(f"Consensus: {result.consensus_reached}")
        print(f"Confidence: {result.confidence_score:.0%}")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        # Print detailed agent tool usage
        print("\n" + "=" * 100)
        print(" DETAILED AGENT TOOL USAGE")
        print("=" * 100)

        for agent_id in sorted(AGENT_TOOL_LOG.keys()):
            calls = AGENT_TOOL_LOG[agent_id]
            print(f"\n{'─' * 80}")
            print(f"  AGENT: {agent_id}")
            print(f"  Total tool calls: {len(calls)}")
            print(f"{'─' * 80}")

            for i, call in enumerate(calls, 1):
                tool = call['tool']
                args = call['args']
                result_len = call['result_len']
                preview = call['result_preview']

                if tool == "resolve_library_id":
                    print(f"  [{i}] CONTEXT7 RESOLVE: '{args.get('library_name', '')}'")
                    print(f"      Result: {preview[:100]}...")
                elif tool == "get_library_docs":
                    print(f"  [{i}] CONTEXT7 DOCS: library='{args.get('library_id', '')}' query='{args.get('query', '')[:50]}'")
                    print(f"      Result: {result_len} chars")
                elif tool == "web_search":
                    print(f"  [{i}] WEB SEARCH: '{args.get('query', '')}'")
                    site = args.get('site_filter')
                    if site:
                        print(f"      Site filter: {site}")
                    print(f"      Result: {result_len} chars")
                else:
                    print(f"  [{i}] {tool}: {args}")

            print()

        # Summary table
        print("\n" + "=" * 100)
        print(" SUMMARY TABLE")
        print("=" * 100)
        print(f"{'Agent':<12} {'Context7 Calls':<16} {'Web Searches':<14} {'Total Calls':<12}")
        print("-" * 54)

        for agent_id in sorted(AGENT_TOOL_LOG.keys()):
            calls = AGENT_TOOL_LOG[agent_id]
            c7_calls = sum(1 for c in calls if c['tool'] in ['resolve_library_id', 'get_library_docs'])
            web_calls = sum(1 for c in calls if c['tool'] == 'web_search')
            total = len(calls)
            print(f"{agent_id:<12} {c7_calls:<16} {web_calls:<14} {total:<12}")

        # Show winning patch
        if result.patch:
            print("\n" + "=" * 100)
            print(" WINNING PATCH")
            print("=" * 100)
            print(result.patch[:2000])
            if len(result.patch) > 2000:
                print("... (truncated)")

            # Check patterns
            checks = {
                "asyncio.Lock": "asyncio.Lock" in result.patch,
                "asyncio.sleep": "asyncio.sleep" in result.patch,
                "async with": "async with" in result.patch,
            }
            print("\nFix patterns:")
            for k, v in checks.items():
                print(f"  {k}: {'YES' if v else 'NO'}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 100)
    print(" TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
