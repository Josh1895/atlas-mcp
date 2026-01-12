#!/usr/bin/env python3
"""Test with exact chart output showing all agent details."""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track tool calls per agent
AGENT_DATA = defaultdict(lambda: {
    'style': '',
    'tool_calls': 0,
    'ctx7_chars': 0,
    'ctx7_calls': 0,
    'web_chars': 0,
    'web_calls': 0,
    'patch': '',
})

from atlas.agents import agentic_client
original_execute_tool = agentic_client.AgenticGeminiClient._execute_tool

async def logged_execute_tool(self, name: str, args: dict) -> str:
    result = await original_execute_tool(self, name, args)

    # Find agent_id from stack
    import inspect
    agent_id = "unknown"
    for frame_info in inspect.stack():
        if 'self' in frame_info.frame.f_locals:
            obj = frame_info.frame.f_locals['self']
            if hasattr(obj, 'agent_id'):
                agent_id = obj.agent_id
                break

    result_len = len(result) if result else 0
    AGENT_DATA[agent_id]['tool_calls'] += 1

    if name in ['resolve_library_id', 'get_library_docs']:
        AGENT_DATA[agent_id]['ctx7_calls'] += 1
        AGENT_DATA[agent_id]['ctx7_chars'] += result_len
    elif name == 'web_search':
        AGENT_DATA[agent_id]['web_calls'] += 1
        AGENT_DATA[agent_id]['web_chars'] += result_len

    return result

agentic_client.AgenticGeminiClient._execute_tool = logged_execute_tool

from atlas.core.config import Config
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.core.task import TaskSubmission
from atlas.agents.prompt_styles import PromptStyleName


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


def grade_patch(patch: str) -> str:
    """Grade a patch based on patterns found."""
    if not patch:
        return "F"

    has_lock = "asyncio.Lock" in patch
    has_sleep = "asyncio.sleep" in patch
    has_async_with = "async with" in patch
    has_mono = "monotonic" in patch.lower()

    score = sum([has_lock, has_sleep, has_async_with, has_mono])

    if score == 4:
        return "A+"
    elif score == 3:
        return "A"
    elif score == 2:
        return "B"
    elif score == 1:
        return "C"
    else:
        return "F"


async def main():
    print("=" * 100)
    print(" ATLAS 10-AGENT TEST WITH DETAILED CHART")
    print("=" * 100)

    config = Config.from_env()
    if config.validate():
        print(f"Config errors: {config.validate()}")
        return

    temp_dir = Path(tempfile.mkdtemp())
    try:
        repo_path, branch = create_test_repo(temp_dir)

        task = TaskSubmission(
            description="""Fix rate_limiter.py with 3 bugs:
1. self._lock = None should be asyncio.Lock()
2. Race condition - need async with self._lock
3. time.sleep blocks - use asyncio.sleep
Best practice: use time.monotonic() instead of time.time()""",
            repository_url=repo_path,
            branch=branch,
            relevant_files=["rate_limiter.py"],
            max_cost_usd=1.0,
            timeout_minutes=5,
            initial_samples=10,  # 10 agents = 2 per style
            max_samples=10,
            voting_k=3,
        )

        print(f"\nRunning 10 agents (2 of each style)...")
        print("Styles: SENIOR_ENGINEER, SECURITY_FOCUSED, PERFORMANCE_EXPERT, SYSTEMS_ARCHITECT, CODE_REVIEWER\n")

        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=True,
        )

        # Store patches per agent
        original_run_agent = orchestrator.agent_pool._run_agent_with_rag

        async def capture_run_agent(agent, context):
            result = await original_run_agent(agent, context)
            AGENT_DATA[agent.agent_id]['patch'] = result.patch
            AGENT_DATA[agent.agent_id]['style'] = agent.prompt_style.name.value if agent.prompt_style else 'default'
            return result

        orchestrator.agent_pool._run_agent_with_rag = capture_run_agent

        result = await asyncio.wait_for(
            orchestrator.solve(task),
            timeout=300,
        )

        # Print summary
        print(f"\nStatus: {result.status.value} | Cost: ${result.cost_usd:.4f} | Duration: {result.duration_seconds:.1f}s")
        print(f"Consensus: {result.consensus_reached} | Confidence: {result.confidence_score:.0%}")

        # Print the chart
        print("\n" + "=" * 120)
        print(" AGENT PERFORMANCE CHART")
        print("=" * 120)

        # Header
        print(f"{'Agent':<25} {'Tool Calls':<12} {'Ctx7 Chars':<12} {'Web Calls':<11} {'Web Chars':<11} {'Lock':<6} {'sleep':<7} {'async':<7} {'mono':<6} {'Grade':<6}")
        print("-" * 120)

        # Map agent_id to style name with instance number
        style_counts = defaultdict(int)

        for agent_id in sorted(AGENT_DATA.keys()):
            data = AGENT_DATA[agent_id]
            style = data['style'].upper() if data['style'] else 'UNKNOWN'

            style_counts[style] += 1
            instance = style_counts[style]

            agent_name = f"{style} #{instance}"

            patch = data['patch']
            has_lock = "✓" if patch and "asyncio.Lock" in patch else "✗"
            has_sleep = "✓" if patch and "asyncio.sleep" in patch else "✗"
            has_async = "✓" if patch and "async with" in patch else "✗"
            has_mono = "✓" if patch and "monotonic" in patch.lower() else "✗"

            grade = grade_patch(patch)

            print(f"{agent_name:<25} {data['tool_calls']:<12} {data['ctx7_chars']:<12} {data['web_calls']:<11} {data['web_chars']:<11} {has_lock:<6} {has_sleep:<7} {has_async:<7} {has_mono:<6} {grade:<6}")

        print("-" * 120)

        # Totals
        total_tools = sum(d['tool_calls'] for d in AGENT_DATA.values())
        total_ctx7 = sum(d['ctx7_chars'] for d in AGENT_DATA.values())
        total_web_calls = sum(d['web_calls'] for d in AGENT_DATA.values())
        total_web_chars = sum(d['web_chars'] for d in AGENT_DATA.values())

        print(f"{'TOTALS':<25} {total_tools:<12} {total_ctx7:<12} {total_web_calls:<11} {total_web_chars:<11}")

        # Winning patch preview
        if result.patch:
            print("\n" + "=" * 120)
            print(" WINNING PATCH")
            print("=" * 120)
            print(result.patch[:1500])

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 120)
    print(" TEST COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
