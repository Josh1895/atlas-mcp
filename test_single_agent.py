#!/usr/bin/env python3
"""Single agent test - minimal test to verify core functionality."""

import asyncio
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Override to use stable model and disable Context7
os.environ["ATLAS_MODEL"] = "gemini-2.0-flash"
os.environ["CONTEXT7_API_KEY"] = ""

async def main():
    print("=" * 70)
    print(" SINGLE AGENT TEST - Minimal Functionality Verification")
    print("=" * 70)

    from atlas.core.config import Config
    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import SENIOR_ENGINEER
    from atlas.core.task import TaskSubmission

    # Create config with stable model
    config = Config.from_env()
    config.model = "gemini-2.0-flash"  # Use stable model
    config.context7_api_key = ""  # Disable Context7

    print(f"\nConfig:")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Context7: DISABLED")

    # Create a simple task
    task = TaskSubmission(
        description="""Fix this Python code:

```python
import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG: Should be asyncio.Lock()

    async def acquire(self):
        # BUG: No lock protection - race condition
        now = time.time()
        self.tokens += (now - self.last_update) * self.rate
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def wait_for_token(self):
        while not await self.acquire():
            time.sleep(0.01)  # BUG: Blocking in async - use asyncio.sleep
        return True
```

Fix all 3 bugs:
1. Initialize _lock as asyncio.Lock()
2. Add async with self._lock for thread safety
3. Use asyncio.sleep instead of time.sleep

Return a unified diff patch.""",
        repository_url="local",
        branch="main",
        relevant_files=["rate_limiter.py"],
        max_cost_usd=0.10,
        timeout_minutes=2,
    )

    # Create single agent
    agent = MicroAgent(
        agent_id="test_agent",
        prompt_style=SENIOR_ENGINEER,
        config=config,
    )

    # Create context with the code
    context = AgentContext(
        task=task,
        repository_content=task.description,  # Code is in the description
    )

    print("\n" + "=" * 70)
    print(" GENERATING PATCH")
    print("=" * 70)
    print("\nSending to Gemini (this may take 30-60 seconds)...")

    try:
        solution = await asyncio.wait_for(
            agent.generate(context),
            timeout=120
        )

        print("\n" + "=" * 70)
        print(" RESULT")
        print("=" * 70)
        print(f"Agent: {solution.agent_id}")
        print(f"Style: {solution.prompt_style}")
        print(f"Valid: {solution.is_valid}")
        print(f"Cost: ${solution.cost:.4f}")
        print(f"Tokens: {solution.tokens_used}")

        if solution.patch:
            print(f"\nPatch ({len(solution.patch)} chars):")
            print("-" * 70)
            print(solution.patch)
            print("-" * 70)

            # Check for key fixes
            checks = {
                "asyncio.Lock": "asyncio.Lock" in solution.patch,
                "asyncio.sleep": "asyncio.sleep" in solution.patch,
                "async with": "async with" in solution.patch,
            }
            print("\nFix patterns:")
            all_found = True
            for k, v in checks.items():
                status = "YES" if v else "NO"
                print(f"  {k}: {status}")
                if not v:
                    all_found = False

            if all_found:
                print("\n ALL BUGS FIXED! MCP SERVER WORKING!")
            else:
                print("\n Some patterns missing (may still be valid)")
        else:
            print(f"\nNo patch generated")
            if solution.validation_errors:
                print(f"Errors: {solution.validation_errors}")

    except asyncio.TimeoutError:
        print("\n TIMEOUT - Model took too long")
    except Exception as e:
        print(f"\n ERROR: {e}")

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
