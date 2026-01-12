#!/usr/bin/env python3
"""Test agentic Context7 access.

The AI autonomously decides what to search in Context7 based on the problem.
No hardcoded queries - full AI autonomy.
"""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.agents.agentic_client import AgenticGeminiClient


async def test_autonomous_research():
    """Test that the AI autonomously researches using Context7."""
    print("=" * 80)
    print(" AGENTIC CONTEXT7 TEST")
    print(" AI decides what to search - no hardcoded queries")
    print("=" * 80)

    config = Config.from_env()
    agent = AgenticGeminiClient(config)

    # The buggy code - AI must figure out what docs to search
    buggy_code = '''import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG: Never initialized!

    async def acquire(self):
        # BUG: Race condition - no lock protection!
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
        return True'''

    prompt = f"""Fix this buggy Python async rate limiter. There are 3 bugs:
1. Race condition - multiple coroutines can access self.tokens without synchronization
2. time.sleep blocks the event loop - should use async sleep
3. self._lock is never initialized as an actual Lock

IMPORTANT: Research the relevant Python documentation to find best practices.
Then provide a production-ready fix with explanation.

```python
{buggy_code}
```"""

    print("\n[PROMPT]")
    print(prompt[:500] + "...")

    print("\n" + "-" * 80)
    print(" AI IS NOW AUTONOMOUSLY RESEARCHING...")
    print("-" * 80)

    result = await agent.generate_with_tools(
        prompt=prompt,
        temperature=0.7,
        max_tokens=4000,
        max_iterations=10,
    )

    print("\n" + "=" * 80)
    print(" TOOL CALLS MADE BY AI")
    print("=" * 80)

    for i, call in enumerate(result.tool_calls):
        print(f"\n[{i+1}] {call['name']}")
        print(f"    Args: {call['args']}")
        print(f"    Result: {call['result_preview']}")

    print("\n" + "=" * 80)
    print(" FINAL RESPONSE")
    print("=" * 80)
    print(result.text[:2000] + "..." if len(result.text) > 2000 else result.text)

    print("\n" + "=" * 80)
    print(" STATS")
    print("=" * 80)
    print(f"  Iterations: {result.iterations}")
    print(f"  Tool calls: {len(result.tool_calls)}")
    print(f"  Input tokens: {result.total_input_tokens}")
    print(f"  Output tokens: {result.total_output_tokens}")
    print(f"  Estimated cost: ${result.total_cost:.4f}")

    # Check if AI found the right patterns
    response_lower = result.text.lower()
    patterns = {
        "asyncio.Lock": "asyncio.lock" in response_lower,
        "asyncio.sleep": "asyncio.sleep" in response_lower,
        "time.monotonic": "monotonic" in response_lower,
        "async with": "async with" in response_lower,
    }

    print("\n" + "=" * 80)
    print(" BEST PRACTICES IN RESPONSE")
    print("=" * 80)
    for pattern, found in patterns.items():
        status = "[OK]" if found else "[X]"
        print(f"  {status} {pattern}")


async def test_different_tech():
    """Test with a completely different technology to show generality."""
    print("\n\n" + "=" * 80)
    print(" TEST 2: DIFFERENT TECHNOLOGY (React)")
    print(" AI should autonomously search React docs")
    print("=" * 80)

    config = Config.from_env()
    agent = AgenticGeminiClient(config)

    prompt = """I'm getting a "Cannot update a component while rendering a different component" error
in my React app. I think it's related to how I'm using useState in a callback.

Research the React documentation and explain:
1. What causes this error
2. How to fix it properly
3. Best practices for state updates"""

    print("\n[PROMPT]")
    print(prompt)

    print("\n" + "-" * 80)
    print(" AI IS NOW AUTONOMOUSLY RESEARCHING REACT DOCS...")
    print("-" * 80)

    result = await agent.generate_with_tools(
        prompt=prompt,
        temperature=0.7,
        max_tokens=3000,
        max_iterations=8,
    )

    print("\n" + "=" * 80)
    print(" TOOL CALLS MADE BY AI")
    print("=" * 80)

    for i, call in enumerate(result.tool_calls):
        print(f"\n[{i+1}] {call['name']}")
        print(f"    Args: {call['args']}")

    print("\n" + "=" * 80)
    print(" FINAL RESPONSE (truncated)")
    print("=" * 80)
    print(result.text[:1500] + "..." if len(result.text) > 1500 else result.text)

    print(f"\n  Tool calls: {len(result.tool_calls)}")
    print(f"  Iterations: {result.iterations}")


if __name__ == "__main__":
    asyncio.run(test_autonomous_research())
    # Uncomment to also test React:
    # asyncio.run(test_different_tech())
