#!/usr/bin/env python3
"""Quick 10-agent demo - runs in parallel for speed."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from atlas.core.config import Config
from atlas.agents.gemini_client import GeminiClient
from atlas.agents.prompt_styles import (
    MINIMAL_DIFF, VERBOSE_EXPLAINER, REFACTOR_FIRST, DEBUGGER, REPO_ONLY, BASE_SYSTEM_PROMPT
)

# Tough problem: Race condition in async rate limiter
BUGGY_CODE = '''class AsyncRateLimiter:
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
        return True
'''

ISSUE = "Fix: 1) Race condition - no lock, 2) time.sleep blocks event loop, 3) _lock never initialized"


async def generate_one(client, style, agent_num):
    """Generate one patch."""
    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Buggy Code
```python
{BUGGY_CODE}
```

## Issue
{ISSUE}

Output ONLY a unified diff patch to fix all bugs."""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)
    result = await client.generate(prompt=prompt, max_tokens=3000, temperature=temp)
    return result.text


async def main():
    config = Config.from_env()
    client = GeminiClient(config)

    print("=" * 70)
    print(" QUICK 10-AGENT DEMO (Parallel Execution)")
    print("=" * 70)

    print("\n[BUGGY CODE]")
    print(BUGGY_CODE)

    print("\n[ISSUE]", ISSUE)

    # All 10 agents (2 per style)
    agents = [
        (MINIMAL_DIFF, 0), (MINIMAL_DIFF, 1),
        (VERBOSE_EXPLAINER, 0), (VERBOSE_EXPLAINER, 1),
        (REFACTOR_FIRST, 0), (REFACTOR_FIRST, 1),
        (DEBUGGER, 0), (DEBUGGER, 1),
        (REPO_ONLY, 0), (REPO_ONLY, 1),
    ]

    print("\n" + "=" * 70)
    print(" Deploying 10 agents in PARALLEL...")
    print("=" * 70)

    # Run all 10 in parallel
    tasks = [generate_one(client, style, num) for style, num in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Show each result
    for i, ((style, num), result) in enumerate(zip(agents, results)):
        agent_name = f"agent_{i}_{style.name.value}_v{num+1}"

        print(f"\n{'=' * 70}")
        print(f" AGENT {i+1}/10: {agent_name}")
        print(f" Style: {style.name.value}")
        print("=" * 70)

        if isinstance(result, Exception):
            print(f"[ERROR] {result}")
        else:
            print(result)

    print("\n" + "=" * 70)
    print(" ALL 10 AGENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
