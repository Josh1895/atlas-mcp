#!/usr/bin/env python3
"""Full agent test with autonomous Context7 access.

Each agent has direct tool access and decides what to search.
Shows full responses and detailed stats.
"""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.agents.agentic_client import AgenticGeminiClient
from atlas.agents.prompt_styles import ALL_STYLES

# The buggy code to fix
BUGGY_CODE = '''import time

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

ISSUE = """Fix the following 3 bugs:
1. Race condition - multiple coroutines can access self.tokens simultaneously without synchronization
2. time.sleep(0.01) blocks the event loop - should use async sleep
3. self._lock = None is never initialized as an actual asyncio.Lock()"""

BASE_SYSTEM_PROMPT = """You are an expert Python developer specializing in async/await patterns.
Research the relevant documentation using the tools provided, then generate a production-ready fix."""


def check_patterns(text):
    """Check which best practice patterns are present in the text."""
    text_lower = text.lower()
    return {
        "time.monotonic": "monotonic" in text_lower,
        "asyncio.Lock": "asyncio.lock" in text_lower or "asyncio lock" in text_lower,
        "asyncio.sleep": "asyncio.sleep" in text_lower,
        "async with": "async with" in text_lower,
    }


async def run_agent(agent_client, style, agent_num, delay_seconds=0):
    """Run a single agent with autonomous tool access."""
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    # Clean up style name
    style_name = str(style.name).replace("PromptStyleName.", "")

    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Buggy Code
```python
{BUGGY_CODE}
```

## Issue
{ISSUE}

---
IMPORTANT: Use the tools to research Python asyncio documentation before answering.
Then provide a production-ready fix with explanation."""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)

    try:
        result = await agent_client.generate_with_tools(
            prompt=prompt,
            temperature=temp,
            max_tokens=4000,
            max_iterations=8,
        )
        return {
            "name": f"{style_name} #{agent_num + 1}",
            "style": style_name,
            "success": True,
            "response": result.text,
            "tool_calls": result.tool_calls,
            "iterations": result.iterations,
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "cost": result.total_cost,
            "patterns": check_patterns(result.text),
            "context7_chars": sum(
                len(tc.get("result_preview", ""))
                for tc in result.tool_calls
                if tc["name"] == "get_library_docs"
            ),
        }
    except Exception as e:
        return {
            "name": f"{style_name} #{agent_num + 1}",
            "style": style_name,
            "success": False,
            "error": str(e),
            "response": "",
            "tool_calls": [],
            "patterns": {},
            "context7_chars": 0,
        }


async def main():
    print("=" * 100)
    print(" ATLAS FULL AGENT TEST V2 - Autonomous Context7 Access")
    print(" Each agent decides what to search - no hardcoded queries")
    print("=" * 100)

    config = Config.from_env()
    agent_client = AgenticGeminiClient(config)

    # Run 10 agents (2 per style) with staggered delays
    print("\n" + "=" * 100)
    print(" DEPLOYING 10 AGENTS (2 per style)")
    print(" Each agent has autonomous access to Context7 tools")
    print("=" * 100)

    tasks = []
    agent_num = 0
    for style in ALL_STYLES:
        for i in range(2):
            delay = agent_num * 2.0  # 2 second stagger to avoid rate limits
            tasks.append(run_agent(agent_client, style, i, delay_seconds=delay))
            agent_num += 1

    print(f"\n  Running {len(tasks)} agents with staggered delays...")
    print("  (This may take a few minutes due to rate limiting)\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    agent_results = []
    for r in results:
        if isinstance(r, Exception):
            agent_results.append({
                "name": "Unknown",
                "success": False,
                "error": str(r),
                "response": "",
                "tool_calls": [],
                "patterns": {},
                "context7_chars": 0,
            })
        else:
            agent_results.append(r)

    # Show each agent's response
    print("\n" + "=" * 100)
    print(" INDIVIDUAL AGENT RESPONSES")
    print("=" * 100)

    for i, result in enumerate(agent_results):
        print(f"\n{'─' * 100}")
        print(f" AGENT {i+1}: {result['name']}")
        print(f"{'─' * 100}")

        if result["success"]:
            # Tool calls
            print(f"\n  Tool Calls ({len(result['tool_calls'])}):")
            for tc in result["tool_calls"]:
                print(f"    • {tc['name']}({tc['args']})")
                print(f"      → {tc['result_preview'][:100]}...")

            # Stats
            print(f"\n  Stats:")
            print(f"    Iterations: {result.get('iterations', 'N/A')}")
            print(f"    Context7 chars: {result['context7_chars']}")
            print(f"    Input tokens: {result.get('input_tokens', 0)}")
            print(f"    Output tokens: {result.get('output_tokens', 0)}")
            print(f"    Cost: ${result.get('cost', 0):.4f}")

            # Patterns found
            print(f"\n  Patterns in response:")
            for pattern, found in result["patterns"].items():
                status = "✓" if found else "✗"
                print(f"    [{status}] {pattern}")

            # Full response
            print(f"\n  Response ({len(result['response'])} chars):")
            print("  " + "─" * 96)
            # Show first 1500 chars
            response_preview = result["response"][:1500]
            for line in response_preview.split("\n"):
                print(f"  {line}")
            if len(result["response"]) > 1500:
                print(f"  ... [truncated, {len(result['response']) - 1500} more chars]")
        else:
            print(f"\n  ERROR: {result.get('error', 'Unknown error')}")

    # Summary table
    print("\n" + "=" * 100)
    print(" SUMMARY TABLE")
    print("=" * 100)

    print(f"\n{'Agent':<30} {'Lock':<8} {'sleep':<8} {'async':<8} {'mono':<8} {'Ctx7':<10} {'Grade'}")
    print("─" * 100)

    for result in agent_results:
        if result["success"]:
            p = result["patterns"]
            lock = "✓" if p.get("asyncio.Lock") else "✗"
            sleep = "✓" if p.get("asyncio.sleep") else "✗"
            async_with = "✓" if p.get("async with") else "✗"
            mono = "✓" if p.get("time.monotonic") else "✗"
            ctx7 = f"{result['context7_chars']}"

            # Calculate grade
            score = sum([
                2 if p.get("asyncio.Lock") else 0,
                1 if p.get("asyncio.sleep") else 0,
                1 if p.get("async with") else 0,
                2 if p.get("time.monotonic") else 0,
            ])
            grade = "A+" if score >= 5 else "A" if score >= 4 else "B" if score >= 3 else "C" if score >= 2 else "F"

            print(f"{result['name']:<30} {lock:<8} {sleep:<8} {async_with:<8} {mono:<8} {ctx7:<10} {grade}")
        else:
            print(f"{result['name']:<30} ERROR: {result.get('error', '')[:50]}")

    # Consensus analysis
    print("\n" + "=" * 100)
    print(" CONSENSUS ANALYSIS")
    print("=" * 100)

    successful = [r for r in agent_results if r["success"]]
    print(f"\nSuccessful agents: {len(successful)}/{len(agent_results)}")

    if successful:
        pattern_counts = {
            "asyncio.Lock": sum(1 for r in successful if r["patterns"].get("asyncio.Lock")),
            "asyncio.sleep": sum(1 for r in successful if r["patterns"].get("asyncio.sleep")),
            "async with": sum(1 for r in successful if r["patterns"].get("async with")),
            "time.monotonic": sum(1 for r in successful if r["patterns"].get("time.monotonic")),
        }

        print("\nPattern adoption:")
        for pattern, count in pattern_counts.items():
            pct = (count / len(successful) * 100)
            bar = "█" * int(pct / 5)
            print(f"  {pattern:<20} {count:>2}/{len(successful)} ({pct:>5.1f}%) {bar}")

        # Total Context7 usage
        total_ctx7 = sum(r["context7_chars"] for r in successful)
        total_tool_calls = sum(len(r["tool_calls"]) for r in successful)
        total_cost = sum(r.get("cost", 0) for r in successful)

        print(f"\nTotal Context7 chars retrieved: {total_ctx7}")
        print(f"Total tool calls made: {total_tool_calls}")
        print(f"Total cost: ${total_cost:.4f}")

    print("\n" + "=" * 100)
    print(" TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
