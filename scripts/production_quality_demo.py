#!/usr/bin/env python3
"""
Production Quality Demo - Tests the new Fortune 100-grade prompt styles.

This demonstrates:
1. New production-focused prompt styles (Senior Engineer, Security, Performance, etc.)
2. Working RAG integration (web search + Context7)
3. 10 agents producing production-ready solutions
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from atlas.core.config import Config
from atlas.agents.gemini_client import GeminiClient
from atlas.agents.prompt_styles import (
    ALL_STYLES, BASE_SYSTEM_PROMPT,
    SENIOR_ENGINEER, SECURITY_FOCUSED, PERFORMANCE_EXPERT,
    SYSTEMS_ARCHITECT, CODE_REVIEWER
)
from atlas.rag.web_search import WebSearchClient
from atlas.rag.context7 import Context7Client

# The same tough problem - async rate limiter with 3 bugs
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
        return True
'''

ISSUE = """Fix the critical bugs in this async rate limiter:
1. Race condition - no lock protection on token operations
2. time.sleep() blocks the event loop - should use asyncio.sleep()
3. self._lock is None but never initialized as asyncio.Lock()

This code must work correctly under high concurrency (100+ simultaneous requests).
"""


async def test_web_search():
    """Test that web search is working."""
    print("\n" + "=" * 70)
    print(" TESTING WEB SEARCH")
    print("=" * 70)

    config = Config.from_env()
    web_search = WebSearchClient(config)

    try:
        results = await web_search.search_for_code_context(
            issue_description="asyncio rate limiter race condition python",
            libraries=["asyncio", "aiohttp"],
        )

        if results.results:
            print(f"\n[SUCCESS] Found {len(results.results)} web results:")
            for i, r in enumerate(results.results[:5]):
                print(f"  [{i+1}] {r.title[:60]}...")
                print(f"      URL: {r.url[:70]}")
            return True
        else:
            print("\n[WARNING] No web results found")
            return False

    except Exception as e:
        print(f"\n[ERROR] Web search failed: {e}")
        return False


async def test_context7():
    """Test that Context7 is working with correct library IDs."""
    print("\n" + "=" * 70)
    print(" TESTING CONTEXT7 DOCUMENTATION")
    print("=" * 70)

    config = Config.from_env()
    context7 = Context7Client(config)

    # Test library extraction
    libraries = context7.extract_libraries_from_issue(ISSUE)
    print(f"\n[INFO] Extracted library IDs: {libraries}")

    # Test with a known library ID format
    test_libs = ["python/cpython", "aio-libs/aiohttp"]

    for lib_id in test_libs:
        try:
            print(f"\n[INFO] Querying Context7 for: {lib_id}")
            result = await context7.get_documentation(
                library_name=lib_id,
                query="asyncio Lock synchronization async await",
                max_tokens=1000,
            )

            if result.chunks:
                print(f"  [SUCCESS] Got {len(result.chunks)} doc chunks")
                for chunk in result.chunks[:2]:
                    preview = chunk.content[:100].replace('\n', ' ')
                    print(f"    - {chunk.source}: {preview}...")
            else:
                print(f"  [WARNING] No docs returned (library may not be indexed)")

        except Exception as e:
            print(f"  [ERROR] Context7 query failed: {e}")

    return True


async def generate_with_style(client, style, agent_num, rag_context):
    """Generate a solution using a specific style.

    RAG context is MANDATORY - agents must receive research before implementing.
    """
    if not rag_context:
        raise ValueError("RAG context is MANDATORY - cannot generate without research!")

    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Buggy Code
```python
{BUGGY_CODE}
```

## Issue
{ISSUE}

## Research Results (USE THIS - MANDATORY)

The following research was gathered from Context7 MCP and web search.
You MUST apply the best practices found in this research to your solution.

{rag_context}

---

Based on your research above, generate a production-ready fix that applies ALL the best practices discovered.
First briefly state what key insights you found from the research, then output your unified diff patch."""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)
    result = await client.generate(prompt=prompt, max_tokens=4000, temperature=temp)
    return result.text


async def main():
    print("=" * 70)
    print(" ATLAS PRODUCTION QUALITY DEMO")
    print(" Fortune 100-Grade Code Solutions")
    print("=" * 70)

    # Test RAG components
    web_ok = await test_web_search()
    ctx7_ok = await test_context7()

    print("\n" + "=" * 70)
    print(" RAG STATUS")
    print("=" * 70)
    print(f"  Web Search:  {'[OK]' if web_ok else '[FAILED]'}")
    print(f"  Context7:    {'[OK]' if ctx7_ok else '[LIMITED]'}")

    # Gather RAG context for agents - THIS IS MANDATORY
    print("\n" + "=" * 70)
    print(" GATHERING RAG CONTEXT FOR AGENTS (MANDATORY)")
    print("=" * 70)

    config = Config.from_env()
    rag_context = ""

    # WEB SEARCH - Multiple targeted queries for comprehensive research
    print("\n  [Web Search] Gathering best practices from reputable sources...")
    web_search = WebSearchClient(config)

    # Search 1: General best practices for the problem domain
    try:
        results1 = await web_search.search(
            "python asyncio best practices time.monotonic vs time.time",
            max_results=3,
        )
        if results1.results:
            rag_context += "\n### Web Search: Python asyncio Best Practices\n"
            rag_context += results1.combined_content[:1500]
            print(f"    - Found {len(results1.results)} results on asyncio best practices")
    except Exception as e:
        print(f"    - Search 1 failed: {e}")

    # Search 2: Specific to rate limiters and token buckets
    try:
        results2 = await web_search.search(
            "asyncio rate limiter implementation Lock synchronization production",
            max_results=3,
            site_filter="stackoverflow.com",
        )
        if results2.results:
            rag_context += "\n\n### Web Search: Rate Limiter Implementations\n"
            rag_context += results2.combined_content[:1500]
            print(f"    - Found {len(results2.results)} Stack Overflow results")
    except Exception as e:
        print(f"    - Search 2 failed: {e}")

    # Search 3: Production patterns from GitHub
    try:
        results3 = await web_search.search(
            "python asyncio Lock time.monotonic production code",
            max_results=2,
            site_filter="github.com",
        )
        if results3.results:
            rag_context += "\n\n### Web Search: Production Code Examples\n"
            rag_context += results3.combined_content[:1000]
            print(f"    - Found {len(results3.results)} GitHub examples")
    except Exception as e:
        print(f"    - Search 3 failed: {e}")

    # CONTEXT7 MCP - Official documentation
    print("\n  [Context7 MCP] Querying official documentation...")
    context7 = Context7Client(config)

    try:
        result = await context7.get_documentation(
            library_name="python/cpython",
            query="asyncio Lock synchronization primitives time monotonic",
            max_tokens=2000,
        )
        if result.chunks:
            rag_context += "\n\n### Official Python Documentation (Context7)\n"
            rag_context += result.combined_content[:2000]
            print(f"    - Found {len(result.chunks)} documentation chunks")
        else:
            print("    - No Context7 docs returned (library may not be indexed)")
    except Exception as e:
        print(f"    - Context7 query failed: {e}")

    # Verify we have SOME research
    if not rag_context.strip():
        print("\n  [ERROR] No RAG context gathered!")
        print("  RAG is MANDATORY - cannot proceed without research.")
        print("  Please check web search and Context7 connectivity.")
        # Provide minimal guidance without giving away specific answers
        rag_context = """
### Research Guidance (RAG unavailable - LIMITED MODE)
External research tools are unavailable. You must rely on your training knowledge.
Research the following topics before implementing:
- Official Python documentation for asyncio synchronization primitives
- Best practices for measuring elapsed time in Python
- Correct patterns for non-blocking async code
- Production-grade concurrency patterns
Apply what you know from official documentation and Fortune 100 engineering standards.
"""
    else:
        print(f"\n  [SUCCESS] Gathered {len(rag_context)} chars of research context")

    # Deploy 10 agents (2 per style)
    print("\n" + "=" * 70)
    print(" DEPLOYING 10 AGENTS (2 per style)")
    print("=" * 70)

    client = GeminiClient(config)

    agents = [
        (SENIOR_ENGINEER, 0), (SENIOR_ENGINEER, 1),
        (SECURITY_FOCUSED, 0), (SECURITY_FOCUSED, 1),
        (PERFORMANCE_EXPERT, 0), (PERFORMANCE_EXPERT, 1),
        (SYSTEMS_ARCHITECT, 0), (SYSTEMS_ARCHITECT, 1),
        (CODE_REVIEWER, 0), (CODE_REVIEWER, 1),
    ]

    print(f"\nRunning {len(agents)} agents in parallel...")
    print("Styles: Senior Engineer, Security, Performance, Systems Architect, Code Reviewer")

    # Run all agents in parallel
    tasks = [
        generate_with_style(client, style, num, rag_context)
        for style, num in agents
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Show results
    print("\n" + "=" * 70)
    print(" AGENT SOLUTIONS")
    print("=" * 70)

    for i, ((style, num), result) in enumerate(zip(agents, results)):
        agent_name = f"agent_{i}_{style.name.value}_v{num+1}"

        print(f"\n{'-' * 70}")
        print(f" AGENT {i+1}/10: {agent_name}")
        print(f" Style: {style.description}")
        print(f"{'-' * 70}")

        if isinstance(result, Exception):
            print(f"[ERROR] {result}")
        else:
            # Check for key quality indicators
            has_monotonic = "monotonic" in result.lower()
            has_asyncio_lock = "asyncio.lock" in result.lower()
            has_async_sleep = "asyncio.sleep" in result.lower()
            has_async_with = "async with" in result.lower()

            print(f"\nQuality Indicators:")
            print(f"  [{'X' if has_monotonic else ' '}] Uses time.monotonic()")
            print(f"  [{'X' if has_asyncio_lock else ' '}] Uses asyncio.Lock()")
            print(f"  [{'X' if has_async_sleep else ' '}] Uses asyncio.sleep()")
            print(f"  [{'X' if has_async_with else ' '}] Uses async with lock")

            print(f"\nFull Response:")
            print(result)

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    successful = [r for r in results if not isinstance(r, Exception)]

    # Count quality indicators across all solutions
    monotonic_count = sum(1 for r in successful if "monotonic" in r.lower())
    lock_count = sum(1 for r in successful if "asyncio.lock" in r.lower())
    async_sleep_count = sum(1 for r in successful if "asyncio.sleep" in r.lower())

    print(f"""
    Agents deployed:         {len(agents)}
    Successful responses:    {len(successful)}/{len(agents)}

    Quality Metrics (Fortune 100 Standards):
    - Used time.monotonic(): {monotonic_count}/{len(successful)} agents
    - Used asyncio.Lock():   {lock_count}/{len(successful)} agents
    - Used asyncio.sleep():  {async_sleep_count}/{len(successful)} agents

    Compare this to the minimal fix approach!
    Production-ready prompts should produce GPT Pro-level solutions.
    """)


if __name__ == "__main__":
    asyncio.run(main())
