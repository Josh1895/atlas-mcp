#!/usr/bin/env python3
"""Full agent test with summary table output."""

import asyncio
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.rag.web_search import WebSearchClient
from atlas.rag.context7 import Context7Client
from atlas.agents.gemini_client import GeminiClient
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
Generate a unified diff patch that fixes all bugs. Use production-grade best practices."""


async def gather_rag_context(config):
    """Gather RAG context from all sources."""
    results = {
        "context7": {"success": False, "content": "", "chars": 0},
        "web_search": {"success": False, "content": "", "chars": 0, "sources": []},
    }

    # Web Search
    print("\n[1] Web Search (filter_domains=False)...")
    web_client = WebSearchClient(config)

    try:
        search_results = await web_client.search_for_code_context(
            issue_description="asyncio rate limiter race condition time.sleep blocking Lock",
            libraries=["asyncio"],
        )
        if search_results.results:
            results["web_search"]["success"] = True
            results["web_search"]["content"] = search_results.combined_content
            results["web_search"]["chars"] = len(search_results.combined_content)
            results["web_search"]["sources"] = [
                {"domain": r.url.split("/")[2] if "/" in r.url else r.url, "chars": len(r.snippet)}
                for r in search_results.results[:8]
            ]
            print(f"    Found {len(search_results.results)} results, {results['web_search']['chars']} chars")
    except Exception as e:
        print(f"    Failed: {e}")

    # Context7
    print("\n[2] Context7 MCP...")
    ctx7_client = Context7Client(config)

    try:
        ctx7_result = await ctx7_client.get_documentation(
            library_name="python/cpython",
            query="asyncio Lock synchronization time monotonic",
            max_tokens=2000,
        )
        if ctx7_result.chunks:
            results["context7"]["success"] = True
            results["context7"]["content"] = ctx7_result.combined_content
            results["context7"]["chars"] = len(ctx7_result.combined_content)
            print(f"    Found {len(ctx7_result.chunks)} chunks, {results['context7']['chars']} chars")
        else:
            print("    No chunks returned (library may not be indexed)")
    except Exception as e:
        print(f"    Failed: {e}")

    return results


def check_patterns(text):
    """Check which best practice patterns are present in the text."""
    text_lower = text.lower()
    return {
        "time.monotonic": "monotonic" in text_lower,
        "asyncio.Lock": "asyncio.lock" in text_lower or "asyncio lock" in text_lower,
        "asyncio.sleep": "asyncio.sleep" in text_lower,
        "async with": "async with" in text_lower,
        "await": "await" in text_lower,
    }


async def generate_fix(client, style, agent_num, rag_context, delay_seconds=0):
    """Generate a fix using a specific agent style."""
    # Stagger requests to avoid rate limiting
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Buggy Code
```python
{BUGGY_CODE}
```

## Issue
{ISSUE}

## Research Results (APPLY THESE BEST PRACTICES)
{rag_context}

---
Generate a production-ready unified diff patch. First state key insights from research, then the patch."""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)
    result = await client.generate_with_retry(
        prompt=prompt,
        max_tokens=4000,
        temperature=temp,
        max_retries=5,
        retry_delay=2.0,
    )
    return result.text


async def main():
    print("=" * 80)
    print(" ATLAS FULL AGENT TEST - With Summary Table")
    print("=" * 80)

    config = Config.from_env()

    # Step 1: Gather RAG context
    print("\n" + "=" * 80)
    print(" STEP 1: GATHERING RAG CONTEXT")
    print("=" * 80)

    rag_results = await gather_rag_context(config)

    # Combine RAG context
    rag_context = ""
    if rag_results["web_search"]["success"]:
        rag_context += "### Web Search Results\n" + rag_results["web_search"]["content"][:8000]
    if rag_results["context7"]["success"]:
        rag_context += "\n\n### Python Documentation (Context7)\n" + rag_results["context7"]["content"][:4000]

    if not rag_context:
        rag_context = "Research unavailable. Use your training knowledge for Python asyncio best practices."

    # Check patterns in RAG context
    rag_patterns = check_patterns(rag_context)

    print("\n" + "-" * 80)
    print(" RAG CONTEXT PATTERNS DISCOVERED")
    print("-" * 80)
    for pattern, found in rag_patterns.items():
        status = "FOUND" if found else "NOT FOUND"
        print(f"  [{status}] {pattern}")

    # Step 2: Deploy agents
    print("\n" + "=" * 80)
    print(" STEP 2: DEPLOYING 10 AGENTS (2 per style)")
    print("=" * 80)

    client = GeminiClient(config)
    agent_results = []

    # Create agent tasks with staggered delays to avoid rate limiting
    tasks = []
    agent_info = []
    agent_num = 0
    for style in ALL_STYLES:
        for i in range(2):
            # Clean up agent name (remove enum prefix)
            style_name = str(style.name).replace("PromptStyleName.", "")
            agent_name = f"{style_name} #{i+1}"
            agent_info.append({"name": agent_name, "style": style_name})
            # Stagger by 1 second per agent to avoid rate limiting
            delay = agent_num * 1.0
            tasks.append(generate_fix(client, style, i, rag_context, delay_seconds=delay))
            agent_num += 1

    print(f"\n  Generating fixes from {len(tasks)} agents...")

    # Run all agents in parallel
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, (info, response) in enumerate(zip(agent_info, responses)):
        if isinstance(response, Exception):
            agent_results.append({
                "agent": info["name"],
                "style": info["style"],
                "success": False,
                "error": str(response),
                "patterns": {},
                "response_preview": "",
            })
        else:
            patterns = check_patterns(response)
            agent_results.append({
                "agent": info["name"],
                "style": info["style"],
                "success": True,
                "patterns": patterns,
                "response_preview": response[:500].replace("\n", " ")[:200] + "...",
                "full_response": response,
            })

    # Step 3: Summary Table
    print("\n" + "=" * 80)
    print(" STEP 3: RESULTS SUMMARY TABLE")
    print("=" * 80)

    # RAG Sources table
    print("\n### RAG DATA SOURCES")
    print("-" * 80)
    print(f"{'Source':<20} {'Status':<10} {'Chars':<10} {'Details'}")
    print("-" * 80)

    ctx7_status = "OK" if rag_results["context7"]["success"] else "FAILED"
    print(f"{'Context7':<20} {ctx7_status:<10} {rag_results['context7']['chars']:<10} Python docs")

    web_status = "OK" if rag_results["web_search"]["success"] else "FAILED"
    print(f"{'Web Search':<20} {web_status:<10} {rag_results['web_search']['chars']:<10} DuckDuckGo (no domain filter)")

    if rag_results["web_search"]["sources"]:
        print("\n  Web Search Sources:")
        for src in rag_results["web_search"]["sources"][:6]:
            print(f"    - {src['domain']}: {src['chars']} chars")

    # Patterns discovered table
    print("\n### PATTERNS IN RAG CONTEXT")
    print("-" * 80)
    for pattern, found in rag_patterns.items():
        mark = "[OK]" if found else "[X]"
        print(f"  {mark} {pattern}")

    # Agent results table
    print("\n### AGENT FIX RESULTS")
    print("-" * 80)
    header = f"{'Agent':<25} {'monotonic':<12} {'Lock':<10} {'sleep':<10} {'async with':<12}"
    print(header)
    print("-" * 80)

    for result in agent_results:
        if result["success"]:
            p = result["patterns"]
            mono = "OK" if p.get("time.monotonic") else "X"
            lock = "OK" if p.get("asyncio.Lock") else "X"
            sleep = "OK" if p.get("asyncio.sleep") else "X"
            async_with = "OK" if p.get("async with") else "X"
            print(f"{result['agent']:<25} {mono:<12} {lock:<10} {sleep:<10} {async_with:<12}")
        else:
            print(f"{result['agent']:<25} ERROR: {result['error'][:40]}")

    # Consensus analysis
    print("\n### CONSENSUS ANALYSIS")
    print("-" * 80)

    # Count how many agents used each pattern
    pattern_counts = {
        "time.monotonic": 0,
        "asyncio.Lock": 0,
        "asyncio.sleep": 0,
        "async with": 0,
    }

    successful_agents = [r for r in agent_results if r["success"]]
    for result in successful_agents:
        for pattern in pattern_counts:
            if result["patterns"].get(pattern):
                pattern_counts[pattern] += 1

    print(f"Total successful agents: {len(successful_agents)}")
    print("\nPattern adoption:")
    for pattern, count in pattern_counts.items():
        pct = (count / len(successful_agents) * 100) if successful_agents else 0
        bar = "=" * int(pct / 5)
        print(f"  {pattern:<20} {count:>2}/{len(successful_agents)} ({pct:>5.1f}%) {bar}")

    # Quality score
    print("\n### QUALITY SCORES")
    print("-" * 80)
    print(f"{'Agent':<25} {'Score':<10} {'Grade'}")
    print("-" * 80)

    for result in agent_results:
        if result["success"]:
            p = result["patterns"]
            score = sum([
                2 if p.get("time.monotonic") else 0,  # Most important
                2 if p.get("asyncio.Lock") else 0,
                1 if p.get("asyncio.sleep") else 0,
                1 if p.get("async with") else 0,
            ])
            grade = "A+" if score >= 5 else "A" if score >= 4 else "B" if score >= 3 else "C" if score >= 2 else "F"
            print(f"{result['agent']:<25} {score}/6        {grade}")

    print("\n" + "=" * 80)
    print(" TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
