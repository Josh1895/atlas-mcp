#!/usr/bin/env python3
"""
ATLAS Full RAG Demo - Shows web search + Context7 + all 10 agent outputs.

This demonstrates:
1. A tough real-world coding problem
2. 10 agents deployed (2 per style)
3. Each agent's web search results
4. Each agent's Context7 documentation lookup
5. Each agent's exact solution
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from atlas.core.config import Config
from atlas.core.task import TaskSubmission
from atlas.agents.gemini_client import GeminiClient
from atlas.agents.micro_agent import MicroAgent, AgentContext
from atlas.agents.prompt_styles import (
    PromptStyleName, get_style_by_name, ALL_STYLES,
    MINIMAL_DIFF, VERBOSE_EXPLAINER, REFACTOR_FIRST, DEBUGGER, REPO_ONLY
)
from atlas.rag.web_search import WebSearchClient
from atlas.rag.context7 import Context7Client
from atlas.quality.pipeline import QualitySelectionPipeline, QualitySelectionConfig


# ============================================================================
# TOUGH REAL-WORLD PROBLEM: Race condition in async rate limiter
# ============================================================================

TOUGH_CODE = '''
import asyncio
import time
from typing import Optional
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_second: float = 10.0
    burst_size: int = 5
    timeout: float = 30.0

class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_update = time.time()
        self._lock = None  # BUG: Lock not initialized properly for async

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        # BUG: Race condition - multiple coroutines can read/write tokens simultaneously
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on time elapsed
        self.tokens = min(
            self.config.burst_size,
            self.tokens + elapsed * self.config.requests_per_second
        )
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def wait_for_token(self, timeout: Optional[float] = None) -> bool:
        """Wait until a token is available."""
        timeout = timeout or self.config.timeout
        start = time.time()

        # BUG: Busy-waiting without proper backoff, wastes CPU
        while time.time() - start < timeout:
            if await self.acquire():
                return True
            # BUG: No await here causes blocking
            time.sleep(0.01)  # Should be asyncio.sleep

        return False


class APIClient:
    """HTTP client with rate limiting."""

    def __init__(self, base_url: str, rate_limiter: AsyncRateLimiter):
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self._session = None  # BUG: Session not properly managed

    async def get(self, endpoint: str) -> dict:
        """Make a rate-limited GET request."""
        # BUG: No error handling for rate limit timeout
        await self.rate_limiter.wait_for_token()

        # BUG: Session created but never closed, resource leak
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}{endpoint}") as resp:
                return await resp.json()

    async def batch_get(self, endpoints: list[str]) -> list[dict]:
        """Make multiple requests with rate limiting."""
        # BUG: No concurrency limit, can overwhelm the rate limiter
        tasks = [self.get(ep) for ep in endpoints]
        return await asyncio.gather(*tasks)
'''

TOUGH_ISSUE = """
Fix the critical bugs in this async rate limiter implementation:

1. RACE CONDITION: Multiple coroutines can access `self.tokens` simultaneously without synchronization, causing token counts to become incorrect under concurrent load.

2. BLOCKING CALL: `time.sleep(0.01)` blocks the event loop. Must use `asyncio.sleep()` for proper async operation.

3. LOCK INITIALIZATION: The `_lock` is set to None but never initialized as an `asyncio.Lock()`.

4. RESOURCE LEAK: Each API call creates a new `aiohttp.ClientSession` which is inefficient and can leak connections.

5. NO CONCURRENCY CONTROL: `batch_get` fires all requests at once without respecting a semaphore limit.

6. MISSING ERROR HANDLING: No handling for rate limit timeouts or HTTP errors.

This needs to work correctly under high concurrency (100+ simultaneous requests).
Libraries involved: asyncio, aiohttp, Python async/await patterns.
"""


def print_section(title: str, char: str = "="):
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


async def main():
    config = Config.from_env()

    print_section("ATLAS FULL RAG DEMO: 10 AGENTS WITH WEB SEARCH + CONTEXT7")

    # Show the problem
    print_section("THE TOUGH PROBLEM", "-")
    print("\nCode with multiple serious bugs:")
    print(TOUGH_CODE[:1500] + "\n... [truncated]")

    print_section("BUGS TO FIX", "-")
    print(TOUGH_ISSUE)

    # Create task submission
    task = TaskSubmission(
        description=TOUGH_ISSUE,
        repository_url="",
        relevant_files=["rate_limiter.py"],
    )

    # Initialize RAG clients
    web_search = WebSearchClient(config)
    context7 = Context7Client(config)

    # Define all 10 agents (2 per style)
    styles = [
        (MINIMAL_DIFF, "Smallest possible fix"),
        (MINIMAL_DIFF, "Smallest possible fix v2"),
        (VERBOSE_EXPLAINER, "Step-by-step reasoning"),
        (VERBOSE_EXPLAINER, "Step-by-step reasoning v2"),
        (REFACTOR_FIRST, "Clean up then fix"),
        (REFACTOR_FIRST, "Clean up then fix v2"),
        (DEBUGGER, "Trace execution flow"),
        (DEBUGGER, "Trace execution flow v2"),
        (REPO_ONLY, "Follow existing patterns"),
        (REPO_ONLY, "Follow existing patterns v2"),
    ]

    all_agent_data = []

    print_section("DEPLOYING 10 AGENTS WITH RAG")

    for i, (style, desc) in enumerate(styles):
        agent_id = f"agent_{i}_{style.name.value}"

        print(f"\n{'-' * 80}")
        print(f"AGENT {i+1}/10: {agent_id}")
        print(f"Style: {style.name.value} - {desc}")
        print(f"Uses Web RAG: {style.use_web_rag}")
        print(f"{'-' * 80}")

        agent_data = {
            "id": agent_id,
            "style": style.name.value,
            "description": desc,
            "web_results": None,
            "context7_results": None,
            "solution": None,
        }

        # Create the micro agent
        agent = MicroAgent(
            agent_id=agent_id,
            prompt_style=style,
            config=config,
        )

        # Step 1: Gather RAG context (web search + Context7)
        print(f"\n  [1] Gathering RAG context...")

        if style.use_web_rag:
            # Web Search
            print(f"      Web Search: Searching StackOverflow, GitHub...")
            try:
                web_results = await agent.web_search_client.search_for_code_context(
                    issue_description="asyncio rate limiter race condition token bucket python",
                    libraries=["asyncio", "aiohttp"],
                )
                agent_data["web_results"] = web_results

                if web_results.results:
                    print(f"      Found {len(web_results.results)} web results:")
                    for j, r in enumerate(web_results.results[:3]):
                        print(f"        [{j+1}] {r.title[:60]}...")
                        print(f"            URL: {r.url[:70]}...")
                else:
                    print(f"      No web results found")

            except Exception as e:
                print(f"      Web search failed: {e}")

            # Context7 Documentation
            print(f"\n      Context7: Looking up asyncio/aiohttp docs...")
            try:
                # Try to get asyncio docs
                context7_result = await agent.context7_client.get_documentation(
                    library_name="python",
                    query="asyncio Lock synchronization rate limiting token bucket",
                    max_tokens=2000,
                )
                agent_data["context7_results"] = context7_result

                if context7_result.chunks:
                    print(f"      Found {len(context7_result.chunks)} doc chunks:")
                    for j, chunk in enumerate(context7_result.chunks[:2]):
                        preview = chunk.content[:100].replace('\n', ' ')
                        print(f"        [{j+1}] {chunk.source}: {preview}...")
                else:
                    print(f"      No Context7 docs found (library may not be indexed)")

            except Exception as e:
                print(f"      Context7 lookup failed: {e}")
        else:
            print(f"      Skipping RAG (REPO_ONLY style)")

        # Step 2: Build context and generate solution
        print(f"\n  [2] Generating solution with Gemini...")

        # Build the context with RAG results
        rag_content = ""
        if agent_data["web_results"] and agent_data["web_results"].results:
            rag_content += "\n## Web Search Results\n"
            rag_content += agent_data["web_results"].combined_content[:2000]

        if agent_data["context7_results"] and agent_data["context7_results"].chunks:
            rag_content += "\n\n## Documentation (Context7)\n"
            rag_content += agent_data["context7_results"].combined_content[:2000]

        context = AgentContext(
            task=task,
            repository_content=TOUGH_CODE,
            additional_context=rag_content,
        )

        try:
            solution = await agent.generate(context)
            agent_data["solution"] = solution

            if solution.patch:
                patch_preview = solution.patch[:500]
                print(f"      Generated patch ({len(solution.patch)} chars)")
                print(f"      Tokens used: {solution.tokens_used}")
                print(f"      Cost: ${solution.cost:.4f}")
            else:
                print(f"      No valid patch generated")
                print(f"      Errors: {solution.validation_errors}")

        except Exception as e:
            print(f"      Generation failed: {e}")

        all_agent_data.append(agent_data)

        # Small delay between agents
        await asyncio.sleep(0.5)

    # Show all solutions
    print_section("ALL 10 AGENT SOLUTIONS")

    for agent_data in all_agent_data:
        print(f"\n{'=' * 80}")
        print(f" {agent_data['id'].upper()}")
        print(f" Style: {agent_data['style']} - {agent_data['description']}")
        print(f"{'=' * 80}")

        # Show RAG sources used
        print("\n[RAG SOURCES USED]")
        if agent_data["web_results"] and agent_data["web_results"].results:
            print("  Web Search:")
            for r in agent_data["web_results"].results[:3]:
                print(f"    - {r.title[:50]}... ({r.url[:40]}...)")
        else:
            print("  Web Search: None")

        if agent_data["context7_results"] and agent_data["context7_results"].chunks:
            print("  Context7 Docs:")
            for c in agent_data["context7_results"].chunks[:2]:
                print(f"    - {c.source}")
        else:
            print("  Context7 Docs: None")

        # Show the solution
        print("\n[SOLUTION]")
        if agent_data["solution"] and agent_data["solution"].patch:
            print(agent_data["solution"].patch)
        else:
            print("  No valid solution generated")
            if agent_data["solution"]:
                print(f"  Errors: {agent_data["solution"].validation_errors}")

    # Summary
    print_section("SUMMARY")

    valid_solutions = [a for a in all_agent_data if a["solution"] and a["solution"].patch]
    web_searches = [a for a in all_agent_data if a["web_results"] and a["web_results"].results]
    context7_hits = [a for a in all_agent_data if a["context7_results"] and a["context7_results"].chunks]

    print(f"""
    Agents deployed:        10
    Valid solutions:        {len(valid_solutions)}/10
    Used web search:        {len(web_searches)}/10
    Got Context7 docs:      {len(context7_hits)}/10

    Each agent:
    1. Searched the web (StackOverflow, GitHub) for similar problems
    2. Queried Context7 for asyncio/aiohttp documentation
    3. Combined RAG context with the code problem
    4. Generated a unique solution based on their "thinking style"

    The solutions can now be clustered, scored, and voted on!
    """)


if __name__ == "__main__":
    asyncio.run(main())
