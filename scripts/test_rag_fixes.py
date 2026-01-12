#!/usr/bin/env python3
"""Test script to verify RAG integration fixes.

Tests:
1. Context7 MCP endpoint connectivity
2. Web search functionality (DuckDuckGo fallback)
3. Content quality for asyncio rate limiter problem
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from atlas.core.config import Config
from atlas.rag.web_search import WebSearchClient
from atlas.rag.context7 import Context7Client


async def test_context7():
    """Test Context7 MCP connectivity."""
    print("\n" + "=" * 70)
    print(" TESTING CONTEXT7 MCP")
    print("=" * 70)

    config = Config.from_env()
    client = Context7Client(config)

    print(f"\n[INFO] API Key configured: {'Yes' if config.context7_api_key else 'No'}")
    print(f"[INFO] API Key prefix: {config.context7_api_key[:10]}..." if config.context7_api_key else "")

    # Test 1: Resolve library ID
    print("\n[TEST 1] Resolving library ID for 'python'...")
    lib_id = await client.resolve_library_id("python")
    if lib_id:
        print(f"  [SUCCESS] Resolved to: {lib_id}")
    else:
        print("  [FAILED] Could not resolve library ID")

    # Test 2: Query docs directly with known library ID
    print("\n[TEST 2] Querying docs for /python/cpython...")
    result = await client.query_docs(
        library_id="/python/cpython",
        query="asyncio Lock synchronization time monotonic",
        max_tokens=2000,
    )

    if result.chunks:
        print(f"  [SUCCESS] Got {len(result.chunks)} chunks")
        for i, chunk in enumerate(result.chunks[:2]):
            preview = chunk.content[:150].replace('\n', ' ')
            print(f"  Chunk {i+1}: {preview}...")
    else:
        print("  [FAILED] No documentation chunks returned")

    # Test 3: Get documentation (full flow)
    print("\n[TEST 3] Full documentation flow for 'asyncio'...")
    result = await client.get_documentation(
        library_name="asyncio",
        query="async rate limiter Lock best practices",
        max_tokens=2000,
    )

    if result.chunks:
        print(f"  [SUCCESS] Got {len(result.chunks)} chunks ({result.total_tokens} tokens)")
        print(f"  Combined content length: {len(result.combined_content)} chars")
    else:
        print("  [FAILED] No documentation returned")

    return bool(result.chunks)


async def test_web_search():
    """Test web search functionality."""
    print("\n" + "=" * 70)
    print(" TESTING WEB SEARCH")
    print("=" * 70)

    config = Config.from_env()
    client = WebSearchClient(config)

    print(f"\n[INFO] SerpAPI Key configured: {'Yes' if client.serpapi_key else 'No (using DuckDuckGo fallback)'}")

    # Test 1: Basic search
    print("\n[TEST 1] Basic search: 'python asyncio time.monotonic vs time.time'...")
    results = await client.search(
        "python asyncio time.monotonic vs time.time best practices",
        max_results=5,
    )

    if results.results:
        print(f"  [SUCCESS] Found {len(results.results)} results")
        for r in results.results[:3]:
            print(f"  - {r.title[:50]}...")
            print(f"    URL: {r.url[:60]}...")
            print(f"    Snippet: {r.snippet[:100]}..." if r.snippet else "    (no snippet)")
    else:
        print(f"  [FAILED] No results. Error: {results.error}")

    # Test 2: Stack Overflow search
    print("\n[TEST 2] Stack Overflow search: 'asyncio rate limiter'...")
    so_results = await client.search(
        "asyncio rate limiter implementation",
        max_results=3,
        site_filter="stackoverflow.com",
    )

    if so_results.results:
        print(f"  [SUCCESS] Found {len(so_results.results)} Stack Overflow results")
        for r in so_results.results:
            print(f"  - {r.title[:60]}...")
    else:
        print(f"  [FAILED] No Stack Overflow results")

    # Test 3: Comprehensive code context search
    print("\n[TEST 3] Comprehensive code context search...")
    context_results = await client.search_for_code_context(
        issue_description="asyncio rate limiter race condition time.sleep blocking",
        libraries=["asyncio", "aiohttp"],
    )

    if context_results.results:
        print(f"  [SUCCESS] Found {len(context_results.results)} results from multiple sources")
        total_content = sum(len(r.snippet) for r in context_results.results)
        print(f"  Total content gathered: {total_content} chars")
        print(f"  Combined content length: {len(context_results.combined_content)} chars")

        print("\n  Sample sources:")
        for r in context_results.results[:5]:
            source_domain = r.url.split('/')[2] if '/' in r.url else r.url
            print(f"  - [{source_domain}] {r.title[:50]}...")
    else:
        print(f"  [FAILED] No context gathered")

    return bool(context_results.results)


async def test_rate_limiter_research():
    """Test if we can gather enough context to discover best practices."""
    print("\n" + "=" * 70)
    print(" RESEARCH TEST: Can we discover time.monotonic()?")
    print("=" * 70)

    config = Config.from_env()
    web_client = WebSearchClient(config)

    # Specific searches that should reveal best practices
    searches = [
        "python time.time vs time.monotonic which to use",
        "python asyncio.Lock vs threading.Lock async",
        "python asyncio.sleep vs time.sleep blocking event loop",
        "python rate limiter implementation production best practices",
    ]

    all_content = []
    for query in searches:
        print(f"\n[SEARCH] {query[:50]}...")
        results = await web_client.search(query, max_results=3)
        if results.results:
            print(f"  Found {len(results.results)} results")
            for r in results.results:
                if r.snippet:
                    all_content.append(r.snippet)
        else:
            print("  No results")

    # Check if we found key best practices
    combined = " ".join(all_content).lower()

    print("\n" + "-" * 70)
    print(" BEST PRACTICES DISCOVERY CHECK")
    print("-" * 70)

    checks = [
        ("time.monotonic()", "monotonic" in combined),
        ("asyncio.Lock()", "asyncio.lock" in combined or "asyncio lock" in combined),
        ("asyncio.sleep()", "asyncio.sleep" in combined),
        ("async with", "async with" in combined),
        ("clock drift/adjustment", "clock" in combined and ("drift" in combined or "adjust" in combined or "backward" in combined)),
    ]

    all_found = True
    for practice, found in checks:
        status = "[FOUND]" if found else "[NOT FOUND]"
        print(f"  {status} {practice}")
        if not found:
            all_found = False

    print(f"\n  Total research content: {len(combined)} chars")

    return all_found


async def main():
    print("=" * 70)
    print(" ATLAS RAG INTEGRATION TEST")
    print("=" * 70)

    # Run all tests
    ctx7_ok = await test_context7()
    web_ok = await test_web_search()
    research_ok = await test_rate_limiter_research()

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
    Context7 MCP:        {'[OK]' if ctx7_ok else '[FAILED]'}
    Web Search:          {'[OK]' if web_ok else '[FAILED]'}
    Best Practice Discovery: {'[OK]' if research_ok else '[PARTIAL]'}

    Next steps:
    - If Context7 failed: Check API key and MCP endpoint connectivity
    - If Web Search failed: DuckDuckGo may be blocking; consider adding SerpAPI key
    - If Discovery failed: Need more targeted searches or additional sources
    """)


if __name__ == "__main__":
    asyncio.run(main())
