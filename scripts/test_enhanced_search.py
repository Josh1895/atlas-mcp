#!/usr/bin/env python3
"""Test enhanced search with page content fetching."""

import asyncio
import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.rag.web_search import WebSearchClient
from atlas.core.config import Config


async def test():
    config = Config.from_env()
    client = WebSearchClient(config)

    print("Testing search_for_code_context with enhanced page fetching...")
    print("=" * 70)

    results = await client.search_for_code_context(
        issue_description="asyncio rate limiter race condition time.sleep blocking",
        libraries=["asyncio", "aiohttp"],
    )

    print(f"\nResults: {len(results.results)}")
    total_content = sum(len(r.snippet) for r in results.results)
    print(f"Total content: {total_content} chars")

    # Show what we got
    print("\n--- Top Results ---")
    for i, r in enumerate(results.results[:5]):
        domain = r.url.split("/")[2] if "/" in r.url else r.url
        print(f"\n[{i+1}] {domain}")
        print(f"    Snippet length: {len(r.snippet)} chars")
        preview = r.snippet[:150].replace("\n", " ")
        print(f"    Preview: {preview}...")

    # Check for best practices
    combined = results.combined_content.lower()
    print("\n" + "=" * 70)
    print("--- Best Practices Discovery Check ---")
    print("=" * 70)

    checks = [
        ("time.monotonic()", "monotonic" in combined),
        ("asyncio.Lock()", "asyncio.lock" in combined or "asyncio lock" in combined),
        ("asyncio.sleep()", "asyncio.sleep" in combined),
        ("async with", "async with" in combined),
        ("clock drift/backward", "clock" in combined and ("drift" in combined or "backward" in combined)),
        ("await", "await" in combined),
    ]

    for practice, found in checks:
        status = "[FOUND]" if found else "[NOT FOUND]"
        print(f"  {status} {practice}")

    print(f"\nTotal combined content: {len(combined)} chars")

    # Compare with simple search
    print("\n" + "=" * 70)
    print("--- Comparison: Simple search (no page fetching) ---")
    print("=" * 70)

    simple_results = await client.search(
        "python asyncio rate limiter time.monotonic best practices",
        max_results=5,
    )
    simple_content = sum(len(r.snippet) for r in simple_results.results)
    print(f"Simple search results: {len(simple_results.results)}")
    print(f"Simple search content: {simple_content} chars")


if __name__ == "__main__":
    asyncio.run(test())
