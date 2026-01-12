#!/usr/bin/env python3
"""Debug page fetching to see why it's failing."""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.rag.web_search import WebSearchClient
from atlas.core.config import Config


async def test():
    config = Config.from_env()
    client = WebSearchClient(config)

    # Test URLs from different sources
    test_urls = [
        "https://stackoverflow.com/questions/tagged/asyncio",
        "https://realpython.com/async-io-python/",
        "https://docs.python.org/3/library/time.html",
        "https://superfastpython.com/asyncio-lock/",
    ]

    print("Testing page content fetching...")
    print("=" * 70)

    for url in test_urls:
        print(f"\nFetching: {url}")
        content = await client.fetch_page_content(url, max_chars=5000)
        if content:
            print(f"  SUCCESS: {len(content)} chars")
            # Check for keywords
            lower = content.lower()
            keywords = ["monotonic", "asyncio.lock", "asyncio.sleep", "async with"]
            found = [k for k in keywords if k in lower]
            if found:
                print(f"  Keywords found: {found}")
            print(f"  Preview: {content[:200].replace(chr(10), ' ')}...")
        else:
            print("  FAILED: Empty content returned")


if __name__ == "__main__":
    asyncio.run(test())
