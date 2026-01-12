#!/usr/bin/env python3
"""Test Context7 MCP client connection."""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from atlas.core.config import Config
from atlas.rag.context7 import Context7Client


async def test_raw_api():
    """Test the raw API directly to debug."""
    print("\n--- RAW API TEST ---")

    config = Config.from_env()
    print(f"API Key configured: {'Yes' if config.context7_api_key else 'No'}")

    base_url = "https://context7.com/api/v2"

    headers = {"Accept": "application/json"}
    if config.context7_api_key:
        headers["Authorization"] = f"Bearer {config.context7_api_key}"

    async with httpx.AsyncClient() as client:
        # Test resolve
        print("\n[RAW] Testing /libs/search...")
        try:
            resp = await client.get(
                f"{base_url}/libs/search",
                params={"libraryName": "react"},
                headers=headers,
                timeout=30.0
            )
            print(f"  Status: {resp.status_code}")
            print(f"  Response: {resp.text[:500]}...")
        except Exception as e:
            print(f"  Error: {e}")

        # Test context
        print("\n[RAW] Testing /context...")
        try:
            resp = await client.get(
                f"{base_url}/context",
                params={
                    "libraryId": "/websites/react_dev",
                    "query": "hooks",
                    "tokens": 2000
                },
                headers=headers,
                timeout=30.0
            )
            print(f"  Status: {resp.status_code}")
            print(f"  Content-Type: {resp.headers.get('content-type')}")
            print(f"  Response length: {len(resp.text)} chars")
            if resp.text:
                print(f"  Preview: {resp.text[:300]}...")
        except Exception as e:
            print(f"  Error: {e}")


async def test():
    print("=" * 70)
    print(" CONTEXT7 MCP CLIENT TEST")
    print("=" * 70)

    # First test raw API
    await test_raw_api()

    config = Config.from_env()
    client = Context7Client(config)

    # Test 1: Resolve library ID
    print("\n" + "=" * 70)
    print("[TEST 1] Resolve library ID for 'react'...")
    lib_id = await client.resolve_library_id("react")
    if lib_id:
        print(f"  SUCCESS: Resolved to '{lib_id}'")
    else:
        print("  FAILED: Could not resolve")

    # Test 2: Query React docs (using resolved ID)
    print("\n[TEST 2] Query React docs...")
    if lib_id:
        result = await client.get_documentation(
            library_name=lib_id,
            query="hooks useState useEffect",
            max_tokens=2000,
        )

        if result.chunks:
            print(f"  SUCCESS: Got {len(result.chunks)} chunks")
            print(f"  Total chars: {len(result.combined_content)}")
            preview = result.combined_content[:300].replace("\n", " ")
            print(f"  Preview: {preview}...")
        else:
            print("  FAILED: No documentation returned")
    else:
        print("  SKIPPED: No library ID to query")

    # Test 3: Try Next.js (resolve first)
    print("\n[TEST 3] Resolve and query Next.js docs...")
    nextjs_id = await client.resolve_library_id("next.js")
    if nextjs_id:
        print(f"  Resolved Next.js to: {nextjs_id}")
        result2 = await client.get_documentation(
            library_name=nextjs_id,
            query="app router server components",
            max_tokens=2000,
        )
        if result2.chunks:
            print(f"  SUCCESS: Got {len(result2.chunks)} chunks, {len(result2.combined_content)} chars")
        else:
            print("  FAILED: No documentation returned")
    else:
        print("  FAILED: Could not resolve next.js")

    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test())
