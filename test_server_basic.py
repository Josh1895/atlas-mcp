#!/usr/bin/env python3
"""Basic MCP server test - validates config and simple operations."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

async def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print(" TEST 1: Configuration")
    print("=" * 60)

    from atlas.core.config import Config, get_config

    config = get_config()
    print(f"  Model: {config.model}")
    print(f"  Voting K: {config.voting_k}")
    print(f"  Max samples: {config.max_samples}")
    print(f"  Has Gemini Key: {bool(config.gemini_api_key)}")
    print(f"  Has Context7 Key: {bool(config.context7_api_key)}")

    errors = config.validate()
    if errors:
        print(f"  Errors: {errors}")
        return False
    print("  Config: VALID")
    return True


async def test_server_import():
    """Test that the MCP server can be imported."""
    print("\n" + "=" * 60)
    print(" TEST 2: Server Import")
    print("=" * 60)

    try:
        from atlas.server import mcp, get_config_info
        print(f"  MCP server name: {mcp.name}")
        print(f"  Instructions: {getattr(mcp, 'instructions', 'N/A')}")

        # List available tools via internal attributes
        tools = []
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            tools = list(mcp._tool_manager._tools.keys())
        print(f"  Available tools: {len(tools)}")
        for tool in tools[:8]:
            print(f"    - {tool}")
        if len(tools) > 8:
            print(f"    ... and {len(tools) - 8} more")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_get_config_info():
    """Test the get_config_info tool."""
    print("\n" + "=" * 60)
    print(" TEST 3: get_config_info() Tool")
    print("=" * 60)

    try:
        from atlas.server import get_config_info
        result = await get_config_info()
        print(f"  Result: {result}")
        return result.get("is_valid", False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_agents():
    """Test that agents can be instantiated."""
    print("\n" + "=" * 60)
    print(" TEST 4: Agent Styles")
    print("=" * 60)

    try:
        from atlas.agents.prompt_styles import ALL_STYLES, get_style_by_name, PromptStyleName
        print(f"  Available styles: {len(ALL_STYLES)}")
        for style in ALL_STYLES:
            print(f"    - {style.name.value}: {style.description}")

        # Test creating one
        style = get_style_by_name(PromptStyleName.SENIOR_ENGINEER)
        print(f"\n  Sample agent 'senior_engineer':")
        print(f"    Temperature offset: {style.temperature_offset}")
        print(f"    Use Web RAG: {style.use_web_rag}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_gemini_connection():
    """Test Gemini API connection."""
    print("\n" + "=" * 60)
    print(" TEST 5: Gemini API Connection")
    print("=" * 60)

    try:
        from atlas.core.config import get_config
        import google.generativeai as genai

        config = get_config()
        genai.configure(api_key=config.gemini_api_key)

        model = genai.GenerativeModel(config.model)
        response = await model.generate_content_async(
            "Say 'Hello from ATLAS!' in exactly those words.",
            generation_config={"max_output_tokens": 50}
        )

        print(f"  Model: {config.model}")
        print(f"  Response: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    print("=" * 60)
    print(" ATLAS MCP SERVER BASIC TEST")
    print("=" * 60)

    results = {}

    results["config"] = await test_config()
    results["server_import"] = await test_server_import()
    results["get_config_info"] = await test_get_config_info()
    results["agents"] = await test_agents()
    results["gemini"] = await test_gemini_connection()

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL TESTS PASSED!")
    else:
        print(" SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
