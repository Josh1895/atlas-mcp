#!/usr/bin/env python3
"""Test ATLAS on a local coding issue.

This script demonstrates ATLAS solving a real bug without needing GitHub.
It creates a temporary file with a bug, then uses ATLAS agents to fix it.

Usage:
    python scripts/solve_local_issue.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from atlas.core.config import Config
from atlas.agents.gemini_client import GeminiClient
from atlas.agents.prompt_styles import PromptStyleName, get_style_by_name, BASE_SYSTEM_PROMPT
from atlas.quality.pipeline import QualitySelectionPipeline, QualitySelectionConfig


# Buggy code to fix
BUGGY_CODE = '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def get_user_display_name(user):
    """Get the display name for a user."""
    return user["first_name"] + " " + user["last_name"]


def find_item_by_id(items, target_id):
    """Find an item in a list by its ID."""
    for item in items:
        if item["id"] == target_id:
            return item
    # Bug: returns None implicitly, caller might not handle it
'''

ISSUE_DESCRIPTION = """
Fix the following bugs in utils.py:

1. calculate_average() crashes with ZeroDivisionError when given an empty list
2. get_user_display_name() crashes with KeyError if user dict is missing keys
3. find_item_by_id() returns None silently which can cause downstream errors

Please add proper error handling and return sensible defaults or raise clear exceptions.
"""


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


async def generate_patch(client: GeminiClient, style_name: PromptStyleName, code: str, issue: str) -> str:
    """Generate a patch using a specific prompt style."""
    style = get_style_by_name(style_name)
    system_prompt = style.get_system_prompt(BASE_SYSTEM_PROMPT)

    prompt = f"""{system_prompt}

## Current Code (utils.py)
```python
{code}
```

## Issue to Fix
{issue}

Generate a unified diff patch that fixes the issues. Output ONLY the patch in this format:

```diff
--- a/utils.py
+++ b/utils.py
@@ ... @@
 context line
-removed line
+added line
```

Patch:"""

    temp = 0.7 + style.temperature_offset
    result = await client.generate(
        prompt=prompt,
        max_tokens=2000,
        temperature=temp,
    )

    return result.text


async def main():
    print_header("ATLAS Local Issue Solver")

    # Initialize
    config = Config.from_env()
    if not config.gemini_api_key:
        print("[ERROR] GEMINI_API_KEY not set in .env file")
        return

    client = GeminiClient(config)

    print("\n[BUGGY CODE]")
    print(BUGGY_CODE)

    print("\n[ISSUE TO FIX]")
    print(ISSUE_DESCRIPTION)

    # Generate patches with different styles
    print_header("Stage 1: Generating Patches (5 different approaches)")

    styles = [
        PromptStyleName.MINIMAL_DIFF,
        PromptStyleName.VERBOSE_EXPLAINER,
        PromptStyleName.REFACTOR_FIRST,
        PromptStyleName.DEBUGGER,
        PromptStyleName.REPO_ONLY,
    ]

    patches = {}
    for i, style in enumerate(styles):
        print(f"\n  [{i+1}/5] Generating with {style.value} style...", end=" ", flush=True)
        try:
            patch = await generate_patch(client, style, BUGGY_CODE, ISSUE_DESCRIPTION)
            patches[f"agent_{i}_{style.value}"] = patch
            print("Done")
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\n  Generated {len(patches)} patches")

    # Show the patches
    print_header("Stage 2: Generated Patches Preview")
    for patch_id, patch in patches.items():
        print(f"\n--- {patch_id} ---")
        # Show first 500 chars
        preview = patch[:500] + "..." if len(patch) > 500 else patch
        print(preview)

    # Run quality selection
    print_header("Stage 3: Quality Selection")

    quality_config = QualitySelectionConfig(
        enable_llm_review=True,
        enable_tournament=len(patches) >= 3,
        max_patches_for_tournament=5,
    )

    pipeline = QualitySelectionPipeline(
        config=quality_config,
        llm_client=client,
    )

    # Build file maps
    original_files = {"utils.py": BUGGY_CODE}
    patched_files_map = {pid: {"utils.py": BUGGY_CODE} for pid in patches.keys()}

    print("\n  Running clustering, scoring, and LLM review...")

    result = await pipeline.select_best_patch(
        patches=patches,
        issue_description=ISSUE_DESCRIPTION,
        original_files=original_files,
        patched_files_map=patched_files_map,
        context_code=BUGGY_CODE,
    )

    # Show results
    print_header("Stage 4: Results")

    print(f"\n  Patches analyzed: {result.total_patches}")
    print(f"  Approach families found: {result.families_found}")
    print(f"  Patches after quality gates: {result.patches_after_gates}")

    print(f"\n  [WINNER]: {result.selection.best_patch_id}")
    print(f"  Family: {result.selection.approach_family}")
    print(f"  Reason: {result.selection.selection_reason}")

    if result.selection.alternates:
        print(f"\n  Alternates:")
        for alt_id, _ in result.selection.alternates[:3]:
            print(f"    - {alt_id}")

    print_header("Winning Patch")
    print(result.selection.best_patch_content)

    # Show quality scores
    print_header("Quality Scores Summary")
    sorted_scores = sorted(
        result.quality_scores.items(),
        key=lambda x: x[1].overall_score,
        reverse=True
    )
    for patch_id, score in sorted_scores[:5]:
        risk_flag = " [RISK]" if score.risk_flags else ""
        print(f"  {patch_id}: {score.overall_score:.1f} (style={score.style_score:.0f}, maint={score.maintainability_score:.0f}, risk={score.risk_score:.0f}){risk_flag}")

    print("\n[DONE] ATLAS successfully analyzed and selected the best patch!")


if __name__ == "__main__":
    asyncio.run(main())
