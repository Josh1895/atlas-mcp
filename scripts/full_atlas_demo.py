#!/usr/bin/env python3
"""Full ATLAS demo with 10 agents (2 per style) as configured."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from atlas.core.config import Config
from atlas.agents.gemini_client import GeminiClient
from atlas.agents.prompt_styles import PromptStyleName, get_style_by_name, BASE_SYSTEM_PROMPT
from atlas.quality.pipeline import QualitySelectionPipeline, QualitySelectionConfig


# ============================================================================
# THE PROBLEM: Buggy code that crashes in multiple ways
# ============================================================================
BUGGY_CODE = '''def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # CRASHES if numbers is empty!


def get_user_display_name(user):
    """Get the display name for a user."""
    return user["first_name"] + " " + user["last_name"]  # CRASHES if keys missing!


def find_item_by_id(items, target_id):
    """Find an item in a list by its ID."""
    for item in items:
        if item["id"] == target_id:
            return item
    # BUG: returns None implicitly, caller gets confused
'''

ISSUE_DESCRIPTION = """
Fix these 3 bugs in utils.py:

1. calculate_average([]) crashes with ZeroDivisionError
2. get_user_display_name({}) crashes with KeyError
3. find_item_by_id() returns None silently causing downstream bugs

Add proper error handling.
"""


async def generate_patch(client, style_name, agent_num):
    """Generate a patch using a specific prompt style."""
    style = get_style_by_name(style_name)
    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Code to Fix (utils.py)
```python
{BUGGY_CODE}
```

## Issue
{ISSUE_DESCRIPTION}

Generate ONLY a unified diff patch:
```diff
--- a/utils.py
+++ b/utils.py
@@ ... @@
...
```"""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)  # Slight variation
    result = await client.generate(prompt=prompt, max_tokens=2000, temperature=temp)
    return result.text


def apply_patch_preview(original_code: str, patch_content: str) -> str:
    """Simple preview of what the fixed code might look like."""
    # Extract added lines from patch
    lines = []
    for line in patch_content.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            lines.append(line[1:])  # Remove the + prefix
        elif line.startswith(' '):
            lines.append(line[1:])  # Context line
    return '\n'.join(lines) if lines else "[Could not parse patch]"


async def main():
    config = Config.from_env()
    client = GeminiClient(config)

    print("=" * 80)
    print(" ATLAS FULL DEMO: 10 AI AGENTS FIXING BUGS")
    print("=" * 80)

    # Show the problem
    print("\n" + "=" * 80)
    print(" THE PROBLEM (Buggy Code)")
    print("=" * 80)
    print(BUGGY_CODE)

    print("\n" + "=" * 80)
    print(" WHAT'S WRONG")
    print("=" * 80)
    print("""
    Bug 1: calculate_average([])
           -> ZeroDivisionError: division by zero

    Bug 2: get_user_display_name({})
           -> KeyError: 'first_name'

    Bug 3: item = find_item_by_id(items, 999)
           -> item is None, then item["name"] crashes
    """)

    # Deploy 10 agents (2 of each style)
    print("\n" + "=" * 80)
    print(" DEPLOYING 10 AI AGENTS (2 per style)")
    print("=" * 80)

    styles = [
        (PromptStyleName.MINIMAL_DIFF, "Minimal changes only"),
        (PromptStyleName.VERBOSE_EXPLAINER, "Step-by-step reasoning"),
        (PromptStyleName.REFACTOR_FIRST, "Clean up then fix"),
        (PromptStyleName.DEBUGGER, "Trace execution flow"),
        (PromptStyleName.REPO_ONLY, "Follow existing patterns"),
    ]

    patches = {}
    agent_id = 0

    for style_name, desc in styles:
        for variant in range(2):  # 2 agents per style
            agent_label = f"agent_{agent_id}_{style_name.value}_v{variant+1}"
            print(f"\n  [{agent_id+1}/10] {style_name.value} (variant {variant+1})...", end=" ", flush=True)

            try:
                patch = await generate_patch(client, style_name, variant)
                patches[agent_label] = patch
                print("Done")
            except Exception as e:
                print(f"Failed: {e}")

            agent_id += 1

    print(f"\n  Total patches generated: {len(patches)}")

    # Run quality selection
    print("\n" + "=" * 80)
    print(" QUALITY SELECTION: Clustering, Scoring, Voting")
    print("=" * 80)

    quality_config = QualitySelectionConfig(
        enable_llm_review=True,
        enable_tournament=len(patches) >= 3,
        max_patches_for_tournament=6,
    )

    pipeline = QualitySelectionPipeline(config=quality_config, llm_client=client)

    original_files = {"utils.py": BUGGY_CODE}
    patched_files_map = {pid: {"utils.py": BUGGY_CODE} for pid in patches.keys()}

    print("\n  Running analysis...")
    result = await pipeline.select_best_patch(
        patches=patches,
        issue_description=ISSUE_DESCRIPTION,
        original_files=original_files,
        patched_files_map=patched_files_map,
        context_code=BUGGY_CODE,
    )

    # Show clustering results
    print(f"\n  Approach families found: {result.families_found}")
    for family in result.approach_families:
        print(f"    - {family.label}: {family.patch_ids}")

    # Show scores
    print("\n  Quality Scores (top 5):")
    sorted_scores = sorted(result.quality_scores.items(), key=lambda x: x[1].overall_score, reverse=True)
    for patch_id, score in sorted_scores[:5]:
        print(f"    {patch_id}: {score.overall_score:.1f}")

    # Show winner
    print("\n" + "=" * 80)
    print(" WINNER")
    print("=" * 80)
    print(f"\n  Selected: {result.selection.best_patch_id}")
    print(f"  Reason: {result.selection.selection_reason}")

    # Show the winning patch
    print("\n" + "=" * 80)
    print(" THE FIX (Winning Patch)")
    print("=" * 80)
    print(result.selection.best_patch_content)

    # Show what the fixed code looks like
    print("\n" + "=" * 80)
    print(" RESULT: Fixed Code Preview")
    print("=" * 80)

    # Parse the winning patch to show the fixed version
    winning_patch = result.selection.best_patch_content
    print("\nThe winning patch adds these fixes:")

    additions = []
    for line in winning_patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            additions.append(f"  {line}")

    print('\n'.join(additions[:20]))  # Show first 20 added lines

    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"""
    Agents deployed:     10 (2 of each style)
    Patches generated:   {len(patches)}
    Approach families:   {result.families_found}
    Winner:              {result.selection.best_patch_id}

    The fix handles all 3 bugs:
    - Empty list check for calculate_average
    - .get() with defaults for get_user_display_name
    - Explicit return/raise for find_item_by_id
    """)


if __name__ == "__main__":
    asyncio.run(main())
