#!/usr/bin/env python3
"""Manual test script for quality selection.

Run with:
    python scripts/test_quality_selection.py

Or with LLM review (requires GEMINI_API_KEY):
    python scripts/test_quality_selection.py --with-llm
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from atlas.quality.fingerprinting import TokenFingerprinter, cluster_by_similarity
from atlas.quality.quality_scorer import QualityScorer
from atlas.quality.selector import FinalSelector
from atlas.quality.pr_reviewer import PRReviewResult
from atlas.quality.pipeline import QualitySelectionPipeline, QualitySelectionConfig


# Example patches for testing
EXAMPLE_PATCHES = {
    # Good: Clean guard clause
    "agent_0_guard_clean": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,9 @@ class UserService:
     def get_display_name(self, user):
+        if user is None:
+            return "Guest"
+
         return user.first_name + " " + user.last_name
""",

    # Good: Similar guard clause (convergence)
    "agent_1_guard_similar": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,8 @@ class UserService:
     def get_display_name(self, user):
+        if not user:
+            return "Guest"
         return user.first_name + " " + user.last_name
""",

    # Okay: Try-except approach
    "agent_2_try_except": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,11 @@ class UserService:
     def get_display_name(self, user):
+        try:
+            return user.first_name + " " + user.last_name
+        except AttributeError:
+            return "Guest"
-        return user.first_name + " " + user.last_name
""",

    # Bad: Bare except (code smell)
    "agent_3_bare_except": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,11 @@ class UserService:
     def get_display_name(self, user):
+        try:
+            return user.first_name + " " + user.last_name
+        except:
+            return "Guest"
-        return user.first_name + " " + user.last_name
""",

    # Bad: Hacky workaround
    "agent_4_hacky": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,9 @@ class UserService:
     def get_display_name(self, user):
+        # HACK: quick fix for prod issue
+        if str(user) == "None":
+            return "Guest"
         return user.first_name + " " + user.last_name
""",

    # Good: getattr approach (Pythonic)
    "agent_5_getattr": """--- a/user_service.py
+++ b/user_service.py
@@ -10,6 +10,8 @@ class UserService:
     def get_display_name(self, user):
+        if user is None:
+            return "Guest"
-        return user.first_name + " " + user.last_name
+        return f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip() or "Guest"
""",
}

ORIGINAL_CODE = """class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        return self.db.find_user(user_id)

    def get_display_name(self, user):
        return user.first_name + " " + user.last_name
"""

ISSUE_DESCRIPTION = """
Fix NullPointerException in get_display_name

When a user is not found, get_display_name raises an AttributeError
because user is None. We should handle this gracefully and return
a default name like "Guest".
"""


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


async def run_quality_selection(with_llm: bool = False):
    """Run the quality selection pipeline on example patches."""

    print_header("ATLAS Quality Selection Test")
    print(f"\nTesting with {len(EXAMPLE_PATCHES)} patches")
    print(f"LLM Review: {'Enabled' if with_llm else 'Disabled'}")

    # Step 1: Token Fingerprinting & Clustering
    print_header("Stage 1: Clustering by Similarity")

    fingerprinter = TokenFingerprinter()
    fingerprints = [
        fingerprinter.fingerprint(patch, patch_id)
        for patch_id, patch in EXAMPLE_PATCHES.items()
    ]

    clusters = cluster_by_similarity(fingerprints, threshold=0.25)

    print(f"\nFound {len(clusters)} approach families:")
    for i, cluster in enumerate(clusters):
        patch_ids = [fp.patch_id for fp in cluster]
        print(f"  Family {i+1} (size {len(cluster)}): {patch_ids}")

    # Step 2: Quality Scoring
    print_header("Stage 2: Objective Quality Scoring")

    scorer = QualityScorer()
    quality_scores = {}

    original_files = {"user_service.py": ORIGINAL_CODE}

    for patch_id, patch in EXAMPLE_PATCHES.items():
        # Simple patched version (in production would apply patch properly)
        patched_files = {"user_service.py": ORIGINAL_CODE + "\n# patched"}

        score = await scorer.score(patch, patch_id, original_files, patched_files)
        quality_scores[patch_id] = score

        print(f"\n{patch_id}:")
        print(f"  Style: {score.style_score:.0f}  Maint: {score.maintainability_score:.0f}  Risk: {score.risk_score:.0f}")
        if score.risk_flags:
            print(f"  Flags: {', '.join(score.risk_flags[:2])}")

    # Step 3: LLM Review (optional)
    review_results = {}

    if with_llm:
        print_header("Stage 3: LLM PR Review")

        try:
            import os
            if not os.getenv("GEMINI_API_KEY"):
                print("  [WARNING] GEMINI_API_KEY not set, skipping LLM review")
            else:
                from atlas.core.config import Config
                from atlas.agents.gemini_client import GeminiClient
                from atlas.quality.pr_reviewer import LLMPRReviewer

                config = Config.from_env()
                llm_client = GeminiClient(config)
                reviewer = LLMPRReviewer(llm_client)

                # Review top 3 candidates
                for patch_id in list(EXAMPLE_PATCHES.keys())[:3]:
                    print(f"\n  Reviewing {patch_id}...")
                    review = await reviewer.review(
                        patch=EXAMPLE_PATCHES[patch_id],
                        patch_id=patch_id,
                        issue_description=ISSUE_DESCRIPTION,
                        surrounding_code=ORIGINAL_CODE,
                    )
                    review_results[patch_id] = review
                    print(f"    Verdict: {review.verdict} (score: {review.quality_score}/10)")
                    if review.top_issues:
                        print(f"    Issues: {review.top_issues[0]}")
        except Exception as e:
            print(f"  [WARNING] LLM review failed: {e}")
    else:
        print_header("Stage 3: LLM PR Review (Skipped)")
        print("  Use --with-llm to enable")

    # Step 4: Final Selection
    print_header("Stage 4: Final Selection")

    # Build approach families dict
    approach_families = {}
    for i, cluster in enumerate(clusters):
        label = f"approach_{i}"
        approach_families[label] = [fp.patch_id for fp in cluster]

    # Compute overall scores
    for qs in quality_scores.values():
        qs.compute_overall()

    selector = FinalSelector()
    result = selector.select(
        patches=EXAMPLE_PATCHES,
        approach_families=approach_families,
        quality_scores=quality_scores,
        review_results=review_results if review_results else None,
    )

    print(f"\n[WINNER]: {result.best_patch_id}")
    print(f"   Family: {result.approach_family}")
    print(f"\n   Reason: {result.selection_reason}")

    if result.alternates:
        print(f"\n   Alternates from other families:")
        for alt_id, _ in result.alternates:
            print(f"     - {alt_id}")

    print(f"\n   Score breakdown (top 3):")
    sorted_scores = sorted(result.scores_summary.items(), key=lambda x: x[1], reverse=True)
    for patch_id, score in sorted_scores[:3]:
        print(f"     {patch_id}: {score:.1f}")

    # Show the winning patch
    print_header("Winning Patch")
    print(result.best_patch_content)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test ATLAS quality selection")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM PR review")
    args = parser.parse_args()

    asyncio.run(run_quality_selection(with_llm=args.with_llm))


if __name__ == "__main__":
    main()
