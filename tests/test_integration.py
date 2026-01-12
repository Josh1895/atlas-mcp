"""Integration tests for ATLAS pipeline.

These tests require API keys to be set in environment variables:
- GEMINI_API_KEY
- CONTEXT7_API_KEY (optional)

Run with: pytest tests/test_integration.py -v -s
"""

import asyncio
import os
import pytest

from atlas.core.config import Config
from atlas.core.task import TaskSubmission
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.quality.pipeline import QualitySelectionPipeline, QualitySelectionConfig
from atlas.quality.fingerprinting import TokenFingerprinter, cluster_by_similarity
from atlas.quality.quality_scorer import QualityScorer
from atlas.quality.selector import FinalSelector
from atlas.quality.pr_reviewer import PRReviewResult
from atlas.quality.quality_scorer import QualityScore


# Skip if no API key
requires_api_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


class TestQualityPipelineIntegration:
    """Integration tests for the quality selection pipeline."""

    def test_clustering_with_real_patches(self):
        """Test clustering with realistic patch examples."""
        fingerprinter = TokenFingerprinter()

        # Simulate 6 patches: 3 use guard clause, 2 use try-except, 1 unique
        patches = {
            # Guard clause family
            "agent_0": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,8 @@
 def get_user_name(user):
+    if user is None:
+        return "Unknown"
     return user.name
""",
            "agent_1": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,8 @@
 def get_user_name(user):
+    if not user:
+        return "Unknown"
     return user.name
""",
            "agent_2": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,8 @@
 def get_user_name(user):
+    if user is None:
+        return "Guest"
     return user.name
""",
            # Try-except family
            "agent_3": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,10 @@
 def get_user_name(user):
+    try:
+        return user.name
+    except AttributeError:
+        return "Unknown"
-    return user.name
""",
            "agent_4": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,10 @@
 def get_user_name(user):
+    try:
+        name = user.name
+        return name
+    except (AttributeError, TypeError):
+        return "Unknown"
-    return user.name
""",
            # Unique approach
            "agent_5": """--- a/utils.py
+++ b/utils.py
@@ -5,6 +5,6 @@
 def get_user_name(user):
-    return user.name
+    return getattr(user, 'name', 'Unknown')
""",
        }

        # Create fingerprints
        fingerprints = [
            fingerprinter.fingerprint(patch, patch_id)
            for patch_id, patch in patches.items()
        ]

        # Cluster
        clusters = cluster_by_similarity(fingerprints, threshold=0.25)

        print(f"\n=== Clustering Results ===")
        print(f"Total patches: {len(patches)}")
        print(f"Clusters found: {len(clusters)}")

        for i, cluster in enumerate(clusters):
            patch_ids = [fp.patch_id for fp in cluster]
            print(f"  Cluster {i}: {patch_ids} (size: {len(cluster)})")

        # We expect at least 2 clusters (guard clause vs try-except)
        assert len(clusters) >= 2, "Should find at least 2 approach families"

    @pytest.mark.asyncio
    async def test_quality_scoring_no_api(self):
        """Test quality scoring without API calls."""
        scorer = QualityScorer()

        # Good clean patch
        clean_patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 def process(data):
+    if data is None:
+        return []
     return [x * 2 for x in data]
"""
        original = {"file.py": "def process(data):\n    return [x * 2 for x in data]\n"}
        patched = {"file.py": "def process(data):\n    if data is None:\n        return []\n    return [x * 2 for x in data]\n"}

        clean_score = await scorer.score(clean_patch, "clean", original, patched)

        print(f"\n=== Clean Patch Score ===")
        print(f"Style: {clean_score.style_score:.1f}")
        print(f"Maintainability: {clean_score.maintainability_score:.1f}")
        print(f"Risk: {clean_score.risk_score:.1f}")
        print(f"Overall: {clean_score.overall_score:.1f}")

        # Risky patch with bad patterns
        risky_patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,8 @@
 def process(data):
+    # HACK: workaround for bug
+    try:
+        eval(data)  # dangerous!
+    except:
+        pass
     return [x * 2 for x in data]
"""
        risky_patched = {"file.py": "def process(data):\n    # HACK\n    try:\n        eval(data)\n    except:\n        pass\n    return [x * 2 for x in data]\n"}

        risky_score = await scorer.score(risky_patch, "risky", original, risky_patched)

        print(f"\n=== Risky Patch Score ===")
        print(f"Style: {risky_score.style_score:.1f}")
        print(f"Maintainability: {risky_score.maintainability_score:.1f}")
        print(f"Risk: {risky_score.risk_score:.1f}")
        print(f"Risk flags: {risky_score.risk_flags}")
        print(f"Overall: {risky_score.overall_score:.1f}")

        # Clean patch should score better
        assert clean_score.overall_score > risky_score.overall_score
        assert risky_score.risk_score > clean_score.risk_score
        assert len(risky_score.risk_flags) > 0

    def test_final_selection_logic(self):
        """Test the final selection logic."""
        selector = FinalSelector()

        patches = {
            "p1": "patch 1 - guard clause",
            "p2": "patch 2 - guard clause similar",
            "p3": "patch 3 - try except",
            "p4": "patch 4 - unique",
        }

        # p1 and p2 are in same family (convergence signal)
        approach_families = {
            "guard_clause": ["p1", "p2"],
            "try_except": ["p3"],
            "getattr": ["p4"],
        }

        # All have similar base quality
        quality_scores = {
            "p1": QualityScore(patch_id="p1", style_score=80, maintainability_score=80, risk_score=10),
            "p2": QualityScore(patch_id="p2", style_score=75, maintainability_score=75, risk_score=15),
            "p3": QualityScore(patch_id="p3", style_score=85, maintainability_score=70, risk_score=10),
            "p4": QualityScore(patch_id="p4", style_score=90, maintainability_score=90, risk_score=5),
        }

        for qs in quality_scores.values():
            qs.compute_overall()

        # Without LLM reviews
        result = selector.select(
            patches=patches,
            approach_families=approach_families,
            quality_scores=quality_scores,
        )

        print(f"\n=== Selection Result (no LLM) ===")
        print(f"Winner: {result.best_patch_id}")
        print(f"Family: {result.approach_family}")
        print(f"Reason: {result.selection_reason}")
        print(f"Scores: {result.scores_summary}")

        # p4 has best individual quality, but p1/p2 have convergence
        # The winner depends on weight balance
        assert result.best_patch_id in patches

        # Now add LLM reviews that favor p1
        review_results = {
            "p1": PRReviewResult(patch_id="p1", verdict="approve", quality_score=9.0),
            "p2": PRReviewResult(patch_id="p2", verdict="approve", quality_score=7.0),
            "p3": PRReviewResult(patch_id="p3", verdict="needs_discussion", quality_score=6.0, has_code_smells=True),
            "p4": PRReviewResult(patch_id="p4", verdict="approve", quality_score=8.0),
        }

        result_with_reviews = selector.select(
            patches=patches,
            approach_families=approach_families,
            quality_scores=quality_scores,
            review_results=review_results,
        )

        print(f"\n=== Selection Result (with LLM reviews) ===")
        print(f"Winner: {result_with_reviews.best_patch_id}")
        print(f"Family: {result_with_reviews.approach_family}")
        print(f"Reason: {result_with_reviews.selection_reason}")

        # p1 should win with best review + convergence
        assert result_with_reviews.best_patch_id == "p1"


@requires_api_key
class TestWithLLM:
    """Tests that require actual LLM API calls."""

    @pytest.mark.asyncio
    async def test_full_quality_pipeline(self):
        """Test the full quality selection pipeline with real LLM."""
        from atlas.agents.gemini_client import GeminiClient

        config = Config.from_env()
        llm_client = GeminiClient(config)

        pipeline_config = QualitySelectionConfig(
            enable_llm_review=True,
            enable_tournament=True,
            max_patches_for_tournament=4,
        )

        pipeline = QualitySelectionPipeline(
            config=pipeline_config,
            llm_client=llm_client,
        )

        # Test patches
        patches = {
            "agent_0": """--- a/utils.py
+++ b/utils.py
@@ -1,3 +1,5 @@
 def get_name(user):
+    if user is None:
+        return "Unknown"
     return user.name
""",
            "agent_1": """--- a/utils.py
+++ b/utils.py
@@ -1,3 +1,6 @@
 def get_name(user):
+    try:
+        return user.name
+    except:
+        return "Unknown"
-    return user.name
""",
        }

        original_files = {"utils.py": "def get_name(user):\n    return user.name\n"}
        patched_files_map = {
            "agent_0": {"utils.py": "def get_name(user):\n    if user is None:\n        return 'Unknown'\n    return user.name\n"},
            "agent_1": {"utils.py": "def get_name(user):\n    try:\n        return user.name\n    except:\n        return 'Unknown'\n"},
        }

        result = await pipeline.select_best_patch(
            patches=patches,
            issue_description="Fix NullPointerException when user is None",
            original_files=original_files,
            patched_files_map=patched_files_map,
            context_code="",
        )

        print(f"\n=== Full Pipeline Result ===")
        print(f"Winner: {result.selection.best_patch_id}")
        print(f"Reason: {result.selection.selection_reason}")
        print(f"Families found: {len(result.approach_families)}")
        print(f"Reviews: {list(result.review_results.keys())}")

        # agent_0 should likely win (guard clause is cleaner than bare except)
        assert result.selection.best_patch_id in patches


class TestMockEndToEnd:
    """End-to-end tests with mocked external services."""

    @pytest.mark.asyncio
    async def test_orchestrator_quality_selection_flow(self):
        """Test the orchestrator's quality selection integration."""
        # This is a structural test - verifies the flow works
        # without actually calling external services

        from atlas.core.task import Solution
        from atlas.quality.pipeline import QualitySelectionResult
        from atlas.quality.selector import SelectionResult

        # Create mock solutions
        solutions = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal_diff",
                patch="--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n+# fix\n x = 1",
                is_valid=True,
            ),
            Solution(
                agent_id="agent_1",
                prompt_style="verbose",
                patch="--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n+# another fix\n x = 1",
                is_valid=True,
            ),
        ]

        # Verify the quality pipeline can be instantiated
        config = QualitySelectionConfig(
            enable_llm_review=False,  # Disable LLM for mock test
            enable_tournament=False,
        )

        pipeline = QualitySelectionPipeline(config=config, llm_client=None)

        patches = {s.agent_id: s.patch for s in solutions}
        original_files = {"f.py": "x = 1\n"}
        patched_files_map = {
            "agent_0": {"f.py": "# fix\nx = 1\n"},
            "agent_1": {"f.py": "# another fix\nx = 1\n"},
        }

        result = await pipeline.select_best_patch(
            patches=patches,
            issue_description="Test issue",
            original_files=original_files,
            patched_files_map=patched_files_map,
        )

        print(f"\n=== Mock E2E Result ===")
        print(f"Winner: {result.selection.best_patch_id}")
        print(f"Total candidates: {result.total_patches}")
        print(f"Families: {len(result.approach_families)}")

        assert result.selection.best_patch_id in patches
        assert result.total_patches == 2


if __name__ == "__main__":
    # Run a quick manual test
    print("Running quick integration tests...")

    # Test 1: Clustering
    test = TestQualityPipelineIntegration()
    test.test_clustering_with_real_patches()

    # Test 2: Quality scoring (async)
    asyncio.run(test.test_quality_scoring_no_api())

    # Test 3: Selection logic
    test.test_final_selection_logic()

    # Test 4: Mock E2E
    asyncio.run(TestMockEndToEnd().test_orchestrator_quality_selection_flow())

    print("\n✓ All quick tests passed!")
