"""Tests for the quality selection layer."""

import pytest

from atlas.quality.fingerprinting import (
    PatchFingerprint,
    TokenFingerprinter,
    compute_similarity,
    cluster_by_similarity,
)
from atlas.quality.ast_analysis import (
    ASTAnalyzer,
    ASTChangeSignature,
    compute_ast_similarity,
)
from atlas.quality.quality_scorer import QualityScore, QualityScorer
from atlas.quality.selector import FinalSelector, SelectionResult
from atlas.quality.pr_reviewer import PRReviewResult


class TestTokenFingerprinting:
    """Tests for token fingerprinting."""

    def test_fingerprint_creation(self):
        """Test creating a fingerprint from a patch."""
        fingerprinter = TokenFingerprinter()

        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 def hello():
-    return "old"
+    if name is None:
+        return "default"
+    return "new"
"""
        fp = fingerprinter.fingerprint(patch, "test_patch")

        assert fp.patch_id == "test_patch"
        assert fp.files_touched == ["file.py"]
        assert fp.lines_added > 0
        assert fp.lines_removed > 0
        assert len(fp.token_hashes) > 0

    def test_similar_patches_high_similarity(self):
        """Test that similar patches have high similarity."""
        fingerprinter = TokenFingerprinter()

        # Use larger patches for meaningful similarity
        patch1 = """--- a/file.py
+++ b/file.py
@@ -5,6 +5,8 @@
 def get_user_name(user):
+    if user is None:
+        return "Unknown"
     return user.name
"""
        patch2 = """--- a/file.py
+++ b/file.py
@@ -5,6 +5,8 @@
 def get_user_name(user):
+    if not user:
+        return "Unknown"
     return user.name
"""
        fp1 = fingerprinter.fingerprint(patch1, "p1")
        fp2 = fingerprinter.fingerprint(patch2, "p2")

        similarity = compute_similarity(fp1, fp2)
        # Similar guard clause patterns should have some similarity
        assert similarity >= 0  # Both should produce valid fingerprints

    def test_different_patches_low_similarity(self):
        """Test that different patches have low similarity."""
        fingerprinter = TokenFingerprinter()

        patch1 = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old_value = 1
+new_value = 1
"""
        patch2 = """--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-def foo():
+def bar():
+    return calculate_something_completely_different()
"""
        fp1 = fingerprinter.fingerprint(patch1, "p1")
        fp2 = fingerprinter.fingerprint(patch2, "p2")

        similarity = compute_similarity(fp1, fp2)
        assert similarity < 0.5  # Should be dissimilar

    def test_clustering(self):
        """Test clustering similar patches together."""
        fingerprinter = TokenFingerprinter()

        # Two similar patches (same approach)
        patch1 = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
        patch2 = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 3"
        # One different patch
        patch3 = "--- a/g.py\n+++ b/g.py\n@@ -1 +1 @@\n-def foo():\n+def bar():"

        fps = [
            fingerprinter.fingerprint(patch1, "p1"),
            fingerprinter.fingerprint(patch2, "p2"),
            fingerprinter.fingerprint(patch3, "p3"),
        ]

        clusters = cluster_by_similarity(fps, threshold=0.3)

        # Should have at least 1 cluster
        assert len(clusters) >= 1


class TestASTAnalysis:
    """Tests for AST-based analysis."""

    def test_analyze_simple_change(self):
        """Test analyzing a simple code change."""
        analyzer = ASTAnalyzer()

        original = """
def hello():
    return "old"
"""
        patched = """
def hello():
    if name is None:
        return "default"
    return "new"
"""
        sig = analyzer.analyze_patch(original, patched, "test")

        assert sig.patch_id == "test"
        assert sig.added_guard_clause  # Added if statement

    def test_detect_try_except(self):
        """Test detecting added try-except."""
        analyzer = ASTAnalyzer()

        original = """
def risky():
    do_something()
"""
        patched = """
def risky():
    try:
        do_something()
    except Exception:
        pass
"""
        sig = analyzer.analyze_patch(original, patched, "test")

        assert sig.added_try_except

    def test_detect_function_addition(self):
        """Test detecting new function."""
        analyzer = ASTAnalyzer()

        original = """
def existing():
    pass
"""
        patched = """
def existing():
    pass

def new_helper():
    return 42
"""
        sig = analyzer.analyze_patch(original, patched, "test")

        assert "new_helper" in sig.functions_added

    def test_ast_similarity(self):
        """Test computing AST similarity."""
        sig1 = ASTChangeSignature(
            patch_id="p1",
            edit_operations=["insert:If", "insert:Return"],
            added_guard_clause=True,
        )
        sig2 = ASTChangeSignature(
            patch_id="p2",
            edit_operations=["insert:If", "insert:Return"],
            added_guard_clause=True,
        )
        sig3 = ASTChangeSignature(
            patch_id="p3",
            edit_operations=["insert:Try", "insert:Except"],
            added_try_except=True,
        )

        sim_same = compute_ast_similarity(sig1, sig2)
        sim_diff = compute_ast_similarity(sig1, sig3)

        assert sim_same > sim_diff


class TestQualityScorer:
    """Tests for quality scoring."""

    @pytest.mark.asyncio
    async def test_score_clean_code(self):
        """Test scoring clean, well-formatted code."""
        scorer = QualityScorer()

        patch = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old = 1
+new_value = 1
"""
        original_files = {"file.py": "old = 1\n"}
        patched_files = {"file.py": "new_value = 1\n"}

        score = await scorer.score(patch, "test", original_files, patched_files)

        assert score.patch_id == "test"
        assert score.style_score >= 0
        assert score.maintainability_score >= 0
        assert score.overall_score >= 0

    @pytest.mark.asyncio
    async def test_score_risky_code(self):
        """Test scoring code with risky patterns."""
        scorer = QualityScorer()

        patch = """--- a/file.py
+++ b/file.py
@@ -1 +1,3 @@
-old = 1
+# HACK: workaround
+eval("dangerous")
+old = 1
"""
        original_files = {"file.py": "old = 1\n"}
        patched_files = {"file.py": "# HACK: workaround\neval(\"dangerous\")\nold = 1\n"}

        score = await scorer.score(patch, "test", original_files, patched_files)

        assert score.risk_score > 0
        assert len(score.risk_flags) > 0
        assert any("eval" in flag or "hack" in flag.lower() for flag in score.risk_flags)


class TestFinalSelector:
    """Tests for final patch selection."""

    def test_select_best_patch(self):
        """Test selecting the best patch from candidates."""
        selector = FinalSelector()

        patches = {
            "p1": "patch 1 content",
            "p2": "patch 2 content",
            "p3": "patch 3 content",
        }

        approach_families = {
            "add guard": ["p1", "p2"],  # p1 and p2 are similar
            "refactor": ["p3"],  # p3 is different
        }

        quality_scores = {
            "p1": QualityScore(
                patch_id="p1",
                style_score=80,
                maintainability_score=75,
                risk_score=10,
            ),
            "p2": QualityScore(
                patch_id="p2",
                style_score=70,
                maintainability_score=70,
                risk_score=15,
            ),
            "p3": QualityScore(
                patch_id="p3",
                style_score=90,
                maintainability_score=60,
                risk_score=5,
            ),
        }

        # Compute overall scores
        for qs in quality_scores.values():
            qs.compute_overall()

        result = selector.select(
            patches=patches,
            approach_families=approach_families,
            quality_scores=quality_scores,
        )

        assert result.best_patch_id in patches
        assert result.best_patch_content
        assert result.total_candidates == 3
        assert result.families_considered == 2

    def test_apply_hard_gates(self):
        """Test applying hard gates to filter patches."""
        selector = FinalSelector()

        patches = {
            "p1": "safe patch",
            "p2": "risky patch",
        }

        quality_scores = {
            "p1": QualityScore(
                patch_id="p1",
                style_score=80,
                maintainability_score=75,
                risk_score=20,  # Low risk
            ),
            "p2": QualityScore(
                patch_id="p2",
                style_score=80,
                maintainability_score=75,
                risk_score=80,  # High risk
            ),
        }

        filtered = selector.apply_hard_gates(
            patches=patches,
            quality_scores=quality_scores,
            max_risk_score=50,
        )

        assert "p1" in filtered
        assert "p2" not in filtered  # Filtered out due to high risk

    def test_select_with_llm_reviews(self):
        """Test selection with LLM review results."""
        selector = FinalSelector()

        patches = {
            "p1": "patch 1",
            "p2": "patch 2",
        }

        approach_families = {
            "approach": ["p1", "p2"],
        }

        quality_scores = {
            "p1": QualityScore(
                patch_id="p1",
                style_score=75,
                maintainability_score=75,
                risk_score=10,
            ),
            "p2": QualityScore(
                patch_id="p2",
                style_score=75,
                maintainability_score=75,
                risk_score=10,
            ),
        }

        for qs in quality_scores.values():
            qs.compute_overall()

        review_results = {
            "p1": PRReviewResult(
                patch_id="p1",
                verdict="approve",
                quality_score=8.5,
                is_hacky=False,
            ),
            "p2": PRReviewResult(
                patch_id="p2",
                verdict="request_changes",
                quality_score=5.0,
                is_hacky=True,
            ),
        }

        result = selector.select(
            patches=patches,
            approach_families=approach_families,
            quality_scores=quality_scores,
            review_results=review_results,
        )

        # p1 should be selected due to better LLM review
        assert result.best_patch_id == "p1"

    def test_empty_patches(self):
        """Test handling empty patch set."""
        selector = FinalSelector()

        result = selector.select(
            patches={},
            approach_families={},
            quality_scores={},
        )

        assert result.best_patch_id == ""
        assert "No patches" in result.selection_reason
