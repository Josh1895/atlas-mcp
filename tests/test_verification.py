"""Tests for the verification layer."""

import pytest

from atlas.verification.clustering import SimilarityClustering, calculate_diversity_score
from atlas.verification.patch_validator import PatchValidator, compare_patches
from atlas.verification.static_analysis import StaticAnalyzer
from atlas.core.task import Solution


class TestStaticAnalyzer:
    """Tests for StaticAnalyzer."""

    def test_valid_python(self):
        """Test analysis of valid Python code."""
        analyzer = StaticAnalyzer()

        code = """
def hello(name):
    return f"Hello, {name}!"
"""
        result = analyzer.analyze_python(code)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_syntax_error(self):
        """Test detection of syntax errors."""
        analyzer = StaticAnalyzer()

        code = """
def hello(name)
    return f"Hello, {name}!"
"""
        result = analyzer.analyze_python(code)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert "Syntax error" in result.errors[0]

    def test_common_mistakes(self):
        """Test detection of common mistakes."""
        analyzer = StaticAnalyzer()

        code = """
def risky():
    try:
        do_something()
    except:
        pass
"""
        result = analyzer.analyze_python(code)

        assert result.is_valid  # Warnings don't affect validity
        assert any("bare except" in w for w in result.warnings)

    def test_patch_analysis(self):
        """Test analysis of a patch."""
        analyzer = StaticAnalyzer()

        patch = """
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print "old"
+    print("new")
"""
        result = analyzer.analyze_patch(patch)

        assert result.is_valid


class TestPatchValidator:
    """Tests for PatchValidator."""

    def test_valid_patch(self):
        """Test validation of a valid patch."""
        validator = PatchValidator()

        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    return "old"
+    return "new"
"""
        result = validator.validate(patch)

        assert result.is_valid
        assert result.patch_info.files_modified == ["file.py"]
        assert result.patch_info.lines_added == 1
        assert result.patch_info.lines_removed == 1

    def test_empty_patch(self):
        """Test validation of empty patch."""
        validator = PatchValidator()

        result = validator.validate("")

        assert not result.is_valid
        assert "Empty patch" in result.errors

    def test_malformed_patch(self):
        """Test detection of malformed patch."""
        validator = PatchValidator()

        patch = "This is not a valid patch"
        result = validator.validate(patch)

        assert not result.is_valid

    def test_patch_comparison(self):
        """Test patch comparison function."""
        patch1 = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        patch2 = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        patch3 = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+different
"""

        assert compare_patches(patch1, patch2) == 1.0
        assert compare_patches(patch1, patch3) < 1.0
        assert compare_patches("", "") == 0.0


class TestSimilarityClustering:
    """Tests for SimilarityClustering."""

    def test_empty_solutions(self):
        """Test clustering with no solutions."""
        clustering = SimilarityClustering()

        result = clustering.cluster([])

        assert result.cluster_count == 0
        assert result.total_solutions == 0

    def test_identical_patches_same_cluster(self):
        """Test that identical patches are in the same cluster."""
        clustering = SimilarityClustering()

        patch = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        solutions = [
            Solution(agent_id=f"agent_{i}", prompt_style="minimal", patch=patch)
            for i in range(3)
        ]

        result = clustering.cluster(solutions)

        assert result.cluster_count == 1
        assert result.largest_cluster.size == 3

    def test_different_patches_different_clusters(self):
        """Test that different patches are in different clusters."""
        clustering = SimilarityClustering()

        solutions = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
            Solution(
                agent_id="agent_1",
                prompt_style="minimal",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+different",
            ),
        ]

        result = clustering.cluster(solutions)

        assert result.cluster_count == 2

    def test_diversity_score(self):
        """Test diversity score calculation."""
        # Identical patches = no diversity
        patch = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        identical = [
            Solution(agent_id=f"agent_{i}", prompt_style="minimal", patch=patch)
            for i in range(3)
        ]

        score = calculate_diversity_score(identical)
        assert score == 0.0

        # Different patches = high diversity
        different = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+one",
            ),
            Solution(
                agent_id="agent_1",
                prompt_style="minimal",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+two",
            ),
        ]

        score = calculate_diversity_score(different)
        assert score > 0.0
