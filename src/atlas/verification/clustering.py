"""Similarity-based clustering for solutions.

Uses semantic similarity based on:
1. Files modified (same files = likely same approach)
2. Functions/classes touched (extracted from diff)
3. Type of change (additions vs deletions vs modifications)
4. Fuzzy text matching on the actual changes
"""

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from atlas.core.task import Solution
from atlas.verification.patch_validator import compare_patches

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """A cluster of similar solutions."""

    cluster_id: str
    solutions: list[Solution] = field(default_factory=list)
    representative: Solution | None = None

    @property
    def size(self) -> int:
        """Get the number of solutions in this cluster."""
        return len(self.solutions)

    @property
    def is_valid(self) -> bool:
        """Check if any solution in the cluster is valid."""
        return any(s.is_valid for s in self.solutions)

    def add_solution(self, solution: Solution) -> None:
        """Add a solution to this cluster."""
        solution.cluster_id = self.cluster_id
        self.solutions.append(solution)

        # Update representative (prefer valid, smaller patches)
        self._update_representative()

    def _update_representative(self) -> None:
        """Update the representative solution."""
        valid_solutions = [s for s in self.solutions if s.is_valid]

        if valid_solutions:
            # Pick the smallest valid patch
            self.representative = min(valid_solutions, key=lambda s: len(s.patch))
        elif self.solutions:
            # Pick the smallest patch overall
            self.representative = min(self.solutions, key=lambda s: len(s.patch))


@dataclass
class ClusteringResult:
    """Result from clustering solutions."""

    clusters: list[Cluster] = field(default_factory=list)
    total_solutions: int = 0
    similarity_threshold: float = 0.8

    @property
    def cluster_count(self) -> int:
        """Get the number of clusters."""
        return len(self.clusters)

    @property
    def largest_cluster(self) -> Cluster | None:
        """Get the largest cluster."""
        if not self.clusters:
            return None
        return max(self.clusters, key=lambda c: c.size)

    @property
    def valid_clusters(self) -> list[Cluster]:
        """Get clusters with at least one valid solution."""
        return [c for c in self.clusters if c.is_valid]

    def get_cluster_sizes(self) -> dict[str, int]:
        """Get a mapping of cluster IDs to sizes."""
        return {c.cluster_id: c.size for c in self.clusters}


@dataclass
class PatchSignature:
    """Semantic signature of a patch for smart comparison."""

    files_modified: set[str] = field(default_factory=set)
    functions_touched: set[str] = field(default_factory=set)
    lines_added: list[str] = field(default_factory=list)
    lines_removed: list[str] = field(default_factory=list)
    change_type: str = "modify"  # add, remove, modify

    def similarity_to(self, other: "PatchSignature") -> float:
        """Calculate similarity to another patch signature.

        Uses weighted combination of:
        - File overlap (40%)
        - Function overlap (30%)
        - Content similarity (30%)
        """
        if not self.files_modified and not other.files_modified:
            return 0.0

        # File similarity (Jaccard)
        file_sim = 0.0
        if self.files_modified or other.files_modified:
            file_intersection = len(self.files_modified & other.files_modified)
            file_union = len(self.files_modified | other.files_modified)
            file_sim = file_intersection / file_union if file_union > 0 else 0.0

        # Function similarity (Jaccard)
        func_sim = 0.0
        if self.functions_touched or other.functions_touched:
            func_intersection = len(self.functions_touched & other.functions_touched)
            func_union = len(self.functions_touched | other.functions_touched)
            func_sim = func_intersection / func_union if func_union > 0 else 0.0

        # Content similarity (fuzzy matching on added lines)
        content_sim = 0.0
        if self.lines_added and other.lines_added:
            self_content = "\n".join(self.lines_added)
            other_content = "\n".join(other.lines_added)
            content_sim = SequenceMatcher(None, self_content, other_content).ratio()

        # Weighted combination
        # If files match perfectly, that's a strong signal
        if file_sim == 1.0:
            return 0.4 * file_sim + 0.3 * func_sim + 0.3 * content_sim
        else:
            # Different files = likely different approach
            return 0.5 * file_sim + 0.2 * func_sim + 0.3 * content_sim


def extract_patch_signature(patch: str) -> PatchSignature:
    """Extract a semantic signature from a patch.

    Args:
        patch: Unified diff patch text

    Returns:
        PatchSignature with extracted features
    """
    sig = PatchSignature()

    current_file = None
    in_hunk = False

    for line in patch.split("\n"):
        # Track files
        if line.startswith("+++ "):
            match = re.match(r"\+\+\+ (?:b/)?(.+)", line)
            if match:
                current_file = match.group(1)
                sig.files_modified.add(current_file)

        # Track functions (look for def/class in context or changes)
        elif line.startswith("@@"):
            in_hunk = True
            # Extract function name from hunk header if present
            func_match = re.search(r"@@.*@@\s*(?:def|class|function|func)\s+(\w+)", line)
            if func_match:
                sig.functions_touched.add(func_match.group(1))

        elif in_hunk:
            # Look for function definitions in changes
            if line.startswith("+") or line.startswith("-"):
                content = line[1:]
                func_match = re.search(r"(?:def|class|function|func)\s+(\w+)", content)
                if func_match:
                    sig.functions_touched.add(func_match.group(1))

                if line.startswith("+"):
                    sig.lines_added.append(content.strip())
                else:
                    sig.lines_removed.append(content.strip())

    # Determine change type
    if sig.lines_added and not sig.lines_removed:
        sig.change_type = "add"
    elif sig.lines_removed and not sig.lines_added:
        sig.change_type = "remove"
    else:
        sig.change_type = "modify"

    return sig


class SimilarityClustering:
    """Cluster solutions by patch similarity.

    Uses semantic similarity based on:
    - Files modified
    - Functions touched
    - Content of changes (fuzzy matching)
    """

    def __init__(self, similarity_threshold: float = 0.6):
        """Initialize the clustering algorithm.

        Args:
            similarity_threshold: Minimum similarity to be in same cluster (0.0-1.0)
                                  Default lowered to 0.6 for fuzzy matching
        """
        self.similarity_threshold = similarity_threshold

    def cluster(self, solutions: list[Solution]) -> ClusteringResult:
        """Cluster solutions by similarity.

        Args:
            solutions: List of solutions to cluster

        Returns:
            ClusteringResult with clusters
        """
        result = ClusteringResult(
            total_solutions=len(solutions),
            similarity_threshold=self.similarity_threshold,
        )

        if not solutions:
            return result

        # Filter to valid solutions with patches
        valid_solutions = [s for s in solutions if s.patch]

        if not valid_solutions:
            # Create a single cluster for all invalid solutions
            cluster = Cluster(cluster_id="invalid_0")
            for solution in solutions:
                cluster.add_solution(solution)
            result.clusters.append(cluster)
            return result

        # Group by exact match first (fast)
        exact_groups = self._group_by_exact_match(valid_solutions)

        # For each group, create a cluster
        for i, group in enumerate(exact_groups):
            cluster = Cluster(cluster_id=f"cluster_{i}")
            for solution in group:
                cluster.add_solution(solution)
            result.clusters.append(cluster)

        # Add invalid solutions to nearest cluster or new cluster
        invalid_solutions = [s for s in solutions if not s.patch]
        if invalid_solutions:
            invalid_cluster = Cluster(cluster_id="invalid")
            for solution in invalid_solutions:
                invalid_cluster.add_solution(solution)
            result.clusters.append(invalid_cluster)

        return result

    def _group_by_exact_match(self, solutions: list[Solution]) -> list[list[Solution]]:
        """Group solutions by exact patch match.

        Args:
            solutions: Solutions to group

        Returns:
            List of groups (each group has identical patches)
        """
        groups = defaultdict(list)

        for solution in solutions:
            # Hash the normalized patch
            normalized = self._normalize_patch(solution.patch)
            patch_hash = hashlib.md5(normalized.encode()).hexdigest()
            groups[patch_hash].append(solution)

        return list(groups.values())

    def _normalize_patch(self, patch: str) -> str:
        """Normalize a patch for comparison.

        Args:
            patch: The patch text

        Returns:
            Normalized patch
        """
        lines = []
        for line in patch.split("\n"):
            # Skip headers
            if line.startswith("---") or line.startswith("+++"):
                continue
            # Keep hunk headers but normalize
            if line.startswith("@@"):
                lines.append("@@")
                continue
            # Normalize content lines
            stripped = line.rstrip()
            if stripped:
                lines.append(stripped)

        return "\n".join(lines)

    def merge_similar_clusters(
        self,
        result: ClusteringResult,
    ) -> ClusteringResult:
        """Merge clusters that are similar to each other.

        This is a second pass that combines clusters whose
        representative patches are similar.

        Args:
            result: Initial clustering result

        Returns:
            Updated result with merged clusters
        """
        if len(result.clusters) <= 1:
            return result

        # Get clusters with representatives
        clusters_with_reps = [
            c for c in result.clusters
            if c.representative and c.representative.patch
        ]

        if len(clusters_with_reps) <= 1:
            return result

        # Build similarity matrix
        n = len(clusters_with_reps)
        merged = [False] * n
        new_clusters = []

        for i in range(n):
            if merged[i]:
                continue

            # Start a new merged cluster
            base_cluster = clusters_with_reps[i]
            merged_cluster = Cluster(cluster_id=f"merged_{len(new_clusters)}")

            for solution in base_cluster.solutions:
                merged_cluster.add_solution(solution)

            merged[i] = True

            # Find similar clusters to merge
            for j in range(i + 1, n):
                if merged[j]:
                    continue

                similarity = compare_patches(
                    base_cluster.representative.patch,
                    clusters_with_reps[j].representative.patch,
                )

                if similarity >= self.similarity_threshold:
                    # Merge cluster j into the current cluster
                    for solution in clusters_with_reps[j].solutions:
                        merged_cluster.add_solution(solution)
                    merged[j] = True

            new_clusters.append(merged_cluster)

        # Add any clusters without representatives (like invalid cluster)
        for cluster in result.clusters:
            if cluster not in clusters_with_reps:
                new_clusters.append(cluster)

        return ClusteringResult(
            clusters=new_clusters,
            total_solutions=result.total_solutions,
            similarity_threshold=self.similarity_threshold,
        )


def calculate_diversity_score(solutions: list[Solution]) -> float:
    """Calculate a diversity score for a set of solutions.

    Higher score means more diverse approaches.

    Args:
        solutions: List of solutions

    Returns:
        Diversity score between 0.0 and 1.0
    """
    if len(solutions) <= 1:
        return 0.0

    # Get valid patches
    patches = [s.patch for s in solutions if s.patch]

    if len(patches) <= 1:
        return 0.0

    # Calculate pairwise similarities
    total_similarity = 0.0
    comparisons = 0

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            similarity = compare_patches(patches[i], patches[j])
            total_similarity += similarity
            comparisons += 1

    if comparisons == 0:
        return 0.0

    avg_similarity = total_similarity / comparisons

    # Diversity is inverse of similarity
    return 1.0 - avg_similarity
