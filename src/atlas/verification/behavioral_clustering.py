"""Behavioral clustering for ATLAS.

This module clusters patches by their test outcomes (behavioral similarity)
rather than just patch content similarity. This is the primary correctness
oracle - patches that pass the same tests and fail the same tests are
likely implementing the same (correct or incorrect) solution.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from atlas.verification.test_runner import TestResult

logger = logging.getLogger(__name__)


@dataclass
class BehavioralSignature:
    """Signature of a patch's behavior based on test outcomes."""

    patch_id: str
    all_tests_pass: bool = False
    passing_tests: Set[str] = field(default_factory=set)
    failing_tests: Set[str] = field(default_factory=set)
    error_tests: Set[str] = field(default_factory=set)

    # Counts for when we don't have individual test names
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0

    # Execution status
    tests_ran: bool = True
    execution_error: str = ""
    exit_code: int = 0

    @property
    def failure_signature(self) -> str:
        """Generate a string signature of failures for fast comparison."""
        if not self.tests_ran:
            return f"ERROR:{self.execution_error[:50]}"
        if self.all_tests_pass:
            return "ALL_PASS"
        # Sort for consistency
        return "|".join(sorted(self.failing_tests)) or f"FAIL_COUNT:{self.failed_count}"

    @property
    def outcome_tuple(self) -> Tuple[int, int, int, int, bool]:
        """Return a hashable tuple representing the outcome."""
        return (
            self.passed_count,
            self.failed_count,
            self.error_count,
            self.skipped_count,
            self.tests_ran,
        )

    @classmethod
    def from_test_result(cls, patch_id: str, result: TestResult) -> "BehavioralSignature":
        """Create a signature from a TestResult.

        Args:
            patch_id: ID of the patch
            result: TestResult from running tests

        Returns:
            BehavioralSignature
        """
        sig = cls(patch_id=patch_id)

        if result.execution_error:
            sig.tests_ran = False
            sig.execution_error = result.execution_error
            return sig

        sig.passed_count = result.passed
        sig.failed_count = result.failed
        sig.error_count = result.errors
        sig.skipped_count = result.skipped
        sig.exit_code = result.exit_code
        sig.all_tests_pass = result.success

        # Extract individual test names if available
        for test_case in result.test_cases:
            if test_case.passed:
                sig.passing_tests.add(test_case.name)
            else:
                sig.failing_tests.add(test_case.name)

        # Add failed tests
        for test_case in result.failed_tests:
            sig.failing_tests.add(test_case.name)

        return sig


@dataclass
class BehavioralCluster:
    """A cluster of patches with similar behavioral signatures."""

    cluster_id: str
    signature: str  # The shared failure signature
    patch_ids: List[str] = field(default_factory=list)
    all_pass: bool = False
    representative_patch_id: str = ""

    # Aggregated metrics
    avg_test_count: float = 0.0
    common_failures: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.patch_ids)


@dataclass
class BehavioralClusteringResult:
    """Result from behavioral clustering."""

    clusters: List[BehavioralCluster] = field(default_factory=list)
    passing_clusters: List[BehavioralCluster] = field(default_factory=list)
    failing_clusters: List[BehavioralCluster] = field(default_factory=list)
    error_clusters: List[BehavioralCluster] = field(default_factory=list)

    # Statistics
    total_patches: int = 0
    patches_all_pass: int = 0
    patches_some_fail: int = 0
    patches_error: int = 0

    # Best cluster (most votes among passing)
    best_cluster: Optional[BehavioralCluster] = None
    confidence: float = 0.0

    @property
    def has_passing_patches(self) -> bool:
        return self.patches_all_pass > 0


def compute_behavioral_similarity(
    sig1: BehavioralSignature,
    sig2: BehavioralSignature,
) -> float:
    """Compute similarity between two behavioral signatures.

    Args:
        sig1: First signature
        sig2: Second signature

    Returns:
        Similarity score from 0.0 to 1.0
    """
    # If neither ran tests, compare by error message
    if not sig1.tests_ran and not sig2.tests_ran:
        return 1.0 if sig1.execution_error == sig2.execution_error else 0.5

    # If one ran and one didn't, they're different
    if sig1.tests_ran != sig2.tests_ran:
        return 0.0

    # If both all pass, they're identical behaviorally
    if sig1.all_tests_pass and sig2.all_tests_pass:
        return 1.0

    # If we have individual test names, use Jaccard on failing tests
    if sig1.failing_tests and sig2.failing_tests:
        intersection = len(sig1.failing_tests & sig2.failing_tests)
        union = len(sig1.failing_tests | sig2.failing_tests)
        if union > 0:
            return intersection / union

    # Fall back to comparing counts
    if sig1.outcome_tuple == sig2.outcome_tuple:
        return 0.9  # Same counts but no names

    # Partial similarity based on counts
    total1 = sig1.passed_count + sig1.failed_count
    total2 = sig2.passed_count + sig2.failed_count

    if total1 == 0 or total2 == 0:
        return 0.0

    # Compare pass rates
    rate1 = sig1.passed_count / total1
    rate2 = sig2.passed_count / total2

    return 1.0 - abs(rate1 - rate2)


class BehavioralClusterer:
    """Clusters patches by behavioral similarity (test outcomes)."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        prioritize_passing: bool = True,
    ):
        """Initialize the clusterer.

        Args:
            similarity_threshold: Minimum similarity to be in same cluster
            prioritize_passing: If True, passing clusters are ranked higher
        """
        self.similarity_threshold = similarity_threshold
        self.prioritize_passing = prioritize_passing

    def cluster(
        self,
        signatures: List[BehavioralSignature],
    ) -> BehavioralClusteringResult:
        """Cluster patches by behavioral similarity.

        Args:
            signatures: List of behavioral signatures to cluster

        Returns:
            BehavioralClusteringResult
        """
        result = BehavioralClusteringResult(total_patches=len(signatures))

        if not signatures:
            return result

        # First pass: group by failure signature (exact match)
        signature_groups: Dict[str, List[BehavioralSignature]] = defaultdict(list)
        for sig in signatures:
            signature_groups[sig.failure_signature].append(sig)

        # Create clusters from groups
        cluster_id = 0
        for failure_sig, group in signature_groups.items():
            cluster = BehavioralCluster(
                cluster_id=f"behavioral_{cluster_id}",
                signature=failure_sig,
                patch_ids=[sig.patch_id for sig in group],
                all_pass=(failure_sig == "ALL_PASS"),
            )

            # Set representative as first patch
            if cluster.patch_ids:
                cluster.representative_patch_id = cluster.patch_ids[0]

            # Compute common failures
            if group:
                cluster.common_failures = group[0].failing_tests.copy()
                for sig in group[1:]:
                    cluster.common_failures &= sig.failing_tests

            result.clusters.append(cluster)
            cluster_id += 1

            # Categorize
            if cluster.all_pass:
                result.passing_clusters.append(cluster)
                result.patches_all_pass += cluster.size
            elif failure_sig.startswith("ERROR:"):
                result.error_clusters.append(cluster)
                result.patches_error += cluster.size
            else:
                result.failing_clusters.append(cluster)
                result.patches_some_fail += cluster.size

        # Sort clusters by size
        result.clusters.sort(key=lambda c: c.size, reverse=True)
        result.passing_clusters.sort(key=lambda c: c.size, reverse=True)
        result.failing_clusters.sort(key=lambda c: c.size, reverse=True)

        # Determine best cluster
        if self.prioritize_passing and result.passing_clusters:
            result.best_cluster = result.passing_clusters[0]
        elif result.clusters:
            result.best_cluster = result.clusters[0]

        # Compute confidence
        if result.best_cluster and result.total_patches > 0:
            # Confidence based on cluster size relative to total
            result.confidence = result.best_cluster.size / result.total_patches

            # Boost confidence if all tests pass
            if result.best_cluster.all_pass:
                result.confidence = min(1.0, result.confidence * 1.2)

        return result

    def merge_with_similarity_clustering(
        self,
        behavioral_result: BehavioralClusteringResult,
        similarity_clusters: Dict[str, List[str]],  # {cluster_id: [patch_ids]}
    ) -> BehavioralClusteringResult:
        """Merge behavioral clustering with similarity-based clustering.

        This allows using similarity clustering for diversity analysis while
        keeping behavioral clustering as the primary correctness signal.

        Args:
            behavioral_result: Result from behavioral clustering
            similarity_clusters: Clusters from similarity-based clustering

        Returns:
            Enhanced BehavioralClusteringResult with diversity info
        """
        # For now, just return behavioral result
        # In the future, we could use similarity clusters to:
        # 1. Break ties between equally-passing behavioral clusters
        # 2. Identify diverse approaches within a passing cluster
        # 3. Ensure we don't over-commit to one implementation style
        return behavioral_result


def cluster_by_test_outcomes(
    patch_test_results: Dict[str, TestResult],
    prioritize_passing: bool = True,
) -> BehavioralClusteringResult:
    """Convenience function to cluster patches by test outcomes.

    Args:
        patch_test_results: Dict of {patch_id: TestResult}
        prioritize_passing: If True, passing clusters ranked higher

    Returns:
        BehavioralClusteringResult
    """
    signatures = [
        BehavioralSignature.from_test_result(patch_id, result)
        for patch_id, result in patch_test_results.items()
    ]

    clusterer = BehavioralClusterer(prioritize_passing=prioritize_passing)
    return clusterer.cluster(signatures)


def select_best_patches(
    behavioral_result: BehavioralClusteringResult,
    max_patches: int = 3,
) -> List[str]:
    """Select the best patches from behavioral clustering.

    Strategy:
    1. If there are passing clusters, take representatives from them
    2. Otherwise, take from largest failing clusters
    3. Ensure diversity by taking from different clusters

    Args:
        behavioral_result: Result from behavioral clustering
        max_patches: Maximum patches to return

    Returns:
        List of patch IDs to consider for quality selection
    """
    selected = []

    # First, add all passing cluster representatives
    for cluster in behavioral_result.passing_clusters:
        if len(selected) >= max_patches:
            break
        if cluster.representative_patch_id:
            selected.append(cluster.representative_patch_id)

    # If no passing, add from failing clusters (for debugging/repair)
    if not selected:
        for cluster in behavioral_result.failing_clusters:
            if len(selected) >= max_patches:
                break
            if cluster.representative_patch_id:
                selected.append(cluster.representative_patch_id)

    return selected


@dataclass
class VerificationResult:
    """Combined result from patch verification (apply + test)."""

    patch_id: str
    apply_success: bool = False
    apply_errors: List[str] = field(default_factory=list)

    test_result: Optional[TestResult] = None
    behavioral_signature: Optional[BehavioralSignature] = None

    # Computed fields
    @property
    def all_pass(self) -> bool:
        return (
            self.apply_success
            and self.test_result is not None
            and self.test_result.success
        )

    @property
    def is_valid(self) -> bool:
        """Check if patch is valid (applies cleanly and tests run)."""
        return self.apply_success and (
            self.test_result is None or self.test_result.tests_ran
        )
