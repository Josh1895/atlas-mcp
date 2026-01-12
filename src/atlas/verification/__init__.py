"""Verification and validation components for ATLAS.

Includes:
- Patch validation (structure and static analysis)
- Patch application (apply unified diffs)
- Test execution (auto-detect and run tests)
- Behavioral clustering (cluster by test outcomes - PRIMARY ORACLE)
- Similarity clustering (for diversity analysis - SUPPORTING ROLE)
"""

from atlas.verification.behavioral_clustering import (
    BehavioralCluster,
    BehavioralClusterer,
    BehavioralClusteringResult,
    BehavioralSignature,
    VerificationResult,
    cluster_by_test_outcomes,
    compute_behavioral_similarity,
    select_best_patches,
)
from atlas.verification.clustering import SimilarityClustering
from atlas.verification.patch_applier import (
    PatchApplier,
    PatchApplyResult,
    PatchParser,
    create_patched_checkout,
)
from atlas.verification.patch_composer import PatchComposeResult, apply_patches, diff_files
from atlas.verification.patch_validator import PatchValidator
from atlas.verification.oracles import OracleResult, OracleRunner
from atlas.verification.static_analysis import StaticAnalyzer
from atlas.verification.test_runner import (
    DetectedTestConfig,
    TestCase,
    TestFramework,
    TestFrameworkDetector,
    TestResult,
    TestRunner,
    cleanup_patched_checkout,
    run_tests_on_patch,
)
from atlas.verification.ownership import OwnershipValidator

__all__ = [
    # Patch validation
    "PatchValidator",
    "StaticAnalyzer",
    "OracleRunner",
    "OracleResult",
    # Patch application
    "PatchApplier",
    "PatchApplyResult",
    "PatchParser",
    "create_patched_checkout",
    "PatchComposeResult",
    "apply_patches",
    "diff_files",
    # Test execution
    "TestRunner",
    "TestResult",
    "TestCase",
    "TestFramework",
    "TestFrameworkDetector",
    "DetectedTestConfig",
    "run_tests_on_patch",
    "cleanup_patched_checkout",
    # Behavioral clustering (PRIMARY ORACLE)
    "BehavioralClusterer",
    "BehavioralCluster",
    "BehavioralClusteringResult",
    "BehavioralSignature",
    "VerificationResult",
    "cluster_by_test_outcomes",
    "compute_behavioral_similarity",
    "select_best_patches",
    # Similarity clustering (supporting role)
    "SimilarityClustering",
    "OwnershipValidator",
]
