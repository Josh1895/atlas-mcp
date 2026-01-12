"""Quality Selection Layer for ATLAS.

This module handles selecting the best patch from multiple candidates
that all pass tests. It uses:

1. Code Similarity Clustering - Group patches by implementation approach
2. Objective Quality Scoring - Measurable metrics (style, maintainability, risk)
3. LLM PR Review - Senior developer perspective from a different model
4. Final Selection - Combine all signals to pick the best patch
"""

from atlas.quality.fingerprinting import (
    PatchFingerprint,
    TokenFingerprinter,
    compute_similarity,
    cluster_by_similarity,
)
from atlas.quality.ast_analysis import (
    ASTChangeSignature,
    ASTAnalyzer,
    compute_ast_similarity,
)
from atlas.quality.quality_scorer import (
    QualityScore,
    QualityScorer,
)
from atlas.quality.pr_reviewer import (
    PRReviewResult,
    LLMPRReviewer,
    PairwiseResult,
    TournamentReviewer,
)
from atlas.quality.selector import (
    SelectionResult,
    FinalSelector,
)
from atlas.quality.pipeline import (
    QualitySelectionPipeline,
    QualitySelectionConfig,
)

__all__ = [
    # Fingerprinting
    "PatchFingerprint",
    "TokenFingerprinter",
    "compute_similarity",
    "cluster_by_similarity",
    # AST Analysis
    "ASTChangeSignature",
    "ASTAnalyzer",
    "compute_ast_similarity",
    # Quality Scoring
    "QualityScore",
    "QualityScorer",
    # PR Review
    "PRReviewResult",
    "LLMPRReviewer",
    "PairwiseResult",
    "TournamentReviewer",
    # Selection
    "SelectionResult",
    "FinalSelector",
    # Pipeline
    "QualitySelectionPipeline",
    "QualitySelectionConfig",
]
