"""Quality Selection Pipeline for ATLAS.

This module orchestrates the full quality selection process:
1. Cluster patches by similarity (token fingerprinting + AST)
2. Score each patch objectively (style, maintainability, risk)
3. LLM PR review for top candidates
4. Final selection combining all signals
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from atlas.quality.fingerprinting import (
    TokenFingerprinter,
    cluster_by_similarity,
)
from atlas.quality.ast_analysis import ASTAnalyzer, compute_ast_similarity
from atlas.quality.quality_scorer import QualityScorer, QualityScore
from atlas.quality.pr_reviewer import (
    LLMPRReviewer,
    TournamentReviewer,
    PRReviewResult,
    label_approach_family,
)
from atlas.quality.selector import FinalSelector, SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class QualitySelectionConfig:
    """Configuration for the quality selection pipeline."""

    # Clustering
    similarity_threshold: float = 0.3  # Jaccard threshold for clustering
    use_ast_refinement: bool = True  # Use AST for refined clustering

    # Quality scoring
    max_risk_score: float = 50  # Hard gate for risk

    # LLM review
    enable_llm_review: bool = True
    enable_tournament: bool = True
    max_patches_for_tournament: int = 6  # Limit to reduce API calls
    review_model: str = "gemini-2.5-flash"

    # Selection weights
    style_weight: float = 0.20
    maintainability_weight: float = 0.25
    risk_weight: float = 0.25
    llm_review_weight: float = 0.20
    family_size_weight: float = 0.10


@dataclass
class ApproachFamily:
    """Represents a family of similar patches."""

    family_id: str
    label: str  # Human-readable description
    patch_ids: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.patch_ids)


@dataclass
class QualitySelectionResult:
    """Full result from the quality selection pipeline."""

    # Selection result
    selection: SelectionResult

    # Intermediate results
    approach_families: List[ApproachFamily] = field(default_factory=list)
    quality_scores: Dict[str, QualityScore] = field(default_factory=dict)
    review_results: Dict[str, PRReviewResult] = field(default_factory=dict)
    tournament_ranking: List[str] = field(default_factory=list)

    # Metadata
    total_patches: int = 0
    patches_after_gates: int = 0
    families_found: int = 0


class QualitySelectionPipeline:
    """Orchestrates the full quality selection process."""

    def __init__(
        self,
        config: Optional[QualitySelectionConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Configuration for the pipeline
            llm_client: LLM client for reviews (required if LLM review enabled)
        """
        self.config = config or QualitySelectionConfig()
        self.llm_client = llm_client

        # Initialize components
        self.fingerprinter = TokenFingerprinter()
        self.ast_analyzer = ASTAnalyzer()
        self.quality_scorer = QualityScorer()
        self.selector = FinalSelector(
            style_weight=self.config.style_weight,
            maintainability_weight=self.config.maintainability_weight,
            risk_weight=self.config.risk_weight,
            llm_review_weight=self.config.llm_review_weight,
            family_size_weight=self.config.family_size_weight,
        )

        if llm_client:
            self.pr_reviewer = LLMPRReviewer(llm_client, self.config.review_model)
            self.tournament_reviewer = TournamentReviewer(
                llm_client, self.config.review_model
            )
        else:
            self.pr_reviewer = None
            self.tournament_reviewer = None

    async def select_best_patch(
        self,
        patches: Dict[str, str],  # {patch_id: patch_content}
        issue_description: str,
        original_files: Dict[str, str],  # {filepath: original_content}
        patched_files_map: Dict[str, Dict[str, str]],  # {patch_id: {filepath: content}}
        context_code: str = "",
    ) -> QualitySelectionResult:
        """Run the full quality selection pipeline.

        Args:
            patches: Dict of {patch_id: patch_content}
            issue_description: Description of the issue being fixed
            original_files: Dict of {filepath: original_content}
            patched_files_map: Dict of {patch_id: {filepath: patched_content}}
            context_code: Additional context code

        Returns:
            QualitySelectionResult with the best patch and all intermediate data
        """
        logger.info(f"Starting quality selection for {len(patches)} patches")

        result = QualitySelectionResult(
            selection=SelectionResult(
                best_patch_id="",
                best_patch_content="",
            ),
            total_patches=len(patches),
        )

        if not patches:
            result.selection.selection_reason = "No patches provided"
            return result

        # Stage 1: Cluster patches by similarity
        logger.info("Stage 1: Clustering patches by similarity")
        approach_families = await self._cluster_patches(patches)
        result.approach_families = approach_families
        result.families_found = len(approach_families)

        # Stage 2: Score each patch objectively
        logger.info("Stage 2: Computing objective quality scores")
        quality_scores = await self._score_patches(
            patches, original_files, patched_files_map
        )
        result.quality_scores = quality_scores

        # Apply hard gates
        filtered_patches = self.selector.apply_hard_gates(
            patches,
            quality_scores,
            max_risk_score=self.config.max_risk_score,
        )
        result.patches_after_gates = len(filtered_patches)

        if not filtered_patches:
            logger.warning("All patches filtered out by hard gates")
            result.selection.selection_reason = "All patches failed quality gates"
            return result

        # Stage 3: LLM PR review (if enabled)
        review_results = {}
        tournament_ranking = []

        if self.config.enable_llm_review and self.pr_reviewer:
            logger.info("Stage 3: Running LLM PR reviews")

            # Review top candidates
            top_candidates = self._get_top_candidates(
                filtered_patches, quality_scores, limit=self.config.max_patches_for_tournament
            )

            for patch_id in top_candidates:
                review = await self.pr_reviewer.review(
                    patch=filtered_patches[patch_id],
                    patch_id=patch_id,
                    issue_description=issue_description,
                    surrounding_code=context_code,
                )
                review_results[patch_id] = review

            result.review_results = review_results

            # Run tournament if enabled
            if self.config.enable_tournament and self.tournament_reviewer:
                if len(top_candidates) >= 2:
                    logger.info("Running pairwise tournament")
                    tournament_patches = [
                        (pid, filtered_patches[pid]) for pid in top_candidates
                    ]
                    tournament_ranking = await self.tournament_reviewer.run_tournament(
                        tournament_patches,
                        issue_description,
                        context_code,
                    )
                    result.tournament_ranking = tournament_ranking

        # Stage 4: Final selection
        logger.info("Stage 4: Making final selection")

        # Convert approach families to dict format
        families_dict = {f.label: f.patch_ids for f in approach_families}

        selection = self.selector.select(
            patches=filtered_patches,
            approach_families=families_dict,
            quality_scores=quality_scores,
            review_results=review_results,
            tournament_ranking=tournament_ranking,
        )

        result.selection = selection

        logger.info(f"Selected patch: {selection.best_patch_id}")
        logger.info(f"Reason: {selection.selection_reason}")

        return result

    async def _cluster_patches(
        self,
        patches: Dict[str, str],
    ) -> List[ApproachFamily]:
        """Cluster patches by similarity.

        Args:
            patches: Dict of {patch_id: patch_content}

        Returns:
            List of ApproachFamily objects
        """
        # Create fingerprints
        fingerprints = []
        for patch_id, content in patches.items():
            fp = self.fingerprinter.fingerprint(content, patch_id)
            fingerprints.append(fp)

        # Cluster by token similarity
        clusters = cluster_by_similarity(
            fingerprints, threshold=self.config.similarity_threshold
        )

        # Convert to ApproachFamily objects
        families = []
        for i, cluster in enumerate(clusters):
            patch_ids = [fp.patch_id for fp in cluster]

            # Generate label if LLM available
            label = f"approach_{i}"
            if self.llm_client and patch_ids:
                try:
                    patch_contents = [patches[pid] for pid in patch_ids]
                    label = await label_approach_family(patch_contents, self.llm_client)
                except Exception as e:
                    logger.warning(f"Failed to label family: {e}")

            families.append(
                ApproachFamily(
                    family_id=f"family_{i}",
                    label=label,
                    patch_ids=patch_ids,
                )
            )

        return families

    async def _score_patches(
        self,
        patches: Dict[str, str],
        original_files: Dict[str, str],
        patched_files_map: Dict[str, Dict[str, str]],
    ) -> Dict[str, QualityScore]:
        """Score all patches.

        Args:
            patches: Dict of {patch_id: patch_content}
            original_files: Dict of {filepath: original_content}
            patched_files_map: Dict of {patch_id: {filepath: patched_content}}

        Returns:
            Dict of {patch_id: QualityScore}
        """
        scores = {}

        for patch_id, patch_content in patches.items():
            patched_files = patched_files_map.get(patch_id, {})

            try:
                score = await self.quality_scorer.score(
                    patch=patch_content,
                    patch_id=patch_id,
                    original_files=original_files,
                    patched_files=patched_files,
                )
                scores[patch_id] = score
            except Exception as e:
                logger.warning(f"Failed to score patch {patch_id}: {e}")
                # Create default score
                scores[patch_id] = QualityScore(patch_id=patch_id)

        return scores

    def _get_top_candidates(
        self,
        patches: Dict[str, str],
        quality_scores: Dict[str, QualityScore],
        limit: int = 6,
    ) -> List[str]:
        """Get top candidate patch IDs by quality score.

        Args:
            patches: Dict of {patch_id: patch_content}
            quality_scores: Dict of {patch_id: QualityScore}
            limit: Maximum number of candidates to return

        Returns:
            List of top patch IDs
        """
        # Sort by overall score
        scored = [
            (pid, quality_scores.get(pid, QualityScore(patch_id=pid)).overall_score)
            for pid in patches.keys()
        ]

        sorted_patches = sorted(scored, key=lambda x: x[1], reverse=True)

        return [pid for pid, _ in sorted_patches[:limit]]


async def run_quality_selection(
    patches: Dict[str, str],
    issue_description: str,
    original_files: Dict[str, str],
    patched_files_map: Dict[str, Dict[str, str]],
    llm_client: Optional[Any] = None,
    config: Optional[QualitySelectionConfig] = None,
) -> QualitySelectionResult:
    """Convenience function to run the full quality selection pipeline.

    Args:
        patches: Dict of {patch_id: patch_content}
        issue_description: Description of the issue being fixed
        original_files: Dict of {filepath: original_content}
        patched_files_map: Dict of {patch_id: {filepath: patched_content}}
        llm_client: Optional LLM client for reviews
        config: Optional configuration

    Returns:
        QualitySelectionResult with the best patch
    """
    pipeline = QualitySelectionPipeline(config=config, llm_client=llm_client)

    return await pipeline.select_best_patch(
        patches=patches,
        issue_description=issue_description,
        original_files=original_files,
        patched_files_map=patched_files_map,
    )
