"""Final selection logic for choosing the best patch.

Combines all quality signals:
- Approach family size (larger = more agreement)
- Objective quality scores (style, maintainability, risk)
- LLM PR review scores
- Tournament rankings
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from atlas.quality.pr_reviewer import PRReviewResult
from atlas.quality.quality_scorer import QualityScore

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Final selection result."""

    # The winner
    best_patch_id: str
    best_patch_content: str

    # Runner-ups (different approach families)
    alternates: List[Tuple[str, str]] = field(
        default_factory=list
    )  # [(patch_id, patch_content), ...]

    # Explanation
    selection_reason: str = ""

    # Scores
    scores_summary: Dict[str, float] = field(default_factory=dict)

    # Metadata
    approach_family: str = ""
    total_candidates: int = 0
    families_considered: int = 0


class FinalSelector:
    """Combines all signals to select the best patch."""

    def __init__(
        self,
        style_weight: float = 0.20,
        maintainability_weight: float = 0.25,
        risk_weight: float = 0.25,
        llm_review_weight: float = 0.20,
        family_size_weight: float = 0.10,
    ):
        """Initialize the selector.

        Args:
            style_weight: Weight for style score
            maintainability_weight: Weight for maintainability score
            risk_weight: Weight for risk score (inverted)
            llm_review_weight: Weight for LLM review score
            family_size_weight: Weight for approach family size
        """
        self.weights = {
            "style": style_weight,
            "maintainability": maintainability_weight,
            "risk": risk_weight,
            "llm_review": llm_review_weight,
            "family_size": family_size_weight,
        }

    def select(
        self,
        patches: Dict[str, str],  # {patch_id: patch_content}
        approach_families: Dict[str, List[str]],  # {family_label: [patch_ids]}
        quality_scores: Dict[str, QualityScore],  # {patch_id: QualityScore}
        review_results: Optional[Dict[str, PRReviewResult]] = None,
        tournament_ranking: Optional[List[str]] = None,
    ) -> SelectionResult:
        """Select the best patch using all available signals.

        Args:
            patches: Dict of {patch_id: patch_content}
            approach_families: Dict of {family_label: [patch_ids in family]}
            quality_scores: Dict of {patch_id: QualityScore}
            review_results: Optional dict of {patch_id: PRReviewResult}
            tournament_ranking: Optional list of patch_ids ordered best to worst

        Returns:
            SelectionResult with the best patch and alternates
        """
        if not patches:
            return SelectionResult(
                best_patch_id="",
                best_patch_content="",
                selection_reason="No patches to select from",
            )

        review_results = review_results or {}

        # Get family sizes (larger = more agreement = better)
        family_sizes = {}
        patch_to_family = {}
        for family_label, patch_ids in approach_families.items():
            for pid in patch_ids:
                family_sizes[pid] = len(patch_ids)
                patch_to_family[pid] = family_label

        max_family_size = max(family_sizes.values()) if family_sizes else 1

        # Compute composite score for each patch
        composite_scores = {}

        for patch_id in patches.keys():
            qs = quality_scores.get(patch_id)
            rr = review_results.get(patch_id)

            if not qs:
                composite_scores[patch_id] = 0.0
                continue

            # Normalize scores to 0-100
            style_score = qs.style_score
            maint_score = qs.maintainability_score
            risk_score = 100 - qs.risk_score  # Invert: high risk = low score

            # LLM review score (0-10 -> 0-100)
            llm_score = (rr.quality_score * 10) if rr else 50

            # Penalize if LLM flagged as hacky or has code smells
            if rr:
                if rr.is_hacky:
                    llm_score -= 20
                if rr.has_code_smells:
                    llm_score -= 10
                if rr.is_over_engineered:
                    llm_score -= 5

            llm_score = max(0, min(100, llm_score))

            # Family size score (normalized to 0-100)
            family_score = (
                (family_sizes.get(patch_id, 1) / max_family_size) * 100
                if max_family_size > 0
                else 50
            )

            # Weighted combination
            composite = (
                self.weights["style"] * style_score
                + self.weights["maintainability"] * maint_score
                + self.weights["risk"] * risk_score
                + self.weights["llm_review"] * llm_score
                + self.weights["family_size"] * family_score
            )

            composite_scores[patch_id] = composite

        # Apply tournament ranking boost if available
        if tournament_ranking:
            n = len(tournament_ranking)
            for rank, patch_id in enumerate(tournament_ranking):
                # Higher rank = more boost
                boost = (n - rank) / n * 10  # Up to 10 point boost for winner
                if patch_id in composite_scores:
                    composite_scores[patch_id] += boost

        # Sort by composite score
        ranked = sorted(
            composite_scores.keys(),
            key=lambda x: composite_scores[x],
            reverse=True,
        )

        if not ranked:
            return SelectionResult(
                best_patch_id="",
                best_patch_content="",
                selection_reason="No valid patches after scoring",
            )

        # Get the winner
        best_id = ranked[0]
        best_content = patches.get(best_id, "")
        best_family = patch_to_family.get(best_id, "unknown")

        # Get alternates from different families
        alternates = []
        seen_families = {best_family}

        for patch_id in ranked[1:]:
            family = patch_to_family.get(patch_id, "unknown")
            if family not in seen_families:
                alternates.append((patch_id, patches.get(patch_id, "")))
                seen_families.add(family)

            if len(alternates) >= 2:
                break

        # Build explanation
        best_qs = quality_scores.get(best_id)
        best_rr = review_results.get(best_id)

        reason_parts = [f"Selected patch '{best_id}' from family '{best_family}'."]

        if best_qs:
            reason_parts.append(
                f"Quality: style={best_qs.style_score:.0f}, "
                f"maintainability={best_qs.maintainability_score:.0f}, "
                f"risk={best_qs.risk_score:.0f}"
            )

        if best_rr:
            reason_parts.append(
                f"LLM review: {best_rr.verdict} (score: {best_rr.quality_score:.1f}/10)"
            )

        family_size = family_sizes.get(best_id, 1)
        if family_size > 1:
            reason_parts.append(
                f"Approach agreed upon by {family_size} agents."
            )

        return SelectionResult(
            best_patch_id=best_id,
            best_patch_content=best_content,
            alternates=alternates,
            selection_reason=" ".join(reason_parts),
            scores_summary={
                pid: score for pid, score in list(composite_scores.items())[:5]
            },
            approach_family=best_family,
            total_candidates=len(patches),
            families_considered=len(approach_families),
        )

    def apply_hard_gates(
        self,
        patches: Dict[str, str],
        quality_scores: Dict[str, QualityScore],
        review_results: Optional[Dict[str, PRReviewResult]] = None,
        max_risk_score: float = 50,
        require_llm_approval: bool = False,
    ) -> Dict[str, str]:
        """Apply hard gates to filter out unacceptable patches.

        Args:
            patches: Dict of {patch_id: patch_content}
            quality_scores: Dict of {patch_id: QualityScore}
            review_results: Optional dict of {patch_id: PRReviewResult}
            max_risk_score: Maximum acceptable risk score
            require_llm_approval: If True, only keep LLM-approved patches

        Returns:
            Filtered dict of {patch_id: patch_content}
        """
        review_results = review_results or {}
        filtered = {}

        for patch_id, content in patches.items():
            qs = quality_scores.get(patch_id)
            rr = review_results.get(patch_id)

            # Skip if no quality score
            if not qs:
                continue

            # Gate 1: Risk score must be below threshold
            if qs.risk_score > max_risk_score:
                logger.debug(f"Patch {patch_id} rejected: risk score {qs.risk_score}")
                continue

            # Gate 2: LLM approval if required
            if require_llm_approval and rr:
                if rr.verdict == "request_changes":
                    logger.debug(f"Patch {patch_id} rejected: LLM requested changes")
                    continue

            # Gate 3: Critical risk flags
            critical_flags = ["eval_usage", "exec_usage", "os_system"]
            if qs.risk_flags:
                has_critical = any(
                    any(cf in flag for cf in critical_flags) for flag in qs.risk_flags
                )
                if has_critical:
                    logger.debug(f"Patch {patch_id} rejected: critical risk flag")
                    continue

            filtered[patch_id] = content

        return filtered
