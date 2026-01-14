"""
Consensus Engine for swarm operations.

Orchestrates clustering and voting to select the best output from
multiple agent candidates (P-003: Multi-Agent Consensus).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from .schemas import AgentOutput, ClusterInfo, SwarmMode
from .atlas_adapter import AtlasAdapter, is_atlas_available, create_atlas_adapter
from .answer_clustering import AnswerClusterer, cluster_answers

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result from consensus voting."""
    consensus_reached: bool = False
    selected_output: str = ""
    selected_cluster_id: Optional[str] = None
    confidence_score: float = 0.0
    vote_counts: Dict[str, int] = None
    clusters: List[ClusterInfo] = None
    winning_agent_ids: List[str] = None

    def __post_init__(self):
        if self.vote_counts is None:
            self.vote_counts = {}
        if self.clusters is None:
            self.clusters = []
        if self.winning_agent_ids is None:
            self.winning_agent_ids = []


class ConsensusEngine:
    """
    Engine for building consensus from multiple agent outputs.

    Supports two modes:
    - Patch mode: Uses similarity clustering on patches
    - Answer mode: Uses embedding-based clustering on answers

    Uses first-to-ahead-by-K voting to determine consensus.
    """

    def __init__(
        self,
        consensus_k: int = 2,
        mode: SwarmMode = SwarmMode.PATCH,
    ):
        """
        Initialize the consensus engine.

        Args:
            consensus_k: Votes ahead needed for consensus
            mode: Swarm mode (patch or answer)
        """
        self.consensus_k = consensus_k
        self.mode = mode
        self._atlas_adapter: Optional[AtlasAdapter] = None
        self._answer_clusterer: Optional[AnswerClusterer] = None

    @property
    def atlas_adapter(self) -> Optional[AtlasAdapter]:
        """Get or create Atlas adapter for patch mode."""
        if self._atlas_adapter is None and is_atlas_available():
            self._atlas_adapter = create_atlas_adapter()
        return self._atlas_adapter

    @property
    def answer_clusterer(self) -> AnswerClusterer:
        """Get or create answer clusterer."""
        if self._answer_clusterer is None:
            self._answer_clusterer = AnswerClusterer()
        return self._answer_clusterer

    async def cluster_and_vote(
        self,
        outputs: List[AgentOutput],
        embeddings: Optional[List[List[float]]] = None,
        behavioral_results: Optional[Dict[str, Any]] = None,
    ) -> ConsensusResult:
        """
        Cluster outputs and run consensus voting.

        Args:
            outputs: Agent outputs to process
            embeddings: Optional embeddings for answer mode
            behavioral_results: Optional test results for behavioral clustering

        Returns:
            ConsensusResult with selected output and metadata
        """
        if not outputs:
            return ConsensusResult()

        # Filter to valid outputs
        valid_outputs = [o for o in outputs if o.is_valid]
        if not valid_outputs:
            # Fall back to all outputs if none are valid
            valid_outputs = outputs
            logger.warning("No valid outputs, using all outputs for consensus")

        # Cluster based on mode
        if self.mode == SwarmMode.PATCH:
            clusters = await self._cluster_patch_mode(valid_outputs, behavioral_results)
        else:
            clusters = await self._cluster_answer_mode(valid_outputs, embeddings)

        # Run voting
        return self._vote_on_clusters(clusters, valid_outputs)

    async def _cluster_patch_mode(
        self,
        outputs: List[AgentOutput],
        behavioral_results: Optional[Dict[str, Any]] = None,
    ) -> List[ClusterInfo]:
        """Cluster patches using similarity or behavioral clustering."""
        # If we have test results, try behavioral clustering
        if behavioral_results:
            clusters = self._cluster_by_test_outcomes(outputs, behavioral_results)
            if clusters:
                return clusters

        # Fall back to similarity clustering
        if self.atlas_adapter:
            return self.atlas_adapter.cluster_patches(outputs)

        # Manual fallback clustering
        return self._fallback_patch_clustering(outputs)

    async def _cluster_answer_mode(
        self,
        outputs: List[AgentOutput],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[ClusterInfo]:
        """Cluster answers using embeddings."""
        return await cluster_answers(outputs, embeddings)

    def _cluster_by_test_outcomes(
        self,
        outputs: List[AgentOutput],
        behavioral_results: Dict[str, Any],
    ) -> List[ClusterInfo]:
        """
        Cluster by test outcomes (behavioral clustering).

        Groups patches by:
        1. All tests pass (highest priority)
        2. Failure signature (which tests failed)
        3. Apply failures

        Passing clusters are always preferred (P-002: Tests as Primary Oracle).
        """
        from collections import defaultdict

        # Group by behavior signature
        groups: Dict[str, List[AgentOutput]] = defaultdict(list)

        for output in outputs:
            agent_result = behavioral_results.get(output.agent_id, {})

            if agent_result.get("patch_applied") is False:
                # Patch couldn't be applied
                signature = "apply_failed"
            elif agent_result.get("passed", False):
                # All tests pass
                signature = "all_pass"
            else:
                # Group by failure signature (sorted failed test names)
                failure_sig = agent_result.get("failure_signature", "unknown")
                signature = f"fail:{failure_sig}"

            groups[signature].append(output)

        clusters = []
        cluster_idx = 0

        # Process passing cluster first (highest priority)
        if "all_pass" in groups:
            passing = groups.pop("all_pass")
            cluster_id = "passing"

            for output in passing:
                output.cluster_id = cluster_id

            # Pick best quality patch as representative
            representative = self._select_best_representative(passing, behavioral_results)

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                size=len(passing),
                vote_count=len(passing),
                representative_output=representative.output_text,
                member_agent_ids=[o.agent_id for o in passing],
                behavioral_pass=True,
            ))
            cluster_idx += 1

        # Process failing clusters grouped by failure signature
        fail_groups = [(k, v) for k, v in groups.items() if k.startswith("fail:")]
        # Sort by size (largest first)
        fail_groups.sort(key=lambda x: -len(x[1]))

        for signature, members in fail_groups:
            cluster_id = f"failing_{cluster_idx}"

            for output in members:
                output.cluster_id = cluster_id

            representative = self._select_best_representative(members, behavioral_results)

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                size=len(members),
                vote_count=len(members),
                representative_output=representative.output_text,
                member_agent_ids=[o.agent_id for o in members],
                behavioral_pass=False,
            ))
            cluster_idx += 1

        # Process apply failures last (lowest priority)
        if "apply_failed" in groups:
            failed = groups["apply_failed"]
            cluster_id = "apply_failed"

            for output in failed:
                output.cluster_id = cluster_id

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                size=len(failed),
                vote_count=len(failed),
                representative_output=failed[0].output_text if failed else "",
                member_agent_ids=[o.agent_id for o in failed],
                behavioral_pass=False,
            ))

        return clusters

    def _select_best_representative(
        self,
        members: List[AgentOutput],
        behavioral_results: Dict[str, Any],
    ) -> AgentOutput:
        """
        Select the best representative from a cluster.

        Prefers patches with:
        1. Highest quality score (if available)
        2. Fewest risk flags
        3. Shortest length (simpler is better)
        """
        if len(members) == 1:
            return members[0]

        # Score each candidate
        scored = []
        for member in members:
            agent_result = behavioral_results.get(member.agent_id, {})
            quality = agent_result.get("quality_score", 50.0)
            risk_count = len(agent_result.get("risk_flags", []))
            length = len(member.output_text)

            # Higher quality = better, fewer risks = better, shorter = better
            score = quality - (risk_count * 5) - (length / 1000)
            scored.append((score, member))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def _fallback_patch_clustering(
        self,
        outputs: List[AgentOutput],
    ) -> List[ClusterInfo]:
        """Simple fallback clustering by patch content hash."""
        import hashlib
        from collections import defaultdict

        groups: Dict[str, List[AgentOutput]] = defaultdict(list)

        for output in outputs:
            # Normalize patch
            normalized = output.output_text.strip()
            # Hash for grouping
            patch_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
            groups[patch_hash].append(output)

        clusters = []
        for idx, (hash_key, members) in enumerate(groups.items()):
            cluster_id = f"cluster_{idx}"

            for output in members:
                output.cluster_id = cluster_id

            # Pick shortest valid patch as representative
            valid_members = [m for m in members if m.is_valid]
            representative = min(
                valid_members or members,
                key=lambda o: len(o.output_text)
            )

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                size=len(members),
                vote_count=len(members),
                representative_output=representative.output_text,
                member_agent_ids=[m.agent_id for m in members],
            ))

        return sorted(clusters, key=lambda c: -c.size)

    def _vote_on_clusters(
        self,
        clusters: List[ClusterInfo],
        outputs: List[AgentOutput],
    ) -> ConsensusResult:
        """
        Run first-to-ahead-by-K voting on clusters.

        Args:
            clusters: Clusters to vote on
            outputs: Original outputs (for reference)

        Returns:
            ConsensusResult with voting outcome
        """
        if not clusters:
            return ConsensusResult()

        # Sort clusters by size (votes)
        sorted_clusters = sorted(clusters, key=lambda c: -c.size)

        # Build vote counts
        vote_counts = {c.cluster_id: c.size for c in sorted_clusters}
        total_votes = sum(vote_counts.values())

        result = ConsensusResult(
            vote_counts=vote_counts,
            clusters=clusters,
        )

        if len(sorted_clusters) == 1:
            # Single cluster - automatic consensus
            winner = sorted_clusters[0]
            result.consensus_reached = True
            result.selected_output = winner.representative_output
            result.selected_cluster_id = winner.cluster_id
            result.winning_agent_ids = winner.member_agent_ids
            result.confidence_score = 1.0
            return result

        # Check for behavioral preference (passing > failing)
        passing_clusters = [c for c in sorted_clusters if c.behavioral_pass is True]
        if passing_clusters:
            # Prefer passing clusters
            winner = passing_clusters[0]
            result.selected_output = winner.representative_output
            result.selected_cluster_id = winner.cluster_id
            result.winning_agent_ids = winner.member_agent_ids
            result.consensus_reached = winner.size >= self.consensus_k
            result.confidence_score = winner.size / total_votes if total_votes > 0 else 0.0
            return result

        # Standard first-to-ahead-by-K
        leader = sorted_clusters[0]
        runner_up_size = sorted_clusters[1].size if len(sorted_clusters) > 1 else 0
        margin = leader.size - runner_up_size

        result.selected_output = leader.representative_output
        result.selected_cluster_id = leader.cluster_id
        result.winning_agent_ids = leader.member_agent_ids

        if margin >= self.consensus_k:
            result.consensus_reached = True
            result.confidence_score = margin / total_votes if total_votes > 0 else 0.0
        else:
            result.consensus_reached = False
            # Lower confidence when no clear consensus
            result.confidence_score = (leader.size / total_votes) * 0.5 if total_votes > 0 else 0.0

        return result

    def reset(self) -> None:
        """Reset engine state for new run."""
        if self._atlas_adapter:
            self._atlas_adapter.reset()


async def build_consensus(
    outputs: List[AgentOutput],
    mode: SwarmMode = SwarmMode.PATCH,
    consensus_k: int = 2,
    embeddings: Optional[List[List[float]]] = None,
    behavioral_results: Optional[Dict[str, Any]] = None,
) -> ConsensusResult:
    """
    Convenience function to build consensus from outputs.

    Args:
        outputs: Agent outputs to process
        mode: Swarm mode
        consensus_k: Votes ahead needed
        embeddings: Optional embeddings for answer mode
        behavioral_results: Optional test results

    Returns:
        ConsensusResult
    """
    engine = ConsensusEngine(consensus_k=consensus_k, mode=mode)
    return await engine.cluster_and_vote(outputs, embeddings, behavioral_results)
