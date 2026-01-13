"""First-to-ahead-by-K voting consensus algorithm."""

import logging
from dataclasses import dataclass, field
from typing import Any

from atlas.core.task import Solution
from atlas.verification.clustering import Cluster, ClusteringResult, SimilarityClustering

logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """Result from a voting round."""

    winner: Any | None = None
    winning_solution: Solution | None = None
    consensus_reached: bool = False
    vote_counts: dict[str, int] = field(default_factory=dict)
    total_votes: int = 0
    margin: int = 0
    round_number: int = 0

    @property
    def confidence_score(self) -> float:
        """Calculate a confidence score based on voting margin."""
        if self.total_votes == 0:
            return 0.0

        if self.winner is None:
            return 0.0

        # Higher margin = higher confidence
        # Max confidence when margin is >= half of total votes
        max_margin = self.total_votes / 2
        normalized_margin = min(self.margin / max_margin, 1.0) if max_margin > 0 else 0.0

        return normalized_margin


class VotingManager:
    """Manages consensus voting using first-to-ahead-by-K algorithm.

    The winner is the first cluster to lead by K votes.
    This provides early stopping when there's clear consensus.
    """

    def __init__(self, k: int = 3, similarity_threshold: float = 0.8):
        """Initialize the voting manager.

        Args:
            k: Number of votes a cluster must lead by to win
            similarity_threshold: Threshold for clustering similarity
        """
        self.k = k
        self.clustering = SimilarityClustering(similarity_threshold)
        self._round_number = 0
        self._history: list[VotingResult] = []

    def vote(self, solutions: list[Solution]) -> VotingResult:
        """Perform a voting round on the given solutions.

        Args:
            solutions: List of solutions to vote on

        Returns:
            VotingResult with the outcome
        """
        self._round_number += 1

        if self._has_behavioral_data(solutions):
            result = self._vote_behavioral(solutions)
            self._history.append(result)
            return result

        # Cluster the solutions by similarity
        clustering_result = self.clustering.cluster(solutions)
        clustering_result = self.clustering.merge_similar_clusters(clustering_result)

        vote_counts = clustering_result.get_cluster_sizes()
        total_votes = sum(vote_counts.values())
        sorted_clusters = sorted(
            clustering_result.clusters,
            key=lambda c: c.size,
            reverse=True
        )

        result = VotingResult(
            vote_counts=vote_counts,
            total_votes=total_votes,
            round_number=self._round_number,
        )

        if not sorted_clusters:
            self._history.append(result)
            return result

        leader = sorted_clusters[0]
        runner_up_size = sorted_clusters[1].size if len(sorted_clusters) > 1 else 0

        margin = leader.size - runner_up_size
        result.margin = margin

        if margin >= self.k and leader.is_valid:
            result.winner = leader
            result.winning_solution = leader.representative
            result.consensus_reached = True
            logger.info(
                f"Consensus reached in round {self._round_number}: "
                f"{leader.cluster_id} leads by {margin} votes"
            )
        else:
            result.winner = leader
            # If leader is invalid, find the best valid cluster
            if leader.is_valid:
                result.winning_solution = leader.representative
            else:
                # Find first valid cluster
                for cluster in sorted_clusters:
                    if cluster.is_valid and cluster.representative:
                        result.winning_solution = cluster.representative
                        logger.info(
                            f"Leader invalid, using valid cluster {cluster.cluster_id}"
                        )
                        break
            result.consensus_reached = False

        self._history.append(result)
        return result

    def _has_behavioral_data(self, solutions: list[Solution]) -> bool:
        if not solutions:
            return False
        return all(getattr(s, "test_result", None) is not None for s in solutions)

    def _vote_behavioral(self, solutions: list[Solution]) -> VotingResult:
        from atlas.verification.behavioral_clustering import cluster_by_test_outcomes

        test_results = {
            solution.agent_id: solution.test_result
            for solution in solutions
            if solution.test_result is not None
        }

        clustering = cluster_by_test_outcomes(test_results)
        vote_counts = {c.cluster_id: c.size for c in clustering.clusters}
        total_votes = sum(vote_counts.values())

        result = VotingResult(
            vote_counts=vote_counts,
            total_votes=total_votes,
            round_number=self._round_number,
        )

        if not clustering.clusters:
            return result

        if clustering.passing_clusters:
            sorted_clusters = sorted(
                clustering.passing_clusters,
                key=lambda c: c.size,
                reverse=True,
            )
        else:
            sorted_clusters = sorted(
                clustering.clusters,
                key=lambda c: c.size,
                reverse=True,
            )

        leader = sorted_clusters[0]
        runner_up_size = sorted_clusters[1].size if len(sorted_clusters) > 1 else 0
        margin = leader.size - runner_up_size
        result.margin = margin
        result.winner = leader

        winning_solution = None
        if getattr(leader, "representative_patch_id", ""):
            winning_solution = next(
                (s for s in solutions if s.agent_id == leader.representative_patch_id),
                None,
            )
        if not winning_solution and getattr(leader, "patch_ids", None):
            winning_solution = next(
                (s for s in solutions if s.agent_id == leader.patch_ids[0]),
                None,
            )

        result.winning_solution = winning_solution

        has_passing = bool(clustering.passing_clusters)
        if margin >= self.k and (leader.all_pass or not has_passing):
            result.consensus_reached = True
            logger.info(
                f"Behavioral consensus reached in round {self._round_number}: "
                f"{leader.cluster_id} leads by {margin} votes"
            )

        return result

    def get_best_effort(self) -> Solution | None:
        """Get the best effort solution when no consensus is reached.

        Returns:
            The solution with the most votes, or None
        """
        if not self._history:
            return None

        # Get the latest round
        latest = self._history[-1]

        if latest.winning_solution:
            return latest.winning_solution

        return None

    def should_continue(self, result: VotingResult, max_samples: int) -> bool:
        """Determine if we should continue generating more samples.

        Args:
            result: The latest voting result
            max_samples: Maximum samples allowed

        Returns:
            True if we should generate more samples
        """
        if result.consensus_reached:
            return False

        if result.total_votes >= max_samples:
            return False

        # Continue if we're making progress
        # (margin is increasing or we have few samples)
        if len(self._history) >= 2:
            prev_margin = self._history[-2].margin
            if result.margin <= prev_margin and result.total_votes > self.k * 2:
                # Not making progress, might want to stop
                logger.info("Voting progress stalled, may want to stop")

        return True

    def get_voting_summary(self) -> dict[str, Any]:
        """Get a summary of all voting rounds.

        Returns:
            Dictionary with voting summary
        """
        return {
            "rounds": self._round_number,
            "consensus_reached": any(r.consensus_reached for r in self._history),
            "final_winner": (
                self._history[-1].winner.cluster_id
                if self._history and self._history[-1].winner
                else None
            ),
            "final_margin": self._history[-1].margin if self._history else 0,
            "history": [
                {
                    "round": r.round_number,
                    "votes": r.vote_counts,
                    "margin": r.margin,
                    "consensus": r.consensus_reached,
                }
                for r in self._history
            ],
        }

    def reset(self) -> None:
        """Reset the voting state for a new task."""
        self._round_number = 0
        self._history = []


class IncrementalVoter:
    """Voter that supports incremental voting with early stopping.

    This is useful when generating samples incrementally and
    checking for consensus after each batch.
    """

    def __init__(self, k: int = 3, similarity_threshold: float = 0.8):
        """Initialize the incremental voter.

        Args:
            k: Consensus threshold
            similarity_threshold: Clustering threshold
        """
        self.voting_manager = VotingManager(k, similarity_threshold)
        self._all_solutions: list[Solution] = []

    def add_solutions(self, solutions: list[Solution]) -> VotingResult:
        """Add new solutions and re-run voting.

        Args:
            solutions: New solutions to add

        Returns:
            Updated voting result
        """
        self._all_solutions.extend(solutions)
        return self.voting_manager.vote(self._all_solutions)

    def has_consensus(self) -> bool:
        """Check if consensus has been reached."""
        if not self.voting_manager._history:
            return False
        return self.voting_manager._history[-1].consensus_reached

    def get_winner(self) -> Solution | None:
        """Get the winning solution if consensus is reached."""
        if not self.has_consensus():
            return None
        return self.voting_manager._history[-1].winning_solution

    def get_best_effort(self) -> Solution | None:
        """Get the best effort solution."""
        return self.voting_manager.get_best_effort()

    @property
    def total_solutions(self) -> int:
        """Get the total number of solutions."""
        return len(self._all_solutions)

    def reset(self) -> None:
        """Reset for a new task."""
        self.voting_manager.reset()
        self._all_solutions = []
