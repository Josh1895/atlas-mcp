"""Tests for the voting and consensus system."""

import pytest

from atlas.core.task import Solution
from atlas.voting.consensus import IncrementalVoter, VotingManager


class TestVotingManager:
    """Tests for VotingManager."""

    def test_empty_solutions(self):
        """Test voting with no solutions."""
        manager = VotingManager(k=3)
        result = manager.vote([])

        assert not result.consensus_reached
        assert result.total_votes == 0
        assert result.winner is None

    def test_single_solution(self):
        """Test voting with a single solution."""
        manager = VotingManager(k=3)

        solution = Solution(
            agent_id="agent_0",
            prompt_style="minimal_diff",
            patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        )

        result = manager.vote([solution])

        assert not result.consensus_reached  # Need k=3 margin
        assert result.total_votes == 1

    def test_consensus_reached(self):
        """Test that consensus is reached when k margin is achieved."""
        manager = VotingManager(k=2)

        # Create 4 solutions: 3 identical, 1 different
        solutions = [
            Solution(
                agent_id=f"agent_{i}",
                prompt_style="minimal_diff",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            )
            for i in range(3)
        ]
        solutions.append(Solution(
            agent_id="agent_3",
            prompt_style="verbose",
            patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+different",
        ))

        result = manager.vote(solutions)

        assert result.consensus_reached
        assert result.margin >= 2

    def test_no_consensus(self):
        """Test that no consensus is reached when votes are split."""
        manager = VotingManager(k=3)

        # Create 4 solutions: 2 of each type
        solutions = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal_diff",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
            Solution(
                agent_id="agent_1",
                prompt_style="minimal_diff",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
            Solution(
                agent_id="agent_2",
                prompt_style="verbose",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+different",
            ),
            Solution(
                agent_id="agent_3",
                prompt_style="verbose",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+different",
            ),
        ]

        result = manager.vote(solutions)

        assert not result.consensus_reached
        assert result.margin == 0


class TestIncrementalVoter:
    """Tests for IncrementalVoter."""

    def test_incremental_voting(self):
        """Test adding solutions incrementally."""
        voter = IncrementalVoter(k=2)

        # Add first batch
        batch1 = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal_diff",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
        ]
        result1 = voter.add_solutions(batch1)
        assert not voter.has_consensus()

        # Add second batch with same patch
        batch2 = [
            Solution(
                agent_id="agent_1",
                prompt_style="verbose",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
            Solution(
                agent_id="agent_2",
                prompt_style="debugger",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
        ]
        result2 = voter.add_solutions(batch2)

        assert voter.has_consensus()
        assert voter.total_solutions == 3

    def test_get_winner(self):
        """Test getting the winning solution."""
        voter = IncrementalVoter(k=2)

        patch = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        solutions = [
            Solution(agent_id=f"agent_{i}", prompt_style="minimal_diff", patch=patch)
            for i in range(3)
        ]

        voter.add_solutions(solutions)

        winner = voter.get_winner()
        assert winner is not None
        assert winner.patch == patch

    def test_reset(self):
        """Test resetting the voter."""
        voter = IncrementalVoter(k=2)

        solutions = [
            Solution(
                agent_id="agent_0",
                prompt_style="minimal_diff",
                patch="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            ),
        ]
        voter.add_solutions(solutions)

        assert voter.total_solutions == 1

        voter.reset()

        assert voter.total_solutions == 0
        assert not voter.has_consensus()
