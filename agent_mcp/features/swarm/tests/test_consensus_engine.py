"""
Unit tests for ConsensusEngine.

Tests clustering, voting, and behavioral clustering functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from ..schemas import AgentOutput, ClusterInfo, SwarmMode
from ..consensus_engine import ConsensusEngine, ConsensusResult, build_consensus


class TestConsensusEngine:
    """Tests for ConsensusEngine class."""

    def test_init_default_values(self):
        """Test default initialization."""
        engine = ConsensusEngine()
        assert engine.consensus_k == 2
        assert engine.mode == SwarmMode.PATCH

    def test_init_custom_values(self):
        """Test custom initialization."""
        engine = ConsensusEngine(consensus_k=3, mode=SwarmMode.ANSWER)
        assert engine.consensus_k == 3
        assert engine.mode == SwarmMode.ANSWER

    @pytest.mark.asyncio
    async def test_cluster_and_vote_empty_outputs(self):
        """Test with empty outputs returns empty result."""
        engine = ConsensusEngine()
        result = await engine.cluster_and_vote([])

        assert result.consensus_reached is False
        assert result.selected_output == ""
        assert result.clusters == []

    @pytest.mark.asyncio
    async def test_cluster_and_vote_single_output(self):
        """Test with single output reaches consensus."""
        engine = ConsensusEngine()
        outputs = [
            AgentOutput(
                agent_id="agent_1",
                prompt_style="default",
                output_text="fix: update function",
                is_valid=True,
            )
        ]

        result = await engine.cluster_and_vote(outputs)

        assert result.consensus_reached is True
        assert result.confidence_score == 1.0
        assert result.selected_output == "fix: update function"
        assert len(result.clusters) == 1

    @pytest.mark.asyncio
    async def test_cluster_and_vote_identical_outputs(self):
        """Test with identical outputs all go to same cluster."""
        engine = ConsensusEngine()
        outputs = [
            AgentOutput(agent_id="agent_1", prompt_style="style1", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="agent_2", prompt_style="style2", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="agent_3", prompt_style="style3", output_text="patch A", is_valid=True),
        ]

        result = await engine.cluster_and_vote(outputs)

        assert result.consensus_reached is True
        assert len(result.clusters) == 1
        assert result.clusters[0].size == 3

    @pytest.mark.asyncio
    async def test_cluster_and_vote_diverse_outputs(self):
        """Test with diverse outputs creates multiple clusters."""
        engine = ConsensusEngine(consensus_k=2)
        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a3", prompt_style="s3", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a4", prompt_style="s4", output_text="patch B different", is_valid=True),
        ]

        result = await engine.cluster_and_vote(outputs)

        # Patch A has 3 votes, patch B has 1 vote, margin = 2 >= k=2
        assert result.consensus_reached is True
        assert result.selected_output == "patch A"

    @pytest.mark.asyncio
    async def test_cluster_and_vote_no_consensus(self):
        """Test when margin < k, no consensus."""
        engine = ConsensusEngine(consensus_k=3)
        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a3", prompt_style="s3", output_text="patch B different", is_valid=True),
        ]

        result = await engine.cluster_and_vote(outputs)

        # Margin = 2-1 = 1 < k=3, no consensus
        assert result.consensus_reached is False
        # Still selects the leader
        assert result.selected_output == "patch A"


class TestBehavioralClustering:
    """Tests for behavioral clustering by test outcomes."""

    @pytest.mark.asyncio
    async def test_behavioral_clustering_passing_preferred(self):
        """Test that passing patches are preferred over failing."""
        engine = ConsensusEngine()

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="passing patch", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="failing patch", is_valid=True),
            AgentOutput(agent_id="a3", prompt_style="s3", output_text="failing patch", is_valid=True),
        ]

        behavioral_results = {
            "a1": {"passed": True, "patch_applied": True},
            "a2": {"passed": False, "patch_applied": True, "failure_signature": "test_foo"},
            "a3": {"passed": False, "patch_applied": True, "failure_signature": "test_foo"},
        }

        result = await engine.cluster_and_vote(outputs, behavioral_results=behavioral_results)

        # Passing cluster should win even with only 1 vote
        assert result.selected_output == "passing patch"
        assert any(c.behavioral_pass for c in result.clusters)

    @pytest.mark.asyncio
    async def test_behavioral_clustering_groups_by_failure(self):
        """Test that failures are grouped by signature."""
        engine = ConsensusEngine()

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch A", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch B", is_valid=True),
            AgentOutput(agent_id="a3", prompt_style="s3", output_text="patch C", is_valid=True),
        ]

        behavioral_results = {
            "a1": {"passed": False, "patch_applied": True, "failure_signature": "test_foo,test_bar"},
            "a2": {"passed": False, "patch_applied": True, "failure_signature": "test_foo,test_bar"},
            "a3": {"passed": False, "patch_applied": True, "failure_signature": "test_baz"},
        }

        result = await engine.cluster_and_vote(outputs, behavioral_results=behavioral_results)

        # Should have 2 clusters: one for foo,bar failures, one for baz
        fail_clusters = [c for c in result.clusters if not c.behavioral_pass]
        assert len(fail_clusters) == 2

    @pytest.mark.asyncio
    async def test_behavioral_clustering_apply_failures(self):
        """Test that apply failures are grouped separately."""
        engine = ConsensusEngine()

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="good patch", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="bad patch", is_valid=True),
        ]

        behavioral_results = {
            "a1": {"passed": True, "patch_applied": True},
            "a2": {"passed": False, "patch_applied": False},
        }

        result = await engine.cluster_and_vote(outputs, behavioral_results=behavioral_results)

        # Passing should win
        assert result.selected_output == "good patch"

        # Should have apply_failed cluster
        apply_failed_clusters = [c for c in result.clusters if c.cluster_id == "apply_failed"]
        assert len(apply_failed_clusters) == 1


class TestAnswerModeClustering:
    """Tests for answer mode with embeddings."""

    @pytest.mark.asyncio
    async def test_answer_mode_single_output(self):
        """Test answer mode with single output."""
        engine = ConsensusEngine(mode=SwarmMode.ANSWER)

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="Answer text", is_valid=True),
        ]

        result = await engine.cluster_and_vote(outputs)

        assert result.consensus_reached is True
        assert result.selected_output == "Answer text"

    @pytest.mark.asyncio
    async def test_answer_mode_fallback_without_embeddings(self):
        """Test answer mode falls back to text hash without embeddings."""
        engine = ConsensusEngine(mode=SwarmMode.ANSWER)

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="Same answer", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="Same answer", is_valid=True),
            AgentOutput(agent_id="a3", prompt_style="s3", output_text="Different answer", is_valid=True),
        ]

        result = await engine.cluster_and_vote(outputs, embeddings=None)

        # Without embeddings, should cluster by text hash
        assert len(result.clusters) == 2


class TestBuildConsensus:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_build_consensus_function(self):
        """Test build_consensus convenience function."""
        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch", is_valid=True),
        ]

        result = await build_consensus(
            outputs=outputs,
            mode=SwarmMode.PATCH,
            consensus_k=2,
        )

        assert isinstance(result, ConsensusResult)
        assert result.consensus_reached is True


class TestSelectBestRepresentative:
    """Tests for representative selection."""

    def test_select_best_representative_by_quality(self):
        """Test that higher quality patches are preferred."""
        engine = ConsensusEngine()

        members = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch 1", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch 2", is_valid=True),
        ]

        behavioral_results = {
            "a1": {"quality_score": 60.0, "risk_flags": []},
            "a2": {"quality_score": 80.0, "risk_flags": []},
        }

        best = engine._select_best_representative(members, behavioral_results)

        assert best.agent_id == "a2"  # Higher quality

    def test_select_best_representative_penalizes_risk(self):
        """Test that risk flags reduce score."""
        engine = ConsensusEngine()

        members = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch 1", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch 2", is_valid=True),
        ]

        behavioral_results = {
            "a1": {"quality_score": 70.0, "risk_flags": []},
            "a2": {"quality_score": 80.0, "risk_flags": ["security_concern", "complexity"]},
        }

        best = engine._select_best_representative(members, behavioral_results)

        # a2 has higher quality but 2 risk flags (-10), so a1 wins
        assert best.agent_id == "a1"
