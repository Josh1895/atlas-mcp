"""
Integration tests for end-to-end swarm workflow.

Tests the complete flow from SwarmRequest to SwarmResult,
verifying all components work together correctly.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ..schemas import (
    SwarmRequest,
    SwarmResult,
    SwarmMode,
    SwarmConfig,
    BudgetConfig,
    ToolConfig,
    MemoryConfig,
    RepoConfig,
    AgentOutput,
    ClusterInfo,
)
from ..swarm_manager import SwarmManager, run_swarm
from ..consensus_engine import ConsensusEngine, ConsensusResult
from ..budget_manager import BudgetManager
from ..tool_router import ToolRouter, create_default_router
from ..context_builder import ContextBuilder
from ..memory_writer import MemoryWriter
from ..metrics import get_metrics_collector, reset_metrics


class TestEndToEndSwarmWorkflow:
    """Integration tests for complete swarm workflow."""

    @pytest.fixture(autouse=True)
    def reset_metrics_fixture(self):
        """Reset metrics before each test."""
        reset_metrics()
        yield
        reset_metrics()

    @pytest.fixture
    def mock_atlas_adapter(self):
        """Create a mock Atlas adapter that returns predictable outputs."""
        adapter = MagicMock()
        adapter.generate_candidates = AsyncMock(return_value=[
            AgentOutput(
                agent_id="agent_0",
                prompt_style="direct",
                output_text="def fix():\n    return True",
                is_valid=True,
            ),
            AgentOutput(
                agent_id="agent_1",
                prompt_style="chain_of_thought",
                output_text="def fix():\n    return True",
                is_valid=True,
            ),
            AgentOutput(
                agent_id="agent_2",
                prompt_style="expert",
                output_text="def fix():\n    return True",
                is_valid=True,
            ),
        ])
        adapter.validate_and_score_patch_candidates = AsyncMock(return_value={
            "results": {
                "agent_0": {"passed": True, "patch_applied": True, "quality_score": 85.0},
                "agent_1": {"passed": True, "patch_applied": True, "quality_score": 80.0},
                "agent_2": {"passed": True, "patch_applied": True, "quality_score": 75.0},
            },
            "any_passed": True,
            "best_passing_agent_id": "agent_0",
        })
        return adapter

    @pytest.mark.asyncio
    async def test_full_workflow_patch_mode_consensus(self, mock_atlas_adapter):
        """Test complete workflow in patch mode reaching consensus."""
        request = SwarmRequest(
            token="test_token",
            task_id="task_001",
            description="Fix the bug in function X",
            mode=SwarmMode.PATCH,
            budgets=BudgetConfig(
                max_cost_usd=1.0,
                timeout_seconds=300,
            ),
            swarm=SwarmConfig(
                agent_count=3,
                consensus_k=2,
            ),
        )

        manager = SwarmManager(request)
        manager._atlas_adapter = mock_atlas_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        result = await manager.run()

        # Verify result structure
        assert isinstance(result, SwarmResult)
        assert result.run_id.startswith("swarm_")
        assert result.mode == SwarmMode.PATCH
        assert result.status == "completed"
        assert result.consensus_reached is True
        assert result.selected_output is not None
        assert len(result.agent_outputs) == 3

    @pytest.mark.asyncio
    async def test_full_workflow_answer_mode(self, mock_atlas_adapter):
        """Test complete workflow in answer mode."""
        # Update mock for answer mode
        mock_atlas_adapter.generate_candidates = AsyncMock(return_value=[
            AgentOutput(
                agent_id="agent_0",
                prompt_style="direct",
                output_text="The answer is 42.",
                is_valid=True,
            ),
            AgentOutput(
                agent_id="agent_1",
                prompt_style="chain_of_thought",
                output_text="The answer is 42.",
                is_valid=True,
            ),
        ])

        request = SwarmRequest(
            token="test_token",
            task_id="answer_001",
            description="What is the meaning of life?",
            mode=SwarmMode.ANSWER,
            swarm=SwarmConfig(agent_count=2, consensus_k=1),
        )

        manager = SwarmManager(request)
        manager._atlas_adapter = mock_atlas_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        result = await manager.run()

        assert result.mode == SwarmMode.ANSWER
        assert result.status == "completed"
        assert "42" in result.selected_output

    @pytest.mark.asyncio
    async def test_workflow_with_behavioral_clustering(self, mock_atlas_adapter):
        """Test workflow uses behavioral clustering for patch selection."""
        # Set up mixed passing/failing results
        mock_atlas_adapter.validate_and_score_patch_candidates = AsyncMock(return_value={
            "results": {
                "agent_0": {"passed": True, "patch_applied": True, "quality_score": 70.0},
                "agent_1": {"passed": False, "patch_applied": True, "failure_signature": "test_foo"},
                "agent_2": {"passed": False, "patch_applied": True, "failure_signature": "test_foo"},
            },
            "any_passed": True,
            "best_passing_agent_id": "agent_0",
        })

        request = SwarmRequest(
            token="test_token",
            task_id="behave_001",
            description="Fix test failures",
            mode=SwarmMode.PATCH,
            swarm=SwarmConfig(agent_count=3, consensus_k=1),
        )

        manager = SwarmManager(request)
        manager._atlas_adapter = mock_atlas_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        result = await manager.run()

        # Passing patch should be selected despite being minority
        assert result.selected_output == "def fix():\n    return True"

    @pytest.mark.asyncio
    async def test_workflow_budget_tracking(self, mock_atlas_adapter):
        """Test that budget is tracked throughout workflow."""
        request = SwarmRequest(
            token="test_token",
            task_id="budget_001",
            description="Test budget tracking",
            mode=SwarmMode.PATCH,
            budgets=BudgetConfig(max_cost_usd=0.50),
            swarm=SwarmConfig(agent_count=3),
        )

        manager = SwarmManager(request)
        manager._atlas_adapter = mock_atlas_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        result = await manager.run()

        # Verify budget manager was used
        assert manager.budget_manager is not None
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_workflow_metrics_collection(self, mock_atlas_adapter):
        """Test that metrics are collected during workflow."""
        request = SwarmRequest(
            token="test_token",
            task_id="metrics_001",
            description="Test metrics",
            mode=SwarmMode.PATCH,
        )

        manager = SwarmManager(request)
        manager._atlas_adapter = mock_atlas_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        await manager.run()

        # Check global metrics collector
        collector = get_metrics_collector()
        summary = collector.get_summary()

        assert summary["total_runs"] >= 1
        assert summary["completed_runs"] >= 1


class TestWorkflowErrorHandling:
    """Integration tests for error handling in workflow."""

    @pytest.fixture(autouse=True)
    def reset_metrics_fixture(self):
        """Reset metrics before each test."""
        reset_metrics()
        yield
        reset_metrics()

    @pytest.mark.asyncio
    async def test_workflow_handles_generation_failure(self):
        """Test workflow handles generation failures gracefully."""
        request = SwarmRequest(
            token="test_token",
            task_id="error_001",
            description="Test error handling",
            mode=SwarmMode.PATCH,
        )

        manager = SwarmManager(request)

        # Mock adapter that fails
        failing_adapter = MagicMock()
        failing_adapter.generate_candidates = AsyncMock(side_effect=Exception("Generation failed"))
        manager._atlas_adapter = failing_adapter

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        with patch.object(manager, "_fallback_generate", new_callable=AsyncMock) as mock_fallback:
                            mock_fallback.return_value = [
                                AgentOutput(
                                    agent_id="fallback_0",
                                    prompt_style="default",
                                    output_text="fallback",
                                    is_valid=False,
                                )
                            ]
                            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_agent", new_callable=AsyncMock):
                                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_output", new_callable=AsyncMock):
                                    result = await manager.run()

        # Should still complete with fallback
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_workflow_handles_consensus_failure(self):
        """Test workflow handles consensus engine failures."""
        request = SwarmRequest(
            token="test_token",
            task_id="consensus_error_001",
            description="Test consensus error",
            mode=SwarmMode.PATCH,
        )

        manager = SwarmManager(request)
        manager._outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch", is_valid=True),
        ]

        # Mock consensus engine that fails
        with patch.object(manager.consensus_engine, "cluster_and_vote", side_effect=Exception("Consensus failed")):
            with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            with patch.object(manager, "_generate_candidates", new_callable=AsyncMock) as mock_gen:
                                mock_gen.return_value = manager._outputs
                                result = await manager.run()

        # Should handle gracefully
        assert result is not None


class TestComponentIntegration:
    """Tests for component interactions."""

    def test_budget_manager_consensus_engine_integration(self):
        """Test budget manager integrates with consensus engine."""
        request = SwarmRequest(
            token="test",
            budgets=BudgetConfig(max_cost_usd=1.0),
            swarm=SwarmConfig(consensus_k=3),
        )

        manager = SwarmManager(request)

        # Both should be accessible and configured
        assert manager.budget_manager is not None
        assert manager.consensus_engine is not None
        assert manager.consensus_engine.consensus_k == 3

    def test_tool_router_creation(self):
        """Test tool router is created with correct config."""
        request = SwarmRequest(
            token="test",
            tools=ToolConfig(
                allow_code_execution=False,
                allow_web_search=True,
            ),
        )

        manager = SwarmManager(request)
        router = manager.tool_router

        assert router is not None
        # Router should respect tool config
        assert not router.is_tool_allowed("code_execution")

    def test_memory_writer_creation(self):
        """Test memory writer is created."""
        request = SwarmRequest(
            token="test",
            memory=MemoryConfig(
                write_to_memory=True,
                memory_tags=["test", "integration"],
            ),
        )

        manager = SwarmManager(request)

        assert manager.memory_writer is not None


class TestConcurrencyBehavior:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_parallel_agent_generation(self):
        """Test that agents can be generated in parallel."""
        request = SwarmRequest(
            token="test",
            swarm=SwarmConfig(agent_count=5),
        )

        manager = SwarmManager(request)

        # Track call times
        call_times = []

        async def mock_generate(*args, **kwargs):
            call_times.append(datetime.utcnow())
            await asyncio.sleep(0.01)
            return [AgentOutput(
                agent_id=f"agent_{len(call_times)}",
                prompt_style="default",
                output_text="output",
                is_valid=True,
            )]

        with patch.object(manager, "_atlas_adapter") as mock_adapter:
            mock_adapter.generate_candidates = mock_generate

            with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_agent", new_callable=AsyncMock):
                                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_output", new_callable=AsyncMock):
                                    await manager.run()


class TestRunSwarmConvenienceFunction:
    """Tests for the run_swarm convenience function."""

    @pytest.fixture(autouse=True)
    def reset_metrics_fixture(self):
        """Reset metrics before each test."""
        reset_metrics()
        yield
        reset_metrics()

    @pytest.mark.asyncio
    async def test_run_swarm_function_creates_manager_and_runs(self):
        """Test run_swarm creates manager and executes."""
        request = SwarmRequest(
            token="test_token",
            task_id="func_001",
            description="Test convenience function",
            mode=SwarmMode.PATCH,
        )

        mock_result = SwarmResult(
            run_id="swarm_test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=True,
            selected_output="test output",
        )

        with patch.object(SwarmManager, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            result = await run_swarm(request)

        assert result.status == "completed"
        assert result.run_id == "swarm_test"
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_swarm_function_passes_request_correctly(self):
        """Test run_swarm passes request to manager."""
        request = SwarmRequest(
            token="special_token",
            task_id="special_task",
            description="Special description",
            mode=SwarmMode.ANSWER,
            swarm=SwarmConfig(agent_count=7, consensus_k=4),
        )

        captured_request = None

        original_init = SwarmManager.__init__

        def capture_init(self, req=None):
            nonlocal captured_request
            captured_request = req
            original_init(self, req)

        with patch.object(SwarmManager, "__init__", capture_init):
            with patch.object(SwarmManager, "run", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = SwarmResult(
                    run_id="test",
                    mode=SwarmMode.ANSWER,
                    status="completed",
                )

                await run_swarm(request)

        assert captured_request is not None
        assert captured_request.task_id == "special_task"
        assert captured_request.swarm.agent_count == 7
