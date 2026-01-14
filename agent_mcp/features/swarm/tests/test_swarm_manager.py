"""
Unit tests for SwarmManager.

Tests orchestration workflow, error handling, and metrics collection.
"""

import pytest
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
    AgentOutput,
)
from ..swarm_manager import (
    SwarmManager,
    SwarmLogger,
    SwarmManagerError,
    SwarmDisabledError,
    run_swarm,
)
from ..budget_manager import BudgetExceededError


class TestSwarmLogger:
    """Tests for SwarmLogger class."""

    def test_init_without_run_id(self):
        """Test logger initialization without run_id."""
        log = SwarmLogger()
        assert log.run_id == ""

    def test_init_with_run_id(self):
        """Test logger initialization with run_id."""
        log = SwarmLogger(run_id="test_123")
        assert log.run_id == "test_123"

    def test_set_run_id(self):
        """Test setting run_id after initialization."""
        log = SwarmLogger()
        log.set_run_id("new_id")
        assert log.run_id == "new_id"

    def test_format_msg_with_run_id(self):
        """Test message formatting includes run_id."""
        log = SwarmLogger(run_id="run_abc")
        msg = log._format_msg("Test message")
        assert "[run=run_abc]" in msg
        assert "Test message" in msg

    def test_format_msg_without_run_id(self):
        """Test message formatting without run_id."""
        log = SwarmLogger()
        msg = log._format_msg("Test message")
        assert "[run=" not in msg
        assert "Test message" in msg

    def test_format_msg_with_extra_data(self):
        """Test message formatting with extra data."""
        log = SwarmLogger(run_id="run_xyz")
        msg = log._format_msg("Test", {"key": "value"})
        assert '"key": "value"' in msg


class TestSwarmManager:
    """Tests for SwarmManager class."""

    def test_init_default_request(self):
        """Test initialization with default request."""
        manager = SwarmManager()
        assert manager.request is not None
        assert manager._run_id == ""

    def test_init_custom_request(self):
        """Test initialization with custom request."""
        request = SwarmRequest(
            token="test_token",
            task_id="task_123",
            description="Test task",
            mode=SwarmMode.ANSWER,
        )
        manager = SwarmManager(request)

        assert manager.request.task_id == "task_123"
        assert manager.request.mode == SwarmMode.ANSWER

    def test_budget_manager_lazy_init(self):
        """Test budget manager is lazily initialized."""
        manager = SwarmManager()
        assert manager._budget_manager is None

        # Access property
        bm = manager.budget_manager
        assert bm is not None
        assert manager._budget_manager is bm

    def test_consensus_engine_lazy_init(self):
        """Test consensus engine is lazily initialized."""
        request = SwarmRequest(
            token="test",
            mode=SwarmMode.PATCH,
            swarm=SwarmConfig(consensus_k=3),
        )
        manager = SwarmManager(request)

        ce = manager.consensus_engine
        assert ce is not None
        assert ce.consensus_k == 3
        assert ce.mode == SwarmMode.PATCH

    @pytest.mark.asyncio
    async def test_run_swarm_disabled(self):
        """Test run raises error when swarm is disabled."""
        manager = SwarmManager()

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            with pytest.raises(SwarmDisabledError):
                await manager.run()

    @pytest.mark.asyncio
    async def test_run_generates_run_id(self):
        """Test run generates a unique run_id."""
        manager = SwarmManager(SwarmRequest(token="test"))

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch.object(manager, "_execute_workflow", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = SwarmResult(
                    run_id="",
                    mode=SwarmMode.PATCH,
                    status="completed",
                )
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            result = await manager.run()

        assert manager._run_id.startswith("swarm_")
        assert len(manager._run_id) > 6

    @pytest.mark.asyncio
    async def test_run_handles_timeout(self):
        """Test run handles timeout gracefully."""
        request = SwarmRequest(
            token="test",
            budgets=BudgetConfig(timeout_seconds=0.001),  # Very short timeout
        )
        manager = SwarmManager(request)

        async def slow_workflow(result):
            import asyncio
            await asyncio.sleep(1)  # Longer than timeout
            return result

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch.object(manager, "_execute_workflow", side_effect=slow_workflow):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            with patch.object(manager, "_finalize_best_effort", new_callable=AsyncMock) as mock_best:
                                mock_best.return_value = SwarmResult(
                                    run_id="",
                                    mode=SwarmMode.PATCH,
                                    status="timeout",
                                )
                                result = await manager.run()

        assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_run_handles_budget_exceeded(self):
        """Test run handles budget exceeded gracefully."""
        manager = SwarmManager(SwarmRequest(token="test"))

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch.object(manager, "_execute_workflow", side_effect=BudgetExceededError("Cost exceeded")):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            with patch.object(manager, "_finalize_best_effort", new_callable=AsyncMock) as mock_best:
                                mock_best.return_value = SwarmResult(
                                    run_id="",
                                    mode=SwarmMode.PATCH,
                                    status="budget_exceeded",
                                )
                                result = await manager.run()

        assert result.status == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_finalize_best_effort_with_outputs(self):
        """Test best effort finalization uses existing outputs."""
        manager = SwarmManager()
        manager._outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch", is_valid=True),
        ]

        result = SwarmResult(run_id="test", mode=SwarmMode.PATCH, status="timeout")

        with patch.object(manager.consensus_engine, "cluster_and_vote", new_callable=AsyncMock) as mock_vote:
            from ..consensus_engine import ConsensusResult
            mock_vote.return_value = ConsensusResult(
                consensus_reached=False,
                selected_output="patch",
                confidence_score=0.8,
            )

            result = await manager._finalize_best_effort(result)

        assert result.selected_output == "patch"
        assert result.confidence_score == 0.4  # 0.8 * 0.5
        assert result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_finalize_best_effort_without_outputs(self):
        """Test best effort finalization with no outputs."""
        manager = SwarmManager()
        manager._outputs = []

        result = SwarmResult(run_id="test", mode=SwarmMode.PATCH, status="timeout")
        result = await manager._finalize_best_effort(result)

        assert result.agent_outputs == []

    def test_collect_metrics(self):
        """Test metrics collection."""
        manager = SwarmManager()
        manager._outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="p1", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="p2", is_valid=False),
        ]

        metrics = manager._collect_metrics()

        assert metrics.agents_succeeded == 1
        assert metrics.agents_failed == 1

    def test_serialize_config(self):
        """Test configuration serialization."""
        request = SwarmRequest(
            token="test",
            task_id="task_123",
            description="Test description",
            mode=SwarmMode.ANSWER,
            budgets=BudgetConfig(max_cost_usd=10.0),
            swarm=SwarmConfig(agent_count=5),
        )
        manager = SwarmManager(request)

        import json
        config_json = manager._serialize_config()
        config = json.loads(config_json)

        assert config["task_id"] == "task_123"
        assert config["mode"] == "answer"
        assert config["budgets"]["max_cost_usd"] == 10.0
        assert config["swarm"]["agent_count"] == 5


class TestFallbackGeneration:
    """Tests for fallback generation when Atlas is unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_generate_returns_output(self):
        """Test fallback generation returns placeholder output."""
        manager = SwarmManager()

        outputs = await manager._fallback_generate("Test description")

        assert len(outputs) == 1
        assert outputs[0].agent_id == "fallback_0"
        assert outputs[0].is_valid is False
        assert "Fallback" in outputs[0].output_text

    @pytest.mark.asyncio
    async def test_fallback_used_when_atlas_unavailable(self):
        """Test fallback is used when Atlas adapter is None."""
        manager = SwarmManager()
        manager._atlas_adapter = None

        with patch.object(manager, "_fallback_generate", new_callable=AsyncMock) as mock_fallback:
            mock_fallback.return_value = [
                AgentOutput(agent_id="fb", prompt_style="default", output_text="fallback", is_valid=False),
            ]
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_agent", new_callable=AsyncMock):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_output", new_callable=AsyncMock):
                    outputs = await manager._generate_candidates("test", "", "")

        mock_fallback.assert_called_once()


class TestRunSwarmFunction:
    """Tests for run_swarm convenience function."""

    @pytest.mark.asyncio
    async def test_run_swarm_creates_manager(self):
        """Test run_swarm creates manager and runs."""
        request = SwarmRequest(token="test")

        with patch.object(SwarmManager, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = SwarmResult(
                run_id="test",
                mode=SwarmMode.PATCH,
                status="completed",
            )

            result = await run_swarm(request)

        assert result.status == "completed"
        mock_run.assert_called_once()
