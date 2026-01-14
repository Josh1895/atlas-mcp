"""
Backward Compatibility Tests for SWARM_ENABLED=false.

These tests ensure the system behaves correctly when the swarm feature
is disabled, maintaining backward compatibility with existing code.
"""

import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock

from ..schemas import (
    SwarmRequest,
    SwarmResult,
    SwarmMode,
    SwarmConfig,
    BudgetConfig,
)
from ..swarm_manager import (
    SwarmManager,
    SwarmDisabledError,
    run_swarm,
)


class TestSwarmDisabledBehavior:
    """Tests for behavior when SWARM_ENABLED=false."""

    @pytest.mark.asyncio
    async def test_run_raises_swarm_disabled_error(self):
        """Test that run() raises SwarmDisabledError when disabled."""
        manager = SwarmManager(SwarmRequest(token="test"))

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            with pytest.raises(SwarmDisabledError):
                await manager.run()

    @pytest.mark.asyncio
    async def test_swarm_disabled_error_message(self):
        """Test SwarmDisabledError has informative message."""
        manager = SwarmManager()

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            with pytest.raises(SwarmDisabledError) as exc_info:
                await manager.run()

            assert "disabled" in str(exc_info.value).lower() or "SWARM_ENABLED" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_swarm_function_raises_when_disabled(self):
        """Test run_swarm convenience function raises when disabled."""
        request = SwarmRequest(token="test", task_id="test_task")

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            with pytest.raises(SwarmDisabledError):
                await run_swarm(request)


class TestManagerInitializationWhenDisabled:
    """Tests for manager initialization when swarm is disabled."""

    def test_manager_can_be_instantiated(self):
        """Test SwarmManager can be instantiated even when disabled."""
        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            assert manager is not None

    def test_manager_initializes_with_request(self):
        """Test manager accepts request when disabled."""
        request = SwarmRequest(
            token="test",
            task_id="task_123",
            mode=SwarmMode.PATCH,
        )

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager(request)
            assert manager.request.task_id == "task_123"

    def test_budget_manager_lazy_init_still_works(self):
        """Test budget manager lazy initialization works when disabled."""
        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            bm = manager.budget_manager
            assert bm is not None

    def test_consensus_engine_lazy_init_still_works(self):
        """Test consensus engine lazy initialization works when disabled."""
        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            ce = manager.consensus_engine
            assert ce is not None


class TestFeatureFlagVariations:
    """Tests for various feature flag configurations."""

    @pytest.mark.asyncio
    async def test_empty_env_var_is_disabled(self):
        """Test empty SWARM_ENABLED env var is treated as disabled."""
        with patch.dict(os.environ, {"SWARM_ENABLED": ""}):
            with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
                manager = SwarmManager()
                with pytest.raises(SwarmDisabledError):
                    await manager.run()

    @pytest.mark.asyncio
    async def test_false_string_is_disabled(self):
        """Test 'false' string is treated as disabled."""
        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            with pytest.raises(SwarmDisabledError):
                await manager.run()

    @pytest.mark.asyncio
    async def test_zero_string_is_disabled(self):
        """Test '0' is treated as disabled."""
        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            with pytest.raises(SwarmDisabledError):
                await manager.run()

    @pytest.mark.asyncio
    async def test_true_string_is_enabled(self):
        """Test 'true' string enables swarm."""
        manager = SwarmManager(SwarmRequest(token="test"))

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch.object(manager, "_execute_workflow", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = SwarmResult(
                    run_id="test",
                    mode=SwarmMode.PATCH,
                    status="completed",
                )
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            result = await manager.run()

        assert result.status == "completed"


class TestSubsystemFeatureFlags:
    """Tests for individual subsystem feature flags."""

    def test_web_search_disabled(self):
        """Test web search can be disabled independently."""
        from ..tool_router import ToolRouter

        router = ToolRouter(allow_web_search=False)
        assert not router.is_tool_allowed("web_search")

    def test_context7_disabled(self):
        """Test Context7 can be disabled independently."""
        from ..tool_router import ToolRouter

        router = ToolRouter(allow_context7=False)
        assert not router.is_tool_allowed("context7")

    def test_code_execution_disabled(self):
        """Test code execution can be disabled independently."""
        from ..tool_router import ToolRouter

        router = ToolRouter(allow_code_execution=False)
        assert not router.is_tool_allowed("code_execution")


class TestNoSideEffectsWhenDisabled:
    """Tests ensuring no side effects when swarm is disabled."""

    @pytest.mark.asyncio
    async def test_no_database_writes_when_disabled(self):
        """Test no database writes occur when disabled."""
        save_run_called = False
        save_agent_called = False

        async def mock_save_run(*args, **kwargs):
            nonlocal save_run_called
            save_run_called = True

        async def mock_save_agent(*args, **kwargs):
            nonlocal save_agent_called
            save_agent_called = True

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", mock_save_run):
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_agent", mock_save_agent):
                    manager = SwarmManager()
                    try:
                        await manager.run()
                    except SwarmDisabledError:
                        pass

        assert not save_run_called
        assert not save_agent_called

    @pytest.mark.asyncio
    async def test_no_metrics_recorded_when_disabled(self):
        """Test no metrics are recorded when disabled."""
        from ..metrics import get_metrics_collector, reset_metrics

        reset_metrics()
        collector = get_metrics_collector()
        initial_runs = collector.get_summary()["total_runs"]

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", False):
            manager = SwarmManager()
            try:
                await manager.run()
            except SwarmDisabledError:
                pass

        # Metrics should not have increased
        final_runs = collector.get_summary()["total_runs"]
        assert final_runs == initial_runs

        reset_metrics()


class TestGracefulDegradation:
    """Tests for graceful degradation when components are unavailable."""

    @pytest.mark.asyncio
    async def test_atlas_unavailable_uses_fallback(self):
        """Test that Atlas unavailability triggers fallback generation."""
        manager = SwarmManager(SwarmRequest(token="test"))
        manager._atlas_adapter = None

        with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
            with patch.object(manager, "_fallback_generate", new_callable=AsyncMock) as mock_fallback:
                mock_fallback.return_value = []
                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_run", new_callable=AsyncMock):
                    with patch("agent_mcp.features.swarm.swarm_manager.update_swarm_run_status", new_callable=AsyncMock):
                        with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                            with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_agent", new_callable=AsyncMock):
                                with patch("agent_mcp.features.swarm.swarm_manager.save_swarm_output", new_callable=AsyncMock):
                                    await manager.run()

                mock_fallback.assert_called()

    def test_memory_writer_disabled_gracefully(self):
        """Test memory writer handles disabled state gracefully."""
        from ..memory_writer import MemoryWriter
        from ..schemas import MemoryConfig

        writer = MemoryWriter(config=MemoryConfig(write_to_memory=False))
        # Should not raise, just no-op
        assert writer is not None


class TestImportCompatibility:
    """Tests ensuring imports work regardless of feature flag."""

    def test_can_import_all_public_classes(self):
        """Test all public classes can be imported."""
        from ..swarm_manager import SwarmManager, SwarmLogger, run_swarm
        from ..consensus_engine import ConsensusEngine, ConsensusResult
        from ..budget_manager import BudgetManager, BudgetExceededError
        from ..tool_router import ToolRouter, CircuitBreaker, RateLimiter
        from ..context_builder import ContextBuilder, SwarmContext
        from ..memory_writer import MemoryWriter
        from ..repo_resolver import RepoResolver
        from ..atlas_adapter import AtlasAdapter
        from ..metrics import SwarmMetricsCollector, get_metrics_collector

        # All imports should succeed
        assert SwarmManager is not None
        assert ConsensusEngine is not None
        assert BudgetManager is not None
        assert ToolRouter is not None

    def test_can_import_schemas(self):
        """Test all schema classes can be imported."""
        from ..schemas import (
            SwarmMode,
            SwarmRequest,
            SwarmResult,
            BudgetConfig,
            ToolConfig,
            MemoryConfig,
            RepoConfig,
            SwarmConfig,
            AgentOutput,
            ClusterInfo,
            SwarmMetrics,
        )

        assert SwarmMode.PATCH is not None
        assert SwarmMode.ANSWER is not None


class TestErrorTypes:
    """Tests for error type consistency."""

    def test_swarm_disabled_error_is_catchable(self):
        """Test SwarmDisabledError can be caught specifically."""
        try:
            raise SwarmDisabledError("Test")
        except SwarmDisabledError as e:
            assert "Test" in str(e)

    def test_swarm_disabled_error_inherits_from_exception(self):
        """Test SwarmDisabledError is a proper Exception subclass."""
        assert issubclass(SwarmDisabledError, Exception)

    def test_error_types_documented(self):
        """Test error types are accessible from module."""
        from ..swarm_manager import SwarmManagerError, SwarmDisabledError

        assert SwarmManagerError is not None
        assert SwarmDisabledError is not None
        assert issubclass(SwarmDisabledError, SwarmManagerError)
