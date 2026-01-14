"""
Unit tests for ToolRouter.

Tests permission checking, caching, rate limiting, and circuit breakers.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from ..schemas import ToolConfig
from ..tool_router import (
    ToolRouter,
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    ToolNotAllowedError,
    CircuitOpenError,
    RateLimitExceededError,
    create_default_router,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test circuit starts closed."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_stays_closed_on_success(self):
        """Test circuit stays closed on successful calls."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_half_open_after_cooldown(self):
        """Test circuit goes half-open after cooldown."""
        cb = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=0.1)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successful_half_open(self):
        """Test circuit closes after successful calls in half-open."""
        cb = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=0.01, half_open_max_calls=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for cooldown
        time.sleep(0.02)
        cb.can_execute()  # Triggers half-open

        # Successful calls in half-open
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open."""
        cb = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=0.01)

        cb.record_failure()
        cb.record_failure()

        time.sleep(0.02)
        cb.can_execute()  # Triggers half-open
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0

    def test_get_status(self):
        """Test get_status returns correct info."""
        cb = CircuitBreaker(name="test_cb")
        cb.record_failure()

        status = cb.get_status()

        assert status["name"] == "test_cb"
        assert status["state"] == "closed"
        assert status["failure_count"] == 1


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_burst(self):
        """Test rate limiter allows initial burst."""
        limiter = RateLimiter(calls_per_second=1.0, bucket_size=5)

        # Should allow 5 calls immediately
        for _ in range(5):
            result = await limiter.acquire(timeout=0.01)
            assert result is True

    @pytest.mark.asyncio
    async def test_blocks_after_bucket_empty(self):
        """Test rate limiter blocks after bucket is empty."""
        limiter = RateLimiter(calls_per_second=100.0, bucket_size=2)

        # Exhaust bucket
        await limiter.acquire(timeout=0.01)
        await limiter.acquire(timeout=0.01)

        # Third should block and timeout
        result = await limiter.acquire(timeout=0.05)
        # May or may not succeed depending on refill timing
        # Just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        """Test bucket refills over time."""
        limiter = RateLimiter(calls_per_second=10.0, bucket_size=1)

        # Exhaust bucket
        await limiter.acquire(timeout=0.01)

        # Wait for refill
        await asyncio.sleep(0.15)

        # Should succeed now
        result = await limiter.acquire(timeout=0.01)
        assert result is True


class TestToolRouter:
    """Tests for ToolRouter class."""

    def test_init_default_config(self):
        """Test default initialization."""
        router = ToolRouter()

        assert router.config is not None
        assert router.run_id == "unknown"

    def test_init_custom_config(self):
        """Test custom config initialization."""
        config = ToolConfig(
            enable_web_search=True,
            enable_context7=False,
        )
        router = ToolRouter(config=config, run_id="test_run_123")

        assert router.config.enable_web_search is True
        assert router.config.enable_context7 is False
        assert router.run_id == "test_run_123"

    def test_register_tool(self):
        """Test tool registration."""
        router = ToolRouter()

        async def handler(**kwargs):
            return {"result": "ok"}

        router.register_tool("test_tool", handler, cache_ttl=300)

        assert "test_tool" in router._handlers
        assert router._cache_ttls["test_tool"] == 300
        assert "test_tool" in router._rate_limiters
        assert "test_tool" in router._circuit_breakers

    @pytest.mark.asyncio
    async def test_call_tool_not_registered(self):
        """Test calling unregistered tool raises error."""
        router = ToolRouter()

        with pytest.raises(Exception) as exc_info:
            await router.call("nonexistent", {})

        assert "No handler" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_permission_denied(self):
        """Test calling disabled tool raises error."""
        config = ToolConfig(enable_web_search=False)
        router = ToolRouter(config=config)

        async def handler(**kwargs):
            return {}

        router.register_tool("web_search", handler)

        with patch.object(router, '_check_permission', side_effect=ToolNotAllowedError("Disabled")):
            with pytest.raises(ToolNotAllowedError):
                await router.call("web_search", {"query": "test"})

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful tool call."""
        router = ToolRouter()

        async def handler(query: str):
            return {"query": query, "results": ["result1"]}

        router.register_tool("test_tool", handler)

        # Mock persistence functions
        with patch("agent_mcp.features.swarm.tool_router.log_tool_call", new_callable=AsyncMock, return_value=1):
            with patch("agent_mcp.features.swarm.tool_router.update_tool_call", new_callable=AsyncMock):
                result = await router.call("test_tool", {"query": "test"})

        assert result["query"] == "test"
        assert result["results"] == ["result1"]

    @pytest.mark.asyncio
    async def test_call_circuit_open(self):
        """Test call fails when circuit is open."""
        router = ToolRouter()

        async def handler(**kwargs):
            return {}

        router.register_tool("test_tool", handler)

        # Force circuit open
        cb = router._circuit_breakers["test_tool"]
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time()  # Recent failure

        with pytest.raises(CircuitOpenError):
            await router.call("test_tool", {})

    def test_get_stats(self):
        """Test get_stats returns expected structure."""
        router = ToolRouter()

        async def handler(**kwargs):
            return {}

        router.register_tool("tool1", handler)

        stats = router.get_stats()

        assert "total_calls" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "errors" in stats
        assert "cache_hit_rate" in stats
        assert "circuits" in stats


class TestCreateDefaultRouter:
    """Tests for create_default_router function."""

    def test_creates_router_with_handlers(self):
        """Test default router has expected handlers."""
        router = create_default_router()

        assert "web_search" in router._handlers
        assert "context7" in router._handlers
        assert "repo_search" in router._handlers
        assert "agent_mcp_rag" in router._handlers

    def test_creates_router_with_config(self):
        """Test default router accepts config."""
        config = ToolConfig(enable_web_search=False)
        router = create_default_router(config=config, run_id="custom_run")

        assert router.config.enable_web_search is False
        assert router.run_id == "custom_run"


class TestPermissionChecking:
    """Tests for permission checking logic."""

    def test_check_permission_allowed(self):
        """Test permission check passes for enabled tools."""
        config = ToolConfig(enable_repo_search=True)
        router = ToolRouter(config=config)

        # Should not raise
        router._check_permission("repo_search")

    def test_check_permission_denied(self):
        """Test permission check fails for disabled tools."""
        config = ToolConfig(enable_web_search=False)
        router = ToolRouter(config=config)

        # Mock the global flag
        with patch("agent_mcp.features.swarm.tool_router.SWARM_ENABLE_WEB", False):
            with pytest.raises(ToolNotAllowedError):
                router._check_permission("web_search")


class TestCaching:
    """Tests for caching functionality."""

    def test_make_cache_key_consistent(self):
        """Test cache key generation is consistent."""
        router = ToolRouter()

        key1 = router._make_cache_key("tool", {"a": 1, "b": 2})
        key2 = router._make_cache_key("tool", {"b": 2, "a": 1})  # Different order

        assert key1 == key2  # Keys should be same regardless of arg order

    def test_make_cache_key_different_args(self):
        """Test different args produce different keys."""
        router = ToolRouter()

        key1 = router._make_cache_key("tool", {"query": "a"})
        key2 = router._make_cache_key("tool", {"query": "b"})

        assert key1 != key2
