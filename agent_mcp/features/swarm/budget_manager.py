"""
Budget Manager for swarm operations.

Enforces hard limits on time, cost, tokens, and tool calls (P-005: Budget Enforcement).
Budget overruns trigger graceful best-effort results, never runaway costs.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from ...core.config import (
    logger,
    SWARM_DEFAULT_TIMEOUT_SECONDS,
    SWARM_DEFAULT_MAX_COST_USD,
    SWARM_DEFAULT_MAX_TOKENS,
    SWARM_DEFAULT_MAX_TOOL_CALLS,
)
from .schemas import BudgetConfig


@dataclass
class BudgetState:
    """Current state of budget consumption."""
    cost_usd: float = 0.0
    tokens_used: int = 0
    tool_calls: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(self, limit_type: str, current: float, limit: float):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(f"Budget exceeded: {limit_type} ({current:.2f} >= {limit:.2f})")


class BudgetManager:
    """
    Manages budget limits for a swarm run.

    Tracks cost, tokens, tool calls, and elapsed time.
    Provides methods to check remaining budget and record consumption.

    Usage:
        budget = BudgetManager(config)

        # Before expensive operation
        if budget.can_spend_cost(0.10):
            # do operation
            budget.record_cost(0.10)

        # Or check all limits at once
        budget.check_budget()  # Raises BudgetExceededError if exceeded
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        """
        Initialize budget manager with limits.

        Args:
            config: Budget configuration. Uses defaults if not provided.
        """
        self.config = config or BudgetConfig(
            timeout_seconds=SWARM_DEFAULT_TIMEOUT_SECONDS,
            max_cost_usd=SWARM_DEFAULT_MAX_COST_USD,
            max_tokens=SWARM_DEFAULT_MAX_TOKENS,
            max_tool_calls=SWARM_DEFAULT_MAX_TOOL_CALLS,
        )
        self.state = BudgetState()
        self._lock = asyncio.Lock()

        logger.debug(
            f"BudgetManager initialized: timeout={self.config.timeout_seconds}s, "
            f"max_cost=${self.config.max_cost_usd}, max_tokens={self.config.max_tokens}, "
            f"max_tool_calls={self.config.max_tool_calls}"
        )

    @property
    def is_exceeded(self) -> bool:
        """Check if any budget limit is exceeded."""
        return (
            self.is_timeout_exceeded or
            self.is_cost_exceeded or
            self.is_tokens_exceeded or
            self.is_tool_calls_exceeded
        )

    @property
    def is_timeout_exceeded(self) -> bool:
        return self.state.elapsed_seconds >= self.config.timeout_seconds

    @property
    def is_cost_exceeded(self) -> bool:
        return self.state.cost_usd >= self.config.max_cost_usd

    @property
    def is_tokens_exceeded(self) -> bool:
        return self.state.tokens_used >= self.config.max_tokens

    @property
    def is_tool_calls_exceeded(self) -> bool:
        return self.state.tool_calls >= self.config.max_tool_calls

    def get_exceeded_type(self) -> Optional[str]:
        """Return the type of limit that was exceeded, or None."""
        if self.is_timeout_exceeded:
            return "timeout"
        if self.is_cost_exceeded:
            return "cost"
        if self.is_tokens_exceeded:
            return "tokens"
        if self.is_tool_calls_exceeded:
            return "tool_calls"
        return None

    def check_budget(self) -> None:
        """
        Check all budget limits and raise if exceeded.

        Raises:
            BudgetExceededError: If any limit is exceeded
        """
        if self.is_timeout_exceeded:
            raise BudgetExceededError(
                "timeout",
                self.state.elapsed_seconds,
                self.config.timeout_seconds
            )
        if self.is_cost_exceeded:
            raise BudgetExceededError(
                "cost",
                self.state.cost_usd,
                self.config.max_cost_usd
            )
        if self.is_tokens_exceeded:
            raise BudgetExceededError(
                "tokens",
                self.state.tokens_used,
                self.config.max_tokens
            )
        if self.is_tool_calls_exceeded:
            raise BudgetExceededError(
                "tool_calls",
                self.state.tool_calls,
                self.config.max_tool_calls
            )

    def get_remaining(self) -> dict:
        """
        Get remaining budget for each limit type.

        Returns:
            Dictionary with remaining values for each limit
        """
        return {
            "timeout_seconds": max(0, self.config.timeout_seconds - self.state.elapsed_seconds),
            "cost_usd": max(0, self.config.max_cost_usd - self.state.cost_usd),
            "tokens": max(0, self.config.max_tokens - self.state.tokens_used),
            "tool_calls": max(0, self.config.max_tool_calls - self.state.tool_calls),
        }

    def can_spend_cost(self, amount: float) -> bool:
        """Check if spending this amount would exceed cost budget."""
        return (self.state.cost_usd + amount) <= self.config.max_cost_usd

    def can_use_tokens(self, count: int) -> bool:
        """Check if using this many tokens would exceed token budget."""
        return (self.state.tokens_used + count) <= self.config.max_tokens

    def can_make_tool_call(self) -> bool:
        """Check if another tool call would exceed limit."""
        return (self.state.tool_calls + 1) <= self.config.max_tool_calls

    async def record_cost(self, amount: float) -> None:
        """
        Record cost expenditure.

        Args:
            amount: Cost in USD
        """
        async with self._lock:
            self.state.cost_usd += amount
            logger.debug(f"Budget: recorded ${amount:.4f}, total ${self.state.cost_usd:.4f}")

    async def record_tokens(self, count: int) -> None:
        """
        Record token usage.

        Args:
            count: Number of tokens used
        """
        async with self._lock:
            self.state.tokens_used += count
            logger.debug(f"Budget: recorded {count} tokens, total {self.state.tokens_used}")

    async def record_tool_call(self) -> None:
        """Record a tool call."""
        async with self._lock:
            self.state.tool_calls += 1
            logger.debug(f"Budget: recorded tool call, total {self.state.tool_calls}")

    async def record_agent_completion(
        self,
        tokens: int,
        cost: float,
        tool_calls: int = 0,
    ) -> None:
        """
        Record completion of an agent's work.

        Args:
            tokens: Tokens used by the agent
            cost: Cost in USD
            tool_calls: Number of tool calls made
        """
        async with self._lock:
            self.state.tokens_used += tokens
            self.state.cost_usd += cost
            self.state.tool_calls += tool_calls
            logger.debug(
                f"Budget: agent completed - tokens={tokens}, cost=${cost:.4f}, "
                f"tools={tool_calls}. Totals: tokens={self.state.tokens_used}, "
                f"cost=${self.state.cost_usd:.4f}, tools={self.state.tool_calls}"
            )

    def get_status_for_result(self) -> str:
        """
        Get the status string for SwarmResult based on budget state.

        Returns:
            'timeout' if timeout exceeded, 'budget_exceeded' if other limit exceeded,
            otherwise 'completed'
        """
        if self.is_timeout_exceeded:
            return "timeout"
        if self.is_exceeded:
            return "budget_exceeded"
        return "completed"

    def get_summary(self) -> dict:
        """Get a summary of budget consumption."""
        remaining = self.get_remaining()
        return {
            "elapsed_seconds": round(self.state.elapsed_seconds, 2),
            "cost_usd": round(self.state.cost_usd, 4),
            "tokens_used": self.state.tokens_used,
            "tool_calls": self.state.tool_calls,
            "remaining": {
                "timeout_seconds": round(remaining["timeout_seconds"], 2),
                "cost_usd": round(remaining["cost_usd"], 4),
                "tokens": remaining["tokens"],
                "tool_calls": remaining["tool_calls"],
            },
            "limits": {
                "timeout_seconds": self.config.timeout_seconds,
                "max_cost_usd": self.config.max_cost_usd,
                "max_tokens": self.config.max_tokens,
                "max_tool_calls": self.config.max_tool_calls,
            },
            "exceeded": self.get_exceeded_type(),
        }


class IncrementalBudgetChecker:
    """
    Helper for checking budget at regular intervals during long operations.

    Usage:
        checker = IncrementalBudgetChecker(budget_manager, check_interval=5.0)

        async for item in items:
            checker.check()  # Raises if budget exceeded
            await process(item)
    """

    def __init__(self, manager: BudgetManager, check_interval: float = 5.0):
        """
        Args:
            manager: The budget manager to check
            check_interval: Minimum seconds between checks
        """
        self.manager = manager
        self.check_interval = check_interval
        self._last_check = 0.0

    def check(self) -> None:
        """
        Check budget if enough time has passed since last check.

        Raises:
            BudgetExceededError: If any limit is exceeded
        """
        now = time.time()
        if now - self._last_check >= self.check_interval:
            self.manager.check_budget()
            self._last_check = now

    def force_check(self) -> None:
        """Force an immediate budget check."""
        self.manager.check_budget()
        self._last_check = time.time()
