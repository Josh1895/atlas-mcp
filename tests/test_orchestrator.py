"""Tests for the orchestrator."""

import pytest

from atlas.core.config import Config
from atlas.core.task import TaskStatus, TaskSubmission


class TestTaskSubmission:
    """Tests for TaskSubmission dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        task = TaskSubmission(
            description="Fix the bug",
            repository_url="https://github.com/test/repo",
        )

        assert task.task_id  # Should have auto-generated ID
        assert task.branch == "main"
        assert task.max_cost_usd == 2.0
        assert task.timeout_minutes == 15
        assert task.voting_k == 3
        assert task.initial_samples == 5

    def test_custom_values(self):
        """Test setting custom values."""
        task = TaskSubmission(
            description="Fix the bug",
            repository_url="https://github.com/test/repo",
            branch="develop",
            max_cost_usd=5.0,
            timeout_minutes=30,
            voting_k=5,
        )

        assert task.branch == "develop"
        assert task.max_cost_usd == 5.0
        assert task.timeout_minutes == 30
        assert task.voting_k == 5


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test that all expected statuses exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.GENERATING.value == "generating"
        assert TaskStatus.VOTING.value == "voting"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.TIMEOUT.value == "timeout"


class TestConfig:
    """Tests for Config."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.model == "gemini-2.5-flash"
        assert config.temperature == 0.7
        assert config.voting_k == 3
        assert config.max_samples == 5

    def test_validation_missing_keys(self):
        """Test validation with missing API keys."""
        config = Config(gemini_api_key="", context7_api_key="")

        errors = config.validate()

        assert len(errors) == 2
        assert any("GEMINI_API_KEY" in e for e in errors)
        assert any("CONTEXT7_API_KEY" in e for e in errors)

    def test_validation_valid_config(self):
        """Test validation with valid config."""
        config = Config(
            gemini_api_key="test-key",
            context7_api_key="test-key",
        )

        errors = config.validate()
        assert len(errors) == 0
        assert config.is_valid()

    def test_token_cost_calculation(self):
        """Test token cost calculation."""
        config = Config()

        # 1M input tokens + 1M output tokens
        cost = config.calculate_token_cost(1_000_000, 1_000_000)

        expected = 0.30 + 2.50  # Input + output
        assert cost == expected

    def test_token_cost_small(self):
        """Test token cost for small amounts."""
        config = Config()

        # 1000 input + 1000 output
        cost = config.calculate_token_cost(1000, 1000)

        expected = (1000 / 1_000_000) * 0.30 + (1000 / 1_000_000) * 2.50
        assert abs(cost - expected) < 0.0001


# Note: Integration tests for ATLASOrchestrator would require
# mocking the Gemini API and git operations.
# These are left as integration tests for a separate test suite.
