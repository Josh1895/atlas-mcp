"""Tests for the agent system."""

import pytest

from atlas.agents.prompt_styles import (
    ALL_STYLES,
    MINIMAL_DIFF,
    VERBOSE_EXPLAINER,
    PromptStyleName,
    get_diverse_styles,
    get_style_by_name,
)
from atlas.agents.agent_pool import AgentPoolManager
from atlas.core.config import Config


class TestPromptStyles:
    """Tests for prompt styles."""

    def test_all_styles_defined(self):
        """Test that all expected styles are defined."""
        assert len(ALL_STYLES) == 5

        names = [s.name for s in ALL_STYLES]
        assert PromptStyleName.MINIMAL_DIFF in names
        assert PromptStyleName.VERBOSE_EXPLAINER in names
        assert PromptStyleName.REFACTOR_FIRST in names
        assert PromptStyleName.DEBUGGER in names
        assert PromptStyleName.REPO_ONLY in names

    def test_get_style_by_name(self):
        """Test getting styles by name."""
        style = get_style_by_name("minimal_diff")
        assert style == MINIMAL_DIFF

        style = get_style_by_name(PromptStyleName.VERBOSE_EXPLAINER)
        assert style == VERBOSE_EXPLAINER

    def test_get_style_by_invalid_name(self):
        """Test getting style with invalid name raises error."""
        with pytest.raises(ValueError):
            get_style_by_name("nonexistent")

    def test_get_diverse_styles(self):
        """Test getting diverse styles."""
        styles = get_diverse_styles(3)
        assert len(styles) == 3

        # Should include MINIMAL_DIFF first
        assert styles[0] == MINIMAL_DIFF

        # All should be unique
        assert len(set(s.name for s in styles)) == 3

    def test_get_diverse_styles_more_than_available(self):
        """Test requesting more styles than available."""
        styles = get_diverse_styles(10)
        assert len(styles) == len(ALL_STYLES)

    def test_system_prompt_generation(self):
        """Test that system prompts are generated correctly."""
        base = "You are an AI assistant."

        prompt = MINIMAL_DIFF.get_system_prompt(base)

        assert base in prompt
        assert "minimal" in prompt.lower() or "surgical" in prompt.lower()


class TestAgentPoolManager:
    """Tests for AgentPoolManager."""

    def test_create_swarm(self):
        """Test creating a diverse swarm."""
        config = Config(
            gemini_api_key="test-key",
            context7_api_key="test-key",
        )
        manager = AgentPoolManager(config=config)

        agents = manager.create_diverse_swarm(5)

        assert len(agents) == 5

        # Check diversity
        styles = set(a.prompt_style.name.value for a in agents if a.prompt_style)
        assert len(styles) >= 3  # At least 3 different styles

    def test_validate_diversity(self):
        """Test diversity validation."""
        config = Config(
            gemini_api_key="test-key",
            context7_api_key="test-key",
        )
        manager = AgentPoolManager(config=config, min_prompt_styles=3)

        # Create diverse swarm
        manager.create_diverse_swarm(5)
        is_valid, errors = manager.validate_diversity()
        assert is_valid
        assert len(errors) == 0

    def test_validate_diversity_empty(self):
        """Test diversity validation with no agents."""
        config = Config(
            gemini_api_key="test-key",
            context7_api_key="test-key",
        )
        manager = AgentPoolManager(config=config)

        is_valid, errors = manager.validate_diversity()
        assert not is_valid
        assert "No agents" in errors[0]

    def test_include_repo_only_agent(self):
        """Test that repo-only agent is included."""
        config = Config(
            gemini_api_key="test-key",
            context7_api_key="test-key",
        )
        manager = AgentPoolManager(config=config)

        agents = manager.create_diverse_swarm(5, include_repo_only=True)

        # Check for repo-only agent
        has_repo_only = any(
            a.prompt_style and not a.prompt_style.use_web_rag
            for a in agents
        )
        assert has_repo_only
