"""Agent pool manager for coordinating multiple micro-agents.

Supports two agent types:
1. MicroAgent - Pre-fetches RAG context (faster, less autonomous)
2. AgenticMicroAgent - AI autonomously decides what to search (more thorough)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Union

from atlas.agents.micro_agent import AgentContext, MicroAgent, AgenticMicroAgent
from atlas.agents.prompt_styles import ALL_STYLES, PromptStyle, get_diverse_styles
from atlas.core.config import Config, get_config
from atlas.core.task import Solution, TaskSubmission

logger = logging.getLogger(__name__)

# Type alias for any agent type
AgentType = Union[MicroAgent, AgenticMicroAgent]


@dataclass
class SwarmResult:
    """Result from running a swarm of agents."""

    solutions: list[Solution] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    successful_count: int = 0
    failed_count: int = 0

    @property
    def valid_solutions(self) -> list[Solution]:
        """Get only valid solutions."""
        return [s for s in self.solutions if s.is_valid]


class AgentPoolManager:
    """Manages a pool of diverse micro-agents.

    Ensures diversity across:
    - Prompt styles (different reasoning approaches)
    - RAG configurations (some with web, some without)

    Supports two modes:
    - Standard (MicroAgent): Pre-fetches RAG context, faster
    - Agentic (AgenticMicroAgent): AI autonomously searches, more thorough
    """

    def __init__(
        self,
        config: Config | None = None,
        min_prompt_styles: int = 3,
        use_agentic: bool = True,
    ):
        """Initialize the agent pool manager.

        Args:
            config: Optional Config instance
            min_prompt_styles: Minimum number of different prompt styles to use
            use_agentic: If True, use AgenticMicroAgent (autonomous tool access)
                         If False, use MicroAgent (pre-fetched RAG)
        """
        self.config = config or get_config()
        self.min_prompt_styles = min_prompt_styles
        self.use_agentic = use_agentic
        self._agents: list[AgentType] = []

    def create_diverse_swarm(
        self,
        size: int,
        include_repo_only: bool = True,
    ) -> list[AgentType]:
        """Create a swarm of diverse agents.

        Creates exactly N agents per prompt style (round-robin) to ensure
        balanced diversity. For example, with 5 styles and size=10, creates
        2 agents of each style.

        Uses AgenticMicroAgent if use_agentic=True (default), otherwise MicroAgent.

        Args:
            size: Number of agents to create
            include_repo_only: Whether to include a repo-only agent

        Returns:
            List of agent instances with diverse configurations
        """
        agents = []

        # Select agent class based on mode
        AgentClass = AgenticMicroAgent if self.use_agentic else MicroAgent
        logger.info(f"Creating swarm with {AgentClass.__name__} (use_agentic={self.use_agentic})")

        # Calculate how many agents per style
        num_styles = len(ALL_STYLES)
        agents_per_style = max(1, size // num_styles)
        remainder = size % num_styles

        agent_idx = 0
        for style_idx, style in enumerate(ALL_STYLES):
            # Give extra agent to first 'remainder' styles if not evenly divisible
            count = agents_per_style + (1 if style_idx < remainder else 0)

            for _ in range(count):
                agent = AgentClass(
                    agent_id=f"agent_{agent_idx}",
                    prompt_style=style,
                    config=self.config,
                )
                agents.append(agent)
                agent_idx += 1

                if agent_idx >= size:
                    break

            if agent_idx >= size:
                break

        self._agents = agents
        return agents

    async def run_swarm(
        self,
        context: AgentContext,
        parallel: bool = True,
    ) -> SwarmResult:
        """Run the swarm of agents to generate solutions.

        Args:
            context: The agent context with task and code
            parallel: Whether to run agents in parallel

        Returns:
            SwarmResult with all solutions
        """
        if not self._agents:
            raise ValueError("No agents in pool. Call create_diverse_swarm first.")

        result = SwarmResult()

        if parallel:
            # Run all agents in parallel
            solutions = await self._run_parallel(context)
        else:
            # Run agents sequentially
            solutions = await self._run_sequential(context)

        # Aggregate results
        for solution in solutions:
            result.solutions.append(solution)
            result.total_cost += solution.cost
            result.total_tokens += solution.tokens_used

            if solution.is_valid:
                result.successful_count += 1
            else:
                result.failed_count += 1

        return result

    async def _run_parallel(self, context: AgentContext) -> list[Solution]:
        """Run agents in parallel."""
        tasks = []

        for agent in self._agents:
            # Create a task for each agent
            task = self._run_agent_with_rag(agent, context)
            tasks.append(task)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        solutions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {self._agents[i].agent_id} failed: {result}")
                # Create a failed solution
                solutions.append(Solution(
                    agent_id=self._agents[i].agent_id,
                    prompt_style=self._agents[i].prompt_style.name.value if self._agents[i].prompt_style else "default",
                    patch="",
                    is_valid=False,
                    validation_errors=[str(result)],
                ))
            else:
                solutions.append(result)

        return solutions

    async def _run_sequential(self, context: AgentContext) -> list[Solution]:
        """Run agents sequentially."""
        solutions = []

        for agent in self._agents:
            try:
                solution = await self._run_agent_with_rag(agent, context)
                solutions.append(solution)
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {e}")
                solutions.append(Solution(
                    agent_id=agent.agent_id,
                    prompt_style=agent.prompt_style.name.value if agent.prompt_style else "default",
                    patch="",
                    is_valid=False,
                    validation_errors=[str(e)],
                ))

        return solutions

    async def _run_agent_with_rag(
        self,
        agent: AgentType,
        base_context: AgentContext,
    ) -> Solution:
        """Run a single agent.

        For MicroAgent: Gathers RAG context first, then generates.
        For AgenticMicroAgent: Just generates - agent handles RAG autonomously.

        Args:
            agent: The agent to run
            base_context: The base context (without RAG)

        Returns:
            The generated solution
        """
        # AgenticMicroAgent handles RAG autonomously via tool calls
        if isinstance(agent, AgenticMicroAgent):
            return await agent.generate(base_context)

        # MicroAgent needs RAG pre-fetched
        # Gather RAG context for this agent (Context7 + web search)
        rag_context, web_search_context = await agent.gather_rag_context(base_context.task)

        # Create agent-specific context with RAG
        context = AgentContext(
            task=base_context.task,
            repository_content=base_context.repository_content,
            rag_context=rag_context,
            web_search_context=web_search_context,
            additional_context=base_context.additional_context,
        )

        # Generate the solution
        return await agent.generate(context)

    async def run_incremental(
        self,
        context: AgentContext,
        batch_size: int = 5,
        max_batches: int = 3,
        should_stop: callable = None,
    ) -> SwarmResult:
        """Run agents incrementally until a stopping condition is met.

        This allows for early stopping when consensus is reached.

        Args:
            context: The agent context
            batch_size: Number of agents per batch
            max_batches: Maximum number of batches
            should_stop: Callback that takes SwarmResult and returns True to stop

        Returns:
            SwarmResult with all solutions
        """
        result = SwarmResult()

        for batch_num in range(max_batches):
            # Create a batch of agents
            start_idx = batch_num * batch_size
            self.create_diverse_swarm(batch_size)

            # Rename agents for this batch
            for i, agent in enumerate(self._agents):
                agent.agent_id = f"agent_{start_idx + i}"

            # Run the batch
            batch_result = await self.run_swarm(context, parallel=True)

            # Aggregate results
            result.solutions.extend(batch_result.solutions)
            result.total_cost += batch_result.total_cost
            result.total_tokens += batch_result.total_tokens
            result.successful_count += batch_result.successful_count
            result.failed_count += batch_result.failed_count

            # Check stopping condition
            if should_stop and should_stop(result):
                logger.info(f"Early stopping after batch {batch_num + 1}")
                break

            # Check cost limit
            if result.total_cost >= self.config.max_cost_usd:
                logger.warning("Cost limit reached, stopping swarm")
                break

        return result

    def validate_diversity(self) -> tuple[bool, list[str]]:
        """Validate that the swarm has sufficient diversity.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not self._agents:
            return False, ["No agents in pool"]

        errors = []

        # Check prompt style diversity
        styles = set()
        for agent in self._agents:
            if agent.prompt_style:
                styles.add(agent.prompt_style.name.value)

        if len(styles) < self.min_prompt_styles:
            errors.append(
                f"Insufficient prompt diversity: have {len(styles)}, need {self.min_prompt_styles}"
            )

        # Check for repo-only agent
        has_repo_only = any(
            agent.prompt_style and not agent.prompt_style.use_web_rag
            for agent in self._agents
        )

        if not has_repo_only:
            errors.append("Missing repo-only agent for diversity")

        return len(errors) == 0, errors
