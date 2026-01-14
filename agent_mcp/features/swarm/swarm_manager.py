"""
Swarm Manager for orchestrating multi-agent consensus.

The single orchestration class that coordinates all swarm components (AD-002):
- AtlasAdapter: Agent generation
- ConsensusEngine: Clustering and voting
- ToolRouter: Managed tool access
- BudgetManager: Cost/time limits
- Persistence: DB operations
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from ...core.config import logger as config_logger, SWARM_ENABLED


class SwarmLogger:
    """
    Structured logger with run_id correlation.

    All log messages include run_id for tracing and filtering.
    Supports structured logging format for easier parsing.
    """

    def __init__(self, run_id: str = "", base_logger: Optional[logging.Logger] = None):
        self.run_id = run_id
        self._logger = base_logger or logging.getLogger(__name__)

    def set_run_id(self, run_id: str) -> None:
        """Set the run_id for all subsequent log messages."""
        self.run_id = run_id

    def _format_msg(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with run_id and optional structured data."""
        parts = [f"[run={self.run_id}]"] if self.run_id else []
        parts.append(msg)
        if extra:
            # Add structured data as JSON suffix
            parts.append(f" | {json.dumps(extra)}")
        return " ".join(parts)

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(self._format_msg(msg, kwargs if kwargs else None))

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(self._format_msg(msg, kwargs if kwargs else None))

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(self._format_msg(msg, kwargs if kwargs else None))

    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        self._logger.error(self._format_msg(msg, kwargs if kwargs else None), exc_info=exc_info)

    def log_event(self, event: str, **data) -> None:
        """Log a structured event with timestamp."""
        payload = {
            "event": event,
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }
        self._logger.info(f"SWARM_EVENT: {json.dumps(payload)}")

from .schemas import (
    SwarmRequest,
    SwarmResult,
    SwarmMode,
    SwarmConfig,
    BudgetConfig,
    ToolConfig,
    MemoryConfig,
    AgentOutput,
    ClusterInfo,
    SwarmMetrics,
)
from .budget_manager import BudgetManager, BudgetExceededError
from .repo_resolver import RepoResolver, RepoResolverError
from .context_builder import ContextBuilder, build_swarm_context
from .tool_router import ToolRouter, create_default_router
from .atlas_adapter import AtlasAdapter, create_atlas_adapter, is_atlas_available
from .consensus_engine import ConsensusEngine, ConsensusResult
from .memory_writer import MemoryWriter
from .persistence import (
    save_swarm_run,
    update_swarm_run_status,
    save_swarm_agent,
    update_swarm_agent,
    save_swarm_output,
)
from .metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class SwarmManagerError(Exception):
    """Base error for swarm manager."""
    pass


class SwarmDisabledError(SwarmManagerError):
    """Raised when swarm is disabled."""
    pass


class SwarmManager:
    """
    Orchestrates the full swarm workflow.

    Workflow:
    1. Resolve repository (local or remote)
    2. Build context from task/memory
    3. Generate candidates via Atlas agents
    4. Cluster outputs by similarity/behavior
    5. Vote on clusters for consensus
    6. Write results back to Agent-MCP

    Handles timeout, budget limits, and partial failures gracefully.
    """

    def __init__(
        self,
        request: Optional[SwarmRequest] = None,
    ):
        """
        Initialize swarm manager.

        Args:
            request: Swarm request with all configuration
        """
        self.request = request or SwarmRequest(token="")

        # Initialize components
        self._budget_manager: Optional[BudgetManager] = None
        self._repo_resolver: Optional[RepoResolver] = None
        self._context_builder: Optional[ContextBuilder] = None
        self._tool_router: Optional[ToolRouter] = None
        self._atlas_adapter: Optional[AtlasAdapter] = None
        self._consensus_engine: Optional[ConsensusEngine] = None
        self._memory_writer: Optional[MemoryWriter] = None

        # State
        self._run_id: str = ""
        self._started_at: Optional[str] = None
        self._start_time: float = 0.0
        self._outputs: List[AgentOutput] = []
        self._clusters: List[ClusterInfo] = []

        # Structured logger with run_id correlation
        self._log = SwarmLogger()

    @property
    def budget_manager(self) -> BudgetManager:
        if self._budget_manager is None:
            self._budget_manager = BudgetManager(self.request.budgets)
        return self._budget_manager

    @property
    def repo_resolver(self) -> RepoResolver:
        if self._repo_resolver is None:
            self._repo_resolver = RepoResolver(self.request.repo)
        return self._repo_resolver

    @property
    def context_builder(self) -> ContextBuilder:
        if self._context_builder is None:
            self._context_builder = ContextBuilder(
                max_tokens=self.request.budgets.max_tokens // 10,  # Reserve most for generation
            )
        return self._context_builder

    @property
    def tool_router(self) -> ToolRouter:
        if self._tool_router is None:
            self._tool_router = create_default_router(
                self.request.tools,
                self._run_id,
            )
        return self._tool_router

    @property
    def atlas_adapter(self) -> Optional[AtlasAdapter]:
        if self._atlas_adapter is None and is_atlas_available():
            self._atlas_adapter = create_atlas_adapter(self.request.swarm)
        return self._atlas_adapter

    @property
    def consensus_engine(self) -> ConsensusEngine:
        if self._consensus_engine is None:
            self._consensus_engine = ConsensusEngine(
                consensus_k=self.request.swarm.consensus_k,
                mode=self.request.mode,
            )
        return self._consensus_engine

    @property
    def memory_writer(self) -> MemoryWriter:
        if self._memory_writer is None:
            self._memory_writer = MemoryWriter(self.request.memory)
        return self._memory_writer

    async def run(self) -> SwarmResult:
        """
        Execute the full swarm workflow.

        Returns:
            SwarmResult with consensus output and metadata
        """
        # Check if swarm is enabled
        if not SWARM_ENABLED:
            raise SwarmDisabledError("Swarm feature is disabled. Set SWARM_ENABLED=true")

        # Generate run ID and set up logging
        self._run_id = f"swarm_{uuid.uuid4().hex[:12]}"
        self._started_at = datetime.utcnow().isoformat()
        self._start_time = time.time()
        self._log.set_run_id(self._run_id)

        self._log.log_event(
            "swarm_started",
            mode=self.request.mode.value,
            task_id=self.request.task_id,
            agent_count=self.request.swarm.agent_count,
            timeout_seconds=self.request.budgets.timeout_seconds,
        )

        # Record metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.record_run_started()

        # Initialize result
        result = SwarmResult(
            run_id=self._run_id,
            task_id=self.request.task_id,
            mode=self.request.mode,
            status="running",
            started_at=self._started_at,
        )

        try:
            # Save initial run record
            await save_swarm_run(
                run_id=self._run_id,
                task_id=self.request.task_id,
                mode=self.request.mode.value,
                config_json=self._serialize_config(),
                started_at=self._started_at,
            )

            # Execute workflow with timeout
            timeout = self.request.budgets.timeout_seconds
            result = await asyncio.wait_for(
                self._execute_workflow(result),
                timeout=timeout,
            )

        except asyncio.TimeoutError:
            self._log.warning("Swarm run timed out", timeout_seconds=timeout)
            self._log.log_event("swarm_timeout", timeout_seconds=timeout)
            result.status = "timeout"
            result.warnings.append(f"Run timed out after {timeout}s")
            result = await self._finalize_best_effort(result)

        except BudgetExceededError as e:
            self._log.warning("Swarm run exceeded budget", error=str(e))
            self._log.log_event("swarm_budget_exceeded", error=str(e))
            result.status = "budget_exceeded"
            result.warnings.append(str(e))
            result = await self._finalize_best_effort(result)

        except SwarmDisabledError:
            raise

        except Exception as e:
            self._log.error("Swarm run failed", exc_info=True, error=str(e))
            self._log.log_event("swarm_error", error=str(e), error_type=type(e).__name__)
            result.status = "failed"
            result.errors.append(str(e))

        # Finalize
        result.completed_at = datetime.utcnow().isoformat()
        result.metrics = self._collect_metrics()

        # Update DB record
        await update_swarm_run_status(
            run_id=self._run_id,
            status=result.status,
            completed_at=result.completed_at,
            consensus_reached=result.consensus_reached,
            confidence_score=result.confidence_score,
            selected_output=result.selected_output,
            selected_variant_id=result.selected_variant_id,
            vote_counts=result.vote_counts,
            metrics=result.metrics,
            errors=result.errors,
            warnings=result.warnings,
        )

        # Write back to memory systems
        try:
            await self.memory_writer.write_result(result)
        except Exception as e:
            self._log.error("Failed to write result to memory", error=str(e))
            result.warnings.append(f"Memory write failed: {e}")

        # Log completion event with full metrics
        self._log.log_event(
            "swarm_completed",
            status=result.status,
            consensus_reached=result.consensus_reached,
            confidence_score=result.confidence_score,
            duration_ms=result.metrics.duration_ms if result.metrics else 0,
            cost_usd=result.metrics.cost_usd if result.metrics else 0,
            tokens_total=result.metrics.tokens_total if result.metrics else 0,
            agents_succeeded=result.metrics.agents_succeeded if result.metrics else 0,
            agents_failed=result.metrics.agents_failed if result.metrics else 0,
            cluster_count=len(result.clusters) if result.clusters else 0,
        )

        # Record completion metrics
        metrics_collector.record_run_completed(
            status=result.status,
            duration_ms=result.metrics.duration_ms if result.metrics else 0,
            consensus_reached=result.consensus_reached,
            confidence_score=result.confidence_score,
            cost_usd=result.metrics.cost_usd if result.metrics else 0,
            tokens_used=result.metrics.tokens_total if result.metrics else 0,
            agents_succeeded=result.metrics.agents_succeeded if result.metrics else 0,
            agents_failed=result.metrics.agents_failed if result.metrics else 0,
            cluster_count=len(result.clusters) if result.clusters else 0,
        )

        return result

    async def _execute_workflow(self, result: SwarmResult) -> SwarmResult:
        """Execute the main workflow steps."""
        # 1. Resolve repository
        self._log.log_event("workflow_step", step="repo_resolve", status="started")
        repo_content = ""
        if self.request.mode == SwarmMode.PATCH:
            try:
                resolved = self.repo_resolver.resolve()
                repo_content = f"Repository: {resolved.path}\n"
                if resolved.branch:
                    repo_content += f"Branch: {resolved.branch}\n"
                self._log.debug("Repository resolved", path=str(resolved.path))
            except RepoResolverError as e:
                self._log.warning("Repo resolution failed", error=str(e))
                result.warnings.append(f"Repo resolution: {e}")

        # 2. Build context
        self._log.log_event("workflow_step", step="context_build", status="started")
        context = self.context_builder.build(
            task_id=self.request.task_id,
            description=self.request.description,
            include_rag=self.request.tools.enable_agent_mcp_rag,
        )
        full_context = context.get_full_context()
        self._log.debug("Context built", context_length=len(full_context))

        # Check budget
        self.budget_manager.check_budget()

        # 3. Generate candidates
        self._log.log_event("workflow_step", step="generation", status="started")
        gen_start = time.time()
        self._outputs = await self._generate_candidates(
            description=self.request.description or "",
            repo_content=repo_content,
            additional_context=full_context,
        )
        gen_duration = time.time() - gen_start

        self._log.log_event(
            "generation_completed",
            output_count=len(self._outputs),
            valid_count=len([o for o in self._outputs if o.is_valid]),
            duration_seconds=round(gen_duration, 2),
        )

        # Record generation latency
        get_metrics_collector().record_generation_latency(int(gen_duration * 1000))

        if not self._outputs:
            result.status = "failed"
            result.errors.append("No outputs generated")
            return result

        # Check budget again
        self.budget_manager.check_budget()

        # 4. Cluster and vote
        self._log.log_event("workflow_step", step="consensus", status="started")
        consensus_start = time.time()
        consensus_result = await self.consensus_engine.cluster_and_vote(self._outputs)
        consensus_duration = time.time() - consensus_start

        self._log.log_event(
            "consensus_completed",
            consensus_reached=consensus_result.consensus_reached,
            confidence=consensus_result.confidence_score,
            cluster_count=len(consensus_result.clusters) if consensus_result.clusters else 0,
            duration_seconds=round(consensus_duration, 2),
        )

        # Record consensus latency
        get_metrics_collector().record_consensus_latency(int(consensus_duration * 1000))

        # 5. Update result
        result.consensus_reached = consensus_result.consensus_reached
        result.selected_output = consensus_result.selected_output
        result.selected_variant_id = consensus_result.selected_cluster_id
        result.confidence_score = consensus_result.confidence_score
        result.vote_counts = consensus_result.vote_counts
        result.clusters = consensus_result.clusters
        result.agent_outputs = self._outputs
        result.status = "completed"

        return result

    async def _generate_candidates(
        self,
        description: str,
        repo_content: str,
        additional_context: str,
    ) -> List[AgentOutput]:
        """Generate candidate outputs using Atlas or fallback."""
        outputs: List[AgentOutput] = []

        if self.atlas_adapter:
            # Use Atlas for generation
            try:
                self._log.debug("Using Atlas adapter for generation")
                if self.request.mode == SwarmMode.PATCH:
                    gen_result = await self.atlas_adapter.generate_candidates_patch_mode(
                        task_description=description,
                        repository_content=repo_content,
                        additional_context=additional_context,
                        agent_count=self.request.swarm.agent_count,
                    )
                else:
                    gen_result = await self.atlas_adapter.generate_candidates_answer_mode(
                        question=description,
                        context=additional_context,
                        agent_count=self.request.swarm.agent_count,
                    )

                outputs = gen_result.outputs

                # Record costs
                await self.budget_manager.record_cost(gen_result.total_cost)
                await self.budget_manager.record_tokens(gen_result.total_tokens)

                self._log.debug(
                    "Atlas generation completed",
                    cost_usd=gen_result.total_cost,
                    tokens=gen_result.total_tokens,
                    succeeded=gen_result.successful_count,
                    failed=gen_result.failed_count,
                )

            except Exception as e:
                self._log.error("Atlas generation failed", error=str(e))
                # Fall through to fallback

        if not outputs:
            # Fallback: Generate simple placeholder outputs
            self._log.warning("Using fallback generation (Atlas not available)")
            outputs = await self._fallback_generate(description)

        # Save agent records
        for output in outputs:
            await save_swarm_agent(
                run_id=self._run_id,
                swarm_agent_id=output.agent_id,
                prompt_style=output.prompt_style,
                model=self.request.swarm.model,
                temperature=self.request.swarm.temperature,
                status="success" if output.is_valid else "failed",
                started_at=self._started_at,
            )

            await save_swarm_output(
                run_id=self._run_id,
                swarm_agent_id=output.agent_id,
                output_type=self.request.mode.value,
                output_text=output.output_text,
                explanation=output.explanation,
                is_valid=output.is_valid,
                validation_errors=output.validation_errors,
            )

        return outputs

    async def _fallback_generate(self, description: str) -> List[AgentOutput]:
        """Fallback generation when Atlas is not available."""
        # This is a placeholder - in production, could use a simpler LLM call
        return [
            AgentOutput(
                agent_id="fallback_0",
                prompt_style="default",
                output_text=f"[Fallback] Unable to generate proper output for: {description[:100]}",
                is_valid=False,
                validation_errors=["Atlas not available for generation"],
            )
        ]

    async def _finalize_best_effort(self, result: SwarmResult) -> SwarmResult:
        """Finalize with best effort when timeout/budget exceeded."""
        self._log.info("Attempting best-effort consensus", output_count=len(self._outputs))

        if self._outputs:
            # Try to get consensus from what we have
            try:
                consensus_result = await self.consensus_engine.cluster_and_vote(self._outputs)
                result.selected_output = consensus_result.selected_output
                result.selected_variant_id = consensus_result.selected_cluster_id
                result.vote_counts = consensus_result.vote_counts
                result.clusters = consensus_result.clusters
                result.confidence_score = consensus_result.confidence_score * 0.5  # Lower confidence
                result.consensus_reached = False  # Not full consensus
                self._log.info(
                    "Best-effort consensus completed",
                    confidence=result.confidence_score,
                )
            except Exception as e:
                self._log.error("Best effort consensus failed", error=str(e))

        result.agent_outputs = self._outputs
        return result

    def _collect_metrics(self) -> SwarmMetrics:
        """Collect metrics from the run."""
        return SwarmMetrics(
            duration_ms=int(self.budget_manager.state.elapsed_seconds * 1000),
            cost_usd=self.budget_manager.state.cost_usd,
            tokens_total=self.budget_manager.state.tokens_used,
            tool_calls_count=self.budget_manager.state.tool_calls,
            cache_hits=self.tool_router.get_stats().get("cache_hits", 0) if self._tool_router else 0,
            agents_succeeded=len([o for o in self._outputs if o.is_valid]),
            agents_failed=len([o for o in self._outputs if not o.is_valid]),
        )

    def _serialize_config(self) -> str:
        """Serialize configuration for DB storage."""
        import json
        return json.dumps({
            "task_id": self.request.task_id,
            "description": self.request.description,
            "mode": self.request.mode.value,
            "budgets": {
                "timeout_seconds": self.request.budgets.timeout_seconds,
                "max_cost_usd": self.request.budgets.max_cost_usd,
                "max_tokens": self.request.budgets.max_tokens,
                "max_tool_calls": self.request.budgets.max_tool_calls,
            },
            "swarm": {
                "agent_count": self.request.swarm.agent_count,
                "consensus_k": self.request.swarm.consensus_k,
                "model": self.request.swarm.model,
                "temperature": self.request.swarm.temperature,
            },
        })


async def run_swarm(request: SwarmRequest) -> SwarmResult:
    """
    Convenience function to run a swarm.

    Args:
        request: Swarm request configuration

    Returns:
        SwarmResult with consensus output
    """
    manager = SwarmManager(request)
    return await manager.run()
