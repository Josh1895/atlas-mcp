"""
Atlas Adapter for Agent-MCP integration.

Wraps Atlas classes (AgentPoolManager, VotingManager, SimilarityClustering)
to isolate Agent-MCP from Atlas internals (AD-003). Enables future Atlas
upgrades without breaking integration.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from ...core.config import logger as config_logger

# Try to import Atlas components
try:
    from atlas.agents.agent_pool import AgentPoolManager, SwarmResult as AtlasSwarmResult
    from atlas.agents.micro_agent import AgentContext, MicroAgent, AgenticMicroAgent
    from atlas.agents.prompt_styles import ALL_STYLES, PromptStyle, get_diverse_styles
    from atlas.core.task import Solution, TaskSubmission
    from atlas.voting.consensus import VotingManager, VotingResult, IncrementalVoter
    from atlas.verification.clustering import SimilarityClustering, ClusteringResult, Cluster
    from atlas.verification.patch_applier import PatchApplier, PatchApplyResult, create_patched_checkout
    from atlas.verification.test_runner import TestRunner, TestResult, TestFramework
    from atlas.quality.quality_scorer import QualityScorer, QualityScore
    ATLAS_AVAILABLE = True
except ImportError as e:
    ATLAS_AVAILABLE = False
    config_logger.warning(f"Atlas components not available: {e}")

from .schemas import (
    SwarmConfig,
    AgentOutput,
    ClusterInfo,
    SwarmMetrics,
    SwarmMode,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from agent generation."""
    outputs: List[AgentOutput]
    total_cost: float = 0.0
    total_tokens: int = 0
    successful_count: int = 0
    failed_count: int = 0


@dataclass
class PatchTestResult:
    """Test result for a single patch."""
    agent_id: str
    patch_applied: bool = False
    apply_errors: List[str] = field(default_factory=list)
    tests_passed: bool = False
    tests_total: int = 0
    tests_failed: int = 0
    test_duration_seconds: float = 0.0
    failure_signature: str = ""  # Sorted failed test names for clustering
    quality_score: Optional[float] = None
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class PatchValidationResult:
    """Result from validating and scoring all patches."""
    results: Dict[str, PatchTestResult] = field(default_factory=dict)  # agent_id -> result
    any_passed: bool = False
    best_passing_agent_id: Optional[str] = None
    total_duration_seconds: float = 0.0


class AtlasAdapterError(Exception):
    """Base error for Atlas adapter."""
    pass


class AtlasNotAvailableError(AtlasAdapterError):
    """Raised when Atlas components are not available."""
    pass


class AtlasAdapter:
    """
    Adapter wrapping Atlas swarm components for Agent-MCP integration.

    Provides a clean interface to:
    - AgentPoolManager: For parallel agent generation
    - SimilarityClustering: For patch clustering
    - VotingManager: For consensus voting

    Handles conversion between Atlas and Agent-MCP data types.
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        """
        Initialize the Atlas adapter.

        Args:
            config: Swarm configuration

        Raises:
            AtlasNotAvailableError: If Atlas components aren't installed
        """
        if not ATLAS_AVAILABLE:
            raise AtlasNotAvailableError(
                "Atlas components are not available. Install atlas package."
            )

        self.config = config or SwarmConfig()
        self._pool_manager: Optional[AgentPoolManager] = None
        self._voting_manager: Optional[VotingManager] = None
        self._clustering: Optional[SimilarityClustering] = None

    @property
    def pool_manager(self) -> "AgentPoolManager":
        """Get or create the agent pool manager."""
        if self._pool_manager is None:
            self._pool_manager = AgentPoolManager(
                min_prompt_styles=min(3, self.config.agent_count),
                use_agentic=True,  # Use autonomous agents
            )
        return self._pool_manager

    @property
    def voting_manager(self) -> "VotingManager":
        """Get or create the voting manager."""
        if self._voting_manager is None:
            self._voting_manager = VotingManager(
                k=self.config.consensus_k,
                similarity_threshold=0.6,
            )
        return self._voting_manager

    @property
    def clustering(self) -> "SimilarityClustering":
        """Get or create the similarity clustering."""
        if self._clustering is None:
            self._clustering = SimilarityClustering(
                similarity_threshold=0.6,
            )
        return self._clustering

    async def generate_candidates_patch_mode(
        self,
        task_description: str,
        repository_content: str,
        additional_context: str = "",
        agent_count: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate patch candidates using swarm of agents.

        Args:
            task_description: Description of the task/bug to fix
            repository_content: Relevant code from the repository
            additional_context: Additional context (RAG results, etc.)
            agent_count: Number of agents (defaults to config)

        Returns:
            GenerationResult with all agent outputs
        """
        count = agent_count or self.config.agent_count

        # Create task submission for Atlas
        task = TaskSubmission(
            task_id=f"swarm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=task_description,
            repository_url="local",  # Local mode
        )

        # Create agent context
        context = AgentContext(
            task=task,
            repository_content=repository_content,
            additional_context=additional_context,
        )

        # Create diverse swarm
        self.pool_manager.create_diverse_swarm(count)

        # Run swarm
        atlas_result = await self.pool_manager.run_swarm(context, parallel=True)

        # Convert to our format
        return self._convert_atlas_result(atlas_result)

    async def generate_candidates_answer_mode(
        self,
        question: str,
        context: str = "",
        agent_count: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate answer candidates for answer mode.

        For answer mode, we use simpler generation without patches.

        Args:
            question: The question to answer
            context: Supporting context
            agent_count: Number of agents

        Returns:
            GenerationResult with answers
        """
        count = agent_count or self.config.agent_count

        # Create task for answer mode
        task = TaskSubmission(
            task_id=f"answer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            description=question,
            repository_url="",
        )

        context_obj = AgentContext(
            task=task,
            repository_content="",
            additional_context=context,
        )

        # Create diverse swarm
        self.pool_manager.create_diverse_swarm(count)

        # Run swarm
        atlas_result = await self.pool_manager.run_swarm(context_obj, parallel=True)

        # Convert - treat explanations as answers in answer mode
        return self._convert_atlas_result(atlas_result, is_answer_mode=True)

    def _convert_atlas_result(
        self,
        atlas_result: "AtlasSwarmResult",
        is_answer_mode: bool = False,
    ) -> GenerationResult:
        """Convert Atlas SwarmResult to our GenerationResult."""
        outputs = []

        for solution in atlas_result.solutions:
            # In answer mode, use explanation as the output
            output_text = solution.explanation if is_answer_mode else solution.patch

            output = AgentOutput(
                agent_id=solution.agent_id,
                prompt_style=solution.prompt_style,
                output_text=output_text,
                explanation=solution.explanation if not is_answer_mode else None,
                is_valid=solution.is_valid,
                validation_errors=solution.validation_errors,
                tokens_used=solution.tokens_used,
                cost_usd=solution.cost,
            )
            outputs.append(output)

        return GenerationResult(
            outputs=outputs,
            total_cost=atlas_result.total_cost,
            total_tokens=atlas_result.total_tokens,
            successful_count=atlas_result.successful_count,
            failed_count=atlas_result.failed_count,
        )

    def cluster_patches(
        self,
        outputs: List[AgentOutput],
    ) -> List[ClusterInfo]:
        """
        Cluster patch outputs by similarity.

        Args:
            outputs: Agent outputs to cluster

        Returns:
            List of ClusterInfo with cluster assignments
        """
        # Convert to Atlas Solution format
        solutions = []
        for output in outputs:
            solution = Solution(
                agent_id=output.agent_id,
                prompt_style=output.prompt_style,
                patch=output.output_text,
                explanation=output.explanation or "",
                is_valid=output.is_valid,
                validation_errors=output.validation_errors,
                tokens_used=output.tokens_used,
                cost=output.cost_usd,
            )
            solutions.append(solution)

        # Cluster using Atlas
        result = self.clustering.cluster(solutions)
        result = self.clustering.merge_similar_clusters(result)

        # Convert to ClusterInfo
        clusters = []
        for cluster in result.clusters:
            info = ClusterInfo(
                cluster_id=cluster.cluster_id,
                size=cluster.size,
                vote_count=cluster.size,  # Initial vote = size
                representative_output=cluster.representative.patch if cluster.representative else "",
                member_agent_ids=[s.agent_id for s in cluster.solutions],
            )
            clusters.append(info)

            # Update output cluster assignments
            for output in outputs:
                if output.agent_id in info.member_agent_ids:
                    output.cluster_id = cluster.cluster_id

        return clusters

    def vote_on_clusters(
        self,
        outputs: List[AgentOutput],
    ) -> tuple[Optional[str], bool, float, Dict[str, int]]:
        """
        Run consensus voting on outputs.

        Args:
            outputs: Agent outputs (should have cluster_id assigned)

        Returns:
            Tuple of (winning_output, consensus_reached, confidence, vote_counts)
        """
        # Convert to Atlas Solution format
        solutions = []
        for output in outputs:
            solution = Solution(
                agent_id=output.agent_id,
                prompt_style=output.prompt_style,
                patch=output.output_text,
                explanation=output.explanation or "",
                is_valid=output.is_valid,
                validation_errors=output.validation_errors,
                cluster_id=output.cluster_id,
            )
            solutions.append(solution)

        # Reset voting manager for new vote
        self.voting_manager.reset()

        # Run voting
        result = self.voting_manager.vote(solutions)

        winning_output = None
        if result.winning_solution:
            winning_output = result.winning_solution.patch

        return (
            winning_output,
            result.consensus_reached,
            result.confidence_score,
            result.vote_counts,
        )

    def reset(self) -> None:
        """Reset adapter state for new run."""
        if self._voting_manager:
            self._voting_manager.reset()
        self._pool_manager = None

    async def validate_and_score_patch_candidates(
        self,
        outputs: List[AgentOutput],
        repo_path: Path,
        original_files: Dict[str, str],
        test_command: Optional[str] = None,
        timeout_seconds: int = 300,
        run_quality_scoring: bool = True,
    ) -> PatchValidationResult:
        """
        Validate and score patch candidates by applying them and running tests.

        Args:
            outputs: Agent outputs containing patches
            repo_path: Path to the repository
            original_files: Dict of {filepath: content} for files
            test_command: Optional explicit test command
            timeout_seconds: Test execution timeout per patch
            run_quality_scoring: Whether to compute quality scores

        Returns:
            PatchValidationResult with test results and quality scores
        """
        import time
        start_time = time.time()

        result = PatchValidationResult()
        applier = PatchApplier()
        test_runner = TestRunner(timeout_seconds=timeout_seconds)
        quality_scorer = QualityScorer(
            repo_path=str(repo_path),
            language="python",  # TODO: Auto-detect
        )

        best_quality = -1.0
        best_passing_id = None

        for output in outputs:
            if not output.is_valid or not output.output_text:
                result.results[output.agent_id] = PatchTestResult(
                    agent_id=output.agent_id,
                    patch_applied=False,
                    apply_errors=output.validation_errors or ["Invalid patch"],
                )
                continue

            test_result = await self._validate_single_patch(
                output=output,
                repo_path=repo_path,
                original_files=original_files,
                applier=applier,
                test_runner=test_runner,
                quality_scorer=quality_scorer if run_quality_scoring else None,
                test_command=test_command,
            )

            result.results[output.agent_id] = test_result

            # Track best passing patch
            if test_result.tests_passed:
                result.any_passed = True
                if test_result.quality_score is not None and test_result.quality_score > best_quality:
                    best_quality = test_result.quality_score
                    best_passing_id = output.agent_id

            # Store test result in output for clustering
            output.test_result = {
                "passed": test_result.tests_passed,
                "total": test_result.tests_total,
                "failed": test_result.tests_failed,
                "failure_signature": test_result.failure_signature,
            }
            output.quality_score = {
                "overall": test_result.quality_score,
                "risk_flags": test_result.risk_flags,
            } if test_result.quality_score is not None else None

        result.best_passing_agent_id = best_passing_id
        result.total_duration_seconds = time.time() - start_time

        logger.info(
            f"Patch validation complete: {len(result.results)} patches, "
            f"{sum(1 for r in result.results.values() if r.tests_passed)} passing, "
            f"duration={result.total_duration_seconds:.1f}s"
        )

        return result

    async def _validate_single_patch(
        self,
        output: AgentOutput,
        repo_path: Path,
        original_files: Dict[str, str],
        applier: "PatchApplier",
        test_runner: "TestRunner",
        quality_scorer: Optional["QualityScorer"],
        test_command: Optional[str],
    ) -> PatchTestResult:
        """Validate a single patch - apply, test, and score."""
        result = PatchTestResult(agent_id=output.agent_id)

        try:
            # Apply patch in memory first to get patched files
            apply_result = applier.apply_patch(output.output_text, original_files)

            if not apply_result.success:
                result.patch_applied = False
                result.apply_errors = apply_result.errors
                return result

            result.patch_applied = True

            # Create temporary checkout with patch applied
            patched_path, checkout_result = create_patched_checkout(
                repo_path, output.output_text
            )

            if patched_path is None:
                result.apply_errors = checkout_result.errors
                return result

            try:
                # Run tests
                test_result = await test_runner.run_tests(
                    patched_path,
                    test_command=test_command,
                    run_setup=True,
                )

                result.tests_passed = test_result.success
                result.tests_total = test_result.total
                result.tests_failed = test_result.failed
                result.test_duration_seconds = test_result.duration_seconds
                result.failure_signature = test_result.failure_signature

                # Quality scoring
                if quality_scorer:
                    quality_result = await quality_scorer.score(
                        patch=output.output_text,
                        patch_id=output.agent_id,
                        original_files=original_files,
                        patched_files=apply_result.patched_files,
                    )
                    result.quality_score = quality_result.overall_score
                    result.risk_flags = quality_result.risk_flags

            finally:
                # Clean up temporary checkout
                if patched_path and patched_path.exists():
                    shutil.rmtree(patched_path, ignore_errors=True)

        except Exception as e:
            logger.error(f"Patch validation failed for {output.agent_id}: {e}")
            result.apply_errors = [str(e)]

        return result

    def get_behavioral_groups(
        self,
        validation_result: PatchValidationResult,
    ) -> Dict[str, List[str]]:
        """
        Group patches by test behavior (passing vs failing, failure signature).

        Args:
            validation_result: Results from validate_and_score_patch_candidates

        Returns:
            Dict mapping behavior key to list of agent_ids
        """
        groups: Dict[str, List[str]] = {}

        for agent_id, test_result in validation_result.results.items():
            if not test_result.patch_applied:
                key = "apply_failed"
            elif test_result.tests_passed:
                key = "all_pass"
            else:
                # Group by failure signature (which tests failed)
                key = f"fail:{test_result.failure_signature}"

            if key not in groups:
                groups[key] = []
            groups[key].append(agent_id)

        return groups


def create_atlas_adapter(config: Optional[SwarmConfig] = None) -> Optional[AtlasAdapter]:
    """
    Create an Atlas adapter if available.

    Args:
        config: Swarm configuration

    Returns:
        AtlasAdapter instance or None if Atlas isn't available
    """
    if not ATLAS_AVAILABLE:
        logger.warning("Atlas not available, returning None")
        return None

    try:
        return AtlasAdapter(config)
    except Exception as e:
        logger.error(f"Failed to create Atlas adapter: {e}")
        return None


def is_atlas_available() -> bool:
    """Check if Atlas components are available."""
    return ATLAS_AVAILABLE
