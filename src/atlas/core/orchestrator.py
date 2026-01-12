"""Main orchestrator for ATLAS code generation pipeline."""

import asyncio
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import git

from atlas.agents.agent_pool import AgentPoolManager, SwarmResult
from atlas.agents.micro_agent import AgentContext
from atlas.core.config import Config, get_config
from atlas.core.task import (
    CostBreakdown,
    ExecutionTrace,
    Solution,
    TaskResult,
    TaskStatus,
    TaskSubmission,
)
from atlas.quality.pipeline import (
    QualitySelectionConfig,
    QualitySelectionPipeline,
    QualitySelectionResult,
)
from atlas.verification.clustering import calculate_diversity_score
from atlas.verification.patch_validator import PatchValidator
from atlas.verification.static_analysis import StaticAnalyzer
from atlas.voting.consensus import IncrementalVoter, VotingResult

logger = logging.getLogger(__name__)


class ATLASOrchestrator:
    """Main orchestrator for the ATLAS code generation pipeline.

    Coordinates:
    1. Repository cloning
    2. Context gathering (RAG) - autonomous via agentic agents
    3. Parallel patch generation with diverse prompt styles
    4. Validation and clustering
    5. Consensus voting (first-to-ahead-by-K)
    6. Quality selection (when multiple patches pass)
    7. Result compilation
    """

    def __init__(
        self,
        config: Config | None = None,
        enable_quality_selection: bool = True,
        use_agentic: bool = True,
    ):
        """Initialize the orchestrator.

        Args:
            config: Optional Config instance
            enable_quality_selection: Whether to run quality selection on passing patches
            use_agentic: If True (default), agents autonomously search Context7 and web.
                         If False, use pre-fetched RAG (faster but less thorough).
        """
        self.config = config or get_config()
        self.enable_quality_selection = enable_quality_selection
        self.use_agentic = use_agentic
        self.agent_pool = AgentPoolManager(config=self.config, use_agentic=use_agentic)
        self.voter = IncrementalVoter(
            k=self.config.voting_k,
            similarity_threshold=0.8,
        )
        self.patch_validator = PatchValidator()
        self.static_analyzer = StaticAnalyzer()

        # Quality selection pipeline (initialized lazily with LLM client)
        self._quality_pipeline: Optional[QualitySelectionPipeline] = None

        # State tracking
        self._tasks: dict[str, TaskResult] = {}
        self._active_task: str | None = None
        self._repo_files: Dict[str, str] = {}  # Cache for quality selection

    async def solve(self, task: TaskSubmission) -> TaskResult:
        """Solve a coding task using multi-agent consensus.

        Args:
            task: The task submission

        Returns:
            TaskResult with the solution
        """
        start_time = time.time()
        trace = ExecutionTrace()
        cost_breakdown = CostBreakdown()

        # Initialize result
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.PENDING,
            execution_trace=trace,
            cost_breakdown=cost_breakdown,
        )
        self._tasks[task.task_id] = result
        self._active_task = task.task_id

        try:
            # Phase 1: Clone repository
            trace.add_phase("cloning")
            result.status = TaskStatus.CLONING
            repo_content = await self._clone_and_extract(task, trace)

            if not repo_content:
                result.status = TaskStatus.FAILED
                result.error_message = "Failed to clone repository"
                return result

            # Phase 2: Generate patches with diverse agents
            trace.add_phase("generating")
            result.status = TaskStatus.GENERATING

            # Create agent context
            context = AgentContext(
                task=task,
                repository_content=repo_content,
            )

            # Run swarm and vote incrementally
            self.voter.reset()
            total_samples = 0
            max_batches = (task.max_samples + task.initial_samples - 1) // task.initial_samples

            for batch in range(max_batches):
                batch_size = task.initial_samples if batch == 0 else task.initial_samples

                # Check cost limit
                if cost_breakdown.total >= task.max_cost_usd:
                    logger.warning("Cost limit reached")
                    break

                # Generate batch of solutions
                self.agent_pool.create_diverse_swarm(batch_size)
                swarm_result = await self.agent_pool.run_swarm(context)

                # Update costs
                cost_breakdown.add_model_cost("gemini", swarm_result.total_cost)
                result.cost_usd = cost_breakdown.total

                # Validate solutions
                trace.add_phase("validating")
                result.status = TaskStatus.VALIDATING
                validated_solutions = self._validate_solutions(swarm_result.solutions)

                # Record agent outputs
                for solution in validated_solutions:
                    trace.add_agent_output(
                        agent_id=solution.agent_id,
                        prompt_style=solution.prompt_style,
                        output=solution.patch[:500] if solution.patch else "",
                        tokens_used=solution.tokens_used,
                        cost=solution.cost,
                    )

                # Vote on solutions
                trace.add_phase("voting")
                result.status = TaskStatus.VOTING
                voting_result = self.voter.add_solutions(validated_solutions)
                total_samples += len(validated_solutions)

                trace.add_voting_round(
                    round_number=batch + 1,
                    clusters=voting_result.vote_counts,
                    winner=voting_result.winner.cluster_id if voting_result.winner else None,
                    consensus_reached=voting_result.consensus_reached,
                )

                # Check for consensus
                if voting_result.consensus_reached:
                    logger.info(f"Consensus reached after {total_samples} samples")
                    break

            # Finalize result
            result.samples_generated = total_samples
            result.votes_cast = total_samples

            # Get all valid solutions for quality selection
            valid_solutions = [s for s in self.voter._all_solutions if s.is_valid and s.patch]

            if self.voter.has_consensus():
                winner = self.voter.get_winner()
                if winner:
                    # If we have multiple valid patches, run quality selection
                    if self.enable_quality_selection and len(valid_solutions) > 1:
                        trace.add_phase("quality_selection")
                        result.status = TaskStatus.VALIDATING

                        quality_result = await self._run_quality_selection(
                            solutions=valid_solutions,
                            task=task,
                            repo_content=repo_content,
                        )

                        if quality_result and quality_result.selection.best_patch_content:
                            result.patch = quality_result.selection.best_patch_content
                            # Boost confidence if quality selection agrees with voting
                            if quality_result.selection.best_patch_id == winner.agent_id:
                                result.confidence_score = min(1.0, voting_result.confidence_score * 1.1)
                            else:
                                result.confidence_score = voting_result.confidence_score
                            logger.info(f"Quality selection chose: {quality_result.selection.best_patch_id}")
                            logger.info(f"Reason: {quality_result.selection.selection_reason}")
                        else:
                            result.patch = winner.patch
                            result.confidence_score = voting_result.confidence_score
                    else:
                        result.patch = winner.patch
                        result.confidence_score = voting_result.confidence_score

                    result.consensus_reached = True
                    result.status = TaskStatus.COMPLETED
                    result.models_used = list(set(
                        s.model for s in self.voter._all_solutions if s.model
                    ))
            else:
                # No consensus - run quality selection on valid patches
                if self.enable_quality_selection and len(valid_solutions) > 1:
                    trace.add_phase("quality_selection")
                    result.status = TaskStatus.VALIDATING

                    quality_result = await self._run_quality_selection(
                        solutions=valid_solutions,
                        task=task,
                        repo_content=repo_content,
                    )

                    if quality_result and quality_result.selection.best_patch_content:
                        result.patch = quality_result.selection.best_patch_content
                        result.confidence_score = voting_result.confidence_score * 0.7
                        result.consensus_reached = False
                        result.status = TaskStatus.COMPLETED
                        result.models_used = list(set(
                            s.model for s in self.voter._all_solutions if s.model
                        ))
                        logger.info(f"Quality selection chose (no consensus): {quality_result.selection.best_patch_id}")
                    else:
                        # Fallback to best effort
                        best_effort = self.voter.get_best_effort()
                        if best_effort:
                            result.patch = best_effort.patch
                            result.confidence_score = voting_result.confidence_score * 0.5
                            result.consensus_reached = False
                            result.status = TaskStatus.COMPLETED
                            result.models_used = list(set(
                                s.model for s in self.voter._all_solutions if s.model
                            ))
                        else:
                            result.status = TaskStatus.FAILED
                            result.error_message = "No valid patches generated"
                else:
                    # Quality selection disabled or only one valid solution
                    best_effort = self.voter.get_best_effort()
                    if best_effort:
                        result.patch = best_effort.patch
                        result.confidence_score = voting_result.confidence_score * 0.5
                        result.consensus_reached = False
                        result.status = TaskStatus.COMPLETED
                        result.models_used = list(set(
                            s.model for s in self.voter._all_solutions if s.model
                        ))
                    else:
                        result.status = TaskStatus.FAILED
                        result.error_message = "No valid patches generated"

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error_message = f"Task timed out after {task.timeout_minutes} minutes"

        except Exception as e:
            logger.exception(f"Task {task.task_id} failed with error")
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            trace.errors.append(str(e))

        finally:
            # Record completion
            result.duration_seconds = time.time() - start_time
            trace.completed_at = datetime.now()
            self._active_task = None

        return result

    async def _clone_and_extract(
        self,
        task: TaskSubmission,
        trace: ExecutionTrace,
    ) -> str:
        """Clone the repository and extract relevant code.

        Args:
            task: The task submission
            trace: Execution trace for logging

        Returns:
            Extracted code content as a string
        """
        try:
            # Create temporary directory for clone
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / "repo"

                # Clone the repository
                logger.info(f"Cloning {task.repository_url}")

                # Run git clone in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: git.Repo.clone_from(
                        task.repository_url,
                        repo_path,
                        branch=task.branch,
                        depth=1,  # Shallow clone for speed
                    ),
                )

                # If a specific commit is requested, check it out
                if task.base_commit:
                    repo = git.Repo(repo_path)
                    repo.git.checkout(task.base_commit)

                # Extract relevant files
                content = await self._extract_relevant_code(
                    repo_path,
                    task.relevant_files,
                )

                trace.add_phase("cloning", {"files_extracted": len(content.split("\n"))})
                return content

        except git.GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            trace.errors.append(f"Git clone failed: {e}")
            return ""

        except Exception as e:
            logger.error(f"Repository extraction failed: {e}")
            trace.errors.append(f"Repository extraction failed: {e}")
            return ""

    async def _extract_relevant_code(
        self,
        repo_path: Path,
        relevant_files: list[str] | None,
    ) -> str:
        """Extract relevant code from the repository.

        Args:
            repo_path: Path to the cloned repository
            relevant_files: Optional list of files to focus on

        Returns:
            Combined code content
        """
        content_parts = []

        if relevant_files:
            # Extract specified files
            for file_path in relevant_files:
                full_path = repo_path / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text()
                        content_parts.append(f"# File: {file_path}\n{content}")
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")
        else:
            # Extract Python files (limited for context window)
            python_files = list(repo_path.rglob("*.py"))[:10]  # Limit to 10 files

            for file_path in python_files:
                try:
                    rel_path = file_path.relative_to(repo_path)
                    content = file_path.read_text()
                    # Limit file size
                    if len(content) > 10000:
                        content = content[:10000] + "\n# ... (truncated)"
                    content_parts.append(f"# File: {rel_path}\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        return "\n\n".join(content_parts)

    async def _run_quality_selection(
        self,
        solutions: list[Solution],
        task: TaskSubmission,
        repo_content: str,
    ) -> Optional[QualitySelectionResult]:
        """Run quality selection on valid solutions.

        Args:
            solutions: List of valid solutions to choose from
            task: The task submission
            repo_content: Repository content for context

        Returns:
            QualitySelectionResult or None if selection fails
        """
        try:
            # Build patches dict
            patches = {s.agent_id: s.patch for s in solutions if s.patch}

            if not patches:
                return None

            # Initialize quality pipeline with LLM client if not already done
            if self._quality_pipeline is None:
                from atlas.agents.gemini_client import GeminiClient

                llm_client = GeminiClient(self.config)
                quality_config = QualitySelectionConfig(
                    enable_llm_review=True,
                    enable_tournament=len(patches) >= 3,  # Only tournament with 3+ patches
                    max_patches_for_tournament=6,
                )
                self._quality_pipeline = QualitySelectionPipeline(
                    config=quality_config,
                    llm_client=llm_client,
                )

            # Build original files dict from repo content
            original_files = self._parse_repo_content(repo_content)

            # Build patched files map (simplified - in production would apply patches)
            patched_files_map = {}
            for patch_id in patches.keys():
                # For now, just use original files as base
                # In production, would properly apply each patch
                patched_files_map[patch_id] = dict(original_files)

            # Run quality selection
            result = await self._quality_pipeline.select_best_patch(
                patches=patches,
                issue_description=task.description,
                original_files=original_files,
                patched_files_map=patched_files_map,
                context_code=repo_content[:5000],  # Limit context size
            )

            return result

        except Exception as e:
            logger.error(f"Quality selection failed: {e}")
            return None

    def _parse_repo_content(self, repo_content: str) -> Dict[str, str]:
        """Parse repository content string into file dict.

        Args:
            repo_content: Combined repository content with file markers

        Returns:
            Dict of {filepath: content}
        """
        files = {}
        current_file = None
        current_content = []

        for line in repo_content.split("\n"):
            if line.startswith("# File: "):
                # Save previous file
                if current_file:
                    files[current_file] = "\n".join(current_content)

                # Start new file
                current_file = line[8:].strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last file
        if current_file:
            files[current_file] = "\n".join(current_content)

        return files

    def _validate_solutions(self, solutions: list[Solution]) -> list[Solution]:
        """Validate solutions using static analysis.

        Args:
            solutions: List of solutions to validate

        Returns:
            List of validated solutions (with validation status updated)
        """
        validated = []

        for solution in solutions:
            if not solution.patch:
                solution.is_valid = False
                solution.validation_errors.append("Empty patch")
                validated.append(solution)
                continue

            # Validate patch structure
            patch_result = self.patch_validator.validate(solution.patch)

            if not patch_result.is_valid:
                solution.is_valid = False
                solution.validation_errors.extend(patch_result.errors)

            # Run static analysis on patch
            analysis_result = self.static_analyzer.analyze_patch(solution.patch)

            if not analysis_result.is_valid:
                solution.is_valid = False
                solution.validation_errors.extend(analysis_result.errors)

            # Add warnings (don't affect validity)
            solution.validation_errors.extend(analysis_result.warnings)
            solution.validation_errors.extend(patch_result.warnings)

            validated.append(solution)

        return validated

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a task.

        Args:
            task_id: The task ID

        Returns:
            Dictionary with task status
        """
        if task_id not in self._tasks:
            return {"status": "not_found", "task_id": task_id}

        result = self._tasks[task_id]

        return {
            "task_id": task_id,
            "status": result.status.value,
            "is_active": self._active_task == task_id,
            "samples_generated": result.samples_generated,
            "cost_usd": result.cost_usd,
            "duration_seconds": result.duration_seconds,
        }

    def get_task_result(self, task_id: str) -> TaskResult | None:
        """Get the full result of a task.

        Args:
            task_id: The task ID

        Returns:
            TaskResult or None if not found
        """
        return self._tasks.get(task_id)


# Singleton instance
_orchestrator: ATLASOrchestrator | None = None


def get_orchestrator() -> ATLASOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ATLASOrchestrator()
    return _orchestrator
