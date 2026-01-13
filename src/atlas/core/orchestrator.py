"""Main orchestrator for ATLAS code generation pipeline.

Enhanced with:
- Test execution loop (apply patch → run tests → record outcomes)
- Behavioral clustering (cluster by test outcomes, not just similarity)
- Intelligent context extraction (replaces "first 10 files")
- Proper patch application for quality scoring
"""

import asyncio
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import git

from atlas.agents.agent_pool import AgentPoolManager, SwarmResult
from atlas.agents.micro_agent import AgentContext
from atlas.context.repo_analyzer import RepoAnalyzer, RepoContext, AnalysisConfig, FileRelevance
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
from atlas.verification.behavioral_clustering import (
    BehavioralClusterer,
    BehavioralClusteringResult,
    BehavioralSignature,
    VerificationResult,
    cluster_by_test_outcomes,
    select_best_patches,
)
from atlas.verification.clustering import calculate_diversity_score
from atlas.scout.repo_scout import RepoScout
from atlas.scout.repo_tools import RepoTools
from atlas.verification.patch_applier import PatchApplier, PatchParser, create_patched_checkout
from atlas.verification.patch_validator import PatchValidator
from atlas.verification.static_analysis import StaticAnalyzer
from atlas.verification.test_runner import TestRunner, TestResult
from atlas.voting.consensus import IncrementalVoter, VotingResult

logger = logging.getLogger(__name__)


class ATLASOrchestrator:
    """Main orchestrator for the ATLAS code generation pipeline.

    Coordinates:
    1. Repository cloning (with proper base_commit support)
    2. Intelligent context extraction (replaces "first 10 files")
    3. Parallel patch generation with diverse prompt styles
    4. Patch application and test execution (primary correctness oracle)
    5. Behavioral clustering (cluster by test outcomes)
    6. Similarity-based voting (for diversity analysis)
    7. Quality selection (when multiple patches pass tests)
    8. Result compilation

    Key improvements:
    - Tests are the PRIMARY correctness oracle, not patch similarity
    - Patches are actually applied and tested, not just validated structurally
    - Quality scoring uses actually patched files, not originals
    """

    def __init__(
        self,
        config: Config | None = None,
        enable_quality_selection: bool = True,
        enable_test_execution: bool = True,
        use_agentic: bool = True,
    ):
        """Initialize the orchestrator.

        Args:
            config: Optional Config instance
            enable_quality_selection: Whether to run quality selection on passing patches
            enable_test_execution: Whether to run tests on patches (HIGHLY RECOMMENDED)
            use_agentic: If True (default), agents autonomously search Context7 and web.
                         If False, use pre-fetched RAG (faster but less thorough).
        """
        self.config = config or get_config()
        self.enable_quality_selection = enable_quality_selection
        self.enable_test_execution = enable_test_execution
        self.use_agentic = use_agentic
        self.agent_pool = AgentPoolManager(config=self.config, use_agentic=use_agentic)
        self.voter = IncrementalVoter(
            k=self.config.voting_k,
            similarity_threshold=0.8,
        )
        self.patch_validator = PatchValidator()
        self.static_analyzer = StaticAnalyzer()
        self.patch_applier = PatchApplier()
        self.test_runner = TestRunner(timeout_seconds=300)
        self.repo_analyzer = RepoAnalyzer()
        self.repo_scout = RepoScout()
        self.behavioral_clusterer = BehavioralClusterer(prioritize_passing=True)

        # Quality selection pipeline (initialized lazily with LLM client)
        self._quality_pipeline: Optional[QualitySelectionPipeline] = None

        # State tracking
        self._tasks: dict[str, TaskResult] = {}
        self._active_task: str | None = None
        self._repo_path: Optional[Path] = None  # Current repo path for test execution
        self._repo_context: Optional[RepoContext] = None  # Intelligent context
        self._repo_report = None
        self._repo_tools: Optional[RepoTools] = None
        self._repo_files: Dict[str, str] = {}  # Cache for quality selection
        self._verification_results: Dict[str, VerificationResult] = {}  # Test results per patch

    async def solve(self, task: TaskSubmission) -> TaskResult:
        """Solve a coding task using multi-agent consensus with test verification.

        The key insight: tests are the PRIMARY correctness oracle, not patch similarity.
        Similarity clustering is demoted to a supporting role for diversity analysis.

        Flow:
        1. Clone repository (with proper base_commit support)
        2. Extract intelligent context (not just "first 10 files")
        3. Generate patches with diverse agents
        4. Apply patches and run tests (behavioral verification)
        5. Cluster by test outcomes (behavioral clustering)
        6. Run quality selection on PASSING patches only
        7. Return best patch with confidence based on test results

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
        self._verification_results = {}

        temp_dir = None

        try:
            # Phase 1: Clone repository (with proper base_commit support)
            trace.add_phase("cloning")
            result.status = TaskStatus.CLONING

            temp_dir, repo_path, repo_context = await self._clone_and_analyze(task, trace)

            if not repo_path:
                result.status = TaskStatus.FAILED
                result.error_message = "Failed to clone repository"
                return result

            self._repo_path = repo_path
            self._repo_context = repo_context
            self._repo_files = repo_context.files

            try:
                keywords = self._extract_keywords_for_scout(task.description)
                repo_report = await self.repo_scout.scan(
                    repo_path,
                    keywords=keywords,
                    component_names=[],
                )
                self._repo_report = repo_report
                self._repo_tools = RepoTools(
                    indexer=self.repo_scout.indexer,
                    report=repo_report,
                )
            except Exception as e:
                trace.warnings.append(f"RepoScout failed: {e}")
                self._repo_report = None
                self._repo_tools = None

            # Format repo content for agents
            repo_content = self._format_context_for_agents(repo_context)
            if self._repo_report:
                repo_content = (
                    f"{repo_content}\n\n"
                    f"{self._repo_report.to_context_pack(max_chars=2500)}"
                )

            # Phase 2: Generate patches with diverse agents
            trace.add_phase("generating")
            result.status = TaskStatus.GENERATING

            # Create agent context
            context = AgentContext(
                task=task,
                repository_content=repo_content,
                repo_tools=self._repo_tools,
            )

            # Run swarm and vote incrementally
            self.voter.reset()
            all_solutions: List[Solution] = []
            total_samples = 0
            max_batches = (task.max_samples + task.initial_samples - 1) // task.initial_samples

            for batch in range(max_batches):
                batch_size = task.initial_samples

                # Check cost limit
                if cost_breakdown.total >= task.max_cost_usd:
                    logger.warning("Cost limit reached")
                    break

                # Check time limit
                elapsed = time.time() - start_time
                if elapsed > task.timeout_minutes * 60:
                    logger.warning("Time limit reached")
                    break

                # Generate batch of solutions
                self.agent_pool.create_diverse_swarm(batch_size)
                swarm_result = await self.agent_pool.run_swarm(context)

                # Update costs
                cost_breakdown.add_model_cost("gemini", swarm_result.total_cost)
                result.cost_usd = cost_breakdown.total

                # Validate solutions (structural validation)
                trace.add_phase("validating")
                result.status = TaskStatus.VALIDATING
                validated_solutions = self._validate_solutions(swarm_result.solutions)
                all_solutions.extend(validated_solutions)

                # Record agent outputs
                for solution in validated_solutions:
                    trace.add_agent_output(
                        agent_id=solution.agent_id,
                        prompt_style=solution.prompt_style,
                        output=solution.patch[:500] if solution.patch else "",
                        tokens_used=solution.tokens_used,
                        cost=solution.cost,
                    )

                total_samples += len(validated_solutions)

                # Run similarity-based voting (for diversity tracking, NOT correctness)
                trace.add_phase("voting")
                result.status = TaskStatus.VOTING
                voting_result = self.voter.add_solutions(validated_solutions)

                trace.add_voting_round(
                    round_number=batch + 1,
                    clusters=voting_result.vote_counts,
                    winner=voting_result.winner.cluster_id if voting_result.winner else None,
                    consensus_reached=voting_result.consensus_reached,
                )

                # Early stopping based on similarity consensus (but we'll still verify with tests)
                if voting_result.consensus_reached:
                    logger.info(f"Similarity consensus reached after {total_samples} samples")
                    break

            # Phase 3: Test execution and behavioral clustering (THE PRIMARY ORACLE)
            valid_patches = [s for s in all_solutions if s.is_valid and s.patch]

            if self.enable_test_execution and valid_patches:
                trace.add_phase("test_execution")
                result.status = TaskStatus.VALIDATING
                logger.info(f"Running tests on {len(valid_patches)} valid patches")

                # Run tests on all valid patches
                test_results = await self._run_tests_on_patches(
                    patches=valid_patches,
                    repo_path=repo_path,
                    test_command=task.test_command,
                )

                # Behavioral clustering (cluster by test outcomes)
                trace.add_phase("behavioral_clustering")
                behavioral_result = cluster_by_test_outcomes(test_results)

                trace.add_phase("behavioral_clustering", {
                    "total_patches": behavioral_result.total_patches,
                    "passing_patches": behavioral_result.patches_all_pass,
                    "failing_patches": behavioral_result.patches_some_fail,
                    "error_patches": behavioral_result.patches_error,
                    "clusters_found": len(behavioral_result.clusters),
                })

                behavioral_voter = IncrementalVoter(
                    k=self.config.voting_k,
                    similarity_threshold=0.8,
                )
                behavioral_voting_result = behavioral_voter.add_solutions(valid_patches)

                # Select best patches from passing clusters
                if behavioral_result.has_passing_patches:
                    logger.info(f"{behavioral_result.patches_all_pass} patches pass all tests!")

                    # Get passing solutions
                    passing_patch_ids = set()
                    for cluster in behavioral_result.passing_clusters:
                        passing_patch_ids.update(cluster.patch_ids)

                    passing_solutions = [
                        s for s in valid_patches if s.agent_id in passing_patch_ids
                    ]

                    # Run quality selection on passing patches only
                    if self.enable_quality_selection and len(passing_solutions) > 1:
                        trace.add_phase("quality_selection")
                        quality_result = await self._run_quality_selection(
                            solutions=passing_solutions,
                            task=task,
                            repo_context=repo_context,
                            repo_path=repo_path,
                        )

                        if quality_result and quality_result.selection.best_patch_content:
                            result.patch = quality_result.selection.best_patch_content
                            result.confidence_score = max(
                                behavioral_result.confidence,
                                behavioral_voting_result.confidence_score,
                            )
                            logger.info(
                                "Quality selection chose: %s",
                                quality_result.selection.best_patch_id,
                            )
                        else:
                            # Use representative from largest passing cluster
                            best_cluster = behavioral_result.passing_clusters[0]
                            best_patch_id = best_cluster.representative_patch_id
                            best_solution = next(
                                (s for s in passing_solutions if s.agent_id == best_patch_id),
                                passing_solutions[0]
                            )
                            result.patch = best_solution.patch
                            result.confidence_score = max(
                                behavioral_result.confidence,
                                behavioral_voting_result.confidence_score,
                            )
                    else:
                        # Single passing patch or quality selection disabled
                        result.patch = passing_solutions[0].patch
                        result.confidence_score = max(
                            behavioral_result.confidence,
                            behavioral_voting_result.confidence_score,
                        )

                    result.consensus_reached = behavioral_voting_result.consensus_reached
                    result.status = TaskStatus.COMPLETED
                else:
                    # No patches pass all tests
                    logger.warning("No patches pass all tests")
                    trace.warnings.append("No patches passed all tests")

                    # Fall back to behavioral consensus on failing clusters
                    if behavioral_voting_result.winning_solution:
                        result.patch = behavioral_voting_result.winning_solution.patch
                        result.confidence_score = behavioral_voting_result.confidence_score * 0.3
                        result.consensus_reached = False
                        result.status = TaskStatus.COMPLETED
                        trace.warnings.append(
                            "Falling back to behavioral consensus "
                            "(no test-passing patches)"
                        )
                    else:
                        result.status = TaskStatus.FAILED
                        result.error_message = "No patches pass tests and no similarity consensus"
            else:
                # Test execution disabled - fall back to original behavior
                logger.info("Test execution disabled, using similarity-based selection")

                if voting_result.winning_solution:
                    # Run quality selection if enabled
                    if self.enable_quality_selection and len(valid_patches) > 1:
                        trace.add_phase("quality_selection")
                        quality_result = await self._run_quality_selection(
                            solutions=valid_patches,
                            task=task,
                            repo_context=repo_context,
                            repo_path=repo_path,
                        )

                        if quality_result and quality_result.selection.best_patch_content:
                            result.patch = quality_result.selection.best_patch_content
                            result.confidence_score = voting_result.confidence_score * 0.8
                        else:
                            result.patch = voting_result.winning_solution.patch
                            result.confidence_score = voting_result.confidence_score * 0.8
                    else:
                        result.patch = voting_result.winning_solution.patch
                        result.confidence_score = voting_result.confidence_score * 0.8

                    result.consensus_reached = voting_result.consensus_reached
                    result.status = TaskStatus.COMPLETED
                    trace.warnings.append("Test execution disabled - confidence reduced")
                else:
                    result.status = TaskStatus.FAILED
                    result.error_message = "No valid patches generated"

            # Finalize result
            result.samples_generated = total_samples
            result.votes_cast = total_samples
            result.models_used = list(set(s.model for s in all_solutions if s.model))

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error_message = f"Task timed out after {task.timeout_minutes} minutes"

        except Exception as e:
            logger.exception(f"Task {task.task_id} failed with error")
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            trace.errors.append(str(e))

        finally:
            # Cleanup
            result.duration_seconds = time.time() - start_time
            trace.completed_at = datetime.now()
            self._active_task = None
            self._repo_path = None
            self._repo_context = None
            self._repo_report = None
            self._repo_tools = None

            # Clean up temp directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp dir: {e}")

        return result

    async def _clone_and_analyze(
        self,
        task: TaskSubmission,
        trace: ExecutionTrace,
    ) -> Tuple[Optional[str], Optional[Path], Optional[RepoContext]]:
        """Clone the repository and extract intelligent context.

        This replaces the old "first 10 files" approach with:
        - Proper base_commit support (full clone if needed)
        - Intelligent file selection based on issue keywords
        - Symbol and import analysis

        Args:
            task: The task submission
            trace: Execution trace for logging

        Returns:
            Tuple of (temp_dir, repo_path, RepoContext) or (None, None, None) on failure
        """
        try:
            # Create temporary directory for clone (caller must clean up)
            temp_dir = tempfile.mkdtemp(prefix="atlas_repo_")
            repo_path = Path(temp_dir) / "repo"

            # Clone the repository
            logger.info(f"Cloning {task.repository_url}")

            # Determine clone depth
            # If base_commit is specified, we need a full clone (or fetch that commit)
            use_shallow = task.base_commit is None
            clone_depth = 1 if use_shallow else None

            # Run git clone in executor
            loop = asyncio.get_event_loop()

            if clone_depth:
                await loop.run_in_executor(
                    None,
                    lambda: git.Repo.clone_from(
                        task.repository_url,
                        repo_path,
                        branch=task.branch,
                        depth=clone_depth,
                    ),
                )
            else:
                # Full clone for base_commit support
                await loop.run_in_executor(
                    None,
                    lambda: git.Repo.clone_from(
                        task.repository_url,
                        repo_path,
                        branch=task.branch,
                    ),
                )

            # If a specific commit is requested, check it out
            if task.base_commit:
                repo = git.Repo(repo_path)
                try:
                    repo.git.checkout(task.base_commit)
                    logger.info(f"Checked out commit {task.base_commit}")
                except git.GitCommandError as e:
                    logger.warning(f"Failed to checkout {task.base_commit}: {e}")
                    trace.warnings.append(f"Could not checkout {task.base_commit}, using HEAD")

            # Use intelligent context extraction
            logger.info("Analyzing repository for relevant context")
            analysis_config = AnalysisConfig(
                max_files=15,
                max_file_size=50000,
                max_total_size=200000,
                include_tests=True,
                test_file_limit=3,
            )
            analyzer = RepoAnalyzer(analysis_config)

            repo_context = analyzer.analyze(
                repo_path=repo_path,
                issue_description=task.description,
                relevant_files=task.relevant_files,
            )

            trace.add_phase("cloning", {
                "files_scanned": repo_context.total_files_scanned,
                "files_selected": repo_context.files_selected,
                "primary_language": repo_context.primary_language,
            })

            logger.info(
                f"Selected {repo_context.files_selected} files from "
                f"{repo_context.total_files_scanned} scanned "
                f"(language: {repo_context.primary_language})"
            )

            return temp_dir, repo_path, repo_context

        except git.GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            trace.errors.append(f"Git clone failed: {e}")
            return None, None, None

        except Exception as e:
            logger.error(f"Repository extraction failed: {e}")
            trace.errors.append(f"Repository extraction failed: {e}")
            return None, None, None

    def _format_context_for_agents(self, repo_context: RepoContext) -> str:
        """Format repository context for agents.

        Args:
            repo_context: The analyzed repository context

        Returns:
            Formatted string with file contents and relevance info
        """
        parts = []
        parts.append(f"# Repository Context (Language: {repo_context.primary_language})")
        parts.append(
            f"# Files: {repo_context.files_selected} selected from "
            f"{repo_context.total_files_scanned} scanned"
        )
        parts.append("")

        # Sort files by relevance score
        sorted_files = sorted(
            repo_context.files.items(),
            key=lambda x: repo_context.file_relevance.get(x[0], FileRelevance(path=x[0])).score,
            reverse=True,
        )

        for path, content in sorted_files:
            # Normalize path to forward slashes for consistency with patches
            norm_path = path.replace("\\", "/")
            relevance = repo_context.file_relevance.get(path)
            if relevance:
                reasons = (
                    ", ".join(relevance.reasons[:3])
                    if relevance.reasons
                    else "auto-selected"
                )
                parts.append(f"# File: {norm_path} (Relevance: {relevance.score:.1f}, {reasons})")
            else:
                parts.append(f"# File: {norm_path}")
            parts.append(content)
            parts.append("")

        return "\n".join(parts)

    async def _run_tests_on_patches(
        self,
        patches: List[Solution],
        repo_path: Path,
        test_command: Optional[str] = None,
    ) -> Dict[str, TestResult]:
        """Apply patches and run tests on each.

        This is the PRIMARY correctness oracle. Patches that don't pass tests
        should not be selected regardless of similarity consensus.

        Args:
            patches: List of patches to test
            repo_path: Path to the original repository
            test_command: Optional explicit test command

        Returns:
            Dict of {patch_id: TestResult}
        """
        results = {}

        # Run tests in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent test runs

        async def test_patch(solution: Solution) -> Tuple[str, TestResult]:
            async with semaphore:
                patch_id = solution.agent_id
                patch_content = solution.patch

                logger.info(f"Testing patch {patch_id}")

                try:
                    # Create patched checkout
                    patched_path, apply_result = create_patched_checkout(
                        repo_path, patch_content
                    )

                    if not patched_path:
                        test_result = TestResult(
                            success=False,
                            execution_error=(
                                "Failed to apply patch: "
                                f"{'; '.join(apply_result.errors)}"
                            ),
                        )
                        solution.test_result = test_result
                        return patch_id, test_result

                    try:
                        # Run tests
                        test_result = await self.test_runner.run_tests(
                            patched_path,
                            test_command=test_command,
                        )
                        solution.test_result = test_result

                        logger.info(
                            f"Patch {patch_id}: {'PASS' if test_result.success else 'FAIL'} "
                            f"({test_result.passed}/{test_result.total} tests passed)"
                        )

                        # Store verification result
                        self._verification_results[patch_id] = VerificationResult(
                            patch_id=patch_id,
                            apply_success=True,
                            test_result=test_result,
                        )

                        return patch_id, test_result

                    finally:
                        # Clean up patched checkout
                        if patched_path and patched_path.exists():
                            shutil.rmtree(patched_path)

                except Exception as e:
                    logger.error(f"Error testing patch {patch_id}: {e}")
                    test_result = TestResult(
                        success=False,
                        execution_error=str(e),
                    )
                    solution.test_result = test_result
                    return patch_id, test_result

        # Run all tests
        tasks = [test_patch(solution) for solution in patches]
        test_results = await asyncio.gather(*tasks)

        for patch_id, result in test_results:
            results[patch_id] = result

        return results

    async def _run_quality_selection(
        self,
        solutions: List[Solution],
        task: TaskSubmission,
        repo_context: RepoContext,
        repo_path: Path,
    ) -> Optional[QualitySelectionResult]:
        """Run quality selection on valid solutions.

        IMPORTANT: This now properly applies patches before scoring,
        fixing the bug where original files were used for quality scoring.

        Args:
            solutions: List of valid solutions to choose from
            task: The task submission
            repo_context: Repository context with original files

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

            original_files = self._load_original_files_for_patches(
                repo_path=repo_path,
                patches=patches.values(),
            )

            # Build ACTUALLY patched files map (FIX: no longer uses original files!)
            patched_files_map = {}
            for patch_id, patch_content in patches.items():
                try:
                    # Apply the patch to get actual patched files
                    apply_result = self.patch_applier.apply_patch(
                        patch_content,
                        original_files,
                    )

                    if apply_result.success:
                        patched_files_map[patch_id] = apply_result.patched_files
                    else:
                        # If patch fails to apply, use original with warning
                        logger.warning(
                            f"Patch {patch_id} failed to apply for quality scoring: "
                            f"{'; '.join(apply_result.errors)}"
                        )
                        patched_files_map[patch_id] = dict(original_files)

                except Exception as e:
                    logger.warning(f"Error applying patch {patch_id}: {e}")
                    patched_files_map[patch_id] = dict(original_files)

            # Format context for quality selection
            context_parts = []
            for path, content in list(repo_context.files.items())[:5]:
                context_parts.append(f"# {path}\n{content[:2000]}")
            context_code = "\n\n".join(context_parts)

            # Run quality selection
            result = await self._quality_pipeline.select_best_patch(
                patches=patches,
                issue_description=task.description,
                original_files=original_files,
                patched_files_map=patched_files_map,
                context_code=context_code,
            )

            return result

        except Exception as e:
            logger.error(f"Quality selection failed: {e}")
            return None

    def _load_original_files_for_patches(
        self,
        repo_path: Path,
        patches: List[str] | Iterable[str],
    ) -> Dict[str, str]:
        parser = PatchParser()
        file_paths: list[str] = []

        for patch in patches:
            for file_patch in parser.parse(patch):
                file_paths.append(file_patch.target_path)

        unique_paths = list(dict.fromkeys(file_paths))
        originals: Dict[str, str] = {}

        for path in unique_paths:
            full_path = repo_path / path
            if full_path.exists():
                originals[path] = full_path.read_text(encoding="utf-8", errors="replace")
            else:
                originals[path] = ""

        return originals

    def _extract_keywords_for_scout(self, description: str) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "when",
            "where", "what", "which", "should", "could", "would", "there", "their",
            "about", "these", "those", "have", "has", "had", "been", "being",
            "will", "can", "just", "only", "your", "you", "our", "are", "was",
            "were", "use", "using", "used", "please", "fix", "bug", "issue",
            "error", "problem", "feature", "request", "implement", "add", "update",
            "change", "modify",
        }

        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", description.lower())
        keywords = []
        seen = set()
        for word in words:
            if word in stopwords or len(word) < 3:
                continue
            if word in seen:
                continue
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= 12:
                break

        return keywords

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
                logger.warning(f"Agent {solution.agent_id}: Empty patch")
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

            # Verify patch applies cleanly against the repo
            if self._repo_path:
                apply_result = self.patch_validator.try_apply(
                    solution.patch,
                    self._repo_path,
                    dry_run=True,
                )
                if not apply_result.can_apply:
                    solution.is_valid = False
                    solution.validation_errors.extend(apply_result.errors)
                solution.validation_errors.extend(apply_result.warnings)

            # Log validation result
            if solution.is_valid:
                logger.info(f"Agent {solution.agent_id}: VALID")
            else:
                logger.warning(f"Agent {solution.agent_id}: INVALID - {solution.validation_errors[:3]}")

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
