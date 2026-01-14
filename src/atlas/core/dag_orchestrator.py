"""Task DAG orchestrator for multi-agent decomposition and assembly."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import git

from atlas.agents.agent_pool import AgentPoolManager
from atlas.agents.micro_agent import AgentContext
from atlas.core.config import Config, get_config
from atlas.core.task import TaskSubmission
from atlas.core.task_dag import TaskDAG, TaskSpec
from atlas.core.task_decomposer import TaskDecomposer, TaskDecomposerConfig
from atlas.quality.pipeline import QualitySelectionConfig, QualitySelectionPipeline
from atlas.scout.repo_scout import RepoScout
from atlas.scout.repo_tools import RepoTools
from atlas.verification.ownership import OwnershipValidator
from atlas.verification.patch_applier import PatchApplier, PatchApplyResult, PatchParser
from atlas.verification.patch_composer import diff_files
from atlas.verification.patch_validator import PatchValidator
from atlas.verification.static_analysis import StaticAnalyzer
from atlas.verification.oracles import OracleResult, OracleRunner

logger = logging.getLogger(__name__)


@dataclass
class TaskDAGSubmission:
    """Submission for a DAG-based task."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    repository_url: str = ""
    repo_path: Path | None = None
    branch: str = "main"
    base_commit: str | None = None
    max_cost_usd: float = 10.0
    timeout_minutes: int = 60
    max_tasks: int = 12
    keywords: list[str] = field(default_factory=list)
    component_names: list[str] = field(default_factory=list)
    test_command: str | None = None
    review_only: bool = False
    dag_override: dict | list | str | None = None


@dataclass
class TaskExecutionConfig:
    """Execution config for DAG orchestration."""

    agents_per_task: int = 6
    top_k_per_task: int = 3
    beam_width: int = 4
    max_context_chars: int = 12000
    max_file_chars: int = 2000
    max_files_per_task: int = 12
    enable_quality_selection: bool = True
    run_oracles: bool = True
    fail_on_oracle: bool = False


@dataclass
class TaskCandidate:
    """Candidate patch for a task."""

    task_id: str
    candidate_id: str
    patch: str
    score: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class TaskExecutionResult:
    """Result for an executed task."""

    task_id: str
    selected_patch_id: str = ""
    selected_patch: str = ""
    candidates: list[TaskCandidate] = field(default_factory=list)
    oracles: list[OracleResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


@dataclass
class AssemblyBranch:
    """Intermediate branch in beam search assembly."""

    patch_ids: list[str] = field(default_factory=list)
    patches: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class AssemblyResult:
    """Result of assembling task patches."""

    success: bool
    combined_patch: str = ""
    selected_patch_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class TaskDAGResult:
    """Result from DAG orchestration."""

    task_id: str
    status: str
    dag: TaskDAG | None = None
    task_results: dict[str, TaskExecutionResult] = field(default_factory=dict)
    final_patch: str | None = None
    assembly: AssemblyResult | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class RepoState:
    """Mutable repo state based on applying patches in memory."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._cache: dict[str, str] = {}
        self._parser = PatchParser()
        self._applier = PatchApplier()

    def get_files(self, paths: Iterable[str]) -> dict[str, str]:
        files: dict[str, str] = {}
        for path in paths:
            if path not in self._cache:
                self._cache[path] = self._read_file(path)
            files[path] = self._cache[path]
        return files

    def preview_apply(self, patch: str) -> PatchApplyResult:
        target_files = self._files_from_patch(patch)
        originals = self.get_files(target_files)
        return self._applier.apply_patch(patch, originals)

    def apply_patch(self, patch: str) -> PatchApplyResult:
        result = self.preview_apply(patch)
        if result.success:
            for path in result.files_deleted:
                self._cache.pop(path, None)
            for path, content in result.patched_files.items():
                self._cache[path] = content
        return result

    def _read_file(self, path: str) -> str:
        target = self.repo_path / path
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8", errors="replace")

    def _files_from_patch(self, patch: str) -> list[str]:
        file_patches = self._parser.parse(patch)
        return [fp.target_path for fp in file_patches]


class TaskDAGOrchestrator:
    """Orchestrates task decomposition, swarms, and assembly."""

    def __init__(
        self,
        config: Config | None = None,
        execution_config: TaskExecutionConfig | None = None,
        use_agentic: bool = True,
    ):
        self.config = config or get_config()
        self.execution_config = execution_config or TaskExecutionConfig()
        self.agent_pool = AgentPoolManager(config=self.config, use_agentic=use_agentic)
        self.repo_scout = RepoScout()
        self.decomposer = TaskDecomposer(
            llm_client=None,
            config=TaskDecomposerConfig(),
        )
        self.patch_validator = PatchValidator()
        self.static_analyzer = StaticAnalyzer()
        self.ownership_validator = OwnershipValidator()
        self.oracle_runner = OracleRunner()
        self._quality_pipeline: QualitySelectionPipeline | None = None
        self._repo_files: list[str] = []

    async def solve(self, submission: TaskDAGSubmission) -> TaskDAGResult:
        start_time = time.time()
        result = TaskDAGResult(task_id=submission.task_id, status="pending")

        try:
            repo_path = await self._resolve_repo(submission)
            scout_report = await self.repo_scout.scan(
                repo_path,
                keywords=submission.keywords,
                component_names=submission.component_names,
            )
            repo_tools = RepoTools(indexer=self.repo_scout.indexer, report=scout_report)
            # Normalize paths to forward slashes for consistency with patches
            self._repo_files = [file_info.path.replace("\\", "/") for file_info in scout_report.codebase_index.files]

            self.decomposer = TaskDecomposer(
                llm_client=self._maybe_llm(),
                config=TaskDecomposerConfig(max_tasks=submission.max_tasks),
            )
            dag = await self.decomposer.decompose(
                submission.description,
                scout_report,
                keywords=submission.keywords,
                component_names=submission.component_names,
                dag_override=submission.dag_override,
            )
            result.dag = dag
            logger.info(f"Decomposed into {len(dag.tasks)} tasks")

            dag_errors = dag.validate()
            if dag_errors:
                logger.warning(f"DAG validation errors: {dag_errors}")
                result.errors.extend(dag_errors)
                if submission.review_only:
                    result.status = "needs_review"
                    return result
                result.status = "failed"
                return result

            if submission.review_only:
                result.status = "needs_review"
                return result

            topo_order = dag.topological_order()
            logger.info(f"Topological order: {[t.task_id for t in topo_order]}")

            base_state = RepoState(repo_path)
            working_state = RepoState(repo_path)

            candidates_by_task: dict[str, list[TaskCandidate]] = {}
            task_results: dict[str, TaskExecutionResult] = {}

            for task in topo_order:
                logger.info(f"Executing task: {task.task_id}")
                task_result = await self._execute_task(
                    task=task,
                    submission=submission,
                    repo_state=working_state,
                    repo_tools=repo_tools,
                    dependencies=task_results,
                )
                task_results[task.task_id] = task_result
                result.cost_usd += task_result.cost_usd
                if result.cost_usd >= submission.max_cost_usd:
                    result.status = "failed"
                    result.errors.append("Cost limit exceeded")
                    return result
                candidates_by_task[task.task_id] = self._select_top_k(task_result)

                if task_result.selected_patch:
                    apply_result = working_state.apply_patch(task_result.selected_patch)
                    if not apply_result.success:
                        task_result.errors.extend(apply_result.errors)
                        result.errors.extend(apply_result.errors)
                        result.status = "failed"
                        return result

            result.task_results = task_results

            assembly = await self._assemble(
                dag,
                candidates_by_task,
                base_state,
            )
            result.assembly = assembly
            result.final_patch = assembly.combined_patch if assembly.success else None
            result.status = "completed" if assembly.success else "failed"

        except Exception as exc:
            logger.exception("DAG orchestration failed")
            result.status = "failed"
            result.errors.append(str(exc))

        result.duration_seconds = time.time() - start_time
        return result

    async def _execute_task(
        self,
        task: TaskSpec,
        submission: TaskDAGSubmission,
        repo_state: RepoState,
        repo_tools: RepoTools,
        dependencies: dict[str, TaskExecutionResult],
    ) -> TaskExecutionResult:
        task_result = TaskExecutionResult(task_id=task.task_id)
        repository_content = self._build_repository_content(task, repo_state)
        additional_context = self._build_additional_context(task, dependencies)

        task_submission = TaskSubmission(
            description=task.description,
            repository_url=submission.repository_url or str(submission.repo_path or ""),
            branch=submission.branch,
            base_commit=submission.base_commit,
            relevant_files=self._ownership_files(task),
            test_command=submission.test_command,
            max_cost_usd=submission.max_cost_usd,
            timeout_minutes=submission.timeout_minutes,
        )

        context = AgentContext(
            task=task_submission,
            repository_content=repository_content,
            additional_context=additional_context,
            repo_tools=repo_tools,
        )

        self.agent_pool.create_diverse_swarm(self.execution_config.agents_per_task)
        swarm_result = await self.agent_pool.run_swarm(context)
        task_result.cost_usd = swarm_result.total_cost

        candidates = self._evaluate_candidates(task, swarm_result.solutions, repo_state)

        if not candidates:
            task_result.errors.append("No valid candidates for task")
            return task_result

        await self._score_candidates(task, candidates, repo_state)
        selected = max(candidates, key=lambda c: c.score)
        task_result.selected_patch_id = selected.candidate_id
        task_result.selected_patch = selected.patch
        task_result.candidates = candidates

        if task.oracles and self.execution_config.run_oracles:
            oracle_results = await self.oracle_runner.run(
                repo_state.repo_path,
                selected.patch,
                task.oracles,
            )
            task_result.oracles = oracle_results
            if self.execution_config.fail_on_oracle:
                failed = [o for o in oracle_results if not o.success]
                if failed:
                    task_result.errors.append("Oracle checks failed")

        return task_result

    def _ownership_files(self, task: TaskSpec) -> list[str]:
        files: list[str] = []
        files.extend(task.ownership.allowed_files)
        files.extend(task.ownership.allowed_dirs)
        files.extend(task.ownership.allowed_globs)
        return list(dict.fromkeys([f for f in files if f]))

    def _build_repository_content(self, task: TaskSpec, repo_state: RepoState) -> str:
        files = self._select_files_for_context(task, repo_state)
        logger.info(f"Selected files for context: {files}")
        logger.info(f"Available repo files: {self._repo_files[:10]}")  # First 10
        parts: list[str] = []
        total = 0

        for path in files:
            # Normalize path to forward slashes
            norm_path = path.replace("\\", "/")
            content = repo_state.get_files([path]).get(path, "")
            if not content:
                # Try with normalized path
                content = repo_state.get_files([norm_path]).get(norm_path, "")
            snippet = content[: self.execution_config.max_file_chars]
            chunk = f"# File: {norm_path}\n{snippet}"
            if total + len(chunk) > self.execution_config.max_context_chars:
                break
            parts.append(chunk)
            total += len(chunk)

        return "\n\n".join(parts)

    def _select_files_for_context(self, task: TaskSpec, repo_state: RepoState) -> list[str]:
        ownership = task.ownership
        selected = []

        if ownership.allowed_files:
            selected.extend(ownership.allowed_files)

        if ownership.allowed_dirs:
            for allowed_dir in ownership.allowed_dirs:
                directory = repo_state.repo_path / allowed_dir
                if directory.exists():
                    for file_path in directory.rglob("*"):
                        if file_path.is_file():
                            rel_path = str(file_path.relative_to(repo_state.repo_path))
                            selected.append(rel_path)
                            if len(selected) >= self.execution_config.max_files_per_task * 2:
                                break
                if len(selected) >= self.execution_config.max_files_per_task * 2:
                    break

        if ownership.allowed_globs:
            for path in self._repo_files:
                for pattern in ownership.allowed_globs:
                    # Handle **/* specially - it should match all files
                    if pattern == "**/*" or pattern == "*":
                        selected.append(path)
                    elif Path(path).match(pattern):
                        selected.append(path)

        deduped = list(dict.fromkeys(selected))
        return deduped[: self.execution_config.max_files_per_task]

    def _build_additional_context(
        self,
        task: TaskSpec,
        dependencies: dict[str, TaskExecutionResult],
    ) -> str:
        parts = []
        parts.append("## Task Contract")
        parts.append(task.contract)
        parts.append("")

        parts.append("## Ownership")
        parts.append(f"Allowed files: {task.ownership.allowed_files}")
        parts.append(f"Allowed dirs: {task.ownership.allowed_dirs}")
        parts.append(f"Allowed globs: {task.ownership.allowed_globs}")
        parts.append("")

        if task.oracles:
            parts.append("## Oracles")
            for oracle in task.oracles:
                parts.append(f"- {oracle.oracle_type.value}: {oracle.command or 'auto-detect'}")
            parts.append("")

        if task.dependencies:
            parts.append("## Dependency Outputs")
            for dep_id in task.dependencies:
                dep_result = dependencies.get(dep_id)
                if dep_result:
                    parts.append(f"- {dep_id}: selected {dep_result.selected_patch_id}")
            parts.append("")

        return "\n".join(parts)

    def _evaluate_candidates(
        self,
        task: TaskSpec,
        solutions: list,
        repo_state: RepoState,
    ) -> list[TaskCandidate]:
        candidates: list[TaskCandidate] = []

        for solution in solutions:
            if not solution.patch:
                logger.debug(f"Agent {solution.agent_id}: No patch generated")
                continue

            candidate = TaskCandidate(
                task_id=task.task_id,
                candidate_id=solution.agent_id,
                patch=solution.patch,
            )

            patch_result = self.patch_validator.validate(solution.patch)
            if not patch_result.is_valid:
                candidate.is_valid = False
                candidate.validation_errors.extend(patch_result.errors)
                logger.info(f"Agent {solution.agent_id}: Patch validation failed: {patch_result.errors}")
            candidate.warnings.extend(patch_result.warnings)

            analysis_result = self.static_analyzer.analyze_patch(solution.patch)
            if not analysis_result.is_valid:
                candidate.is_valid = False
                candidate.validation_errors.extend(analysis_result.errors)
                logger.info(f"Agent {solution.agent_id}: Static analysis failed: {analysis_result.errors}")
            candidate.warnings.extend(analysis_result.warnings)

            ownership_result = self.ownership_validator.validate(solution.patch, task.ownership)
            if not ownership_result.is_valid:
                candidate.is_valid = False
                candidate.validation_errors.extend(ownership_result.errors)
                logger.info(f"Agent {solution.agent_id}: Ownership validation failed: {ownership_result.errors}")

            apply_result = repo_state.preview_apply(solution.patch)
            if not apply_result.success:
                candidate.is_valid = False
                candidate.validation_errors.extend(apply_result.errors)
                logger.info(f"Agent {solution.agent_id}: Patch apply failed: {apply_result.errors}")
                # Log the patch target files for debugging
                logger.debug(f"Patch targets: {self._union_touched_files([solution.patch])}")

            if candidate.is_valid:
                logger.info(f"Agent {solution.agent_id}: Valid candidate!")
                candidates.append(candidate)
            else:
                logger.info(f"Agent {solution.agent_id}: REJECTED - {candidate.validation_errors}")

        return candidates

    async def _score_candidates(
        self,
        task: TaskSpec,
        candidates: list[TaskCandidate],
        repo_state: RepoState,
    ) -> None:
        if not candidates:
            return

        if not self.execution_config.enable_quality_selection or len(candidates) < 2:
            for candidate in candidates:
                candidate.score = 100 - len(candidate.warnings) * 2
            return

        patches = {c.candidate_id: c.patch for c in candidates}
        touched_files = self._union_touched_files(patches.values())
        original_files = repo_state.get_files(touched_files)

        patched_files_map = {}
        applier = PatchApplier()
        for patch_id, patch in patches.items():
            apply_result = applier.apply_patch(patch, original_files)
            if apply_result.success:
                patched_files_map[patch_id] = apply_result.patched_files
            else:
                patched_files_map[patch_id] = dict(original_files)

        pipeline = await self._get_quality_pipeline()
        quality_result = await pipeline.select_best_patch(
            patches=patches,
            issue_description=f"{task.title}\n{task.description}",
            original_files=original_files,
            patched_files_map=patched_files_map,
            context_code="",
        )

        for candidate in candidates:
            score = quality_result.quality_scores.get(candidate.candidate_id)
            candidate.score = score.overall_score if score else 50.0

    def _select_top_k(self, task_result: TaskExecutionResult) -> list[TaskCandidate]:
        if not task_result.candidates:
            return []
        sorted_candidates = sorted(task_result.candidates, key=lambda c: c.score, reverse=True)
        return sorted_candidates[: self.execution_config.top_k_per_task]

    async def _assemble(
        self,
        dag: TaskDAG,
        candidates_by_task: dict[str, list[TaskCandidate]],
        base_state: RepoState,
    ) -> AssemblyResult:
        order = dag.topological_order()
        base_files = base_state.get_files(self._union_touched_files(
            candidate.patch
            for candidates in candidates_by_task.values()
            for candidate in candidates
        ))

        branches = [AssemblyBranch(files=dict(base_files))]
        applier = PatchApplier()

        for task in order:
            next_branches: list[AssemblyBranch] = []
            candidates = candidates_by_task.get(task.task_id, [])
            if not candidates:
                return AssemblyResult(success=False, errors=[f"No candidates for {task.task_id}"])

            for branch in branches:
                for candidate in candidates:
                    apply_result = applier.apply_patch(candidate.patch, branch.files)
                    if not apply_result.success:
                        continue
                    next_branches.append(
                        AssemblyBranch(
                            patch_ids=branch.patch_ids + [candidate.candidate_id],
                            patches=branch.patches + [candidate.patch],
                            files=dict(apply_result.patched_files),
                            score=branch.score + candidate.score,
                        )
                    )

            if not next_branches:
                return AssemblyResult(success=False, errors=[f"Assembly failed at {task.task_id}"])

            next_branches.sort(key=lambda b: b.score, reverse=True)
            branches = next_branches[: self.execution_config.beam_width]

        best = max(branches, key=lambda b: b.score)
        combined_patch = diff_files(base_files, best.files)

        return AssemblyResult(
            success=True,
            combined_patch=combined_patch,
            selected_patch_ids=best.patch_ids,
            score=best.score,
        )

    async def _resolve_repo(self, submission: TaskDAGSubmission) -> Path:
        if submission.repo_path:
            return submission.repo_path

        if not submission.repository_url:
            raise ValueError("repository_url or repo_path is required")

        temp_dir = tempfile.mkdtemp(prefix="atlas_repo_")
        repo_path = Path(temp_dir) / "repo"

        # Check if repository_url is a local path
        local_path = Path(submission.repository_url)
        is_local = local_path.exists() and local_path.is_dir()

        loop = asyncio.get_running_loop()

        if is_local:
            # Local repository - copy to temp directory
            logger.info(f"Using local repository: {submission.repository_url}")
            await loop.run_in_executor(
                None,
                lambda: shutil.copytree(
                    local_path,
                    repo_path,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', 'node_modules', '.venv', 'venv'),
                ),
            )
            # Initialize as git repo if not already
            if not (repo_path / ".git").exists():
                git.Repo.init(repo_path)
                repo = git.Repo(repo_path)
                repo.index.add("*")
                repo.index.commit("ATLAS: Initial commit from local")
        else:
            # Remote repository - clone it
            use_shallow = submission.base_commit is None
            if use_shallow:
                await loop.run_in_executor(
                    None,
                    lambda: git.Repo.clone_from(
                        submission.repository_url,
                        repo_path,
                        branch=submission.branch,
                        depth=1,
                    ),
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: git.Repo.clone_from(
                        submission.repository_url,
                        repo_path,
                        branch=submission.branch,
                    ),
                )

            if submission.base_commit:
                repo = git.Repo(repo_path)
                repo.git.checkout(submission.base_commit)

        return repo_path

    def _union_touched_files(self, patches: Iterable[str]) -> list[str]:
        parser = PatchParser()
        paths = []
        for patch in patches:
            if not patch:
                continue
            for file_patch in parser.parse(patch):
                paths.append(file_patch.target_path)
        return list(dict.fromkeys(paths))

    async def _get_quality_pipeline(self) -> QualitySelectionPipeline:
        if self._quality_pipeline is None:
            config = QualitySelectionConfig(
                enable_llm_review=False,
                enable_tournament=False,
            )
            self._quality_pipeline = QualitySelectionPipeline(
                config=config,
                llm_client=None,
            )
        return self._quality_pipeline

    def _maybe_llm(self):
        if not self.config.gemini_api_key:
            return None
        from atlas.agents.gemini_client import GeminiClient

        return GeminiClient(self.config)


_dag_orchestrator: TaskDAGOrchestrator | None = None


def get_dag_orchestrator() -> TaskDAGOrchestrator:
    """Get a singleton TaskDAGOrchestrator instance."""
    global _dag_orchestrator
    if _dag_orchestrator is None:
        _dag_orchestrator = TaskDAGOrchestrator()
    return _dag_orchestrator
