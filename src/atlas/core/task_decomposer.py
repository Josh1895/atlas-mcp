"""LLM-assisted task decomposer for contract-driven DAGs.

Enhanced with 9-variant diversity (3 personas × 3 temperatures) and
mathematical ensemble merging using Jaccard clustering and set cover.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from graphlib import TopologicalSorter, CycleError
from pathlib import PurePosixPath
from typing import Any

from atlas.agents.gemini_client import GeminiClient
from atlas.core.task_dag import OracleType, OwnershipRules, TaskDAG, TaskOracle, TaskSpec
from atlas.scout.repo_scout import RepoScoutReport

logger = logging.getLogger(__name__)


# =============================================================================
# T-001: 9-Variant Configuration
# =============================================================================

DECOMPOSITION_PERSONAS: dict[str, str] = {
    "ATOMIC_ARCHITECT": (
        "You decompose into smallest independent units. Each task should be "
        "completable in isolation with minimal dependencies. Focus on granular, "
        "bottom-up decomposition. Every task must touch exactly one concern."
    ),
    "INTEGRATION_LEAD": (
        "You think about interfaces and data flow first. Define tasks around "
        "integration points and API boundaries. Focus on middle-out decomposition "
        "where connections between components are explicit. Tasks should be defined "
        "by the data they consume and produce."
    ),
    "TEST_FIRST_QA": (
        "You start from test scenarios and work backward. First identify how to "
        "verify the feature, then decompose into tasks that enable those verifications. "
        "Focus on top-down decomposition where every task has clear acceptance criteria "
        "based on testable outcomes."
    ),
}

DECOMPOSITION_TEMPERATURES: list[float] = [0.3, 0.5, 0.8]


@dataclass
class MergeMetrics:
    """Metrics from ensemble merge process (T-009)."""

    variants_generated: int = 0
    variants_valid: int = 0
    clusters_formed: int = 0
    tasks_selected: int = 0
    gaps_filled: int = 0
    final_task_count: int = 0
    persona_contributions: dict[str, int] = field(default_factory=dict)
    temperature_contributions: dict[float, int] = field(default_factory=dict)


@dataclass
class TaskDecomposerConfig:
    """Configuration for task decomposition."""

    # Core settings
    max_tasks: int = 12
    require_oracles: bool = True
    require_contract: bool = True
    require_ownership: bool = True

    # Variant generation (T-001)
    num_variants: int = 9  # 3 personas × 3 temperatures for optimal diversity
    num_personas: int = 3
    num_temperatures: int = 3
    temperatures: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])

    # Ensemble merging (T-001)
    enable_ensemble_merge: bool = True
    cluster_similarity_threshold: float = 0.6
    min_oracle_count: int = 1

    # Consensus (legacy, used when ensemble_merge disabled)
    min_consensus: float = 0.4
    min_coverage_ratio: float = 1.0

    # Coverage targets
    max_keyword_hits: int = 5
    max_component_paths: int = 5
    include_entrypoints_in_coverage: bool = True


@dataclass
class TaskDecompositionCandidate:
    """Candidate DAG with validation and coverage metadata."""

    dag: TaskDAG
    source: str
    errors: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    coverage_missing: list[str] = field(default_factory=list)
    consensus_score: float = 0.0

    # T-002: Track variant origin for ensemble merge
    persona: str = ""
    temperature: float = 0.0


class TaskDecomposer:
    """Decompose a PRD into a contract-driven task DAG."""

    def __init__(
        self,
        llm_client: GeminiClient | None = None,
        config: TaskDecomposerConfig | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or TaskDecomposerConfig()

    async def decompose(
        self,
        description: str,
        repo_report: RepoScoutReport,
        keywords: list[str] | None = None,
        component_names: list[str] | None = None,
        dag_override: Any | None = None,
    ) -> TaskDAG:
        """Decompose a description into a TaskDAG."""
        keywords = keywords or []
        component_names = component_names or []

        if dag_override is not None:
            dag = self._build_override_dag(dag_override)
            override_coverage, override_missing = self._coverage_score(
                dag,
                repo_report,
                keywords,
                component_names,
            )
            override_errors = self._validate_candidate(
                dag,
                override_coverage,
                override_missing,
            )
            if override_errors:
                raise ValueError(f"DAG override failed validation: {override_errors}")
            return dag

        if not self.llm_client:
            logger.warning("No LLM client provided; using fallback DAG")
            return self._fallback_dag(description, repo_report)

        prompt = self._build_prompt(description, repo_report, keywords, component_names)

        # T-002: Generate 3×3 variant matrix (personas × temperatures) in parallel
        candidates = await self._generate_variant_matrix(
            prompt, description, repo_report, keywords, component_names
        )

        # T-008: Use ensemble merge or legacy selection
        if self.config.enable_ensemble_merge:
            merged_dag, metrics = self._merge_candidates(candidates, repo_report, keywords, component_names)
            if merged_dag is None:
                logger.warning("Ensemble merge failed; using fallback DAG")
                return self._fallback_dag(description, repo_report)

            logger.info(
                "Merged DAG: %d tasks from %d/%d valid variants, %d clusters, %d gaps filled",
                metrics.final_task_count,
                metrics.variants_valid,
                metrics.variants_generated,
                metrics.clusters_formed,
                metrics.gaps_filled,
            )
            return merged_dag
        else:
            # Legacy single-winner selection
            selected = self._select_candidate(candidates)
            if selected is None:
                logger.warning("Decomposition too weak; using fallback DAG")
                return self._fallback_dag(description, repo_report)

            logger.info(
                "Selected DAG from %s (consensus=%.2f, coverage=%.2f)",
                selected.source,
                selected.consensus_score,
                selected.coverage_score,
            )
            return selected.dag

    async def _generate_variant_matrix(
        self,
        prompt: str,
        description: str,
        repo_report: RepoScoutReport,
        keywords: list[str],
        component_names: list[str],
    ) -> list[TaskDecompositionCandidate]:
        """T-002: Generate 3×3 variant matrix (personas × temperatures) in parallel."""
        personas = list(DECOMPOSITION_PERSONAS.items())[:self.config.num_personas]
        temperatures = self.config.temperatures[:self.config.num_temperatures]

        async def generate_variant(
            persona_name: str,
            persona_prompt: str,
            temperature: float,
            variant_idx: int,
        ) -> TaskDecompositionCandidate:
            """Generate a single variant with specific persona and temperature."""
            source = f"{persona_name}_{temperature}"
            system_prompt = (
                f"You are a senior staff engineer decomposing work into verified tasks. "
                f"{persona_prompt}"
            )

            try:
                result = await self.llm_client.generate_with_retry(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=4000,
                )

                data = self._parse_json(result.text)
                if not data:
                    return TaskDecompositionCandidate(
                        dag=self._fallback_dag(description, repo_report),
                        source=source,
                        errors=["Failed to parse decomposition JSON"],
                        persona=persona_name,
                        temperature=temperature,
                    )

                dag = self._build_dag_from_json(data)
                coverage_score, coverage_missing = self._coverage_score(
                    dag, repo_report, keywords, component_names
                )
                errors = self._validate_candidate(dag, coverage_score, coverage_missing)

                return TaskDecompositionCandidate(
                    dag=dag,
                    source=source,
                    errors=errors,
                    coverage_score=coverage_score,
                    coverage_missing=coverage_missing,
                    persona=persona_name,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.warning("Variant %s failed: %s", source, exc)
                return TaskDecompositionCandidate(
                    dag=self._fallback_dag(description, repo_report),
                    source=source,
                    errors=[f"Generation failed: {exc}"],
                    persona=persona_name,
                    temperature=temperature,
                )

        # Build all combinations and run in parallel
        tasks = []
        variant_idx = 0
        for persona_name, persona_prompt in personas:
            for temperature in temperatures:
                tasks.append(
                    generate_variant(persona_name, persona_prompt, temperature, variant_idx)
                )
                variant_idx += 1

        candidates = await asyncio.gather(*tasks)
        logger.info(
            "Generated %d variants: %d valid, %d with errors",
            len(candidates),
            sum(1 for c in candidates if not c.errors),
            sum(1 for c in candidates if c.errors),
        )
        return list(candidates)

    def _build_prompt(
        self,
        description: str,
        repo_report: RepoScoutReport,
        keywords: list[str],
        component_names: list[str],
    ) -> str:
        context = repo_report.to_context_pack(max_chars=6000)
        schema = {
            "tasks": [
                {
                    "id": "string",
                    "title": "string",
                    "description": "string",
                    "contract": "string",
                    "ownership": {
                        "allowed_files": ["path/one.py"],
                        "allowed_globs": ["src/**/grid*.ts"],
                        "allowed_dirs": ["src/components"],
                        "blocked_globs": ["tests/**"],
                    },
                    "oracles": [
                        {
                            "type": "test|lint|typecheck|command",
                            "command": "pytest tests/test_grid.py",
                            "description": "string",
                            "timeout_seconds": 300,
                        }
                    ],
                    "inputs": ["string"],
                    "outputs": ["string"],
                    "dependencies": ["task_id"],
                    "risk_level": "low|medium|high",
                    "priority": 0,
                }
            ]
        }

        focus = []
        if keywords:
            focus.append(f"Keywords: {', '.join(keywords)}")
        if component_names:
            focus.append(f"Components: {', '.join(component_names)}")
        focus_block = "\n".join(focus) if focus else "None provided."

        return (
            "## RepoScout Context\n"
            f"{context}\n\n"
            "## Feature Request\n"
            f"{description}\n\n"
            "## Focus Areas\n"
            f"{focus_block}\n\n"
            "## Instructions\n"
            "Decompose into minimally verifiable, composable tasks.\n"
            f"Max tasks: {self.config.max_tasks}.\n"
            "Every task must include a contract, ownership scope, and oracles.\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )

    def _parse_json(self, text: str) -> dict | None:
        if not text:
            return None

        # Strategy 1: Look for ```json ... ``` block with balanced braces
        fenced = re.search(r"```json\s*(\{[\s\S]*\})\s*```", text)
        if fenced:
            payload = self._extract_balanced_json(fenced.group(1))
            if payload:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Find outermost { } in entire text using balanced extraction
        start = text.find("{")
        if start != -1:
            payload = self._extract_balanced_json(text[start:])
            if payload:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Fallback - try simple start/end with trailing comma fix
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            payload = text[start : end + 1]
            # Remove trailing commas before } or ]
            payload = re.sub(r",\s*([}\]])", r"\1", payload)
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON from LLM output after all strategies")

        return None

    def _build_override_dag(self, override: Any) -> TaskDAG:
        if isinstance(override, TaskDAG):
            return override

        if isinstance(override, str):
            try:
                data = json.loads(override)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid override JSON: {exc}") from exc
        elif isinstance(override, list):
            data = {"tasks": override}
        elif isinstance(override, dict):
            data = override
        else:
            raise ValueError("Override must be a dict, list, JSON string, or TaskDAG")

        if "tasks" not in data and isinstance(data, dict):
            data = {"tasks": [data]}

        return self._build_dag_from_json(data)

    def _validate_candidate(
        self,
        dag: TaskDAG,
        coverage_score: float,
        coverage_missing: list[str],
    ) -> list[str]:
        errors = dag.validate()

        for task in dag.tasks.values():
            if self.config.require_contract and not task.contract.strip():
                errors.append(f"Task {task.task_id} missing contract")

            if self.config.require_ownership and not (
                task.ownership.allowed_files
                or task.ownership.allowed_globs
                or task.ownership.allowed_dirs
            ):
                errors.append(f"Task {task.task_id} missing ownership scope")

            if self.config.require_oracles and not task.oracles:
                errors.append(f"Task {task.task_id} missing oracles")

        if coverage_score < self.config.min_coverage_ratio:
            preview = ", ".join(coverage_missing[:5])
            errors.append(
                "Coverage below threshold "
                f"({coverage_score:.2f} < {self.config.min_coverage_ratio:.2f}). "
                f"Missing: {preview}"
            )

        return errors

    def _coverage_score(
        self,
        dag: TaskDAG,
        repo_report: RepoScoutReport,
        keywords: list[str],
        component_names: list[str],
    ) -> tuple[float, list[str]]:
        targets = self._collect_coverage_targets(repo_report, keywords, component_names)
        if not targets:
            return 1.0, []

        covered: list[str] = []
        missing: list[str] = []

        for target in targets:
            if self._dag_covers_target(dag, target):
                covered.append(target)
            else:
                missing.append(target)

        return len(covered) / len(targets), missing

    def _collect_coverage_targets(
        self,
        repo_report: RepoScoutReport,
        keywords: list[str],
        component_names: list[str],
    ) -> list[str]:
        targets: list[str] = []

        for keyword in keywords:
            hits = repo_report.keyword_hits.get(keyword, [])
            for hit in hits[: self.config.max_keyword_hits]:
                targets.append(hit.path)

        for name in component_names:
            paths = repo_report.component_map.get(name, [])
            for path in paths[: self.config.max_component_paths]:
                targets.append(path)

        if (
            not targets
            and self.config.include_entrypoints_in_coverage
            and repo_report.entrypoints
        ):
            targets.extend(repo_report.entrypoints)

        normalized = [self._normalize_path(target) for target in targets if target]
        return sorted(set(normalized))

    def _dag_covers_target(self, dag: TaskDAG, target: str) -> bool:
        for task in dag.tasks.values():
            if self._task_covers_target(task, target):
                return True
        return False

    def _task_covers_target(self, task: TaskSpec, target: str) -> bool:
        normalized = self._normalize_path(target)

        for path in task.ownership.allowed_files:
            if normalized == self._normalize_path(path):
                return True

        for path in task.ownership.allowed_dirs:
            norm_dir = self._normalize_path(path).rstrip("/")
            if normalized == norm_dir or normalized.startswith(f"{norm_dir}/"):
                return True

        for pattern in task.ownership.allowed_globs:
            if self._glob_matches(normalized, pattern):
                return True

        return False

    def _glob_matches(self, path: str, pattern: str) -> bool:
        normalized = self._normalize_path(pattern)
        if normalized in {"*", "**", "**/*"}:
            return True
        try:
            return PurePosixPath(path).match(normalized)
        except ValueError:
            return False

    def _normalize_path(self, path: str) -> str:
        """Normalize file paths for cross-platform comparison."""
        normalized = path.replace("\\", "/").strip()

        # Remove ./ prefix
        if normalized.startswith("./"):
            normalized = normalized[2:]

        # Remove leading slash
        normalized = normalized.lstrip("/")

        # Compress multiple consecutive slashes (src//api -> src/api)
        while "//" in normalized:
            normalized = normalized.replace("//", "/")

        # Remove trailing slash for directories
        normalized = normalized.rstrip("/")

        return normalized

    def _select_candidate(
        self,
        candidates: list[TaskDecompositionCandidate],
    ) -> TaskDecompositionCandidate | None:
        valid = [candidate for candidate in candidates if not candidate.errors]
        if not valid:
            return None

        if len(valid) == 1:
            valid[0].consensus_score = 1.0
            return valid[0]

        signatures = [self._dag_signature(candidate.dag) for candidate in valid]

        for idx, candidate in enumerate(valid):
            sims = [
                self._jaccard(signatures[idx], signatures[other_idx])
                for other_idx in range(len(valid))
                if other_idx != idx
            ]
            candidate.consensus_score = sum(sims) / len(sims) if sims else 1.0

        best = max(
            valid,
            key=lambda c: (c.consensus_score, c.coverage_score, len(c.dag.tasks)),
        )

        if len(valid) > 1 and best.consensus_score < self.config.min_consensus:
            return None

        return best

    def _dag_signature(self, dag: TaskDAG) -> set[str]:
        tokens: set[str] = set()
        for task in dag.tasks.values():
            tokens.update(self._task_scope_tokens(task))
            title = self._slugify(task.title)
            if title:
                tokens.add(f"title:{title}")
        return tokens

    def _task_scope_tokens(self, task: TaskSpec) -> set[str]:
        tokens: set[str] = set()
        scope_items = [
            *task.ownership.allowed_files,
            *task.ownership.allowed_dirs,
            *task.ownership.allowed_globs,
        ]

        for item in scope_items:
            normalized = self._normalize_path(item)
            if not normalized:
                continue
            tokens.add(normalized)

            prefix = normalized.split("*", 1)[0].rstrip("/")
            if prefix:
                tokens.add(prefix)

            top = normalized.split("/", 1)[0]
            if top:
                tokens.add(top)

        return tokens

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    # =========================================================================
    # T-003 to T-009: Mathematical Ensemble Merging
    # =========================================================================

    def _merge_candidates(
        self,
        candidates: list[TaskDecompositionCandidate],
        repo_report: RepoScoutReport,
        keywords: list[str],
        component_names: list[str],
    ) -> tuple[TaskDAG | None, MergeMetrics]:
        """T-008: Merge best tasks from all valid variants using mathematical ensemble."""
        metrics = MergeMetrics(variants_generated=len(candidates))

        # Filter to valid candidates
        valid = [c for c in candidates if not c.errors]
        metrics.variants_valid = len(valid)

        if not valid:
            logger.warning("No valid variants to merge")
            return None, metrics

        # T-003: Extract all scopes across all variants
        all_scopes = self._extract_all_scopes(valid)

        # Collect all tasks from all valid variants with source tracking
        all_tasks: list[tuple[TaskSpec, str, str, float]] = []
        for candidate in valid:
            for task in candidate.dag.tasks.values():
                all_tasks.append((task, candidate.source, candidate.persona, candidate.temperature))

        if not all_tasks:
            logger.warning("No tasks found across valid variants")
            return None, metrics

        # T-004: Cluster similar tasks by Jaccard similarity on scopes
        clusters = self._cluster_tasks(all_tasks)
        metrics.clusters_formed = len(clusters)

        # T-005: Vote for best task in each cluster
        selected_tasks: list[TaskSpec] = []
        for cluster in clusters:
            representative, source, persona, temp = self._select_representative(cluster)
            selected_tasks.append(representative)
            metrics.persona_contributions[persona] = metrics.persona_contributions.get(persona, 0) + 1
            metrics.temperature_contributions[temp] = metrics.temperature_contributions.get(temp, 0) + 1

        metrics.tasks_selected = len(selected_tasks)

        # T-006: Fill gaps with greedy set cover
        covered_scopes = self._get_covered_scopes(selected_tasks)
        gaps = all_scopes - covered_scopes

        if gaps:
            gap_fillers = self._fill_gaps(gaps, all_tasks, selected_tasks)
            for filler, source, persona, temp in gap_fillers:
                selected_tasks.append(filler)
                metrics.persona_contributions[persona] = metrics.persona_contributions.get(persona, 0) + 1
                metrics.temperature_contributions[temp] = metrics.temperature_contributions.get(temp, 0) + 1
            metrics.gaps_filled = len(gap_fillers)

        # T-007: Resolve dependencies with topological sort
        sorted_tasks = self._resolve_dependencies(selected_tasks)

        # Deduplicate task IDs to avoid conflicts
        sorted_tasks = self._deduplicate_task_ids(sorted_tasks)

        metrics.final_task_count = len(sorted_tasks)

        # Build final DAG
        try:
            merged_dag = TaskDAG.from_list(sorted_tasks)
            return merged_dag, metrics
        except Exception as exc:
            logger.error("Failed to build merged DAG: %s", exc)
            return None, metrics

    def _extract_all_scopes(
        self,
        candidates: list[TaskDecompositionCandidate],
    ) -> set[str]:
        """T-003: Union of all file scopes across all variants."""
        all_scopes: set[str] = set()
        for candidate in candidates:
            for task in candidate.dag.tasks.values():
                all_scopes.update(
                    self._normalize_path(f) for f in task.ownership.allowed_files
                )
                all_scopes.update(
                    self._normalize_path(d) for d in task.ownership.allowed_dirs
                )
                # Don't include broad globs like **/* as explicit scopes
                for glob in task.ownership.allowed_globs:
                    normalized = self._normalize_path(glob)
                    if normalized not in {"*", "**", "**/*"}:
                        all_scopes.add(normalized)
        return all_scopes

    def _task_scope_set(self, task: TaskSpec) -> set[str]:
        """Get normalized scope set for a task (for Jaccard comparison)."""
        scopes: set[str] = set()
        scopes.update(self._normalize_path(f) for f in task.ownership.allowed_files)
        scopes.update(self._normalize_path(d) for d in task.ownership.allowed_dirs)
        for glob in task.ownership.allowed_globs:
            normalized = self._normalize_path(glob)
            if normalized not in {"*", "**", "**/*"}:
                scopes.add(normalized)
        return scopes

    def _task_jaccard(self, task_a: TaskSpec, task_b: TaskSpec) -> float:
        """T-004: Jaccard similarity between two tasks based on ownership scopes."""
        scopes_a = self._task_scope_set(task_a)
        scopes_b = self._task_scope_set(task_b)

        if not scopes_a and not scopes_b:
            return 1.0

        union = scopes_a | scopes_b
        if not union:
            return 0.0

        return len(scopes_a & scopes_b) / len(union)

    def _cluster_tasks(
        self,
        all_tasks: list[tuple[TaskSpec, str, str, float]],
    ) -> list[list[tuple[TaskSpec, str, str, float]]]:
        """T-004: Group similar tasks into clusters using Jaccard threshold."""
        if not all_tasks:
            return []

        threshold = self.config.cluster_similarity_threshold
        clusters: list[list[tuple[TaskSpec, str, str, float]]] = []
        assigned: set[int] = set()

        for i, (task_i, source_i, persona_i, temp_i) in enumerate(all_tasks):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [(task_i, source_i, persona_i, temp_i)]
            assigned.add(i)

            # Find all similar tasks
            for j, (task_j, source_j, persona_j, temp_j) in enumerate(all_tasks):
                if j in assigned:
                    continue

                similarity = self._task_jaccard(task_i, task_j)
                if similarity >= threshold:
                    cluster.append((task_j, source_j, persona_j, temp_j))
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _select_representative(
        self,
        cluster: list[tuple[TaskSpec, str, str, float]],
    ) -> tuple[TaskSpec, str, str, float]:
        """T-005: Pick best task from cluster by scoring."""

        def score(entry: tuple[TaskSpec, str, str, float]) -> float:
            task = entry[0]
            # Oracle score: more oracles = better verifiability
            oracle_score = len(task.oracles) * 10

            # Specificity score: penalize wildcards
            specificity = 10
            for glob in task.ownership.allowed_globs:
                if "**" in glob:
                    specificity -= 2
                elif "*" in glob:
                    specificity -= 1

            # Priority score (higher priority = higher score)
            priority_score = task.priority

            # Contract clarity: longer contracts often more specific
            contract_score = min(len(task.contract) / 50, 5)

            return oracle_score + specificity + priority_score + contract_score

        return max(cluster, key=score)

    def _get_covered_scopes(self, tasks: list[TaskSpec]) -> set[str]:
        """Get all scopes covered by a list of tasks."""
        covered: set[str] = set()
        for task in tasks:
            covered.update(self._normalize_path(f) for f in task.ownership.allowed_files)
            covered.update(self._normalize_path(d) for d in task.ownership.allowed_dirs)
            for glob in task.ownership.allowed_globs:
                normalized = self._normalize_path(glob)
                if normalized not in {"*", "**", "**/*"}:
                    covered.add(normalized)
        return covered

    def _fill_gaps(
        self,
        gaps: set[str],
        all_tasks: list[tuple[TaskSpec, str, str, float]],
        already_selected: list[TaskSpec],
    ) -> list[tuple[TaskSpec, str, str, float]]:
        """T-006: Greedily add tasks to cover gaps (set cover algorithm)."""
        selected_ids = {t.task_id for t in already_selected}
        remaining_gaps = gaps.copy()
        fillers: list[tuple[TaskSpec, str, str, float]] = []

        while remaining_gaps:
            best_entry = None
            best_coverage = 0

            for task, source, persona, temp in all_tasks:
                # Skip if already selected
                if task.task_id in selected_ids:
                    continue

                # Count how many gaps this task covers
                task_scopes = self._task_scope_set(task)
                coverage = len(task_scopes & remaining_gaps)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_entry = (task, source, persona, temp)

            if best_entry is None or best_coverage == 0:
                # No task can cover remaining gaps
                break

            fillers.append(best_entry)
            selected_ids.add(best_entry[0].task_id)
            remaining_gaps -= self._task_scope_set(best_entry[0])

        return fillers

    def _resolve_dependencies(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        """T-007: Topological sort to ensure valid task ordering."""
        task_map = {t.task_id: t for t in tasks}
        valid_ids = set(task_map.keys())

        # First pass: sanitize dependencies to only valid IDs
        for task in tasks:
            task.dependencies = [d for d in task.dependencies if d in valid_ids]

        # Build dependency graph
        graph: dict[str, set[str]] = {t.task_id: set(t.dependencies) for t in tasks}

        try:
            sorter = TopologicalSorter(graph)
            sorted_ids = list(sorter.static_order())
            return [task_map[tid] for tid in sorted_ids if tid in task_map]
        except CycleError as e:
            # e.args[1] contains cycle path like ['a', 'b', 'a'] meaning a->b->a
            cycle_nodes = e.args[1] if len(e.args) > 1 else []
            logger.warning("Dependency cycle detected: %s. Breaking cycle.", cycle_nodes)

            if cycle_nodes and len(cycle_nodes) >= 3:
                # Cycle path is [node1, node2, ..., node1] where each depends on next
                # To break a->b->a, remove the last edge: b's dependency on a
                # That's cycle_nodes[-2] depends on cycle_nodes[-1]
                source_id = cycle_nodes[-2]  # The task that has the problematic dependency
                target_id = cycle_nodes[-1]  # The dependency to remove

                if source_id in task_map:
                    task = task_map[source_id]
                    if target_id in task.dependencies:
                        task.dependencies.remove(target_id)
                        logger.info("Removed dependency %s -> %s to break cycle", source_id, target_id)

                # Recursively try again after breaking the edge
                return self._resolve_dependencies(tasks)
            else:
                # Fallback: if we can't identify the cycle, clear all deps and return
                logger.warning("Could not identify cycle nodes, returning tasks without dependency ordering")
                for task in tasks:
                    task.dependencies = []
                return tasks

    def _deduplicate_task_ids(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        """Ensure unique task IDs across merged tasks."""
        seen_ids: dict[str, int] = {}
        result: list[TaskSpec] = []

        for task in tasks:
            original_id = task.task_id
            if original_id in seen_ids:
                seen_ids[original_id] += 1
                # Create new task with unique ID
                new_id = f"{original_id}_{seen_ids[original_id]}"
                task = TaskSpec(
                    task_id=new_id,
                    title=task.title,
                    description=task.description,
                    contract=task.contract,
                    ownership=task.ownership,
                    oracles=task.oracles,
                    inputs=task.inputs,
                    outputs=task.outputs,
                    dependencies=task.dependencies,
                    risk_level=task.risk_level,
                    priority=task.priority,
                )
            else:
                seen_ids[original_id] = 0
            result.append(task)

        return result

    def _extract_balanced_json(self, text: str) -> str | None:
        """Extract JSON object with balanced braces from text starting with {."""
        if not text or not text.strip().startswith("{"):
            return None

        text = text.strip()
        depth = 0
        in_string = False
        escape = False

        for i, char in enumerate(text):
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]

        return None

    def _build_dag_from_json(self, data: dict) -> TaskDAG:
        tasks_data = data.get("tasks", [])
        tasks: list[TaskSpec] = []

        for item in tasks_data[: self.config.max_tasks]:
            task_id = item.get("id") or self._slugify(item.get("title", "task"))
            ownership = item.get("ownership", {})
            oracles = [
                TaskOracle(
                    oracle_type=OracleType(oracle.get("type", "command")),
                    command=oracle.get("command", ""),
                    description=oracle.get("description", ""),
                    timeout_seconds=oracle.get("timeout_seconds", 300),
                )
                for oracle in item.get("oracles", [])
            ]

            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    title=item.get("title", task_id),
                    description=item.get("description", ""),
                    contract=item.get("contract", ""),
                    ownership=OwnershipRules(
                        allowed_files=ownership.get("allowed_files", []),
                        allowed_globs=ownership.get("allowed_globs", []),
                        allowed_dirs=ownership.get("allowed_dirs", []),
                        blocked_globs=ownership.get("blocked_globs", []),
                    ),
                    oracles=oracles,
                    inputs=item.get("inputs", []),
                    outputs=item.get("outputs", []),
                    dependencies=item.get("dependencies", []),
                    risk_level=item.get("risk_level", "medium"),
                    priority=item.get("priority", 0),
                )
            )

        return TaskDAG.from_list(tasks)

    def _fallback_dag(self, description: str, repo_report: RepoScoutReport) -> TaskDAG:
        ownership = OwnershipRules(allowed_globs=["**/*"])
        oracles = []

        if self.config.require_oracles:
            description_text = (
                "Auto-detected tests"
                if repo_report.test_frameworks
                else "Auto-detect tests"
            )
            oracles.append(
                TaskOracle(
                    oracle_type=OracleType.TEST,
                    command="",
                    description=description_text,
                )
            )

        task = TaskSpec(
            task_id="task_0",
            title="Implement feature end-to-end",
            description=description,
            contract="Feature implemented and passes available verification.",
            ownership=ownership,
            oracles=oracles,
            inputs=[],
            outputs=[],
            dependencies=[],
            risk_level="high",
            priority=0,
        )
        return TaskDAG.from_list([task])

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
        return slug.strip("_") or "task"
