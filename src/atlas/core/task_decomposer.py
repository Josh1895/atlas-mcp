"""LLM-assisted task decomposer for contract-driven DAGs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from atlas.agents.gemini_client import GeminiClient
from atlas.core.task_dag import OracleType, OwnershipRules, TaskDAG, TaskOracle, TaskSpec
from atlas.scout.repo_scout import RepoScoutReport

logger = logging.getLogger(__name__)


@dataclass
class TaskDecomposerConfig:
    """Configuration for task decomposition."""

    max_tasks: int = 12
    require_oracles: bool = True
    require_contract: bool = True
    require_ownership: bool = True
    num_variants: int = 3
    min_consensus: float = 0.4
    min_coverage_ratio: float = 1.0
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
        candidates: list[TaskDecompositionCandidate] = []
        num_variants = max(1, min(self.config.num_variants, 5))

        for idx in range(num_variants):
            temperature = 0.2 + (idx * 0.05)
            result = await self.llm_client.generate_with_retry(
                prompt=prompt,
                system_prompt="You are a senior staff engineer decomposing work into verified tasks.",
                temperature=temperature,
                max_tokens=4000,
            )

            data = self._parse_json(result.text)
            if not data:
                candidates.append(
                    TaskDecompositionCandidate(
                        dag=self._fallback_dag(description, repo_report),
                        source=f"variant_{idx + 1}",
                        errors=["Failed to parse decomposition JSON"],
                    )
                )
                continue

            try:
                dag = self._build_dag_from_json(data)
            except Exception as exc:
                candidates.append(
                    TaskDecompositionCandidate(
                        dag=self._fallback_dag(description, repo_report),
                        source=f"variant_{idx + 1}",
                        errors=[f"Failed to build TaskDAG: {exc}"],
                    )
                )
                continue

            coverage_score, coverage_missing = self._coverage_score(
                dag,
                repo_report,
                keywords,
                component_names,
            )
            errors = self._validate_candidate(
                dag,
                coverage_score,
                coverage_missing,
            )
            candidates.append(
                TaskDecompositionCandidate(
                    dag=dag,
                    source=f"variant_{idx + 1}",
                    errors=errors,
                    coverage_score=coverage_score,
                    coverage_missing=coverage_missing,
                )
            )

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
        normalized = path.replace("\\", "/").strip()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.lstrip("/")

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
