"""LLM-assisted task decomposer for contract-driven DAGs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from atlas.agents.gemini_client import GeminiClient
from atlas.core.task_dag import OracleType, OwnershipRules, TaskDAG, TaskOracle, TaskSpec
from atlas.scout.repo_scout import RepoScoutReport

logger = logging.getLogger(__name__)


@dataclass
class TaskDecomposerConfig:
    """Configuration for task decomposition."""

    max_tasks: int = 12
    require_oracles: bool = True


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
    ) -> TaskDAG:
        """Decompose a description into a TaskDAG."""
        if not self.llm_client:
            logger.warning("No LLM client provided; using fallback DAG")
            return self._fallback_dag(description, repo_report)

        prompt = self._build_prompt(description, repo_report)
        result = await self.llm_client.generate_with_retry(
            prompt=prompt,
            system_prompt="You are a senior staff engineer decomposing work into verified tasks.",
            temperature=0.2,
            max_tokens=4000,
        )

        data = self._parse_json(result.text)
        if not data:
            logger.warning("Failed to parse decomposition JSON; using fallback DAG")
            return self._fallback_dag(description, repo_report)

        try:
            dag = self._build_dag_from_json(data)
        except Exception as exc:
            logger.warning("Failed to build TaskDAG (%s); using fallback", exc)
            return self._fallback_dag(description, repo_report)

        errors = dag.validate()
        if errors:
            logger.warning("TaskDAG validation errors: %s", errors)
        return dag

    def _build_prompt(self, description: str, repo_report: RepoScoutReport) -> str:
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

        return (
            "## RepoScout Context\n"
            f"{context}\n\n"
            "## Feature Request\n"
            f"{description}\n\n"
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

        fenced = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", text)
        if fenced:
            payload = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            payload = text[start : end + 1]

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from LLM output")
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

        if self.config.require_oracles and repo_report.test_frameworks:
            oracles.append(
                TaskOracle(
                    oracle_type=OracleType.TEST,
                    command="",
                    description="Auto-detected tests",
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

