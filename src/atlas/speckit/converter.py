"""Converter to transform SpecKit specifications into ATLAS TaskDAGs.

This module bridges SpecKit's specification-driven approach with
ATLAS's multi-agent code generation system.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from atlas.core.task_dag import (
    OracleType,
    OwnershipRules,
    TaskDAG,
    TaskOracle,
    TaskSpec,
)
from atlas.speckit.models import (
    Constitution,
    ImplementationPlan,
    Priority,
    SpecKit,
    Specification,
    Task,
    TaskList,
)

logger = logging.getLogger(__name__)


class SpecKitConverter:
    """Converts SpecKit artifacts to ATLAS TaskDAG format."""

    def __init__(self, constitution: Constitution | None = None):
        """Initialize converter with optional constitution.

        Args:
            constitution: Project constitution for validation rules.
        """
        self.constitution = constitution

    def task_to_taskspec(
        self,
        task: Task,
        spec: Specification | None = None,
        plan: ImplementationPlan | None = None,
    ) -> TaskSpec:
        """Convert a SpecKit Task to an ATLAS TaskSpec.

        Args:
            task: SpecKit task to convert.
            spec: Optional specification for context.
            plan: Optional plan for technical details.

        Returns:
            ATLAS TaskSpec ready for execution.
        """
        # Build contract from task description and acceptance criteria
        contract_parts = [task.description]
        if task.acceptance_criteria:
            contract_parts.append("\nAcceptance Criteria:")
            for criterion in task.acceptance_criteria:
                contract_parts.append(f"- {criterion}")

        # If we have a linked user scenario, add its acceptance criteria
        if task.story_id and spec:
            for scenario in spec.user_scenarios:
                if scenario.id == task.story_id:
                    contract_parts.append(f"\nUser Story: {scenario.title}")
                    if scenario.given:
                        contract_parts.append("Given: " + "; ".join(scenario.given))
                    if scenario.when:
                        contract_parts.append("When: " + "; ".join(scenario.when))
                    if scenario.then:
                        contract_parts.append("Then: " + "; ".join(scenario.then))
                    break

        contract = "\n".join(contract_parts)

        # Build ownership rules from file paths
        allowed_files = task.file_paths.copy()
        allowed_globs = []
        allowed_dirs = []

        # Infer globs from file patterns
        for path in task.file_paths:
            # If path contains wildcard, it's a glob
            if "*" in path:
                allowed_globs.append(path)
                allowed_files.remove(path)
            # If path ends with /, it's a directory
            elif path.endswith("/"):
                allowed_dirs.append(path)
                allowed_files.remove(path)

        ownership = OwnershipRules(
            allowed_files=allowed_files,
            allowed_globs=allowed_globs,
            allowed_dirs=allowed_dirs,
            blocked_globs=[],
        )

        # Add forbidden patterns from constitution
        if self.constitution:
            ownership.blocked_globs = self.constitution.forbidden_patterns.copy()

        # Build oracles from testing requirements
        oracles = []

        # Add test oracle if testing framework is specified
        if plan and plan.technical_context.testing_framework:
            framework = plan.technical_context.testing_framework.lower()
            test_cmd = self._infer_test_command(framework, task.file_paths)
            if test_cmd:
                oracles.append(TaskOracle(
                    oracle_type=OracleType.TEST,
                    command=test_cmd,
                    description=f"Run tests for {task.id}",
                    timeout_seconds=300,
                ))

        # Add lint oracle if we have coding standards
        if self.constitution and self.constitution.coding_standards:
            lang = plan.technical_context.language.lower() if plan else ""
            lint_cmd = self._infer_lint_command(lang)
            if lint_cmd:
                oracles.append(TaskOracle(
                    oracle_type=OracleType.LINT,
                    command=lint_cmd,
                    description="Check code style",
                    timeout_seconds=60,
                ))

        # Add type check oracle for typed languages
        if plan:
            lang = plan.technical_context.language.lower()
            typecheck_cmd = self._infer_typecheck_command(lang)
            if typecheck_cmd:
                oracles.append(TaskOracle(
                    oracle_type=OracleType.TYPECHECK,
                    command=typecheck_cmd,
                    description="Run type checker",
                    timeout_seconds=120,
                ))

        # Determine risk level based on priority and file scope
        risk_level = "medium"
        if task.story_id and spec:
            for scenario in spec.user_scenarios:
                if scenario.id == task.story_id:
                    if scenario.priority == Priority.P1:
                        risk_level = "high"
                    elif scenario.priority == Priority.P3:
                        risk_level = "low"
                    break

        return TaskSpec(
            task_id=task.id,
            title=task.description[:50] + "..." if len(task.description) > 50 else task.description,
            description=task.description,
            contract=contract,
            ownership=ownership,
            oracles=oracles,
            inputs=[],  # Could be inferred from dependencies
            outputs=task.file_paths,
            dependencies=task.dependencies,
            risk_level=risk_level,
            priority=task.phase,
        )

    def tasklist_to_taskdag(
        self,
        tasks: TaskList,
        spec: Specification | None = None,
        plan: ImplementationPlan | None = None,
    ) -> TaskDAG:
        """Convert a SpecKit TaskList to an ATLAS TaskDAG.

        Args:
            tasks: TaskList to convert.
            spec: Optional specification for context.
            plan: Optional plan for technical details.

        Returns:
            ATLAS TaskDAG ready for execution.
        """
        dag = TaskDAG()

        # Process tasks in phase order
        all_tasks = tasks.get_all_tasks()

        for task in all_tasks:
            task_spec = self.task_to_taskspec(task, spec, plan)
            dag.add_task(task_spec)

        # Validate the DAG
        errors = dag.validate()
        if errors:
            logger.warning(f"TaskDAG validation warnings: {errors}")

        return dag

    def speckit_to_taskdag(self, speckit: SpecKit) -> TaskDAG:
        """Convert a complete SpecKit to an ATLAS TaskDAG.

        Args:
            speckit: Complete SpecKit with all artifacts.

        Returns:
            ATLAS TaskDAG ready for execution.
        """
        if not speckit.tasks:
            raise ValueError("SpecKit has no tasks defined")

        # Update constitution if available
        if speckit.constitution:
            self.constitution = speckit.constitution

        return self.tasklist_to_taskdag(
            speckit.tasks,
            speckit.specification,
            speckit.plan,
        )

    def taskdag_to_dict(self, dag: TaskDAG) -> dict[str, Any]:
        """Convert a TaskDAG to a dictionary for serialization.

        This format can be passed to solve_feature_dag as dag_override.

        Args:
            dag: TaskDAG to convert.

        Returns:
            Dictionary representation of the DAG.
        """
        tasks = []
        for task in dag.topological_order():
            tasks.append({
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "contract": task.contract,
                "ownership": {
                    "allowed_files": task.ownership.allowed_files,
                    "allowed_globs": task.ownership.allowed_globs,
                    "allowed_dirs": task.ownership.allowed_dirs,
                    "blocked_globs": task.ownership.blocked_globs,
                },
                "oracles": [
                    {
                        "oracle_type": oracle.oracle_type.value,
                        "command": oracle.command,
                        "description": oracle.description,
                        "timeout_seconds": oracle.timeout_seconds,
                    }
                    for oracle in task.oracles
                ],
                "inputs": task.inputs,
                "outputs": task.outputs,
                "dependencies": task.dependencies,
                "risk_level": task.risk_level,
                "priority": task.priority,
            })

        return {"tasks": tasks}

    def _infer_test_command(self, framework: str, files: list[str]) -> str | None:
        """Infer test command from framework and file paths."""
        commands = {
            "pytest": "pytest",
            "jest": "npm test",
            "mocha": "npm test",
            "vitest": "npm run test",
            "rspec": "bundle exec rspec",
            "unittest": "python -m unittest discover",
            "go": "go test ./...",
            "cargo": "cargo test",
        }

        for key, cmd in commands.items():
            if key in framework:
                return cmd

        return None

    def _infer_lint_command(self, language: str) -> str | None:
        """Infer lint command from language."""
        commands = {
            "python": "ruff check .",
            "javascript": "npm run lint",
            "typescript": "npm run lint",
            "go": "golangci-lint run",
            "rust": "cargo clippy",
            "ruby": "bundle exec rubocop",
        }

        return commands.get(language)

    def _infer_typecheck_command(self, language: str) -> str | None:
        """Infer type check command from language."""
        commands = {
            "python": "mypy .",
            "typescript": "tsc --noEmit",
            "go": "go vet ./...",
            "rust": "cargo check",
        }

        return commands.get(language)


def convert_speckit_to_dag(speckit: SpecKit) -> TaskDAG:
    """Convenience function to convert SpecKit to TaskDAG.

    Args:
        speckit: SpecKit to convert.

    Returns:
        ATLAS TaskDAG.
    """
    converter = SpecKitConverter(speckit.constitution)
    return converter.speckit_to_taskdag(speckit)


def convert_speckit_to_dag_override(speckit: SpecKit) -> dict[str, Any]:
    """Convert SpecKit to dag_override format for solve_feature_dag.

    Args:
        speckit: SpecKit to convert.

    Returns:
        Dictionary suitable for dag_override parameter.
    """
    converter = SpecKitConverter(speckit.constitution)
    dag = converter.speckit_to_taskdag(speckit)
    return converter.taskdag_to_dict(dag)
