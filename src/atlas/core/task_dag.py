"""Task DAG models for contract-driven decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class OracleType(str, Enum):
    """Types of verification oracles."""

    COMMAND = "command"
    TEST = "test"
    LINT = "lint"
    TYPECHECK = "typecheck"


@dataclass
class TaskOracle:
    """Definition of a verification oracle for a task."""

    oracle_type: OracleType
    command: str
    description: str = ""
    timeout_seconds: int = 300


@dataclass
class OwnershipRules:
    """Allowed and disallowed scope for a task."""

    allowed_files: list[str] = field(default_factory=list)
    allowed_globs: list[str] = field(default_factory=list)
    allowed_dirs: list[str] = field(default_factory=list)
    blocked_globs: list[str] = field(default_factory=list)


@dataclass
class TaskSpec:
    """A single decomposed task with contract and ownership."""

    task_id: str
    title: str
    description: str
    contract: str
    ownership: OwnershipRules
    oracles: list[TaskOracle] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    priority: int = 0

    def validate(self) -> list[str]:
        errors = []
        if not self.task_id:
            errors.append("task_id is required")
        if not self.title:
            errors.append(f"Task {self.task_id} missing title")
        if not self.description:
            errors.append(f"Task {self.task_id} missing description")
        if not self.contract:
            errors.append(f"Task {self.task_id} missing contract")
        if not (
            self.ownership.allowed_files
            or self.ownership.allowed_globs
            or self.ownership.allowed_dirs
        ):
            errors.append(f"Task {self.task_id} missing ownership scope")
        return errors


@dataclass
class TaskDAG:
    """Directed acyclic graph of tasks."""

    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def add_task(self, task: TaskSpec) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        self.tasks[task.task_id] = task

    def validate(self) -> list[str]:
        errors: list[str] = []
        for task in self.tasks.values():
            errors.extend(task.validate())

        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task {task.task_id} has unknown dependency {dep}")

        if self._has_cycle():
            errors.append("TaskDAG has a cycle")

        return errors

    def topological_order(self) -> list[TaskSpec]:
        indegree = {task_id: 0 for task_id in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                indegree[task.task_id] += 1

        queue = [task_id for task_id, deg in indegree.items() if deg == 0]
        order: list[TaskSpec] = []

        while queue:
            current = queue.pop(0)
            order.append(self.tasks[current])
            for task in self.tasks.values():
                if current in task.dependencies:
                    indegree[task.task_id] -= 1
                    if indegree[task.task_id] == 0:
                        queue.append(task.task_id)

        if len(order) != len(self.tasks):
            raise ValueError("TaskDAG contains a cycle")

        return order

    def _has_cycle(self) -> bool:
        try:
            self.topological_order()
            return False
        except ValueError:
            return True

    @classmethod
    def from_list(cls, tasks: Iterable[TaskSpec]) -> "TaskDAG":
        dag = cls()
        for task in tasks:
            dag.add_task(task)
        return dag

