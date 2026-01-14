"""SpecKit models for specification-driven development.

Based on GitHub's spec-kit framework for creating executable specifications
that can be transformed into ATLAS TaskDAGs for implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json
import re


class SpecStatus(str, Enum):
    """Status of a specification."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Priority(str, Enum):
    """Priority levels for requirements and stories."""
    P1 = "P1"  # Critical - must have
    P2 = "P2"  # Important - should have
    P3 = "P3"  # Nice to have


@dataclass
class UserScenario:
    """A user scenario with acceptance criteria."""
    id: str
    title: str
    description: str
    priority: Priority
    given: list[str] = field(default_factory=list)
    when: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "given": self.given,
            "when": self.when,
            "then": self.then,
            "edge_cases": self.edge_cases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserScenario":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            priority=Priority(data["priority"]),
            given=data.get("given", []),
            when=data.get("when", []),
            then=data.get("then", []),
            edge_cases=data.get("edge_cases", []),
        )


@dataclass
class FunctionalRequirement:
    """A functional requirement with MUST/SHOULD/MAY language."""
    id: str
    description: str
    requirement_type: str = "MUST"  # MUST, SHOULD, MAY
    needs_clarification: bool = False
    clarification_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "requirement_type": self.requirement_type,
            "needs_clarification": self.needs_clarification,
            "clarification_notes": self.clarification_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionalRequirement":
        return cls(
            id=data["id"],
            description=data["description"],
            requirement_type=data.get("requirement_type", "MUST"),
            needs_clarification=data.get("needs_clarification", False),
            clarification_notes=data.get("clarification_notes", ""),
        )


@dataclass
class DataEntity:
    """A key data entity in the specification."""
    name: str
    description: str
    attributes: dict[str, str] = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "relationships": self.relationships,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataEntity":
        return cls(
            name=data["name"],
            description=data["description"],
            attributes=data.get("attributes", {}),
            relationships=data.get("relationships", []),
        )


@dataclass
class SuccessCriteria:
    """Measurable success criteria for the feature."""
    id: str
    metric: str
    target: str
    measurement_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric,
            "target": self.target,
            "measurement_method": self.measurement_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessCriteria":
        return cls(
            id=data["id"],
            metric=data["metric"],
            target=data["target"],
            measurement_method=data.get("measurement_method", ""),
        )


@dataclass
class Specification:
    """A complete feature specification following SpecKit format."""
    feature_id: str
    title: str
    description: str
    status: SpecStatus = SpecStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # User scenarios
    user_scenarios: list[UserScenario] = field(default_factory=list)

    # Requirements
    functional_requirements: list[FunctionalRequirement] = field(default_factory=list)

    # Data model
    data_entities: list[DataEntity] = field(default_factory=list)

    # Success criteria
    success_criteria: list[SuccessCriteria] = field(default_factory=list)

    # Additional notes
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_scenarios": [s.to_dict() for s in self.user_scenarios],
            "functional_requirements": [r.to_dict() for r in self.functional_requirements],
            "data_entities": [e.to_dict() for e in self.data_entities],
            "success_criteria": [c.to_dict() for c in self.success_criteria],
            "assumptions": self.assumptions,
            "constraints": self.constraints,
            "open_questions": self.open_questions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Specification":
        return cls(
            feature_id=data["feature_id"],
            title=data["title"],
            description=data["description"],
            status=SpecStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            user_scenarios=[UserScenario.from_dict(s) for s in data.get("user_scenarios", [])],
            functional_requirements=[FunctionalRequirement.from_dict(r) for r in data.get("functional_requirements", [])],
            data_entities=[DataEntity.from_dict(e) for e in data.get("data_entities", [])],
            success_criteria=[SuccessCriteria.from_dict(c) for c in data.get("success_criteria", [])],
            assumptions=data.get("assumptions", []),
            constraints=data.get("constraints", []),
            open_questions=data.get("open_questions", []),
        )

    def validate(self) -> list[str]:
        """Validate the specification for completeness."""
        errors = []
        if not self.feature_id:
            errors.append("feature_id is required")
        if not self.title:
            errors.append("title is required")
        if not self.description:
            errors.append("description is required")
        if not self.user_scenarios:
            errors.append("At least one user scenario is required")
        if not self.functional_requirements:
            errors.append("At least one functional requirement is required")

        # Check for unclarified requirements
        unclear = [r.id for r in self.functional_requirements if r.needs_clarification]
        if unclear:
            errors.append(f"Requirements need clarification: {', '.join(unclear)}")

        return errors


@dataclass
class TechnicalContext:
    """Technical context for the implementation plan."""
    language: str = ""
    version: str = ""
    dependencies: list[str] = field(default_factory=list)
    storage: str = ""
    testing_framework: str = ""
    target_platforms: list[str] = field(default_factory=list)
    performance_targets: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "version": self.version,
            "dependencies": self.dependencies,
            "storage": self.storage,
            "testing_framework": self.testing_framework,
            "target_platforms": self.target_platforms,
            "performance_targets": self.performance_targets,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TechnicalContext":
        return cls(
            language=data.get("language", ""),
            version=data.get("version", ""),
            dependencies=data.get("dependencies", []),
            storage=data.get("storage", ""),
            testing_framework=data.get("testing_framework", ""),
            target_platforms=data.get("target_platforms", []),
            performance_targets=data.get("performance_targets", {}),
            constraints=data.get("constraints", []),
        )


@dataclass
class ImplementationPlan:
    """Implementation plan derived from a specification."""
    feature_id: str
    spec_link: str
    summary: str
    technical_context: TechnicalContext = field(default_factory=TechnicalContext)
    created_at: datetime = field(default_factory=datetime.now)

    # Architecture decisions
    architecture_decisions: list[str] = field(default_factory=list)

    # File structure
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    # Complexity tracking
    complexity_notes: list[str] = field(default_factory=list)

    # Contracts
    api_contracts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "spec_link": self.spec_link,
            "summary": self.summary,
            "technical_context": self.technical_context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "architecture_decisions": self.architecture_decisions,
            "source_files": self.source_files,
            "test_files": self.test_files,
            "complexity_notes": self.complexity_notes,
            "api_contracts": self.api_contracts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImplementationPlan":
        return cls(
            feature_id=data["feature_id"],
            spec_link=data.get("spec_link", ""),
            summary=data.get("summary", ""),
            technical_context=TechnicalContext.from_dict(data.get("technical_context", {})),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            architecture_decisions=data.get("architecture_decisions", []),
            source_files=data.get("source_files", []),
            test_files=data.get("test_files", []),
            complexity_notes=data.get("complexity_notes", []),
            api_contracts=data.get("api_contracts", []),
        )


@dataclass
class Task:
    """A single implementation task."""
    id: str
    description: str
    story_id: str = ""  # Links to user scenario
    phase: int = 1
    parallel: bool = False  # Can run in parallel with other tasks
    file_paths: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "story_id": self.story_id,
            "phase": self.phase,
            "parallel": self.parallel,
            "file_paths": self.file_paths,
            "dependencies": self.dependencies,
            "acceptance_criteria": self.acceptance_criteria,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            description=data["description"],
            story_id=data.get("story_id", ""),
            phase=data.get("phase", 1),
            parallel=data.get("parallel", False),
            file_paths=data.get("file_paths", []),
            dependencies=data.get("dependencies", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            completed=data.get("completed", False),
        )


@dataclass
class TaskList:
    """Ordered list of implementation tasks."""
    feature_id: str
    plan_link: str
    created_at: datetime = field(default_factory=datetime.now)
    phases: dict[int, list[Task]] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        if task.phase not in self.phases:
            self.phases[task.phase] = []
        self.phases[task.phase].append(task)

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks in phase order."""
        tasks = []
        for phase in sorted(self.phases.keys()):
            tasks.extend(self.phases[phase])
        return tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "plan_link": self.plan_link,
            "created_at": self.created_at.isoformat(),
            "phases": {
                str(phase): [t.to_dict() for t in tasks]
                for phase, tasks in self.phases.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskList":
        task_list = cls(
            feature_id=data["feature_id"],
            plan_link=data.get("plan_link", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )
        for phase_str, tasks in data.get("phases", {}).items():
            phase = int(phase_str)
            task_list.phases[phase] = [Task.from_dict(t) for t in tasks]
        return task_list


@dataclass
class ConstitutionPrinciple:
    """A single principle in the project constitution."""
    id: str
    title: str
    description: str
    rationale: str = ""
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstitutionPrinciple":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            rationale=data.get("rationale", ""),
            examples=data.get("examples", []),
        )


@dataclass
class Constitution:
    """Project constitution defining governance principles."""
    project_name: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Core principles
    principles: list[ConstitutionPrinciple] = field(default_factory=list)

    # Technical standards
    coding_standards: list[str] = field(default_factory=list)
    testing_requirements: list[str] = field(default_factory=list)
    documentation_standards: list[str] = field(default_factory=list)

    # Constraints
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "principles": [p.to_dict() for p in self.principles],
            "coding_standards": self.coding_standards,
            "testing_requirements": self.testing_requirements,
            "documentation_standards": self.documentation_standards,
            "forbidden_patterns": self.forbidden_patterns,
            "required_patterns": self.required_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Constitution":
        return cls(
            project_name=data["project_name"],
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            principles=[ConstitutionPrinciple.from_dict(p) for p in data.get("principles", [])],
            coding_standards=data.get("coding_standards", []),
            testing_requirements=data.get("testing_requirements", []),
            documentation_standards=data.get("documentation_standards", []),
            forbidden_patterns=data.get("forbidden_patterns", []),
            required_patterns=data.get("required_patterns", []),
        )

    def validate_against(self, plan: ImplementationPlan) -> list[str]:
        """Validate an implementation plan against this constitution."""
        violations = []

        # Check for forbidden patterns in source files
        for pattern in self.forbidden_patterns:
            for file in plan.source_files:
                if re.search(pattern, file):
                    violations.append(f"File {file} matches forbidden pattern: {pattern}")

        return violations


@dataclass
class SpecKit:
    """A complete SpecKit containing all specification artifacts."""
    feature_id: str
    root_path: Path
    constitution: Constitution | None = None
    specification: Specification | None = None
    plan: ImplementationPlan | None = None
    tasks: TaskList | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "root_path": str(self.root_path),
            "constitution": self.constitution.to_dict() if self.constitution else None,
            "specification": self.specification.to_dict() if self.specification else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "tasks": self.tasks.to_dict() if self.tasks else None,
        }

    def is_complete(self) -> bool:
        """Check if all artifacts are present."""
        return all([
            self.constitution,
            self.specification,
            self.plan,
            self.tasks,
        ])

    def get_status(self) -> dict[str, bool]:
        """Get status of each artifact."""
        return {
            "constitution": self.constitution is not None,
            "specification": self.specification is not None,
            "plan": self.plan is not None,
            "tasks": self.tasks is not None,
        }
