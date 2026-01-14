"""MCP tools for SpecKit operations.

These tools expose SpecKit functionality through the ATLAS MCP server,
enabling AI agents to create and manage specifications.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from atlas.speckit.models import (
    Constitution,
    ConstitutionPrinciple,
    DataEntity,
    FunctionalRequirement,
    ImplementationPlan,
    Priority,
    Specification,
    SpecStatus,
    SuccessCriteria,
    Task,
    TaskList,
    TechnicalContext,
    UserScenario,
)
from atlas.speckit.manager import SpecKitManager
from atlas.speckit.converter import convert_speckit_to_dag_override

logger = logging.getLogger(__name__)


async def speckit_init(
    repo_path: str,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Initialize a SpecKit in a repository.

    Creates the .speckit directory structure with default constitution.

    Args:
        repo_path: Path to the repository root.
        project_name: Optional project name for the constitution.

    Returns:
        Dictionary with initialization status and paths.
    """
    manager = SpecKitManager(repo_path)
    result = manager.initialize(project_name)

    logger.info(f"Initialized SpecKit at {repo_path}")
    return result


async def speckit_constitution(
    repo_path: str,
    project_name: str | None = None,
    principles: list[dict[str, Any]] | None = None,
    coding_standards: list[str] | None = None,
    testing_requirements: list[str] | None = None,
    documentation_standards: list[str] | None = None,
    forbidden_patterns: list[str] | None = None,
    required_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Create or update the project constitution.

    The constitution defines governance principles for the project,
    including coding standards, testing requirements, and forbidden patterns.

    Args:
        repo_path: Path to the repository root.
        project_name: Project name for the constitution.
        principles: List of principle definitions.
        coding_standards: List of coding standards.
        testing_requirements: List of testing requirements.
        documentation_standards: List of documentation standards.
        forbidden_patterns: List of forbidden code patterns (regex).
        required_patterns: List of required code patterns (regex).

    Returns:
        Dictionary with the saved constitution.
    """
    manager = SpecKitManager(repo_path)

    # Ensure initialized
    if not manager.is_initialized():
        manager.initialize(project_name)

    # Load existing or create new
    constitution = manager.get_constitution()
    if constitution is None:
        constitution = Constitution(
            project_name=project_name or Path(repo_path).name,
        )
    elif project_name:
        constitution.project_name = project_name

    # Update principles
    if principles:
        constitution.principles = [
            ConstitutionPrinciple(
                id=p.get("id", f"P-{i+1:03d}"),
                title=p.get("title", ""),
                description=p.get("description", ""),
                rationale=p.get("rationale", ""),
                examples=p.get("examples", []),
            )
            for i, p in enumerate(principles)
        ]

    # Update standards
    if coding_standards is not None:
        constitution.coding_standards = coding_standards
    if testing_requirements is not None:
        constitution.testing_requirements = testing_requirements
    if documentation_standards is not None:
        constitution.documentation_standards = documentation_standards
    if forbidden_patterns is not None:
        constitution.forbidden_patterns = forbidden_patterns
    if required_patterns is not None:
        constitution.required_patterns = required_patterns

    # Save
    path = manager.save_constitution(constitution)

    return {
        "status": "saved",
        "path": path,
        "constitution": constitution.to_dict(),
    }


async def speckit_specify(
    repo_path: str,
    feature_id: str,
    title: str,
    description: str,
    user_scenarios: list[dict[str, Any]] | None = None,
    functional_requirements: list[dict[str, Any]] | None = None,
    data_entities: list[dict[str, Any]] | None = None,
    success_criteria: list[dict[str, Any]] | None = None,
    assumptions: list[str] | None = None,
    constraints: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> dict[str, Any]:
    """Create or update a feature specification.

    Defines the WHAT - functional requirements, user scenarios,
    and acceptance criteria for a feature.

    Args:
        repo_path: Path to the repository root.
        feature_id: Unique identifier for the feature (e.g., "AUTH-001").
        title: Human-readable title.
        description: Description of what the feature does.
        user_scenarios: List of user scenario definitions.
        functional_requirements: List of requirement definitions.
        data_entities: List of data entity definitions.
        success_criteria: List of success criteria.
        assumptions: List of assumptions.
        constraints: List of constraints.
        open_questions: List of open questions.

    Returns:
        Dictionary with the saved specification.
    """
    manager = SpecKitManager(repo_path)

    # Ensure initialized
    if not manager.is_initialized():
        manager.initialize()

    # Load existing or create new
    spec = manager.get_specification(feature_id)
    if spec is None:
        spec = Specification(
            feature_id=feature_id,
            title=title,
            description=description,
        )
    else:
        spec.title = title
        spec.description = description

    # Update user scenarios
    if user_scenarios:
        spec.user_scenarios = [
            UserScenario(
                id=s.get("id", f"US-{i+1:03d}"),
                title=s.get("title", ""),
                description=s.get("description", ""),
                priority=Priority(s.get("priority", "P2")),
                given=s.get("given", []),
                when=s.get("when", []),
                then=s.get("then", []),
                edge_cases=s.get("edge_cases", []),
            )
            for i, s in enumerate(user_scenarios)
        ]

    # Update requirements
    if functional_requirements:
        spec.functional_requirements = [
            FunctionalRequirement(
                id=r.get("id", f"FR-{i+1:03d}"),
                description=r.get("description", ""),
                requirement_type=r.get("requirement_type", "MUST"),
                needs_clarification=r.get("needs_clarification", False),
                clarification_notes=r.get("clarification_notes", ""),
            )
            for i, r in enumerate(functional_requirements)
        ]

    # Update data entities
    if data_entities:
        spec.data_entities = [
            DataEntity(
                name=e.get("name", ""),
                description=e.get("description", ""),
                attributes=e.get("attributes", {}),
                relationships=e.get("relationships", []),
            )
            for e in data_entities
        ]

    # Update success criteria
    if success_criteria:
        spec.success_criteria = [
            SuccessCriteria(
                id=c.get("id", f"SC-{i+1:03d}"),
                metric=c.get("metric", ""),
                target=c.get("target", ""),
                measurement_method=c.get("measurement_method", ""),
            )
            for i, c in enumerate(success_criteria)
        ]

    # Update lists
    if assumptions is not None:
        spec.assumptions = assumptions
    if constraints is not None:
        spec.constraints = constraints
    if open_questions is not None:
        spec.open_questions = open_questions

    # Validate
    errors = spec.validate()
    if errors:
        logger.warning(f"Specification has validation warnings: {errors}")

    # Save
    path = manager.save_specification(spec)

    return {
        "status": "saved",
        "path": path,
        "feature_id": feature_id,
        "validation_warnings": errors,
        "specification": spec.to_dict(),
    }


async def speckit_plan(
    repo_path: str,
    feature_id: str,
    summary: str,
    language: str = "",
    version: str = "",
    dependencies: list[str] | None = None,
    storage: str = "",
    testing_framework: str = "",
    target_platforms: list[str] | None = None,
    performance_targets: dict[str, str] | None = None,
    technical_constraints: list[str] | None = None,
    architecture_decisions: list[str] | None = None,
    source_files: list[str] | None = None,
    test_files: list[str] | None = None,
    api_contracts: list[dict[str, Any]] | None = None,
    complexity_notes: list[str] | None = None,
) -> dict[str, Any]:
    """Create or update an implementation plan.

    Defines the HOW - technical approach, architecture decisions,
    and file structure for implementing a feature.

    Args:
        repo_path: Path to the repository root.
        feature_id: Feature identifier (must have existing spec).
        summary: Brief summary of the technical approach.
        language: Programming language.
        version: Language/framework version.
        dependencies: List of dependencies.
        storage: Storage technology.
        testing_framework: Testing framework to use.
        target_platforms: List of target platforms.
        performance_targets: Dict of performance targets.
        technical_constraints: List of technical constraints.
        architecture_decisions: List of architecture decisions.
        source_files: List of source files to create/modify.
        test_files: List of test files.
        api_contracts: List of API contract definitions.
        complexity_notes: Notes about complexity trade-offs.

    Returns:
        Dictionary with the saved plan.
    """
    manager = SpecKitManager(repo_path)

    # Load existing or create new
    plan = manager.get_plan(feature_id)
    if plan is None:
        plan = ImplementationPlan(
            feature_id=feature_id,
            spec_link=f".speckit/specs/{feature_id.lower()}/spec.md",
            summary=summary,
        )
    else:
        plan.summary = summary

    # Update technical context
    plan.technical_context = TechnicalContext(
        language=language,
        version=version,
        dependencies=dependencies or [],
        storage=storage,
        testing_framework=testing_framework,
        target_platforms=target_platforms or [],
        performance_targets=performance_targets or {},
        constraints=technical_constraints or [],
    )

    # Update other fields
    if architecture_decisions is not None:
        plan.architecture_decisions = architecture_decisions
    if source_files is not None:
        plan.source_files = source_files
    if test_files is not None:
        plan.test_files = test_files
    if api_contracts is not None:
        plan.api_contracts = api_contracts
    if complexity_notes is not None:
        plan.complexity_notes = complexity_notes

    # Save
    path = manager.save_plan(plan)

    return {
        "status": "saved",
        "path": path,
        "feature_id": feature_id,
        "plan": plan.to_dict(),
    }


async def speckit_tasks(
    repo_path: str,
    feature_id: str,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create or update the task list for a feature.

    Generates ordered, actionable task breakdowns from the plan.

    Args:
        repo_path: Path to the repository root.
        feature_id: Feature identifier (must have existing spec/plan).
        tasks: List of task definitions with:
            - id: Task identifier
            - description: What needs to be done
            - story_id: Link to user scenario (optional)
            - phase: Phase number (1=setup, 2=foundation, 3+=stories)
            - parallel: Whether task can run in parallel
            - file_paths: Files this task will modify
            - dependencies: Task IDs this depends on
            - acceptance_criteria: Criteria for completion

    Returns:
        Dictionary with the saved task list.
    """
    manager = SpecKitManager(repo_path)

    # Create task list
    task_list = TaskList(
        feature_id=feature_id,
        plan_link=f".speckit/specs/{feature_id.lower()}/plan.md",
    )

    for t in tasks:
        task = Task(
            id=t.get("id", ""),
            description=t.get("description", ""),
            story_id=t.get("story_id", ""),
            phase=t.get("phase", 1),
            parallel=t.get("parallel", False),
            file_paths=t.get("file_paths", []),
            dependencies=t.get("dependencies", []),
            acceptance_criteria=t.get("acceptance_criteria", []),
            completed=t.get("completed", False),
        )
        task_list.add_task(task)

    # Save
    path = manager.save_tasks(task_list)

    return {
        "status": "saved",
        "path": path,
        "feature_id": feature_id,
        "task_count": len(tasks),
        "phases": list(task_list.phases.keys()),
        "tasks": task_list.to_dict(),
    }


async def speckit_implement(
    repo_path: str,
    feature_id: str,
    test_command: str | None = None,
    max_cost_usd: float = 10.0,
    timeout_minutes: int = 60,
) -> dict[str, Any]:
    """Execute the implementation using ATLAS multi-agent system.

    Converts the SpecKit to a TaskDAG and uses ATLAS's solve_feature_dag
    to implement the feature with multi-agent consensus.

    Args:
        repo_path: Path to the repository root.
        feature_id: Feature identifier (must have spec, plan, tasks).
        test_command: Test command to run for validation.
        max_cost_usd: Maximum cost budget.
        timeout_minutes: Maximum time in minutes.

    Returns:
        Dictionary with dag_override for solve_feature_dag or error.
    """
    manager = SpecKitManager(repo_path)

    # Load the complete speckit
    speckit = manager.get_speckit(feature_id)

    # Validate completeness
    status = speckit.get_status()
    missing = [k for k, v in status.items() if not v]
    if missing:
        return {
            "status": "incomplete",
            "error": f"SpecKit is missing: {', '.join(missing)}",
            "status_details": status,
        }

    # Validate specification
    if speckit.specification:
        errors = speckit.specification.validate()
        if errors:
            return {
                "status": "invalid",
                "error": "Specification has validation errors",
                "validation_errors": errors,
            }

    # Convert to dag_override format
    try:
        dag_override = convert_speckit_to_dag_override(speckit)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to convert SpecKit to TaskDAG: {e}",
        }

    # Return the dag_override for use with solve_feature_dag
    # The actual execution should be done by calling solve_feature_dag
    return {
        "status": "ready",
        "feature_id": feature_id,
        "dag_override": dag_override,
        "task_count": len(dag_override.get("tasks", [])),
        "suggested_params": {
            "repo_url": repo_path,
            "description": speckit.specification.description if speckit.specification else "",
            "test_command": test_command,
            "max_cost_usd": max_cost_usd,
            "timeout_minutes": timeout_minutes,
            "dag_override": dag_override,
        },
    }


async def speckit_status(
    repo_path: str,
    feature_id: str | None = None,
) -> dict[str, Any]:
    """Get the status of a SpecKit or list all features.

    Args:
        repo_path: Path to the repository root.
        feature_id: Optional feature ID. If not provided, lists all features.

    Returns:
        Dictionary with status information.
    """
    manager = SpecKitManager(repo_path)

    if not manager.is_initialized():
        return {
            "status": "not_initialized",
            "message": "SpecKit not initialized. Run speckit_init first.",
        }

    if feature_id:
        # Get specific feature status
        speckit = manager.get_speckit(feature_id)
        return {
            "status": "ok",
            "feature_id": feature_id,
            "artifacts": speckit.get_status(),
            "is_complete": speckit.is_complete(),
            "speckit": speckit.to_dict(),
        }
    else:
        # List all features
        features = manager.list_features()
        constitution = manager.get_constitution()

        return {
            "status": "ok",
            "has_constitution": constitution is not None,
            "project_name": constitution.project_name if constitution else None,
            "feature_count": len(features),
            "features": features,
        }


async def speckit_export(
    repo_path: str,
    feature_id: str,
) -> dict[str, Any]:
    """Export a SpecKit as a formatted string for LLM consumption.

    Useful for providing context to AI agents about the feature spec.

    Args:
        repo_path: Path to the repository root.
        feature_id: Feature identifier.

    Returns:
        Dictionary with the exported content.
    """
    manager = SpecKitManager(repo_path)

    if not manager.is_initialized():
        return {
            "status": "error",
            "error": "SpecKit not initialized",
        }

    try:
        content = manager.export_for_llm(feature_id)
        return {
            "status": "ok",
            "feature_id": feature_id,
            "content": content,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
