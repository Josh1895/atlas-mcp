"""SpecKit manager for creating and managing specification kits.

Handles file I/O, directory structure, and lifecycle of spec kits.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from atlas.speckit.models import (
    Constitution,
    ConstitutionPrinciple,
    DataEntity,
    FunctionalRequirement,
    ImplementationPlan,
    Priority,
    SpecKit,
    Specification,
    SpecStatus,
    SuccessCriteria,
    Task,
    TaskList,
    TechnicalContext,
    UserScenario,
)
from atlas.speckit.templates import (
    DEFAULT_ATLAS_CONSTITUTION,
    format_constitution,
    format_plan,
    format_spec,
    format_tasks,
)

logger = logging.getLogger(__name__)


class SpecKitManager:
    """Manages SpecKit lifecycle and file operations."""

    SPECKIT_DIR = ".speckit"
    MEMORY_DIR = "memory"
    SPECS_DIR = "specs"
    TEMPLATES_DIR = "templates"

    def __init__(self, root_path: str | Path):
        """Initialize the manager with a root path.

        Args:
            root_path: Path to the project root (e.g., cloned repository).
        """
        self.root_path = Path(root_path)
        self.speckit_path = self.root_path / self.SPECKIT_DIR

    def initialize(self, project_name: str | None = None) -> dict[str, Any]:
        """Initialize the .speckit directory structure.

        Args:
            project_name: Optional project name for the constitution.

        Returns:
            Dictionary with initialization status and paths.
        """
        project_name = project_name or self.root_path.name

        # Create directory structure
        dirs_created = []
        for subdir in [self.MEMORY_DIR, self.SPECS_DIR, self.TEMPLATES_DIR]:
            dir_path = self.speckit_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            dirs_created.append(str(dir_path))

        # Create default constitution if not exists
        constitution_path = self.speckit_path / self.MEMORY_DIR / "constitution.md"
        constitution_json = self.speckit_path / self.MEMORY_DIR / "constitution.json"

        if not constitution_path.exists():
            # Create default constitution
            default_const = DEFAULT_ATLAS_CONSTITUTION.copy()
            default_const["project_name"] = project_name

            # Write markdown
            md_content = format_constitution(**default_const)
            constitution_path.write_text(md_content, encoding="utf-8")

            # Write JSON for machine reading
            constitution_json.write_text(
                json.dumps(default_const, indent=2),
                encoding="utf-8",
            )

            logger.info(f"Created default constitution for {project_name}")

        return {
            "status": "initialized",
            "root_path": str(self.root_path),
            "speckit_path": str(self.speckit_path),
            "directories_created": dirs_created,
            "constitution_exists": constitution_path.exists(),
        }

    def is_initialized(self) -> bool:
        """Check if .speckit is initialized."""
        return self.speckit_path.exists() and (
            self.speckit_path / self.MEMORY_DIR / "constitution.json"
        ).exists()

    def get_constitution(self) -> Constitution | None:
        """Load the project constitution."""
        constitution_json = self.speckit_path / self.MEMORY_DIR / "constitution.json"

        if not constitution_json.exists():
            return None

        data = json.loads(constitution_json.read_text(encoding="utf-8"))
        return Constitution.from_dict(data)

    def save_constitution(self, constitution: Constitution) -> str:
        """Save the project constitution.

        Args:
            constitution: Constitution to save.

        Returns:
            Path to saved constitution.
        """
        constitution.updated_at = datetime.now()

        # Ensure directory exists
        memory_dir = self.speckit_path / self.MEMORY_DIR
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = memory_dir / "constitution.json"
        json_path.write_text(
            json.dumps(constitution.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Save markdown
        md_path = memory_dir / "constitution.md"
        md_content = format_constitution(
            project_name=constitution.project_name,
            version=constitution.version,
            principles=[p.to_dict() for p in constitution.principles],
            coding_standards=constitution.coding_standards,
            testing_requirements=constitution.testing_requirements,
            documentation_standards=constitution.documentation_standards,
            required_patterns=constitution.required_patterns,
            forbidden_patterns=constitution.forbidden_patterns,
        )
        md_path.write_text(md_content, encoding="utf-8")

        return str(json_path)

    def create_feature(self, feature_id: str, title: str, description: str) -> str:
        """Create a new feature directory with initial spec.

        Args:
            feature_id: Unique identifier for the feature (e.g., "AUTH-001").
            title: Human-readable title.
            description: Brief description of the feature.

        Returns:
            Path to the feature directory.
        """
        # Sanitize feature_id
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "-", feature_id).lower()
        feature_dir = self.speckit_path / self.SPECS_DIR / safe_id

        # Create directories
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "contracts").mkdir(exist_ok=True)

        # Create initial spec
        spec = Specification(
            feature_id=feature_id,
            title=title,
            description=description,
            status=SpecStatus.DRAFT,
        )

        self.save_specification(spec)

        logger.info(f"Created feature directory: {feature_dir}")
        return str(feature_dir)

    def get_feature_path(self, feature_id: str) -> Path:
        """Get the path to a feature directory."""
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "-", feature_id).lower()
        return self.speckit_path / self.SPECS_DIR / safe_id

    def list_features(self) -> list[dict[str, Any]]:
        """List all features in the specs directory."""
        specs_dir = self.speckit_path / self.SPECS_DIR

        if not specs_dir.exists():
            return []

        features = []
        for item in specs_dir.iterdir():
            if item.is_dir():
                spec_json = item / "spec.json"
                if spec_json.exists():
                    spec_data = json.loads(spec_json.read_text(encoding="utf-8"))
                    features.append({
                        "feature_id": spec_data.get("feature_id", item.name),
                        "title": spec_data.get("title", ""),
                        "status": spec_data.get("status", "draft"),
                        "path": str(item),
                    })
                else:
                    features.append({
                        "feature_id": item.name,
                        "title": "",
                        "status": "unknown",
                        "path": str(item),
                    })

        return features

    def get_specification(self, feature_id: str) -> Specification | None:
        """Load a specification by feature ID."""
        feature_path = self.get_feature_path(feature_id)
        spec_json = feature_path / "spec.json"

        if not spec_json.exists():
            return None

        data = json.loads(spec_json.read_text(encoding="utf-8"))
        return Specification.from_dict(data)

    def save_specification(self, spec: Specification) -> str:
        """Save a specification.

        Args:
            spec: Specification to save.

        Returns:
            Path to saved specification.
        """
        spec.updated_at = datetime.now()
        feature_path = self.get_feature_path(spec.feature_id)
        feature_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = feature_path / "spec.json"
        json_path.write_text(
            json.dumps(spec.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Save markdown
        md_path = feature_path / "spec.md"
        md_content = format_spec(
            feature_id=spec.feature_id,
            title=spec.title,
            description=spec.description,
            status=spec.status.value,
            user_scenarios=[s.to_dict() for s in spec.user_scenarios],
            functional_requirements=[r.to_dict() for r in spec.functional_requirements],
            data_entities=[e.to_dict() for e in spec.data_entities],
            success_criteria=[c.to_dict() for c in spec.success_criteria],
            assumptions=spec.assumptions,
            constraints=spec.constraints,
            open_questions=spec.open_questions,
        )
        md_path.write_text(md_content, encoding="utf-8")

        return str(json_path)

    def get_plan(self, feature_id: str) -> ImplementationPlan | None:
        """Load an implementation plan by feature ID."""
        feature_path = self.get_feature_path(feature_id)
        plan_json = feature_path / "plan.json"

        if not plan_json.exists():
            return None

        data = json.loads(plan_json.read_text(encoding="utf-8"))
        return ImplementationPlan.from_dict(data)

    def save_plan(self, plan: ImplementationPlan) -> str:
        """Save an implementation plan.

        Args:
            plan: Implementation plan to save.

        Returns:
            Path to saved plan.
        """
        feature_path = self.get_feature_path(plan.feature_id)
        feature_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = feature_path / "plan.json"
        json_path.write_text(
            json.dumps(plan.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Save markdown
        md_path = feature_path / "plan.md"
        md_content = format_plan(
            feature_id=plan.feature_id,
            spec_link=plan.spec_link,
            summary=plan.summary,
            technical_context=plan.technical_context.to_dict(),
            architecture_decisions=plan.architecture_decisions,
            source_files=plan.source_files,
            test_files=plan.test_files,
            api_contracts=plan.api_contracts,
            complexity_notes=plan.complexity_notes,
        )
        md_path.write_text(md_content, encoding="utf-8")

        return str(json_path)

    def get_tasks(self, feature_id: str) -> TaskList | None:
        """Load a task list by feature ID."""
        feature_path = self.get_feature_path(feature_id)
        tasks_json = feature_path / "tasks.json"

        if not tasks_json.exists():
            return None

        data = json.loads(tasks_json.read_text(encoding="utf-8"))
        return TaskList.from_dict(data)

    def save_tasks(self, tasks: TaskList) -> str:
        """Save a task list.

        Args:
            tasks: Task list to save.

        Returns:
            Path to saved tasks.
        """
        feature_path = self.get_feature_path(tasks.feature_id)
        feature_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = feature_path / "tasks.json"
        json_path.write_text(
            json.dumps(tasks.to_dict(), indent=2),
            encoding="utf-8",
        )

        # Save markdown
        md_path = feature_path / "tasks.md"

        # Convert tasks to dict format for template
        phases_dict = {}
        for phase, phase_tasks in tasks.phases.items():
            phases_dict[phase] = [t.to_dict() for t in phase_tasks]

        md_content = format_tasks(
            feature_id=tasks.feature_id,
            plan_link=tasks.plan_link,
            phases=phases_dict,
        )
        md_path.write_text(md_content, encoding="utf-8")

        return str(json_path)

    def get_speckit(self, feature_id: str) -> SpecKit:
        """Load a complete SpecKit for a feature.

        Args:
            feature_id: Feature identifier.

        Returns:
            SpecKit with all available artifacts.
        """
        return SpecKit(
            feature_id=feature_id,
            root_path=self.root_path,
            constitution=self.get_constitution(),
            specification=self.get_specification(feature_id),
            plan=self.get_plan(feature_id),
            tasks=self.get_tasks(feature_id),
        )

    def save_speckit(self, speckit: SpecKit) -> dict[str, str]:
        """Save all artifacts in a SpecKit.

        Args:
            speckit: SpecKit to save.

        Returns:
            Dictionary mapping artifact names to saved paths.
        """
        saved = {}

        if speckit.constitution:
            saved["constitution"] = self.save_constitution(speckit.constitution)

        if speckit.specification:
            saved["specification"] = self.save_specification(speckit.specification)

        if speckit.plan:
            saved["plan"] = self.save_plan(speckit.plan)

        if speckit.tasks:
            saved["tasks"] = self.save_tasks(speckit.tasks)

        return saved

    def export_for_llm(self, feature_id: str) -> str:
        """Export a SpecKit as a single string for LLM consumption.

        Args:
            feature_id: Feature identifier.

        Returns:
            Formatted string with all spec information.
        """
        speckit = self.get_speckit(feature_id)

        parts = [f"# SpecKit: {feature_id}\n"]

        if speckit.constitution:
            parts.append("## Constitution\n")
            parts.append(f"Project: {speckit.constitution.project_name}\n")
            parts.append(f"Principles: {len(speckit.constitution.principles)}\n")
            for p in speckit.constitution.principles:
                parts.append(f"- {p.title}: {p.description}\n")

        if speckit.specification:
            parts.append("\n## Specification\n")
            parts.append(f"Title: {speckit.specification.title}\n")
            parts.append(f"Status: {speckit.specification.status.value}\n")
            parts.append(f"Description: {speckit.specification.description}\n")

            if speckit.specification.user_scenarios:
                parts.append("\n### User Scenarios\n")
                for scenario in speckit.specification.user_scenarios:
                    parts.append(f"- [{scenario.priority.value}] {scenario.title}\n")

            if speckit.specification.functional_requirements:
                parts.append("\n### Requirements\n")
                for req in speckit.specification.functional_requirements:
                    parts.append(f"- {req.id}: {req.requirement_type} {req.description}\n")

        if speckit.plan:
            parts.append("\n## Implementation Plan\n")
            parts.append(f"Summary: {speckit.plan.summary}\n")

            if speckit.plan.source_files:
                parts.append("\n### Files to Create/Modify\n")
                for f in speckit.plan.source_files:
                    parts.append(f"- {f}\n")

        if speckit.tasks:
            parts.append("\n## Tasks\n")
            for phase, phase_tasks in sorted(speckit.tasks.phases.items()):
                parts.append(f"\n### Phase {phase}\n")
                for task in phase_tasks:
                    status = "[x]" if task.completed else "[ ]"
                    parts.append(f"- {status} {task.id}: {task.description}\n")

        return "".join(parts)


# Singleton instance
_manager: SpecKitManager | None = None


def get_speckit_manager(root_path: str | Path | None = None) -> SpecKitManager:
    """Get or create the singleton SpecKitManager.

    Args:
        root_path: Path to project root. Required on first call.

    Returns:
        SpecKitManager instance.
    """
    global _manager

    if root_path:
        _manager = SpecKitManager(root_path)
    elif _manager is None:
        raise ValueError("root_path is required for first initialization")

    return _manager
