"""SpecKit module for specification-driven development.

This module provides integration with GitHub's spec-kit framework,
enabling ATLAS to generate code from well-defined specifications.

Key components:
- models: Data models for specifications, plans, tasks, and constitutions
- templates: Markdown templates for spec documents
- manager: File I/O and lifecycle management
- converter: Transform specs to ATLAS TaskDAGs
"""

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
from atlas.speckit.manager import (
    SpecKitManager,
    get_speckit_manager,
)
from atlas.speckit.converter import (
    SpecKitConverter,
    convert_speckit_to_dag,
    convert_speckit_to_dag_override,
)

__all__ = [
    # Models
    "Constitution",
    "ConstitutionPrinciple",
    "DataEntity",
    "FunctionalRequirement",
    "ImplementationPlan",
    "Priority",
    "SpecKit",
    "Specification",
    "SpecStatus",
    "SuccessCriteria",
    "Task",
    "TaskList",
    "TechnicalContext",
    "UserScenario",
    # Manager
    "SpecKitManager",
    "get_speckit_manager",
    # Converter
    "SpecKitConverter",
    "convert_speckit_to_dag",
    "convert_speckit_to_dag_override",
]
