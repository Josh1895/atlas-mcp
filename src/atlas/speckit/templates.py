"""SpecKit templates for specification documents.

These templates are based on GitHub's spec-kit framework and provide
structured formats for creating executable specifications.
"""

from __future__ import annotations

from datetime import datetime

# Specification Template
SPEC_TEMPLATE = '''# {feature_id}: {title}

**Branch:** `{branch_name}`
**Created:** {created_date}
**Status:** {status}

## Overview

{description}

---

## User Scenarios & Testing

{user_scenarios_section}

---

## Functional Requirements

{requirements_section}

---

## Data Entities

{data_entities_section}

---

## Success Criteria

{success_criteria_section}

---

## Assumptions

{assumptions_section}

---

## Constraints

{constraints_section}

---

## Open Questions

{open_questions_section}
'''

USER_SCENARIO_TEMPLATE = '''### {id}: {title}
**Priority:** {priority}

{description}

**Acceptance Criteria:**
```gherkin
Given:
{given_clauses}

When:
{when_clauses}

Then:
{then_clauses}
```

**Edge Cases:**
{edge_cases}
'''

REQUIREMENT_TEMPLATE = '''- **{id}**: {requirement_type} {description}{clarification}
'''

# Plan Template
PLAN_TEMPLATE = '''# Implementation Plan: {feature_id}

**Specification:** {spec_link}
**Created:** {created_date}

## Summary

{summary}

---

## Technical Context

| Aspect | Value |
|--------|-------|
| Language | {language} |
| Version | {version} |
| Storage | {storage} |
| Testing Framework | {testing_framework} |
| Target Platforms | {target_platforms} |

### Dependencies
{dependencies_section}

### Performance Targets
{performance_targets_section}

### Constraints
{constraints_section}

---

## Architecture Decisions

{architecture_decisions_section}

---

## Project Structure

### Source Files
```
{source_files_section}
```

### Test Files
```
{test_files_section}
```

---

## API Contracts

{api_contracts_section}

---

## Complexity Notes

{complexity_notes_section}
'''

# Tasks Template
TASKS_TEMPLATE = '''# Tasks: {feature_id}

**Plan:** {plan_link}
**Created:** {created_date}

## Task Format
`[ID] [P?] [Story] Description`
- P = Parallelizable (no file conflicts)
- Story = User story reference

---

{phases_section}
'''

PHASE_TEMPLATE = '''## Phase {phase_num}: {phase_name}

{phase_description}

{tasks_section}
'''

TASK_TEMPLATE = '''- [{id}]{parallel} [{story_id}] {description}
  - Files: {file_paths}
  - Depends on: {dependencies}
  - Acceptance: {acceptance_criteria}
'''

# Constitution Template
CONSTITUTION_TEMPLATE = '''# Project Constitution: {project_name}

**Version:** {version}
**Created:** {created_date}
**Updated:** {updated_date}

---

## Core Principles

{principles_section}

---

## Coding Standards

{coding_standards_section}

---

## Testing Requirements

{testing_requirements_section}

---

## Documentation Standards

{documentation_standards_section}

---

## Patterns

### Required Patterns
{required_patterns_section}

### Forbidden Patterns
{forbidden_patterns_section}
'''

PRINCIPLE_TEMPLATE = '''### {id}: {title}

{description}

**Rationale:** {rationale}

**Examples:**
{examples}
'''


def format_spec(
    feature_id: str,
    title: str,
    description: str,
    branch_name: str = "",
    status: str = "Draft",
    user_scenarios: list | None = None,
    functional_requirements: list | None = None,
    data_entities: list | None = None,
    success_criteria: list | None = None,
    assumptions: list | None = None,
    constraints: list | None = None,
    open_questions: list | None = None,
) -> str:
    """Format a specification document from components."""
    user_scenarios = user_scenarios or []
    functional_requirements = functional_requirements or []
    data_entities = data_entities or []
    success_criteria = success_criteria or []
    assumptions = assumptions or []
    constraints = constraints or []
    open_questions = open_questions or []

    # Format user scenarios
    scenarios_text = ""
    for scenario in user_scenarios:
        given_clauses = "\n".join(f"  - {g}" for g in scenario.get("given", []))
        when_clauses = "\n".join(f"  - {w}" for w in scenario.get("when", []))
        then_clauses = "\n".join(f"  - {t}" for t in scenario.get("then", []))
        edge_cases = "\n".join(f"- {e}" for e in scenario.get("edge_cases", []))

        scenarios_text += USER_SCENARIO_TEMPLATE.format(
            id=scenario.get("id", "US-001"),
            title=scenario.get("title", ""),
            priority=scenario.get("priority", "P2"),
            description=scenario.get("description", ""),
            given_clauses=given_clauses or "  - (none specified)",
            when_clauses=when_clauses or "  - (none specified)",
            then_clauses=then_clauses or "  - (none specified)",
            edge_cases=edge_cases or "- (none specified)",
        )
        scenarios_text += "\n"

    # Format requirements
    requirements_text = ""
    for req in functional_requirements:
        clarification = ""
        if req.get("needs_clarification"):
            clarification = f" [NEEDS CLARIFICATION: {req.get('clarification_notes', '')}]"
        requirements_text += REQUIREMENT_TEMPLATE.format(
            id=req.get("id", "FR-001"),
            requirement_type=req.get("requirement_type", "MUST"),
            description=req.get("description", ""),
            clarification=clarification,
        )

    # Format data entities
    entities_text = ""
    for entity in data_entities:
        entities_text += f"### {entity.get('name', 'Entity')}\n\n"
        entities_text += f"{entity.get('description', '')}\n\n"
        if entity.get("attributes"):
            entities_text += "**Attributes:**\n"
            for attr, desc in entity["attributes"].items():
                entities_text += f"- `{attr}`: {desc}\n"
        if entity.get("relationships"):
            entities_text += "\n**Relationships:**\n"
            for rel in entity["relationships"]:
                entities_text += f"- {rel}\n"
        entities_text += "\n"

    # Format success criteria
    criteria_text = ""
    for criterion in success_criteria:
        criteria_text += f"- **{criterion.get('id', 'SC-001')}**: {criterion.get('metric', '')}\n"
        criteria_text += f"  - Target: {criterion.get('target', '')}\n"
        if criterion.get("measurement_method"):
            criteria_text += f"  - Measurement: {criterion['measurement_method']}\n"

    # Format lists
    assumptions_text = "\n".join(f"- {a}" for a in assumptions) or "- (none)"
    constraints_text = "\n".join(f"- {c}" for c in constraints) or "- (none)"
    questions_text = "\n".join(f"- {q}" for q in open_questions) or "- (none)"

    return SPEC_TEMPLATE.format(
        feature_id=feature_id,
        title=title,
        branch_name=branch_name or f"{feature_id.lower()}-feature",
        created_date=datetime.now().strftime("%Y-%m-%d"),
        status=status,
        description=description,
        user_scenarios_section=scenarios_text or "*(No user scenarios defined)*",
        requirements_section=requirements_text or "*(No requirements defined)*",
        data_entities_section=entities_text or "*(No data entities defined)*",
        success_criteria_section=criteria_text or "*(No success criteria defined)*",
        assumptions_section=assumptions_text,
        constraints_section=constraints_text,
        open_questions_section=questions_text,
    )


def format_plan(
    feature_id: str,
    spec_link: str,
    summary: str,
    technical_context: dict | None = None,
    architecture_decisions: list | None = None,
    source_files: list | None = None,
    test_files: list | None = None,
    api_contracts: list | None = None,
    complexity_notes: list | None = None,
) -> str:
    """Format an implementation plan document."""
    tech = technical_context or {}
    architecture_decisions = architecture_decisions or []
    source_files = source_files or []
    test_files = test_files or []
    api_contracts = api_contracts or []
    complexity_notes = complexity_notes or []

    # Format dependencies
    deps = tech.get("dependencies", [])
    deps_text = "\n".join(f"- {d}" for d in deps) or "- (none)"

    # Format performance targets
    perf = tech.get("performance_targets", {})
    perf_text = ""
    for metric, target in perf.items():
        perf_text += f"- **{metric}**: {target}\n"
    perf_text = perf_text or "- (none specified)"

    # Format constraints
    constraints = tech.get("constraints", [])
    constraints_text = "\n".join(f"- {c}" for c in constraints) or "- (none)"

    # Format architecture decisions
    arch_text = ""
    for i, decision in enumerate(architecture_decisions, 1):
        arch_text += f"{i}. {decision}\n"
    arch_text = arch_text or "*(No architecture decisions documented)*"

    # Format file lists
    source_text = "\n".join(source_files) or "(no source files defined)"
    test_text = "\n".join(test_files) or "(no test files defined)"

    # Format API contracts
    contracts_text = ""
    for contract in api_contracts:
        contracts_text += f"### {contract.get('name', 'API')}\n\n"
        contracts_text += f"```json\n{contract.get('schema', '{}')}\n```\n\n"
    contracts_text = contracts_text or "*(No API contracts defined)*"

    # Format complexity notes
    complexity_text = "\n".join(f"- {n}" for n in complexity_notes) or "- (none)"

    platforms = tech.get("target_platforms", [])

    return PLAN_TEMPLATE.format(
        feature_id=feature_id,
        spec_link=spec_link,
        created_date=datetime.now().strftime("%Y-%m-%d"),
        summary=summary,
        language=tech.get("language", "(not specified)"),
        version=tech.get("version", "(not specified)"),
        storage=tech.get("storage", "(not specified)"),
        testing_framework=tech.get("testing_framework", "(not specified)"),
        target_platforms=", ".join(platforms) if platforms else "(not specified)",
        dependencies_section=deps_text,
        performance_targets_section=perf_text,
        constraints_section=constraints_text,
        architecture_decisions_section=arch_text,
        source_files_section=source_text,
        test_files_section=test_text,
        api_contracts_section=contracts_text,
        complexity_notes_section=complexity_text,
    )


def format_tasks(
    feature_id: str,
    plan_link: str,
    phases: dict | None = None,
) -> str:
    """Format a tasks document."""
    phases = phases or {}

    phase_names = {
        1: ("Setup", "Project initialization and basic structure"),
        2: ("Foundation", "Core infrastructure that MUST be complete before ANY user story"),
        3: ("User Stories", "Individual feature implementations"),
        4: ("Integration", "Cross-cutting concerns and integration"),
        5: ("Polish", "Final improvements and optimizations"),
    }

    phases_text = ""
    for phase_num in sorted(phases.keys()):
        tasks = phases[phase_num]
        name, desc = phase_names.get(phase_num, (f"Phase {phase_num}", ""))

        tasks_text = ""
        for task in tasks:
            parallel = " [P]" if task.get("parallel") else ""
            files = ", ".join(task.get("file_paths", [])) or "(none)"
            deps = ", ".join(task.get("dependencies", [])) or "(none)"
            criteria = "; ".join(task.get("acceptance_criteria", [])) or "(none)"

            tasks_text += TASK_TEMPLATE.format(
                id=task.get("id", "T-001"),
                parallel=parallel,
                story_id=task.get("story_id", "-"),
                description=task.get("description", ""),
                file_paths=files,
                dependencies=deps,
                acceptance_criteria=criteria,
            )

        phases_text += PHASE_TEMPLATE.format(
            phase_num=phase_num,
            phase_name=name,
            phase_description=desc,
            tasks_section=tasks_text or "*(No tasks in this phase)*",
        )
        phases_text += "\n"

    return TASKS_TEMPLATE.format(
        feature_id=feature_id,
        plan_link=plan_link,
        created_date=datetime.now().strftime("%Y-%m-%d"),
        phases_section=phases_text or "*(No phases defined)*",
    )


def format_constitution(
    project_name: str,
    version: str = "1.0",
    principles: list | None = None,
    coding_standards: list | None = None,
    testing_requirements: list | None = None,
    documentation_standards: list | None = None,
    required_patterns: list | None = None,
    forbidden_patterns: list | None = None,
) -> str:
    """Format a constitution document."""
    principles = principles or []
    coding_standards = coding_standards or []
    testing_requirements = testing_requirements or []
    documentation_standards = documentation_standards or []
    required_patterns = required_patterns or []
    forbidden_patterns = forbidden_patterns or []

    # Format principles
    principles_text = ""
    for principle in principles:
        examples = "\n".join(f"- {e}" for e in principle.get("examples", []))
        principles_text += PRINCIPLE_TEMPLATE.format(
            id=principle.get("id", "P-001"),
            title=principle.get("title", ""),
            description=principle.get("description", ""),
            rationale=principle.get("rationale", "(not specified)"),
            examples=examples or "- (none provided)",
        )
        principles_text += "\n"

    principles_text = principles_text or "*(No principles defined)*"

    # Format standards
    coding_text = "\n".join(f"- {s}" for s in coding_standards) or "- (none defined)"
    testing_text = "\n".join(f"- {r}" for r in testing_requirements) or "- (none defined)"
    docs_text = "\n".join(f"- {d}" for d in documentation_standards) or "- (none defined)"

    # Format patterns
    required_text = "\n".join(f"- `{p}`" for p in required_patterns) or "- (none)"
    forbidden_text = "\n".join(f"- `{p}`" for p in forbidden_patterns) or "- (none)"

    now = datetime.now().strftime("%Y-%m-%d")

    return CONSTITUTION_TEMPLATE.format(
        project_name=project_name,
        version=version,
        created_date=now,
        updated_date=now,
        principles_section=principles_text,
        coding_standards_section=coding_text,
        testing_requirements_section=testing_text,
        documentation_standards_section=docs_text,
        required_patterns_section=required_text,
        forbidden_patterns_section=forbidden_text,
    )


# Default constitution for ATLAS projects
DEFAULT_ATLAS_CONSTITUTION = {
    "project_name": "ATLAS Project",
    "version": "1.0",
    "principles": [
        {
            "id": "P-001",
            "title": "Tests as Primary Oracle",
            "description": "Tests are the primary correctness oracle. Patches are validated by running actual tests, not by similarity matching.",
            "rationale": "Behavioral correctness is more important than syntactic similarity.",
            "examples": [
                "Always provide test_command when solving issues",
                "Prefer comprehensive test suites over manual verification",
            ],
        },
        {
            "id": "P-002",
            "title": "Multi-Agent Consensus",
            "description": "Use diverse AI agents with different perspectives to generate solutions, then use voting to select the best.",
            "rationale": "Diverse approaches reduce single-point-of-failure and improve solution quality.",
            "examples": [
                "Use at least 5 agents with different prompt styles",
                "Wait for consensus (K votes ahead) before selecting",
            ],
        },
        {
            "id": "P-003",
            "title": "Contract-Driven Development",
            "description": "Complex features should be decomposed into tasks with clear contracts specifying what MUST be true.",
            "rationale": "Clear contracts enable independent verification and parallel development.",
            "examples": [
                "Each task has a contract describing expected behavior",
                "Ownership rules limit scope to prevent conflicts",
            ],
        },
    ],
    "coding_standards": [
        "Use type hints for all function signatures",
        "Follow PEP 8 style guidelines",
        "Keep functions under 50 lines where possible",
        "Use dataclasses for data structures",
    ],
    "testing_requirements": [
        "All new features must have unit tests",
        "Integration tests for cross-module functionality",
        "Test coverage should not decrease",
    ],
    "documentation_standards": [
        "All public functions must have docstrings",
        "Complex algorithms must have inline comments",
        "Update README when adding new features",
    ],
    "required_patterns": [],
    "forbidden_patterns": [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
    ],
}
