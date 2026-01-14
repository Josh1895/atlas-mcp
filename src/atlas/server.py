"""ATLAS MCP Server - Main entry point.

This module provides the MCP server interface for ATLAS,
exposing tools for code generation via multi-agent consensus.
"""

import asyncio
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from atlas.core.config import Config, get_config
from atlas.core.orchestrator import ATLASOrchestrator, get_orchestrator
from atlas.core.task import TaskSubmission, TaskStatus
from atlas.core.dag_orchestrator import (
    TaskDAGSubmission,
    get_dag_orchestrator,
)
from atlas.core.task_dag import TaskDAG
from atlas.speckit import tools as speckit_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP(
    name="atlas",
    instructions="ATLAS: Multi-agent code generation with consensus voting",
)


def _serialize_dag(dag: TaskDAG) -> dict[str, Any]:
    tasks = []
    try:
        ordered_tasks = dag.topological_order()
    except Exception:
        ordered_tasks = list(dag.tasks.values())

    for task in ordered_tasks:
        tasks.append({
            "id": task.task_id,
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
                    "type": oracle.oracle_type.value,
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


@mcp.tool()
async def solve_issue(
    repo_path: str,
    issue_description: str,
    branch: str = "main",
    base_commit: str | None = None,
    relevant_files: list[str] | None = None,
    test_command: str | None = None,
    max_cost_usd: float = 2.0,
    timeout_minutes: int = 15,
) -> dict[str, Any]:
    """Solve an issue using multi-agent consensus.

    Generates patches using multiple diverse AI agents, validates them,
    and uses voting to select the best solution.

    Works with both local paths and remote git URLs.

    Args:
        repo_path: Local folder path (e.g., "./my-project") OR git URL
        issue_description: Natural language description of the issue to fix
        branch: Target branch (default: main)
        base_commit: Specific commit to base the fix on (optional)
        relevant_files: List of files likely to need changes (optional)
        test_command: Command to run tests (optional, used as primary oracle)
        max_cost_usd: Maximum cost budget in USD (default: 2.0)
        timeout_minutes: Maximum time in minutes (default: 15)

    Returns:
        Dictionary with task_id and initial status.
        Use check_status() and get_result() to monitor progress.
    """
    config = get_config()

    # Validate config
    errors = config.validate()
    if errors:
        return {
            "status": "error",
            "error": "Configuration error",
            "details": errors,
        }

    # Create task submission
    task = TaskSubmission(
        description=issue_description,
        repository_url=repo_path,
        branch=branch,
        base_commit=base_commit,
        relevant_files=relevant_files or [],
        test_command=test_command,
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
        voting_k=config.voting_k,
        initial_samples=config.initial_samples,
        max_samples=config.max_samples * 3,
    )

    logger.info(f"Starting task {task.task_id} for {repo_path}")

    # Get orchestrator and solve
    orchestrator = get_orchestrator()

    try:
        # Run with timeout
        result = await asyncio.wait_for(
            orchestrator.solve(task),
            timeout=timeout_minutes * 60,
        )

        return result.to_dict()

    except asyncio.TimeoutError:
        return {
            "task_id": task.task_id,
            "status": "timeout",
            "error": f"Task timed out after {timeout_minutes} minutes",
        }

    except Exception as e:
        logger.exception(f"Task {task.task_id} failed")
        return {
            "task_id": task.task_id,
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def solve_file(
    file_path: str,
    issue_description: str,
    additional_files: list[str] | None = None,
    max_cost_usd: float = 1.0,
    timeout_minutes: int = 5,
) -> dict[str, Any]:
    """Fix issues in a single file or small set of files.

    Simpler than solve_issue - no git repo required. Just pass a file path
    and describe what needs to be fixed. Works on any local file.

    Args:
        file_path: Path to the main file to fix (e.g., "./my_script.py")
        issue_description: What needs to be fixed or changed
        additional_files: Other files to include as context (optional)
        max_cost_usd: Maximum cost budget (default: 1.0)
        timeout_minutes: Maximum time (default: 5)

    Returns:
        Dictionary with:
        - status: "completed" or "failed"
        - patch: Unified diff patch for the file(s)
        - explanation: What was changed and why
        - cost_usd: Total cost
    """
    from pathlib import Path as FilePath
    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import ALL_STYLES
    from atlas.core.task import TaskSubmission, Solution

    config = get_config()

    # Validate config
    errors = config.validate()
    if errors:
        return {
            "status": "error",
            "error": "Configuration error",
            "details": errors,
        }

    # Read the main file
    main_path = FilePath(file_path)
    if not main_path.exists():
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
        }

    try:
        main_content = main_path.read_text(encoding='utf-8')
    except Exception as e:
        return {
            "status": "error",
            "error": f"Could not read file: {e}",
        }

    # Build file context
    file_context = f"## File: {main_path.name}\n```\n{main_content}\n```\n"

    # Read additional files if provided
    if additional_files:
        for add_file in additional_files:
            add_path = FilePath(add_file)
            if add_path.exists():
                try:
                    add_content = add_path.read_text(encoding='utf-8')
                    file_context += f"\n## File: {add_path.name}\n```\n{add_content}\n```\n"
                except:
                    pass

    # Create a minimal task
    task = TaskSubmission(
        description=f"""Fix the following issue in {main_path.name}:

{issue_description}

Generate a unified diff patch that fixes the issue.""",
        repository_url="local",
        branch="main",
        relevant_files=[str(main_path)],
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
        initial_samples=3,
        max_samples=3,
        voting_k=2,
    )

    # Create context for agents
    context = AgentContext(
        task=task,
        repository_content=file_context,
    )

    # Run 3 diverse agents
    agents = [
        MicroAgent(agent_id=f"agent_{i}", prompt_style=ALL_STYLES[i % len(ALL_STYLES)], config=config)
        for i in range(3)
    ]

    logger.info(f"Running 3 agents on {file_path}")

    try:
        # Run agents in parallel
        solutions = await asyncio.wait_for(
            asyncio.gather(*[agent.generate(context) for agent in agents], return_exceptions=True),
            timeout=timeout_minutes * 60,
        )

        # Filter valid solutions
        valid_solutions = [
            s for s in solutions
            if isinstance(s, Solution) and s.is_valid and s.patch
        ]

        if not valid_solutions:
            return {
                "status": "failed",
                "error": "No valid patches generated",
                "agent_errors": [str(s) if isinstance(s, Exception) else s.validation_errors for s in solutions],
            }

        # Pick the best solution (longest patch as simple heuristic, or first valid)
        best = max(valid_solutions, key=lambda s: len(s.patch))

        total_cost = sum(s.cost for s in solutions if isinstance(s, Solution))

        return {
            "status": "completed",
            "file": str(main_path),
            "patch": best.patch,
            "explanation": best.explanation,
            "cost_usd": total_cost,
            "agents_run": len(solutions),
            "valid_patches": len(valid_solutions),
        }

    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "error": f"Timed out after {timeout_minutes} minutes",
        }

    except Exception as e:
        logger.exception("solve_file failed")
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def solve_folder(
    folder_path: str,
    issue_description: str,
    file_patterns: list[str] | None = None,
    max_files: int = 10,
    max_cost_usd: float = 2.0,
    timeout_minutes: int = 10,
) -> dict[str, Any]:
    """Fix issues in a local folder of files.

    Works on any local folder - no git repo required. Automatically finds
    relevant files based on patterns or common extensions.

    Args:
        folder_path: Path to the folder (e.g., "./src" or "C:/projects/myapp")
        issue_description: What needs to be fixed or changed
        file_patterns: Glob patterns to match files (e.g., ["*.py", "*.js"])
                      Default: common code extensions
        max_files: Maximum number of files to include (default: 10)
        max_cost_usd: Maximum cost budget (default: 2.0)
        timeout_minutes: Maximum time (default: 10)

    Returns:
        Dictionary with:
        - status: "completed" or "failed"
        - patch: Unified diff patch for all files
        - files_analyzed: List of files that were included
        - cost_usd: Total cost
    """
    import glob
    from pathlib import Path as FilePath
    from atlas.agents.micro_agent import MicroAgent, AgentContext
    from atlas.agents.prompt_styles import ALL_STYLES
    from atlas.core.task import TaskSubmission, Solution

    config = get_config()

    # Validate config
    errors = config.validate()
    if errors:
        return {
            "status": "error",
            "error": "Configuration error",
            "details": errors,
        }

    # Check folder exists
    folder = FilePath(folder_path)
    if not folder.exists() or not folder.is_dir():
        return {
            "status": "error",
            "error": f"Folder not found: {folder_path}",
        }

    # Default file patterns for common code files
    if not file_patterns:
        file_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h",
            "*.cs", "*.rb", "*.php", "*.swift", "*.kt",
        ]

    # Find matching files
    all_files = []
    for pattern in file_patterns:
        # Search recursively
        matches = list(folder.rglob(pattern))
        all_files.extend(matches)

    # Filter out common non-source directories
    excluded_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '.next'}
    filtered_files = [
        f for f in all_files
        if not any(excl in f.parts for excl in excluded_dirs)
    ]

    # Sort by size (smaller files first) and limit
    filtered_files.sort(key=lambda f: f.stat().st_size)
    selected_files = filtered_files[:max_files]

    if not selected_files:
        return {
            "status": "error",
            "error": f"No matching files found in {folder_path} with patterns {file_patterns}",
        }

    # Read file contents
    file_context = ""
    files_analyzed = []
    total_size = 0
    max_total_size = 100000  # 100KB limit

    for file_path in selected_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            if total_size + len(content) > max_total_size:
                break
            rel_path = file_path.relative_to(folder)
            file_context += f"\n## File: {rel_path}\n```\n{content}\n```\n"
            files_analyzed.append(str(rel_path))
            total_size += len(content)
        except Exception:
            pass  # Skip files that can't be read

    if not files_analyzed:
        return {
            "status": "error",
            "error": "Could not read any files",
        }

    logger.info(f"Analyzing {len(files_analyzed)} files from {folder_path}")

    # Create task
    task = TaskSubmission(
        description=f"""Fix the following issue in the codebase:

{issue_description}

Files included:
{chr(10).join('- ' + f for f in files_analyzed)}

Generate a unified diff patch that fixes the issue.""",
        repository_url="local",
        branch="main",
        relevant_files=files_analyzed,
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
        initial_samples=3,
        max_samples=3,
        voting_k=2,
    )

    # Create context for agents
    context = AgentContext(
        task=task,
        repository_content=file_context,
    )

    # Run 3 diverse agents
    agents = [
        MicroAgent(agent_id=f"agent_{i}", prompt_style=ALL_STYLES[i % len(ALL_STYLES)], config=config)
        for i in range(3)
    ]

    try:
        # Run agents in parallel
        solutions = await asyncio.wait_for(
            asyncio.gather(*[agent.generate(context) for agent in agents], return_exceptions=True),
            timeout=timeout_minutes * 60,
        )

        # Filter valid solutions
        valid_solutions = [
            s for s in solutions
            if isinstance(s, Solution) and s.is_valid and s.patch
        ]

        if not valid_solutions:
            return {
                "status": "failed",
                "error": "No valid patches generated",
                "files_analyzed": files_analyzed,
                "agent_errors": [str(s) if isinstance(s, Exception) else s.validation_errors for s in solutions],
            }

        # Pick the best solution
        best = max(valid_solutions, key=lambda s: len(s.patch))
        total_cost = sum(s.cost for s in solutions if isinstance(s, Solution))

        return {
            "status": "completed",
            "folder": str(folder),
            "patch": best.patch,
            "explanation": best.explanation,
            "files_analyzed": files_analyzed,
            "cost_usd": total_cost,
            "agents_run": len(solutions),
            "valid_patches": len(valid_solutions),
        }

    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "error": f"Timed out after {timeout_minutes} minutes",
            "files_analyzed": files_analyzed,
        }

    except Exception as e:
        logger.exception("solve_folder failed")
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def solve_feature_dag(
    repo_path: str,
    description: str,
    branch: str = "main",
    base_commit: str | None = None,
    max_tasks: int = 12,
    max_cost_usd: float = 10.0,
    timeout_minutes: int = 60,
    keywords: list[str] | None = None,
    component_names: list[str] | None = None,
    test_command: str | None = None,
    review_only: bool = False,
    dag_override: dict | list | str | None = None,
) -> dict[str, Any]:
    """Solve a feature request using TaskDAG decomposition and assembly.

    Breaks complex features into smaller tasks, solves each with multi-agent
    consensus, then assembles the final patch.

    Works with both local paths and remote git URLs.

    Args:
        repo_path: Local folder path (e.g., "./my-project") OR git URL
        description: Description of the feature to implement
        branch: Target branch (default: main)
        base_commit: Specific commit to base changes on (optional)
        max_tasks: Maximum number of sub-tasks to create (default: 12)
        max_cost_usd: Maximum cost budget in USD (default: 10.0)
        timeout_minutes: Maximum time in minutes (default: 60)
        keywords: Keywords to help identify relevant code (optional)
        component_names: Component names to focus on (optional)
        test_command: Command to run tests (optional)
        review_only: If True, only generate DAG without implementing (default: False)
        dag_override: Override the auto-generated DAG (optional)

    Returns:
        Dictionary with final_patch, cost, and task results.
    """
    submission = TaskDAGSubmission(
        description=description,
        repository_url=repo_path,
        branch=branch,
        base_commit=base_commit,
        max_tasks=max_tasks,
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
        keywords=keywords or [],
        component_names=component_names or [],
        test_command=test_command,
        review_only=review_only,
        dag_override=dag_override,
    )

    orchestrator = get_dag_orchestrator()

    try:
        result = await asyncio.wait_for(
            orchestrator.solve(submission),
            timeout=timeout_minutes * 60,
        )
    except asyncio.TimeoutError:
        return {
            "task_id": submission.task_id,
            "status": "timeout",
            "error": f"Task timed out after {timeout_minutes} minutes",
        }

    if result.status == "needs_review" and result.dag:
        return {
            "task_id": result.task_id,
            "status": result.status,
            "dag": _serialize_dag(result.dag),
            "errors": result.errors,
        }

    dag_summary = []
    if result.dag:
        for task_id, task in result.dag.tasks.items():
            dag_summary.append({
                "task_id": task_id,
                "title": task.title,
                "dependencies": task.dependencies,
            })

    task_results_summary = {
        task_id: {
            "selected_patch_id": task_result.selected_patch_id,
            "candidate_count": len(task_result.candidates),
            "oracle_count": len(task_result.oracles),
            "errors": task_result.errors,
        }
        for task_id, task_result in result.task_results.items()
    }

    return {
        "task_id": result.task_id,
        "status": result.status,
        "final_patch": result.final_patch,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
        "dag": dag_summary,
        "task_results": task_results_summary,
        "errors": result.errors,
    }


@mcp.tool()
async def check_status(task_id: str) -> dict[str, Any]:
    """Check the status of a running or completed task.

    Args:
        task_id: The task ID returned by solve_issue()

    Returns:
        Dictionary with current status, progress, and cost information.
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_task_status(task_id)


@mcp.tool()
async def get_result(task_id: str) -> dict[str, Any]:
    """Get the full result of a completed task.

    Args:
        task_id: The task ID returned by solve_issue()

    Returns:
        Dictionary with the patch, confidence score, cost breakdown,
        and execution trace.
    """
    orchestrator = get_orchestrator()
    result = orchestrator.get_task_result(task_id)

    if result is None:
        return {
            "status": "not_found",
            "task_id": task_id,
            "error": "Task not found",
        }

    # Build detailed response
    response = result.to_dict()

    # Add execution trace summary
    if result.execution_trace:
        response["trace_summary"] = {
            "phases": [p["phase"] for p in result.execution_trace.phases],
            "agent_count": len(result.execution_trace.agent_outputs),
            "voting_rounds": len(result.execution_trace.voting_rounds),
            "errors": result.execution_trace.errors,
            "warnings": result.execution_trace.warnings,
        }

    # Add cost breakdown
    if result.cost_breakdown:
        response["cost_breakdown"] = {
            "model_costs": result.cost_breakdown.model_costs,
            "api_costs": result.cost_breakdown.api_costs,
            "total": result.cost_breakdown.total,
        }

    return response


@mcp.tool()
async def get_config_info() -> dict[str, Any]:
    """Get current ATLAS configuration.

    Returns:
        Dictionary with current configuration values (excluding secrets).
    """
    config = get_config()

    return {
        "model": config.model,
        "temperature": config.temperature,
        "voting_k": config.voting_k,
        "max_samples": config.max_samples,
        "max_cost_usd": config.max_cost_usd,
        "timeout_minutes": config.timeout_minutes,
        "has_gemini_key": bool(config.gemini_api_key),
        "has_context7_key": bool(config.context7_api_key),
        "is_valid": config.is_valid(),
    }


# ============================================================================
# SpecKit Tools - Specification-driven development
# ============================================================================


@mcp.tool()
async def speckit_init(
    repo_path: str,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Initialize a SpecKit in a repository.

    Creates the .speckit directory structure with default constitution.
    This is the first step in spec-driven development.

    Args:
        repo_path: Path to the repository root.
        project_name: Optional project name for the constitution.

    Returns:
        Dictionary with initialization status and paths.
    """
    return await speckit_tools.speckit_init(repo_path, project_name)


@mcp.tool()
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
        principles: List of principle definitions with keys:
            - id: Principle identifier (e.g., "P-001")
            - title: Short title
            - description: Full description
            - rationale: Why this principle exists
            - examples: List of examples
        coding_standards: List of coding standards.
        testing_requirements: List of testing requirements.
        documentation_standards: List of documentation standards.
        forbidden_patterns: List of forbidden code patterns (regex).
        required_patterns: List of required code patterns (regex).

    Returns:
        Dictionary with the saved constitution.
    """
    return await speckit_tools.speckit_constitution(
        repo_path=repo_path,
        project_name=project_name,
        principles=principles,
        coding_standards=coding_standards,
        testing_requirements=testing_requirements,
        documentation_standards=documentation_standards,
        forbidden_patterns=forbidden_patterns,
        required_patterns=required_patterns,
    )


@mcp.tool()
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
        user_scenarios: List of user scenario definitions with keys:
            - id: Scenario identifier
            - title: Scenario title
            - description: What the user is trying to do
            - priority: P1/P2/P3
            - given: List of preconditions
            - when: List of actions
            - then: List of expected outcomes
            - edge_cases: List of edge cases
        functional_requirements: List of requirement definitions with keys:
            - id: Requirement ID (e.g., "FR-001")
            - description: What MUST/SHOULD/MAY happen
            - requirement_type: MUST, SHOULD, or MAY
            - needs_clarification: Boolean
            - clarification_notes: Notes if needs clarification
        data_entities: List of data entity definitions.
        success_criteria: List of success criteria.
        assumptions: List of assumptions.
        constraints: List of constraints.
        open_questions: List of open questions.

    Returns:
        Dictionary with the saved specification.
    """
    return await speckit_tools.speckit_specify(
        repo_path=repo_path,
        feature_id=feature_id,
        title=title,
        description=description,
        user_scenarios=user_scenarios,
        functional_requirements=functional_requirements,
        data_entities=data_entities,
        success_criteria=success_criteria,
        assumptions=assumptions,
        constraints=constraints,
        open_questions=open_questions,
    )


@mcp.tool()
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
    return await speckit_tools.speckit_plan(
        repo_path=repo_path,
        feature_id=feature_id,
        summary=summary,
        language=language,
        version=version,
        dependencies=dependencies,
        storage=storage,
        testing_framework=testing_framework,
        target_platforms=target_platforms,
        performance_targets=performance_targets,
        technical_constraints=technical_constraints,
        architecture_decisions=architecture_decisions,
        source_files=source_files,
        test_files=test_files,
        api_contracts=api_contracts,
        complexity_notes=complexity_notes,
    )


@mcp.tool()
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
        tasks: List of task definitions with keys:
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
    return await speckit_tools.speckit_tasks(
        repo_path=repo_path,
        feature_id=feature_id,
        tasks=tasks,
    )


@mcp.tool()
async def speckit_implement(
    repo_path: str,
    feature_id: str,
    test_command: str | None = None,
    max_cost_usd: float = 10.0,
    timeout_minutes: int = 60,
) -> dict[str, Any]:
    """Prepare SpecKit for implementation using ATLAS.

    Converts the SpecKit to a TaskDAG format that can be passed
    to solve_feature_dag for multi-agent implementation.

    Args:
        repo_path: Path to the repository root.
        feature_id: Feature identifier (must have spec, plan, tasks).
        test_command: Test command to run for validation.
        max_cost_usd: Maximum cost budget.
        timeout_minutes: Maximum time in minutes.

    Returns:
        Dictionary with dag_override for solve_feature_dag or error.
        Use the suggested_params to call solve_feature_dag.
    """
    return await speckit_tools.speckit_implement(
        repo_path=repo_path,
        feature_id=feature_id,
        test_command=test_command,
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
    )


@mcp.tool()
async def speckit_execute(
    repo_path: str,
    feature_id: str,
    branch: str = "main",
    test_command: str | None = None,
    max_cost_usd: float = 10.0,
    timeout_minutes: int = 60,
) -> dict[str, Any]:
    """Execute a complete SpecKit implementation using ATLAS multi-agent system.

    This is the all-in-one tool: takes a SpecKit and implements it fully.
    Combines speckit_implement + solve_feature_dag into a single call.

    Flow:
    1. Loads your SpecKit (spec, plan, tasks)
    2. Converts tasks to a TaskDAG with dependencies
    3. Runs multi-agent consensus on each task in order
    4. Assembles final patch from all task patches
    5. Returns the complete implementation

    Args:
        repo_path: Path to the repository root (local path works).
        feature_id: Feature identifier (must have spec, plan, tasks defined).
        branch: Git branch to base changes on (default: main).
        test_command: Command to run tests for validation (e.g., "pytest").
        max_cost_usd: Maximum cost budget in USD (default: 10.0).
        timeout_minutes: Maximum time in minutes (default: 60).

    Returns:
        Dictionary with:
        - status: "completed", "failed", or "timeout"
        - final_patch: The unified diff patch for the entire feature
        - cost_usd: Total cost of all agent calls
        - duration_seconds: Total execution time
        - task_results: Summary of each task's implementation
        - errors: Any errors encountered
    """
    # Step 1: Prepare the SpecKit (converts to dag_override)
    prep_result = await speckit_tools.speckit_implement(
        repo_path=repo_path,
        feature_id=feature_id,
        test_command=test_command,
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
    )

    # Check if preparation failed
    if prep_result.get("status") != "ready":
        return {
            "status": "failed",
            "error": prep_result.get("error", "SpecKit preparation failed"),
            "details": prep_result,
        }

    # Step 2: Create the DAG submission with the converted tasks
    dag_override = prep_result.get("dag_override")
    description = prep_result.get("suggested_params", {}).get("description", f"Implement {feature_id}")

    submission = TaskDAGSubmission(
        description=description,
        repository_url=repo_path,
        branch=branch,
        max_tasks=len(dag_override.get("tasks", [])) + 5,  # Allow some buffer
        max_cost_usd=max_cost_usd,
        timeout_minutes=timeout_minutes,
        test_command=test_command,
        dag_override=dag_override,
    )

    # Step 3: Execute with the DAG orchestrator
    orchestrator = get_dag_orchestrator()

    try:
        result = await asyncio.wait_for(
            orchestrator.solve(submission),
            timeout=timeout_minutes * 60,
        )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "feature_id": feature_id,
            "error": f"Execution timed out after {timeout_minutes} minutes",
        }

    # Step 4: Format and return results
    task_results_summary = {
        task_id: {
            "selected_patch_id": task_result.selected_patch_id,
            "candidate_count": len(task_result.candidates),
            "errors": task_result.errors,
        }
        for task_id, task_result in result.task_results.items()
    }

    return {
        "status": result.status,
        "feature_id": feature_id,
        "final_patch": result.final_patch,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
        "task_count": len(result.task_results),
        "task_results": task_results_summary,
        "errors": result.errors,
    }


@mcp.tool()
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
    return await speckit_tools.speckit_status(
        repo_path=repo_path,
        feature_id=feature_id,
    )


@mcp.tool()
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
    return await speckit_tools.speckit_export(
        repo_path=repo_path,
        feature_id=feature_id,
    )


def main():
    """Run the ATLAS MCP server."""
    # Validate configuration on startup
    config = get_config()
    errors = config.validate()

    if errors:
        logger.warning("Configuration warnings:")
        for error in errors:
            logger.warning(f"  - {error}")

    logger.info("Starting ATLAS MCP server...")
    logger.info(f"Model: {config.model}")
    logger.info(f"Voting K: {config.voting_k}")
    logger.info(f"Max samples: {config.max_samples}")

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
