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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP(
    "atlas",
    description="ATLAS: Multi-agent code generation with consensus voting",
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
    repo_url: str,
    issue_description: str,
    branch: str = "main",
    base_commit: str | None = None,
    relevant_files: list[str] | None = None,
    test_command: str | None = None,
    max_cost_usd: float = 2.0,
    timeout_minutes: int = 15,
) -> dict[str, Any]:
    """Solve a GitHub issue using multi-agent consensus.

    Generates patches using multiple diverse AI agents, validates them,
    and uses voting to select the best solution.

    Args:
        repo_url: URL of the git repository (e.g., https://github.com/user/repo)
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
        repository_url=repo_url,
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

    logger.info(f"Starting task {task.task_id} for {repo_url}")

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
async def solve_feature_dag(
    repo_url: str,
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
    """Solve a feature request using TaskDAG decomposition and assembly."""
    submission = TaskDAGSubmission(
        description=description,
        repository_url=repo_url,
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
