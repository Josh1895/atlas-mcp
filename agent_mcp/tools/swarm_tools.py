"""
Swarm Tools for Agent-MCP.

Provides the run_swarm_consensus MCP tool that invokes parallel multi-agent
generation, clustering, and voting consensus to solve tasks.
"""

import json
from typing import Dict, Any, List

import mcp.types as mcp_types

from .registry import register_tool
from ..core.config import logger, SWARM_ENABLED
from ..core.auth import verify_token
from ..utils.audit_utils import log_audit


# Input schema for run_swarm_consensus
RUN_SWARM_CONSENSUS_SCHEMA = {
    "type": "object",
    "properties": {
        "token": {
            "type": "string",
            "description": "Authentication token (admin or agent token)"
        },
        "task_id": {
            "type": "string",
            "description": "Optional task ID to build context from"
        },
        "description": {
            "type": "string",
            "description": "Description of the task or question"
        },
        "mode": {
            "type": "string",
            "enum": ["patch", "answer"],
            "description": "Swarm mode: 'patch' for code fixes, 'answer' for research questions",
            "default": "patch"
        },
        "repo": {
            "type": "object",
            "description": "Repository configuration (uses local MCP_PROJECT_DIR)",
            "properties": {
                "branch": {"type": "string"},
                "commit": {"type": "string"},
                "test_command": {"type": "string"}
            }
        },
        "swarm": {
            "type": "object",
            "description": "Swarm agent configuration",
            "properties": {
                "agent_count": {
                    "type": "integer",
                    "description": "Number of agents to run",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "consensus_k": {
                    "type": "integer",
                    "description": "Votes ahead needed for consensus",
                    "default": 2,
                    "minimum": 1
                },
                "model": {
                    "type": "string",
                    "default": "gemini-3-flash-preview"
                },
                "temperature": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0,
                    "maximum": 2
                }
            }
        },
        "budgets": {
            "type": "object",
            "description": "Budget limits for the swarm run",
            "properties": {
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum time for the run",
                    "default": 900
                },
                "max_cost_usd": {
                    "type": "number",
                    "description": "Maximum cost in USD",
                    "default": 5.0
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 500000
                },
                "max_tool_calls": {
                    "type": "integer",
                    "default": 100
                }
            }
        },
        "tools": {
            "type": "object",
            "description": "Tool availability configuration",
            "properties": {
                "enable_web_search": {"type": "boolean", "default": False},
                "enable_context7": {"type": "boolean", "default": False},
                "enable_repo_search": {"type": "boolean", "default": True},
                "enable_agent_mcp_rag": {"type": "boolean", "default": True}
            }
        },
        "memory": {
            "type": "object",
            "description": "Memory write-back configuration",
            "properties": {
                "write_back_to_task": {"type": "boolean", "default": True},
                "write_back_to_project_context": {"type": "boolean", "default": False},
                "index_into_rag": {"type": "boolean", "default": False}
            }
        }
    },
    "required": ["token"]
}


async def run_swarm_consensus_impl(arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
    """
    Implementation of the run_swarm_consensus tool.

    Invokes the swarm manager to run multi-agent consensus.

    Args:
        arguments: Tool arguments including token, task_id, description, etc.

    Returns:
        List of TextContent with the result
    """
    # Check if swarm is enabled
    if not SWARM_ENABLED:
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Swarm feature is disabled. Set SWARM_ENABLED=true to enable."
            })
        )]

    # Extract and verify token
    token = arguments.get("token")
    if not token:
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Missing required parameter: token"
            })
        )]

    # Verify authentication
    try:
        verify_token(token)
    except Exception as e:
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Authentication failed: {str(e)}"
            })
        )]

    # Build SwarmRequest from arguments
    try:
        from ..features.swarm.schemas import (
            SwarmRequest,
            SwarmMode,
            BudgetConfig,
            ToolConfig,
            MemoryConfig,
            RepoConfig,
            SwarmConfig,
        )
        from ..features.swarm.swarm_manager import run_swarm

        # Parse mode
        mode_str = arguments.get("mode", "patch")
        mode = SwarmMode.PATCH if mode_str == "patch" else SwarmMode.ANSWER

        # Parse repo config (local paths only)
        repo_args = arguments.get("repo", {})
        repo_config = RepoConfig(
            branch=repo_args.get("branch"),
            commit=repo_args.get("commit"),
            test_command=repo_args.get("test_command"),
        )

        # Parse swarm config
        swarm_args = arguments.get("swarm", {})
        swarm_config = SwarmConfig(
            agent_count=swarm_args.get("agent_count", 5),
            consensus_k=swarm_args.get("consensus_k", 2),
            model=swarm_args.get("model", "gemini-3-flash-preview"),
            temperature=swarm_args.get("temperature", 0.7),
        )

        # Parse budget config
        budget_args = arguments.get("budgets", {})
        budget_config = BudgetConfig(
            timeout_seconds=budget_args.get("timeout_seconds", 900),
            max_cost_usd=budget_args.get("max_cost_usd", 5.0),
            max_tokens=budget_args.get("max_tokens", 500000),
            max_tool_calls=budget_args.get("max_tool_calls", 100),
        )

        # Parse tool config
        tool_args = arguments.get("tools", {})
        tool_config = ToolConfig(
            enable_web_search=tool_args.get("enable_web_search", False),
            enable_context7=tool_args.get("enable_context7", False),
            enable_repo_search=tool_args.get("enable_repo_search", True),
            enable_agent_mcp_rag=tool_args.get("enable_agent_mcp_rag", True),
        )

        # Parse memory config
        memory_args = arguments.get("memory", {})
        memory_config = MemoryConfig(
            write_back_to_task=memory_args.get("write_back_to_task", True),
            write_back_to_project_context=memory_args.get("write_back_to_project_context", False),
            index_into_rag=memory_args.get("index_into_rag", False),
        )

        # Build request
        request = SwarmRequest(
            token=token,
            task_id=arguments.get("task_id"),
            description=arguments.get("description"),
            mode=mode,
            repo=repo_config,
            swarm=swarm_config,
            budgets=budget_config,
            tools=tool_config,
            memory=memory_config,
        )

        # Log audit
        log_audit(
            "run_swarm_consensus",
            token,
            {
                "task_id": request.task_id,
                "mode": mode_str,
                "agent_count": swarm_config.agent_count,
            }
        )

        # Run swarm
        logger.info(f"Starting swarm consensus run (mode={mode_str})")
        result = await run_swarm(request)

        # Convert result to response
        response = {
            "success": True,
            "run_id": result.run_id,
            "task_id": result.task_id,
            "mode": result.mode.value,
            "status": result.status,
            "consensus_reached": result.consensus_reached,
            "confidence_score": result.confidence_score,
            "selected_output": result.selected_output,
            "selected_variant_id": result.selected_variant_id,
            "vote_counts": result.vote_counts,
            "cluster_count": len(result.clusters),
            "metrics": {
                "duration_ms": result.metrics.duration_ms,
                "cost_usd": result.metrics.cost_usd,
                "tokens_total": result.metrics.tokens_total,
                "agents_succeeded": result.metrics.agents_succeeded,
                "agents_failed": result.metrics.agents_failed,
            } if result.metrics else None,
            "warnings": result.warnings,
            "errors": result.errors,
        }

        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]

    except Exception as e:
        logger.error(f"Swarm consensus failed: {e}", exc_info=True)
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e)
            })
        )]


def register_swarm_tools():
    """Register all swarm-related tools."""
    # Only register if swarm is enabled (or we want the tool to return disabled message)
    register_tool(
        name="run_swarm_consensus",
        description=(
            "Run multi-agent swarm consensus to solve a task. "
            "Generates multiple candidate solutions using diverse AI agents, "
            "clusters them by similarity, and votes to select the best. "
            "Supports 'patch' mode for code fixes and 'answer' mode for research questions. "
            "Requires SWARM_ENABLED=true to be active."
        ),
        input_schema=RUN_SWARM_CONSENSUS_SCHEMA,
        implementation=run_swarm_consensus_impl
    )


# Register tools when module is imported
register_swarm_tools()
