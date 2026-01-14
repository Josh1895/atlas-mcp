"""
Persistence layer for swarm operations.

All database writes go through execute_db_write to maintain serialized
write semantics and avoid SQLite concurrency issues (AD-007).
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

from ...db.connection import get_db_connection, execute_db_write
from ...core.config import logger
from .schemas import SwarmResult, SwarmMetrics, AgentOutput, ClusterInfo, SwarmMode


async def save_swarm_run(
    run_id: str,
    task_id: Optional[str],
    mode: str,
    config_json: str,
    started_at: str,
) -> None:
    """
    Save a new swarm run record.

    Args:
        run_id: Unique identifier for the run
        task_id: Optional associated task ID
        mode: 'patch' or 'answer'
        config_json: JSON-serialized SwarmRequest
        started_at: ISO timestamp
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO swarm_runs (
                    run_id, task_id, mode, status, config_json, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (run_id, task_id, mode, config_json, started_at)
            )
            conn.commit()
            logger.debug(f"Saved swarm run {run_id}")
        finally:
            conn.close()

    await execute_db_write(_write)


async def update_swarm_run_status(
    run_id: str,
    status: str,
    completed_at: Optional[str] = None,
    consensus_reached: bool = False,
    confidence_score: float = 0.0,
    selected_output: Optional[str] = None,
    selected_variant_id: Optional[str] = None,
    vote_counts: Optional[Dict[str, int]] = None,
    metrics: Optional[SwarmMetrics] = None,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """
    Update a swarm run with results.

    Args:
        run_id: The run to update
        status: New status (completed, failed, timeout, budget_exceeded)
        completed_at: ISO timestamp of completion
        consensus_reached: Whether consensus was achieved
        confidence_score: 0.0 to 1.0 confidence
        selected_output: The winning output
        selected_variant_id: ID of the winning variant/cluster
        vote_counts: Mapping of cluster_id to vote count
        metrics: Collected metrics
        errors: List of error messages
        warnings: List of warning messages
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            metrics_json = None
            if metrics:
                metrics_json = json.dumps({
                    "duration_ms": metrics.duration_ms,
                    "cost_usd": metrics.cost_usd,
                    "tokens_total": metrics.tokens_total,
                    "tool_calls_count": metrics.tool_calls_count,
                    "cache_hits": metrics.cache_hits,
                    "agents_succeeded": metrics.agents_succeeded,
                    "agents_failed": metrics.agents_failed,
                })

            cursor.execute(
                """
                UPDATE swarm_runs SET
                    status = ?,
                    completed_at = ?,
                    consensus_reached = ?,
                    confidence_score = ?,
                    selected_output = ?,
                    selected_variant_id = ?,
                    vote_counts_json = ?,
                    metrics_json = ?,
                    errors_json = ?,
                    warnings_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    completed_at,
                    1 if consensus_reached else 0,
                    confidence_score,
                    selected_output,
                    selected_variant_id,
                    json.dumps(vote_counts) if vote_counts else None,
                    metrics_json,
                    json.dumps(errors) if errors else None,
                    json.dumps(warnings) if warnings else None,
                    run_id,
                )
            )
            conn.commit()
            logger.debug(f"Updated swarm run {run_id} status to {status}")
        finally:
            conn.close()

    await execute_db_write(_write)


async def save_swarm_agent(
    run_id: str,
    swarm_agent_id: str,
    prompt_style: str,
    model: str,
    temperature: float,
    status: str = "pending",
    started_at: Optional[str] = None,
) -> None:
    """
    Save a swarm agent record.

    Args:
        run_id: Parent run ID
        swarm_agent_id: Agent identifier within the run
        prompt_style: The prompt style used
        model: Model name
        temperature: Temperature setting
        status: Agent status
        started_at: When the agent started
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO swarm_agents (
                    run_id, swarm_agent_id, prompt_style, model, temperature, status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, swarm_agent_id, prompt_style, model, temperature, status, started_at)
            )
            conn.commit()
            logger.debug(f"Saved swarm agent {swarm_agent_id} for run {run_id}")
        finally:
            conn.close()

    await execute_db_write(_write)


async def update_swarm_agent(
    run_id: str,
    swarm_agent_id: str,
    status: str,
    completed_at: Optional[str] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    error: Optional[str] = None,
) -> None:
    """
    Update a swarm agent with results.
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE swarm_agents SET
                    status = ?,
                    completed_at = ?,
                    tokens_used = ?,
                    cost_usd = ?,
                    error = ?
                WHERE run_id = ? AND swarm_agent_id = ?
                """,
                (status, completed_at, tokens_used, cost_usd, error, run_id, swarm_agent_id)
            )
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)


async def save_swarm_output(
    run_id: str,
    swarm_agent_id: str,
    output_type: str,
    output_text: str,
    explanation: Optional[str] = None,
    is_valid: bool = True,
    validation_errors: Optional[List[str]] = None,
    cluster_id: Optional[str] = None,
    test_result: Optional[Dict[str, Any]] = None,
    quality_score: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a swarm agent's output.
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO swarm_outputs (
                    run_id, swarm_agent_id, output_type, output_text, explanation,
                    is_valid, validation_errors_json, cluster_id, test_result_json, quality_score_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    swarm_agent_id,
                    output_type,
                    output_text,
                    explanation,
                    1 if is_valid else 0,
                    json.dumps(validation_errors) if validation_errors else None,
                    cluster_id,
                    json.dumps(test_result) if test_result else None,
                    json.dumps(quality_score) if quality_score else None,
                )
            )
            conn.commit()
            logger.debug(f"Saved output for agent {swarm_agent_id} in run {run_id}")
        finally:
            conn.close()

    await execute_db_write(_write)


async def update_swarm_output_cluster(
    run_id: str,
    swarm_agent_id: str,
    cluster_id: str,
) -> None:
    """
    Update the cluster assignment for a swarm output.
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE swarm_outputs SET cluster_id = ?
                WHERE run_id = ? AND swarm_agent_id = ?
                """,
                (cluster_id, run_id, swarm_agent_id)
            )
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)


def get_swarm_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a swarm run by ID.

    Returns:
        Dictionary with run data or None if not found
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM swarm_runs WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_swarm_run_with_outputs(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a swarm run with all associated agents and outputs.

    Returns:
        Dictionary with run data, agents, and outputs
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Get run
        cursor.execute("SELECT * FROM swarm_runs WHERE run_id = ?", (run_id,))
        run_row = cursor.fetchone()
        if not run_row:
            return None

        run_data = dict(run_row)

        # Get agents
        cursor.execute(
            "SELECT * FROM swarm_agents WHERE run_id = ? ORDER BY swarm_agent_id",
            (run_id,)
        )
        run_data["agents"] = [dict(row) for row in cursor.fetchall()]

        # Get outputs
        cursor.execute(
            "SELECT * FROM swarm_outputs WHERE run_id = ? ORDER BY swarm_agent_id",
            (run_id,)
        )
        run_data["outputs"] = [dict(row) for row in cursor.fetchall()]

        return run_data
    finally:
        conn.close()


def get_runs_by_task(task_id: str) -> List[Dict[str, Any]]:
    """
    Get all swarm runs for a task.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM swarm_runs WHERE task_id = ? ORDER BY started_at DESC",
            (task_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# --- Tool Cache Functions ---

async def save_tool_cache(
    cache_key: str,
    tool_name: str,
    value: Any,
    ttl_seconds: int = 600,
) -> None:
    """
    Save a tool result to cache.

    Args:
        cache_key: Unique key for this cached result
        tool_name: Name of the tool
        value: The result to cache (will be JSON serialized)
        ttl_seconds: Time to live in seconds
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            now = datetime.utcnow()
            expires_at = datetime.utcnow()
            expires_at = now.replace(
                second=now.second + ttl_seconds
            ) if ttl_seconds < 60 else datetime.fromtimestamp(
                now.timestamp() + ttl_seconds
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO tool_cache (
                    cache_key, tool_name, value_json, created_at, expires_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    cache_key,
                    tool_name,
                    json.dumps(value),
                    now.isoformat(),
                    expires_at.isoformat(),
                )
            )
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)


def get_tool_cache(cache_key: str) -> Optional[Any]:
    """
    Get a cached tool result if it exists and hasn't expired.

    Returns:
        The cached value or None if not found/expired
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            SELECT value_json FROM tool_cache
            WHERE cache_key = ? AND expires_at > ?
            """,
            (cache_key, now)
        )
        row = cursor.fetchone()
        if row:
            # Increment hit count (fire and forget)
            cursor.execute(
                "UPDATE tool_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,)
            )
            conn.commit()
            return json.loads(row["value_json"])
        return None
    finally:
        conn.close()


async def cleanup_expired_cache() -> int:
    """
    Remove expired cache entries.

    Returns:
        Number of entries removed
    """
    removed = 0

    async def _write():
        nonlocal removed
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute(
                "DELETE FROM tool_cache WHERE expires_at <= ?",
                (now,)
            )
            removed = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)
    return removed


# --- Tool Calls Logging ---

async def log_tool_call(
    run_id: str,
    swarm_agent_id: Optional[str],
    tool_name: str,
    args: Dict[str, Any],
    status: str = "pending",
    duration_ms: Optional[int] = None,
    result_meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> int:
    """
    Log a tool call for audit purposes.

    Returns:
        The inserted row ID
    """
    row_id = 0

    async def _write():
        nonlocal row_id
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_calls (
                    run_id, swarm_agent_id, tool_name, args_json,
                    status, duration_ms, result_meta_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    swarm_agent_id,
                    tool_name,
                    json.dumps(args),
                    status,
                    duration_ms,
                    json.dumps(result_meta) if result_meta else None,
                    error,
                    datetime.utcnow().isoformat(),
                )
            )
            row_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)
    return row_id


async def update_tool_call(
    call_id: int,
    status: str,
    duration_ms: int,
    result_meta: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Update a tool call record with results.
    """
    async def _write():
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE tool_calls SET
                    status = ?,
                    duration_ms = ?,
                    result_meta_json = ?,
                    error = ?
                WHERE id = ?
                """,
                (
                    status,
                    duration_ms,
                    json.dumps(result_meta) if result_meta else None,
                    error,
                    call_id,
                )
            )
            conn.commit()
        finally:
            conn.close()

    await execute_db_write(_write)


def get_tool_calls_for_run(run_id: str) -> List[Dict[str, Any]]:
    """
    Get all tool calls for a swarm run.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at",
            (run_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
