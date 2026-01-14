"""
Memory Writer for swarm operations.

Writes swarm results back to Agent-MCP's memory systems:
- Task notes
- Project context
- RAG index (optional)

Respects memory policy flags from MemoryConfig.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from ...db.connection import get_db_connection, execute_db_write
from ...core.config import logger
from .schemas import SwarmResult, MemoryConfig


class MemoryWriter:
    """
    Writes swarm results back to Agent-MCP memory systems.

    Supports writing to:
    - Task notes (append swarm summary)
    - Project context (store swarm insights)
    - RAG index (index selected outputs)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory writer.

        Args:
            config: Memory configuration controlling what gets written
        """
        self.config = config or MemoryConfig()

    async def write_result(
        self,
        result: SwarmResult,
        author: str = "swarm",
    ) -> Dict[str, bool]:
        """
        Write swarm result to configured memory systems.

        Args:
            result: The swarm result to persist
            author: Author identifier for notes

        Returns:
            Dictionary indicating which writes succeeded
        """
        outcomes = {
            "task_notes": False,
            "project_context": False,
            "rag_index": False,
        }

        # Write to task notes
        if self.config.write_back_to_task and result.task_id:
            try:
                await self.write_to_task(result, author)
                outcomes["task_notes"] = True
            except Exception as e:
                logger.error(f"Failed to write to task notes: {e}")

        # Write to project context
        if self.config.write_back_to_project_context:
            try:
                await self.write_to_project_context(result, author)
                outcomes["project_context"] = True
            except Exception as e:
                logger.error(f"Failed to write to project context: {e}")

        # Index into RAG
        if self.config.index_into_rag and result.consensus_reached:
            try:
                await self.index_to_rag(result)
                outcomes["rag_index"] = True
            except Exception as e:
                logger.error(f"Failed to index to RAG: {e}")

        return outcomes

    async def write_to_task(
        self,
        result: SwarmResult,
        author: str = "swarm",
    ) -> None:
        """
        Append swarm summary to task notes.

        Args:
            result: Swarm result to summarize
            author: Author identifier
        """
        if not result.task_id:
            logger.warning("Cannot write to task - no task_id in result")
            return

        # Build summary note
        status_emoji = "✓" if result.consensus_reached else "⚠"
        summary = f"{status_emoji} Swarm run {result.run_id} ({result.mode.value} mode)\n"
        summary += f"Status: {result.status}\n"
        summary += f"Consensus: {'Yes' if result.consensus_reached else 'No'} "
        summary += f"(confidence: {result.confidence_score:.2f})\n"

        if result.metrics:
            summary += f"Metrics: {result.metrics.duration_ms}ms, "
            summary += f"${result.metrics.cost_usd:.4f}, "
            summary += f"{result.metrics.tokens_total} tokens\n"

        if result.clusters:
            summary += f"Clusters: {len(result.clusters)} "
            summary += f"(votes: {result.vote_counts})\n"

        if result.warnings:
            summary += f"Warnings: {', '.join(result.warnings[:3])}\n"

        # Truncate selected output for note
        output_preview = result.selected_output[:500] if result.selected_output else "None"
        if len(result.selected_output or "") > 500:
            output_preview += "... [truncated]"
        summary += f"\nSelected output:\n```\n{output_preview}\n```"

        note = {
            "timestamp": datetime.utcnow().isoformat(),
            "author": author,
            "content": summary,
            "type": "swarm_result",
            "run_id": result.run_id,
        }

        await self._append_task_note(result.task_id, note)
        logger.debug(f"Wrote swarm summary to task {result.task_id}")

    async def _append_task_note(self, task_id: str, note: Dict[str, Any]) -> None:
        """Append a note to task's notes array."""
        async def _write():
            conn = get_db_connection()
            try:
                cursor = conn.cursor()

                # Get existing notes
                cursor.execute(
                    "SELECT notes FROM tasks WHERE task_id = ?",
                    (task_id,)
                )
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"Task {task_id} not found")
                    return

                existing_notes = []
                if row['notes']:
                    try:
                        existing_notes = json.loads(row['notes'])
                    except json.JSONDecodeError:
                        existing_notes = []

                existing_notes.append(note)

                # Update notes
                cursor.execute(
                    """
                    UPDATE tasks
                    SET notes = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    (
                        json.dumps(existing_notes),
                        datetime.utcnow().isoformat(),
                        task_id,
                    )
                )
                conn.commit()
            finally:
                conn.close()

        await execute_db_write(_write)

    async def write_to_project_context(
        self,
        result: SwarmResult,
        author: str = "swarm",
    ) -> None:
        """
        Write swarm insights to project context.

        Creates or updates context entries for swarm results.
        """
        context_key = f"swarm_result_{result.run_id}"
        description = f"Swarm consensus result from run {result.run_id}"

        value = {
            "run_id": result.run_id,
            "task_id": result.task_id,
            "mode": result.mode.value,
            "status": result.status,
            "consensus_reached": result.consensus_reached,
            "confidence_score": result.confidence_score,
            "selected_variant_id": result.selected_variant_id,
            "cluster_count": len(result.clusters),
            "vote_counts": result.vote_counts,
            "metrics": {
                "duration_ms": result.metrics.duration_ms,
                "cost_usd": result.metrics.cost_usd,
                "tokens_total": result.metrics.tokens_total,
            } if result.metrics else None,
            "completed_at": result.completed_at,
        }

        async def _write():
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO project_context
                    (context_key, value, last_updated, updated_by, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        context_key,
                        json.dumps(value),
                        datetime.utcnow().isoformat(),
                        author,
                        description,
                    )
                )
                conn.commit()
            finally:
                conn.close()

        await execute_db_write(_write)
        logger.debug(f"Wrote swarm result to project context: {context_key}")

        # Also update latest swarm result pointer
        await self._update_latest_swarm_context(result, author)

    async def _update_latest_swarm_context(
        self,
        result: SwarmResult,
        author: str,
    ) -> None:
        """Update the 'latest_swarm_result' context entry."""
        async def _write():
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO project_context
                    (context_key, value, last_updated, updated_by, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "latest_swarm_result",
                        json.dumps({
                            "run_id": result.run_id,
                            "task_id": result.task_id,
                            "mode": result.mode.value,
                            "consensus_reached": result.consensus_reached,
                        }),
                        datetime.utcnow().isoformat(),
                        author,
                        "Pointer to the most recent swarm result",
                    )
                )
                conn.commit()
            finally:
                conn.close()

        await execute_db_write(_write)

    async def index_to_rag(self, result: SwarmResult) -> None:
        """
        Index selected swarm output into RAG.

        Only indexes if consensus was reached.
        """
        if not result.consensus_reached or not result.selected_output:
            logger.debug("Skipping RAG index - no consensus or no output")
            return

        async def _write():
            conn = get_db_connection()
            try:
                cursor = conn.cursor()

                # Insert chunk
                cursor.execute(
                    """
                    INSERT INTO rag_chunks
                    (source_type, source_ref, chunk_text, indexed_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "swarm_output",
                        result.run_id,
                        result.selected_output,
                        datetime.utcnow().isoformat(),
                        json.dumps({
                            "mode": result.mode.value,
                            "task_id": result.task_id,
                            "confidence": result.confidence_score,
                            "consensus": result.consensus_reached,
                        }),
                    )
                )

                chunk_id = cursor.lastrowid
                logger.debug(f"Indexed swarm output as chunk {chunk_id}")

                # Note: Embedding generation would be done by the RAG indexer
                # We just store the chunk here

                conn.commit()
            finally:
                conn.close()

        await execute_db_write(_write)
        logger.info(f"Indexed swarm output from run {result.run_id} to RAG")


async def write_swarm_result(
    result: SwarmResult,
    config: Optional[MemoryConfig] = None,
    author: str = "swarm",
) -> Dict[str, bool]:
    """
    Convenience function to write swarm result to memory.

    Args:
        result: The swarm result
        config: Memory configuration
        author: Author identifier

    Returns:
        Dictionary of write outcomes
    """
    writer = MemoryWriter(config)
    return await writer.write_result(result, author)
