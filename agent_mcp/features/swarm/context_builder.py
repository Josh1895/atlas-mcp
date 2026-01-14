"""
Context Builder for swarm operations.

Assembles context from task descriptions, parent task chain, project context,
and RAG hits for swarm agents (AD-006).
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ...db.connection import get_db_connection
from ...core.config import logger


@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    content: str
    source_type: str  # 'task', 'parent_task', 'project_context', 'rag', 'description'
    source_ref: str
    priority: int = 0  # Higher = more important
    tokens_estimate: int = 0


@dataclass
class SwarmContext:
    """Assembled context for swarm agents."""
    chunks: List[ContextChunk] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False

    def get_full_context(self) -> str:
        """Get the full context as a single string."""
        # Sort by priority (highest first)
        sorted_chunks = sorted(self.chunks, key=lambda c: -c.priority)
        return "\n\n".join(
            f"## {chunk.source_type.replace('_', ' ').title()}: {chunk.source_ref}\n{chunk.content}"
            for chunk in sorted_chunks
        )

    def get_context_by_type(self, source_type: str) -> List[ContextChunk]:
        """Get all chunks of a specific type."""
        return [c for c in self.chunks if c.source_type == source_type]


class ContextBuilder:
    """
    Builds context for swarm agents from various sources.

    Sources:
    - Task description and notes
    - Parent task chain (up to max_depth)
    - Project context entries
    - RAG search results

    Respects token budget and deduplicates content.
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        max_parent_depth: int = 3,
        chars_per_token: float = 4.0,  # Rough estimate
    ):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum tokens for combined context
            max_parent_depth: How many parent tasks to include
            chars_per_token: Characters per token estimate
        """
        self.max_tokens = max_tokens
        self.max_parent_depth = max_parent_depth
        self.chars_per_token = chars_per_token
        self._seen_content: set = set()

    def build(
        self,
        task_id: Optional[str] = None,
        description: Optional[str] = None,
        include_project_context: bool = True,
        include_rag: bool = False,
        rag_query: Optional[str] = None,
        rag_limit: int = 5,
    ) -> SwarmContext:
        """
        Build context for swarm agents.

        Args:
            task_id: Task ID to build context for
            description: Direct description (used if no task_id)
            include_project_context: Include project context entries
            include_rag: Include RAG search results
            rag_query: Query for RAG search (defaults to task description)
            rag_limit: Max RAG results to include

        Returns:
            SwarmContext with assembled chunks
        """
        self._seen_content.clear()
        context = SwarmContext()
        remaining_tokens = self.max_tokens

        # 1. Add direct description if provided
        if description:
            chunk = self._add_chunk(
                context,
                content=description,
                source_type="description",
                source_ref="provided",
                priority=100,
                max_tokens=remaining_tokens,
            )
            if chunk:
                remaining_tokens -= chunk.tokens_estimate

        # 2. Add task and its notes
        if task_id:
            remaining_tokens = self._add_task_context(
                context, task_id, remaining_tokens, priority=90
            )

            # 3. Add parent task chain
            remaining_tokens = self._add_parent_chain(
                context, task_id, remaining_tokens, priority_base=70
            )

        # 4. Add project context
        if include_project_context and remaining_tokens > 1000:
            remaining_tokens = self._add_project_context(
                context, remaining_tokens, priority=50
            )

        # 5. Add RAG results
        if include_rag and remaining_tokens > 500:
            query = rag_query or description or self._get_task_description(task_id)
            if query:
                self._add_rag_context(
                    context, query, remaining_tokens, rag_limit, priority=30
                )

        context.total_tokens = self.max_tokens - remaining_tokens
        logger.debug(
            f"Built context with {len(context.chunks)} chunks, "
            f"~{context.total_tokens} tokens"
        )

        return context

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) / self.chars_per_token)

    def _content_hash(self, content: str) -> str:
        """Generate hash for deduplication."""
        # Simple hash of first 200 chars + length
        return f"{hash(content[:200])}_{len(content)}"

    def _add_chunk(
        self,
        context: SwarmContext,
        content: str,
        source_type: str,
        source_ref: str,
        priority: int,
        max_tokens: int,
    ) -> Optional[ContextChunk]:
        """
        Add a chunk to context if not duplicate and within budget.

        Returns the added chunk or None if skipped.
        """
        # Skip duplicates
        content_hash = self._content_hash(content)
        if content_hash in self._seen_content:
            return None
        self._seen_content.add(content_hash)

        # Estimate tokens
        tokens = self._estimate_tokens(content)

        # Truncate if needed
        if tokens > max_tokens:
            # Truncate to fit
            max_chars = int(max_tokens * self.chars_per_token)
            content = content[:max_chars] + "\n... [truncated]"
            tokens = max_tokens
            context.truncated = True

        chunk = ContextChunk(
            content=content,
            source_type=source_type,
            source_ref=source_ref,
            priority=priority,
            tokens_estimate=tokens,
        )
        context.chunks.append(chunk)
        return chunk

    def _add_task_context(
        self,
        context: SwarmContext,
        task_id: str,
        remaining_tokens: int,
        priority: int,
    ) -> int:
        """Add task description and notes."""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title, description, notes FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Task {task_id} not found")
                return remaining_tokens

            # Add title and description
            task_content = f"# Task: {row['title']}\n\n{row['description'] or 'No description'}"
            chunk = self._add_chunk(
                context,
                content=task_content,
                source_type="task",
                source_ref=task_id,
                priority=priority,
                max_tokens=remaining_tokens,
            )
            if chunk:
                remaining_tokens -= chunk.tokens_estimate

            # Add notes
            if row['notes'] and remaining_tokens > 500:
                try:
                    notes = json.loads(row['notes'])
                    if notes:
                        notes_content = "## Task Notes\n\n"
                        for note in notes[-10:]:  # Last 10 notes
                            notes_content += f"- [{note.get('timestamp', 'N/A')}] {note.get('author', 'unknown')}: {note.get('content', '')}\n"

                        chunk = self._add_chunk(
                            context,
                            content=notes_content,
                            source_type="task_notes",
                            source_ref=task_id,
                            priority=priority - 5,
                            max_tokens=remaining_tokens,
                        )
                        if chunk:
                            remaining_tokens -= chunk.tokens_estimate
                except json.JSONDecodeError:
                    pass

            return remaining_tokens
        finally:
            conn.close()

    def _add_parent_chain(
        self,
        context: SwarmContext,
        task_id: str,
        remaining_tokens: int,
        priority_base: int,
    ) -> int:
        """Add parent task chain up to max depth."""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            current_task_id = task_id
            depth = 0

            while depth < self.max_parent_depth and remaining_tokens > 500:
                # Get parent
                cursor.execute(
                    "SELECT parent_task, title, description FROM tasks WHERE task_id = ?",
                    (current_task_id,)
                )
                row = cursor.fetchone()

                if not row or not row['parent_task']:
                    break

                parent_id = row['parent_task']

                # Get parent details
                cursor.execute(
                    "SELECT title, description, notes FROM tasks WHERE task_id = ?",
                    (parent_id,)
                )
                parent_row = cursor.fetchone()

                if parent_row:
                    content = f"# Parent Task (Level {depth + 1}): {parent_row['title']}\n\n{parent_row['description'] or 'No description'}"

                    chunk = self._add_chunk(
                        context,
                        content=content,
                        source_type="parent_task",
                        source_ref=parent_id,
                        priority=priority_base - (depth * 10),
                        max_tokens=remaining_tokens // 2,  # Save space for other parents
                    )
                    if chunk:
                        remaining_tokens -= chunk.tokens_estimate

                current_task_id = parent_id
                depth += 1

            return remaining_tokens
        finally:
            conn.close()

    def _add_project_context(
        self,
        context: SwarmContext,
        remaining_tokens: int,
        priority: int,
    ) -> int:
        """Add relevant project context entries."""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            # Get important project context entries
            cursor.execute(
                """
                SELECT context_key, value, description FROM project_context
                ORDER BY last_updated DESC
                LIMIT 20
                """
            )

            project_content_parts = []
            for row in cursor.fetchall():
                key = row['context_key']
                value = row['value']
                desc = row['description'] or ''

                # Skip very long values
                if len(value) > 2000:
                    value = value[:2000] + "... [truncated]"

                project_content_parts.append(f"### {key}\n{desc}\n```\n{value}\n```")

            if project_content_parts:
                content = "# Project Context\n\n" + "\n\n".join(project_content_parts)

                chunk = self._add_chunk(
                    context,
                    content=content,
                    source_type="project_context",
                    source_ref="global",
                    priority=priority,
                    max_tokens=remaining_tokens,
                )
                if chunk:
                    remaining_tokens -= chunk.tokens_estimate

            return remaining_tokens
        finally:
            conn.close()

    def _add_rag_context(
        self,
        context: SwarmContext,
        query: str,
        remaining_tokens: int,
        limit: int,
        priority: int,
    ) -> int:
        """Add RAG search results. Requires RAG to be set up."""
        # This would integrate with Agent-MCP's RAG system
        # For now, just search rag_chunks by text similarity
        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            # Simple text search in rag_chunks
            # A proper implementation would use vector similarity
            search_term = f"%{query[:50]}%"
            cursor.execute(
                """
                SELECT source_type, source_ref, chunk_text
                FROM rag_chunks
                WHERE chunk_text LIKE ?
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (search_term, limit)
            )

            for row in cursor.fetchall():
                if remaining_tokens < 200:
                    break

                chunk = self._add_chunk(
                    context,
                    content=row['chunk_text'],
                    source_type="rag",
                    source_ref=f"{row['source_type']}:{row['source_ref']}",
                    priority=priority,
                    max_tokens=remaining_tokens // limit,
                )
                if chunk:
                    remaining_tokens -= chunk.tokens_estimate

            return remaining_tokens
        finally:
            conn.close()

    def _get_task_description(self, task_id: Optional[str]) -> Optional[str]:
        """Get task description for RAG query."""
        if not task_id:
            return None

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT description FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            return row['description'] if row else None
        finally:
            conn.close()


def build_swarm_context(
    task_id: Optional[str] = None,
    description: Optional[str] = None,
    max_tokens: int = 100000,
    include_rag: bool = False,
) -> SwarmContext:
    """
    Convenience function to build swarm context.

    Args:
        task_id: Task ID to build context for
        description: Direct description
        max_tokens: Token budget
        include_rag: Include RAG results

    Returns:
        Assembled SwarmContext
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    return builder.build(
        task_id=task_id,
        description=description,
        include_rag=include_rag,
    )
