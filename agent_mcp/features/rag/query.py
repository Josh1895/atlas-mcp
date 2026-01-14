# Agent-MCP/mcp_template/mcp_server_src/features/rag/query.py
# Updated to use Gemini for embeddings and chat (unified under GEMINI_API_KEY)

import json
import sqlite3
from typing import List, Dict, Any, Optional

from ...core.config import (
    logger,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHAT_MODEL,
    MAX_CONTEXT_TOKENS,
)
from ...db.connection import get_db_connection, is_vss_loadable
from ...external.gemini_service import (
    get_gemini_embedding,
    get_gemini_chat_response,
    initialize_gemini_client,
)
from ...core import globals as g


async def query_rag_system(query_text: str) -> str:
    """
    Processes a natural language query using the RAG system.
    Fetches relevant context from live data and indexed knowledge,
    then uses Gemini to synthesize an answer.

    Args:
        query_text: The natural language question from the user.

    Returns:
        A string containing the answer or an error message.
    """
    # Ensure Gemini is initialized
    if not getattr(g, 'gemini_initialized', False):
        if not initialize_gemini_client():
            logger.error("RAG Query: Gemini client is not available. Cannot process query.")
            return "RAG Error: Gemini client not available. Please check server configuration and GEMINI_API_KEY."

    conn = None
    answer = "An unexpected error occurred during the RAG query."

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        live_context_results: List[Dict[str, Any]] = []
        live_task_results: List[Dict[str, Any]] = []
        vector_search_results: List[Dict[str, Any]] = []

        # --- 1. Fetch Live Context (Recently Updated) ---
        try:
            cursor.execute(
                "SELECT meta_value FROM rag_meta WHERE meta_key = ?",
                ("last_indexed_context",),
            )
            last_indexed_context_row = cursor.fetchone()
            last_indexed_context_time = (
                last_indexed_context_row["meta_value"]
                if last_indexed_context_row
                else "1970-01-01T00:00:00Z"
            )

            cursor.execute(
                """
                SELECT context_key, value, description, last_updated
                FROM project_context
                WHERE last_updated > ?
                ORDER BY last_updated DESC
                LIMIT 5
            """,
                (last_indexed_context_time,),
            )
            live_context_results = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e_live_ctx:
            logger.warning(f"RAG Query: Failed to fetch live project context: {e_live_ctx}")
        except Exception as e_live_ctx_other:
            logger.warning(
                f"RAG Query: Unexpected error fetching live project context: {e_live_ctx_other}",
                exc_info=True,
            )

        # --- 2. Fetch Live Tasks (Keyword Search) ---
        try:
            query_keywords = [
                f"%{word.strip().lower()}%"
                for word in query_text.split()
                if len(word.strip()) > 2
            ]
            if query_keywords:
                conditions = []
                sql_params_tasks: List[str] = []
                for kw in query_keywords:
                    conditions.append("LOWER(title) LIKE ?")
                    sql_params_tasks.append(kw)
                    conditions.append("LOWER(description) LIKE ?")
                    sql_params_tasks.append(kw)

                if conditions:
                    safe_conditions = []
                    for condition in conditions:
                        if condition not in [
                            "LOWER(title) LIKE ?",
                            "LOWER(description) LIKE ?",
                        ]:
                            logger.warning(f"RAG Query: Skipping unsafe condition: {condition}")
                            continue
                        safe_conditions.append(condition)

                    if safe_conditions:
                        where_clause = " OR ".join(safe_conditions)
                        task_query_sql = f"""
                            SELECT task_id, title, status, description, updated_at
                            FROM tasks
                            WHERE {where_clause}
                            ORDER BY updated_at DESC
                            LIMIT 5
                        """
                        cursor.execute(task_query_sql, sql_params_tasks)
                    live_task_results = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e_live_task:
            logger.warning(f"RAG Query: Failed to fetch live tasks: {e_live_task}")
        except Exception as e_live_task_other:
            logger.warning(
                f"RAG Query: Unexpected error fetching live tasks: {e_live_task_other}",
                exc_info=True,
            )

        # --- 3. Perform Vector Search (Indexed Knowledge) ---
        if is_vss_loadable():
            try:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_embeddings'"
                )
                if cursor.fetchone() is not None:
                    # Use Gemini for query embedding
                    query_embedding = get_gemini_embedding(query_text)
                    if query_embedding:
                        query_embedding_json = json.dumps(query_embedding)

                        k_results = 13
                        sql_vector_search = """
                            SELECT c.chunk_text, c.source_type, c.source_ref, c.metadata, r.distance
                            FROM rag_embeddings r
                            JOIN rag_chunks c ON r.rowid = c.chunk_id
                            WHERE r.embedding MATCH ? AND k = ?
                            ORDER BY r.distance
                        """
                        cursor.execute(sql_vector_search, (query_embedding_json, k_results))
                        raw_results = cursor.fetchall()

                        for row in raw_results:
                            result = dict(row)
                            if result.get("metadata"):
                                try:
                                    result["metadata"] = json.loads(result["metadata"])
                                except json.JSONDecodeError:
                                    result["metadata"] = None
                            vector_search_results.append(result)
                    else:
                        logger.warning("RAG Query: Failed to generate query embedding")
                else:
                    logger.warning("RAG Query: 'rag_embeddings' table not found.")
            except sqlite3.Error as e_vec_sql:
                logger.error(f"RAG Query: Database error during vector search: {e_vec_sql}")
            except Exception as e_vec_other:
                logger.error(
                    f"RAG Query: Unexpected error during vector search: {e_vec_other}",
                    exc_info=True,
                )
        else:
            logger.warning("RAG Query: Vector search (sqlite-vec) not available.")

        # --- 4. Combine Contexts for LLM ---
        context_parts: List[str] = []
        current_token_count: int = 0

        if live_context_results:
            context_parts.append("--- Recently Updated Project Context (Live) ---")
            for item in live_context_results:
                entry_text = f"Key: {item['context_key']}\nValue: {item['value']}\nDescription: {item.get('description', 'N/A')}\n(Updated: {item['last_updated']})\n"
                entry_tokens = len(entry_text.split())
                if current_token_count + entry_tokens < MAX_CONTEXT_TOKENS:
                    context_parts.append(entry_text)
                    current_token_count += entry_tokens
                else:
                    break
            context_parts.append("---------------------------------------------")

        if live_task_results:
            context_parts.append("--- Potentially Relevant Tasks (Live) ---")
            for task in live_task_results:
                entry_text = f"Task ID: {task['task_id']}\nTitle: {task['title']}\nStatus: {task['status']}\nDescription: {task.get('description', 'N/A')}\n(Updated: {task['updated_at']})\n"
                entry_tokens = len(entry_text.split())
                if current_token_count + entry_tokens < MAX_CONTEXT_TOKENS:
                    context_parts.append(entry_text)
                    current_token_count += entry_tokens
                else:
                    break
            context_parts.append("---------------------------------------")

        if vector_search_results:
            context_parts.append("--- Indexed Project Knowledge (Vector Search Results) ---")
            for i, item in enumerate(vector_search_results):
                chunk_text = item["chunk_text"]
                source_type = item["source_type"]
                source_ref = item["source_ref"]
                metadata = item.get("metadata", {})
                distance = item.get("distance", "N/A")

                source_info = f"Source Type: {source_type}, Reference: {source_ref}"

                if metadata and source_type in ["code", "code_summary"]:
                    if metadata.get("language"):
                        source_info += f", Language: {metadata['language']}"
                    if metadata.get("section_type"):
                        source_info += f", Section: {metadata['section_type']}"
                    if metadata.get("entities"):
                        entity_names = [e.get("name", "") for e in metadata["entities"]]
                        if entity_names:
                            source_info += f", Contains: {', '.join(entity_names[:3])}"
                            if len(entity_names) > 3:
                                source_info += f" (+{len(entity_names)-3} more)"

                entry_text = f"Retrieved Chunk {i+1} (Similarity/Distance: {distance}):\n{source_info}\nContent:\n{chunk_text}\n"
                chunk_tokens = len(entry_text.split())
                if current_token_count + chunk_tokens < MAX_CONTEXT_TOKENS:
                    context_parts.append(entry_text)
                    current_token_count += chunk_tokens
                else:
                    context_parts.append("--- [Indexed knowledge truncated due to token limit] ---")
                    break
            context_parts.append("-------------------------------------------------------")

        if not context_parts:
            logger.info(f"RAG Query: No relevant information found for query: '{query_text}'")
            answer = "No relevant information found in the project knowledge base or live data for your query."
        else:
            combined_context_str = "\n\n".join(context_parts)

            # --- 5. Call Gemini Chat API ---
            system_prompt_for_llm = """You are an AI assistant answering questions about a software project.
Use the provided context, which may include recently updated live data (like project context keys or tasks) and information retrieved from an indexed knowledge base (like documentation or code summaries), to answer the user's query.
Prioritize information from the 'Live' sections if available and relevant for time-sensitive data.
Answer using *only* the information given in the context. If the context doesn't contain the answer, state that clearly.

Be VERBOSE and comprehensive in your responses. It's better to give too much context than too little.
When answering, please also suggest additional context entries and queries that might be helpful for understanding this topic better.
For example, suggest related files to examine, related project context keys to check, or follow-up questions that could provide more insight.
Always err on the side of providing more detailed explanations and comprehensive information rather than brief responses."""

            user_message_for_llm = f"CONTEXT:\n{combined_context_str}\n\nQUERY:\n{query_text}\n\nBased *only* on the CONTEXT provided above, please answer the QUERY."

            logger.debug(f"RAG Query: Combined context for LLM (approx tokens: {current_token_count})")

            # Use Gemini for chat
            answer = get_gemini_chat_response(
                prompt=user_message_for_llm,
                system_instruction=system_prompt_for_llm,
                temperature=0.4,
            )

            if not answer:
                answer = "Error: Failed to get response from Gemini."

    except sqlite3.Error as e_sql:
        logger.error(f"RAG Query: Database error: {e_sql}", exc_info=True)
        answer = f"Error querying RAG database: {e_sql}"
    except Exception as e_unexpected:
        logger.error(f"RAG Query: Unexpected error: {e_unexpected}", exc_info=True)
        answer = f"An unexpected error occurred during the RAG query: {str(e_unexpected)}"
    finally:
        if conn:
            conn.close()

    return answer


async def query_rag_system_with_model(
    query_text: str, model_name: str = None, max_tokens: int = None
) -> str:
    """
    Processes a query using the RAG system with Gemini.
    The model_name parameter is kept for API compatibility but uses Gemini.

    Args:
        query_text: The natural language question from the user.
        model_name: Ignored (uses Gemini)
        max_tokens: The maximum context tokens

    Returns:
        A string containing the answer or an error message.
    """
    # Ensure Gemini is initialized
    if not getattr(g, 'gemini_initialized', False):
        if not initialize_gemini_client():
            logger.error("RAG Query: Gemini client is not available.")
            return "RAG Error: Gemini client not available."

    context_limit = max_tokens if max_tokens else MAX_CONTEXT_TOKENS

    conn = None
    answer = "An unexpected error occurred during the RAG query."

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        live_context_results: List[Dict[str, Any]] = []
        live_task_results: List[Dict[str, Any]] = []
        vector_search_results: List[Dict[str, Any]] = []

        # Get live context
        cursor.execute(
            "SELECT context_key, value, description, last_updated FROM project_context ORDER BY last_updated DESC"
        )
        live_context_results = [dict(row) for row in cursor.fetchall()]

        # Get live tasks
        cursor.execute(
            """
            SELECT task_id, title, description, status, created_by, assigned_to,
                   priority, parent_task, depends_on_tasks, created_at, updated_at
            FROM tasks
            WHERE status IN ('pending', 'in_progress')
            ORDER BY updated_at DESC
        """
        )
        live_task_results = [dict(row) for row in cursor.fetchall()]

        # Get vector search results if VSS is available
        if is_vss_loadable():
            try:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_embeddings'"
                )
                if cursor.fetchone() is not None:
                    # Use Gemini for query embedding
                    query_embedding = get_gemini_embedding(query_text)
                    if query_embedding:
                        query_embedding_json = json.dumps(query_embedding)

                        k_results = 13
                        vector_search_sql = """
                            SELECT c.chunk_text, c.source_type, c.source_ref, c.metadata, r.distance
                            FROM rag_embeddings r
                            JOIN rag_chunks c ON r.rowid = c.chunk_id
                            WHERE r.embedding MATCH ? AND k = ?
                            ORDER BY r.distance
                        """
                        cursor.execute(vector_search_sql, (query_embedding_json, k_results))
                        raw_results = cursor.fetchall()

                        for row in raw_results:
                            result = dict(row)
                            if result.get("metadata"):
                                try:
                                    result["metadata"] = json.loads(result["metadata"])
                                except json.JSONDecodeError:
                                    result["metadata"] = None
                            vector_search_results.append(result)
                else:
                    logger.warning("RAG Query: 'rag_embeddings' table not found.")
            except sqlite3.Error as e_vec_sql:
                logger.error(f"RAG Query: Database error during vector search: {e_vec_sql}")
            except Exception as e_vec_other:
                logger.error(f"RAG Query: Error during vector search: {e_vec_other}", exc_info=True)

        # Build context
        context_parts = []
        current_token_count = 0

        if live_context_results:
            context_parts.append("=== Live Project Context ===")
            for item in live_context_results:
                entry_text = f"Key: {item['context_key']}\nDescription: {item['description']}\nValue: {item['value']}\nLast Updated: {item['last_updated']}\n"
                chunk_tokens = len(entry_text.split())
                if current_token_count + chunk_tokens < context_limit:
                    context_parts.append(entry_text)
                    current_token_count += chunk_tokens
                else:
                    context_parts.append("--- [Live context truncated due to token limit] ---")
                    break

        if live_task_results:
            context_parts.append("\n=== Live Task Information ===")
            for item in live_task_results:
                entry_text = f"Task ID: {item['task_id']}\nTitle: {item['title']}\nDescription: {item['description']}\nStatus: {item['status']}\n"
                entry_text += f"Priority: {item['priority']}\nAssigned To: {item['assigned_to']}\nCreated By: {item['created_by']}\n"
                entry_text += f"Parent Task: {item['parent_task']}\nDependencies: {item['depends_on_tasks']}\n"
                entry_text += f"Created: {item['created_at']}\nUpdated: {item['updated_at']}\n"
                chunk_tokens = len(entry_text.split())
                if current_token_count + chunk_tokens < context_limit:
                    context_parts.append(entry_text)
                    current_token_count += chunk_tokens
                else:
                    context_parts.append("--- [Live tasks truncated due to token limit] ---")
                    break

        if vector_search_results:
            context_parts.append("\n=== Retrieved from Indexed Knowledge ===")
            for i, item in enumerate(vector_search_results):
                chunk_text = item["chunk_text"]
                source_type = item["source_type"]
                source_ref = item["source_ref"]
                metadata = item.get("metadata", {})
                distance = item.get("distance", "N/A")

                source_info = f"Source Type: {source_type}, Reference: {source_ref}"

                if metadata and source_type in ["code", "code_summary"]:
                    if metadata.get("language"):
                        source_info += f", Language: {metadata['language']}"
                    if metadata.get("section_type"):
                        source_info += f", Section: {metadata['section_type']}"
                    if metadata.get("entities"):
                        entity_names = [e.get("name", "") for e in metadata["entities"]]
                        if entity_names:
                            source_info += f", Contains: {', '.join(entity_names[:3])}"
                            if len(entity_names) > 3:
                                source_info += f" (+{len(entity_names)-3} more)"

                entry_text = f"Retrieved Chunk {i+1} (Similarity/Distance: {distance}):\n{source_info}\nContent:\n{chunk_text}\n"
                chunk_tokens = len(entry_text.split())
                if current_token_count + chunk_tokens < context_limit:
                    context_parts.append(entry_text)
                    current_token_count += chunk_tokens
                else:
                    context_parts.append("--- [Indexed knowledge truncated due to token limit] ---")
                    break

        if not context_parts:
            logger.info(f"RAG Query: No relevant information found for query: '{query_text}'")
            answer = "No relevant information found."
        else:
            combined_context_str = "\n\n".join(context_parts)

            system_prompt_for_llm = """You are an AI assistant specializing in task hierarchy analysis and project structure optimization.
You must CRITICALLY THINK about task placement, dependencies, and hierarchical relationships.
Use the provided context to make intelligent recommendations about task organization.
Be strict about the single root task rule and logical task relationships.

Be VERBOSE and comprehensive in your analysis. It's better to give too much context than too little.
When making recommendations, suggest additional context entries and queries that might be helpful for understanding task relationships better.
Consider suggesting related files to examine, project context keys to check, or follow-up questions for deeper task analysis.
Provide detailed explanations for your reasoning and comprehensive information rather than brief responses.
Answer in the exact JSON format requested, but include thorough explanations in your reasoning sections."""

            user_message_for_llm = f"CONTEXT:\n{combined_context_str}\n\nQUERY:\n{query_text}\n\nBased on the CONTEXT provided above, please answer the QUERY."

            logger.info(f"Task Analysis Query: Using Gemini with {context_limit} token limit")

            # Use Gemini for chat
            answer = get_gemini_chat_response(
                prompt=user_message_for_llm,
                system_instruction=system_prompt_for_llm,
                temperature=0.4,
            )

            if not answer:
                answer = "Error: Failed to get response from Gemini."

    except Exception as e:
        logger.error(f"RAG Query: Error: {e}", exc_info=True)
        answer = f"Error during RAG query: {str(e)}"
    finally:
        if conn:
            conn.close()

    return answer
