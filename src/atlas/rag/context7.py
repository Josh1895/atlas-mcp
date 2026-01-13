"""Context7 MCP client for documentation retrieval.

Context7 is an MCP server that provides up-to-date documentation for libraries.
This client connects to Context7 via MCP protocol (SSE transport) for reliable access.

MCP-to-MCP Architecture:
- ATLAS MCP Server acts as both server (to Claude) and client (to Context7)
- Uses SSE transport to connect to Context7's remote MCP endpoint
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from atlas.core.config import Config, get_config

logger = logging.getLogger(__name__)

# Context7 MCP endpoint (SSE transport)
CONTEXT7_MCP_URL = "https://mcp.context7.com/sse"


@dataclass
class DocumentChunk:
    """A chunk of documentation from Context7."""

    content: str
    source: str
    library_id: str
    relevance_score: float = 0.0


@dataclass
class Context7Result:
    """Result from a Context7 query."""

    chunks: list[DocumentChunk] = field(default_factory=list)
    library_id: str = ""
    query: str = ""
    total_tokens: int = 0

    @property
    def combined_content(self) -> str:
        """Get all chunks combined into a single string."""
        return "\n\n---\n\n".join(
            f"Source: {chunk.source}\n{chunk.content}"
            for chunk in self.chunks
        )


class Context7Client:
    """MCP Client for Context7 documentation retrieval.

    Connects to Context7 MCP server using SSE transport and calls
    its tools via MCP protocol.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the Context7 MCP client."""
        self.config = config or get_config()
        self._cache: dict[str, Context7Result] = {}
        self._session = None

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the Context7 MCP server.

        Uses HTTP REST API directly for speed (MCP SSE is slow due to connection overhead).
        """
        # Use HTTP directly - much faster than MCP SSE which creates new connection each time
        return await self._call_tool_http(tool_name, arguments)

    async def _call_tool_http(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Fallback: Call Context7 via HTTP REST API v2 if MCP fails."""
        import httpx

        base_url = "https://context7.com/api/v2"

        # Map MCP tool names to REST API v2 endpoints
        if tool_name == "resolve-library-id":
            url = f"{base_url}/libs/search"
            params = {"libraryName": arguments.get("libraryName", "")}
        elif tool_name == "get-library-docs":
            url = f"{base_url}/context"
            params = {
                "libraryId": arguments.get("context7CompatibleLibraryID", ""),
                "query": arguments.get("topic", ""),
                "tokens": arguments.get("tokens", 5000),
            }
        else:
            return None

        headers = {"Accept": "application/json"}
        if self.config.context7_api_key:
            headers["Authorization"] = f"Bearer {self.config.context7_api_key}"

        # Retry with exponential backoff for rate limiting (429)
        max_retries = 3
        base_delay = 2.0

        try:
            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    response = await client.get(url, params=params, headers=headers, timeout=60.0)

                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                            logger.warning(f"Context7 rate limited (429), retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error("Context7 rate limited after all retries")
                            return None

                    response.raise_for_status()
                    break

                # Check content type to determine how to parse
                content_type = response.headers.get('content-type', '')

                # Handle get-library-docs - returns text/plain directly
                if tool_name == "get-library-docs":
                    if 'text/plain' in content_type:
                        # Direct text response - return as-is
                        return response.text if response.text.strip() else None
                    elif 'application/json' in content_type:
                        data = response.json()
                        if isinstance(data, str):
                            return data
                        elif isinstance(data, dict):
                            content = data.get("context") or data.get("content") or data.get("text")
                            if isinstance(content, list):
                                return "\n\n".join(str(c) for c in content)
                            return str(content) if content else None
                    else:
                        # Unknown content type, try as text
                        return response.text if response.text.strip() else None

                # Handle resolve-library-id - returns JSON
                elif tool_name == "resolve-library-id":
                    data = response.json()
                    if isinstance(data, dict):
                        lib_id = data.get("libraryId") or data.get("library_id")
                        if not lib_id and "results" in data:
                            results = data.get("results", [])
                            if results and isinstance(results[0], dict):
                                lib_id = results[0].get("libraryId") or results[0].get("id")
                        return lib_id
                    elif isinstance(data, list) and data:
                        return data[0].get("libraryId") or data[0].get("id")

                return None

        except Exception as e:
            logger.error(f"HTTP fallback failed: {e}")
            return None

    async def resolve_library_id(self, library_name: str) -> str | None:
        """Resolve a library name to a Context7 library ID.

        Uses Context7's resolve-library-id MCP tool.

        Args:
            library_name: Human-readable library name (e.g., "react", "asyncio")

        Returns:
            Context7 library ID (e.g., "/vercel/next.js") or None if not found
        """
        cache_key = f"resolve:{library_name}"
        if cache_key in self._cache:
            return self._cache[cache_key].library_id

        try:
            result = await self._call_tool(
                "resolve-library-id",
                {"libraryName": library_name}
            )

            if result:
                # Parse the library ID from the response
                # Context7 returns the ID directly or in a structured format
                library_id = None
                if isinstance(result, str):
                    # Look for pattern like /owner/repo
                    import re
                    match = re.search(r'(/[\w-]+/[\w.-]+)', result)
                    if match:
                        library_id = match.group(1)
                    elif result.startswith("/"):
                        library_id = result.strip()

                if library_id:
                    self._cache[cache_key] = Context7Result(library_id=library_id)
                    logger.info(f"Resolved '{library_name}' to '{library_id}'")
                    return library_id

        except Exception as e:
            logger.error(f"Failed to resolve library '{library_name}': {e}")

        return None

    async def query_docs(
        self,
        library_id: str,
        query: str,
        max_tokens: int = 5000,
    ) -> Context7Result:
        """Query documentation for a library.

        Uses Context7's get-library-docs MCP tool.

        Args:
            library_id: Context7 library ID (e.g., "/vercel/next.js")
            query: Natural language query / topic
            max_tokens: Maximum tokens to return

        Returns:
            Context7Result with documentation chunks
        """
        # Normalize library ID
        if not library_id.startswith("/"):
            library_id = f"/{library_id}"

        cache_key = f"query:{library_id}:{query}:{max_tokens}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = await self._call_tool(
                "get-library-docs",
                {
                    "context7CompatibleLibraryID": library_id,
                    "topic": query,
                    "tokens": max_tokens,
                }
            )

            if result and isinstance(result, str) and result.strip():
                # Create a documentation chunk from the result
                chunks = [DocumentChunk(
                    content=result.strip(),
                    source=f"Context7: {library_id}",
                    library_id=library_id,
                    relevance_score=1.0,
                )]

                ctx_result = Context7Result(
                    chunks=chunks,
                    library_id=library_id,
                    query=query,
                    total_tokens=len(result.split()),  # Approximate
                )

                self._cache[cache_key] = ctx_result
                logger.info(f"Got {len(result)} chars of docs for '{library_id}'")
                return ctx_result

        except Exception as e:
            logger.error(f"Failed to query docs for '{library_id}': {e}")

        return Context7Result(library_id=library_id, query=query)

    async def get_documentation(
        self,
        library_name: str,
        query: str,
        max_tokens: int = 5000,
    ) -> Context7Result:
        """Get documentation for a library by name or ID.

        Args:
            library_name: Library name or Context7 ID (e.g., "python/cpython")
            query: Natural language query
            max_tokens: Maximum tokens to return

        Returns:
            Context7Result with documentation chunks
        """
        # If it looks like a Context7 ID (contains /), use directly
        if "/" in library_name:
            library_id = library_name
        else:
            # Try to resolve the human-readable name
            library_id = await self.resolve_library_id(library_name)
            if not library_id:
                logger.warning(f"Could not resolve library: {library_name}")
                return Context7Result(query=query)

        return await self.query_docs(library_id, query, max_tokens)

    async def get_multi_library_docs(
        self,
        libraries: list[str],
        query: str,
        max_tokens_per_lib: int = 2000,
    ) -> dict[str, Context7Result]:
        """Get documentation from multiple libraries in parallel.

        Args:
            libraries: List of library names
            query: Natural language query
            max_tokens_per_lib: Maximum tokens per library

        Returns:
            Dict mapping library names to their results
        """
        tasks = [
            self.get_documentation(lib, query, max_tokens_per_lib)
            for lib in libraries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            lib: result if not isinstance(result, Exception) else Context7Result(query=query)
            for lib, result in zip(libraries, results)
        }

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def extract_libraries_from_issue(self, issue_text: str) -> list[str]:
        """Extract likely library names from an issue description.

        Maps common terms to their Context7 library IDs.

        Args:
            issue_text: The issue description text

        Returns:
            List of Context7 library IDs
        """
        # Map common terms to their Context7 library IDs
        library_mappings = {
            # Python libraries
            "asyncio": "python/cpython",
            "async": "python/cpython",
            "await": "python/cpython",
            "aiohttp": "aio-libs/aiohttp",
            "fastapi": "tiangolo/fastapi",
            "flask": "pallets/flask",
            "django": "django/django",
            "requests": "psf/requests",
            "httpx": "encode/httpx",
            "pydantic": "pydantic/pydantic",
            "sqlalchemy": "sqlalchemy/sqlalchemy",
            "pytest": "pytest-dev/pytest",
            "numpy": "numpy/numpy",
            "pandas": "pandas-dev/pandas",
            "celery": "celery/celery",

            # JavaScript/TypeScript
            "react": "facebook/react",
            "next.js": "vercel/next.js",
            "nextjs": "vercel/next.js",
            "vue": "vuejs/vue",
            "angular": "angular/angular",
            "express": "expressjs/express",
            "typescript": "microsoft/typescript",
            "node": "nodejs/node",
            "nodejs": "nodejs/node",

            # Databases
            "mongodb": "mongodb/mongo",
            "redis": "redis/redis",
            "postgresql": "postgres/postgres",
            "prisma": "prisma/prisma",

            # Infrastructure
            "docker": "docker/docs",
            "kubernetes": "kubernetes/kubernetes",

            # Other
            "graphql": "graphql/graphql-js",
            "tailwind": "tailwindlabs/tailwindcss",
        }

        text_lower = issue_text.lower()
        found = []
        seen_ids = set()

        for keyword, library_id in library_mappings.items():
            if keyword in text_lower and library_id not in seen_ids:
                found.append(library_id)
                seen_ids.add(library_id)

        return found[:5]
