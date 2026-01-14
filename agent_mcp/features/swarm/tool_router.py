"""
Tool Router for swarm operations.

Implements policy enforcement, caching, rate limiting, and circuit breakers
for all external tool calls (AD-004). Single point of control for tool access.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable
from enum import Enum

from ...core.config import logger, SWARM_ENABLE_WEB, SWARM_ENABLE_CONTEXT7
from .schemas import ToolConfig
from .persistence import (
    save_tool_cache,
    get_tool_cache,
    log_tool_call,
    update_tool_call,
)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for a tool provider.

    Opens after N consecutive failures, then enters half-open state
    after cooldown to test if the service has recovered.
    """
    name: str
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                # Recovered
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
                logger.info(f"Circuit {self.name} closed (recovered)")
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during test, reopen
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            logger.warning(f"Circuit {self.name} reopened (failed in half-open)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} opened (failures: {self.failure_count})")

    def can_execute(self) -> bool:
        """Check if calls are allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if cooldown has passed
            if time.time() - self.last_failure_time >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit {self.name} half-open (testing recovery)")
                return True
            return False

        # HALF_OPEN - allow limited calls
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time else None,
        }


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Allows burst up to bucket_size, then rate-limits to calls_per_second.
    """
    calls_per_second: float
    bucket_size: int = 10

    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.bucket_size)

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make a call.

        Args:
            timeout: Max seconds to wait for a token

        Returns:
            True if acquired, False if timed out
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # Wait a bit before retrying
            await asyncio.sleep(0.1)

        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        self._tokens = min(
            self.bucket_size,
            self._tokens + (elapsed * self.calls_per_second)
        )


class ToolRouterError(Exception):
    """Base error for tool router."""
    pass


class ToolNotAllowedError(ToolRouterError):
    """Tool is not allowed by policy."""
    pass


class CircuitOpenError(ToolRouterError):
    """Circuit breaker is open."""
    pass


class RateLimitExceededError(ToolRouterError):
    """Rate limit exceeded."""
    pass


class ToolRouter:
    """
    Routes tool calls with policy enforcement, caching, rate limiting,
    and circuit breakers.

    Usage:
        router = ToolRouter(config, run_id)

        # Register tool handlers
        router.register_tool("web_search", web_search_handler, cache_ttl=604800)

        # Execute tool call
        result = await router.call("web_search", {"query": "python async"})
    """

    def __init__(
        self,
        config: Optional[ToolConfig] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize tool router.

        Args:
            config: Tool availability configuration
            run_id: Swarm run ID for logging
        """
        self.config = config or ToolConfig()
        self.run_id = run_id or "unknown"

        # Tool handlers
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._cache_ttls: Dict[str, int] = {}  # Tool name -> TTL in seconds

        # Rate limiters per tool
        self._rate_limiters: Dict[str, RateLimiter] = {}

        # Circuit breakers per tool
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Stats
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        # Set up default rate limits
        self._default_rate_limits = {
            "web_search": 2.0,  # 2 RPS
            "context7": 1.0,  # 1 RPS
            "repo_search": 10.0,  # 10 RPS
            "agent_mcp_rag": 5.0,  # 5 RPS
        }

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        cache_ttl: int = 0,
        rate_limit: Optional[float] = None,
        circuit_threshold: int = 5,
    ) -> None:
        """
        Register a tool handler.

        Args:
            name: Tool name
            handler: Async function to handle tool calls
            cache_ttl: Cache TTL in seconds (0 = no cache)
            rate_limit: Calls per second (None = use default)
            circuit_threshold: Failures before circuit opens
        """
        self._handlers[name] = handler
        self._cache_ttls[name] = cache_ttl

        # Set up rate limiter
        rps = rate_limit or self._default_rate_limits.get(name, 10.0)
        self._rate_limiters[name] = RateLimiter(calls_per_second=rps)

        # Set up circuit breaker
        self._circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=circuit_threshold,
        )

        logger.debug(f"Registered tool {name} (cache_ttl={cache_ttl}, rps={rps})")

    async def call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        agent_id: Optional[str] = None,
        skip_cache: bool = False,
    ) -> Any:
        """
        Execute a tool call with all protections.

        Args:
            tool_name: Name of the tool to call
            args: Arguments for the tool
            agent_id: Optional agent ID for logging
            skip_cache: Skip cache lookup/write

        Returns:
            Tool result

        Raises:
            ToolNotAllowedError: Tool not allowed by policy
            CircuitOpenError: Circuit breaker is open
            RateLimitExceededError: Rate limit exceeded
        """
        self._stats["total_calls"] += 1

        # 1. Check policy
        self._check_permission(tool_name)

        # 2. Check circuit breaker
        circuit = self._circuit_breakers.get(tool_name)
        if circuit and not circuit.can_execute():
            raise CircuitOpenError(f"Circuit open for {tool_name}")

        # 3. Check cache
        cache_key = None
        if not skip_cache and self._cache_ttls.get(tool_name, 0) > 0:
            cache_key = self._make_cache_key(tool_name, args)
            cached = get_tool_cache(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {tool_name}")
                return cached
            self._stats["cache_misses"] += 1

        # 4. Rate limit
        limiter = self._rate_limiters.get(tool_name)
        if limiter:
            if not await limiter.acquire(timeout=10.0):
                raise RateLimitExceededError(f"Rate limit for {tool_name}")

        # 5. Log call start
        call_id = await log_tool_call(
            run_id=self.run_id,
            swarm_agent_id=agent_id,
            tool_name=tool_name,
            args=args,
            status="started",
        )

        # 6. Execute
        start_time = time.time()
        try:
            handler = self._handlers.get(tool_name)
            if not handler:
                raise ToolRouterError(f"No handler for tool: {tool_name}")

            result = await handler(**args)

            duration_ms = int((time.time() - start_time) * 1000)

            # Record success
            if circuit:
                circuit.record_success()

            # Update call log
            await update_tool_call(
                call_id=call_id,
                status="success",
                duration_ms=duration_ms,
                result_meta={"result_type": type(result).__name__},
            )

            # Cache result
            if cache_key and self._cache_ttls.get(tool_name, 0) > 0:
                await save_tool_cache(
                    cache_key=cache_key,
                    tool_name=tool_name,
                    value=result,
                    ttl_seconds=self._cache_ttls[tool_name],
                )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._stats["errors"] += 1

            # Record failure
            if circuit:
                circuit.record_failure()

            # Update call log
            await update_tool_call(
                call_id=call_id,
                status="error",
                duration_ms=duration_ms,
                error=str(e),
            )

            raise

    def _check_permission(self, tool_name: str) -> None:
        """Check if tool is allowed by policy."""
        # Map tool names to config flags
        permission_map = {
            "web_search": self.config.enable_web_search and SWARM_ENABLE_WEB,
            "context7": self.config.enable_context7 and SWARM_ENABLE_CONTEXT7,
            "repo_search": self.config.enable_repo_search,
            "agent_mcp_rag": self.config.enable_agent_mcp_rag,
        }

        if tool_name in permission_map:
            if not permission_map[tool_name]:
                raise ToolNotAllowedError(f"Tool {tool_name} is disabled")

    def _make_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate cache key from tool name and args."""
        # Sort args for consistent hashing
        args_str = json.dumps(args, sort_keys=True)
        hash_input = f"{tool_name}:{args_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        circuits = {
            name: cb.get_status()
            for name, cb in self._circuit_breakers.items()
        }

        return {
            **self._stats,
            "cache_hit_rate": (
                self._stats["cache_hits"] /
                max(1, self._stats["cache_hits"] + self._stats["cache_misses"])
            ),
            "circuits": circuits,
        }


# --- Default Tool Handlers ---
# These handlers integrate with Atlas and Agent-MCP services

# Singleton clients
_web_search_client = None
_context7_client = None


def _get_web_search_client():
    """Get or create WebSearchClient singleton."""
    global _web_search_client
    if _web_search_client is None:
        try:
            from atlas.rag.web_search import WebSearchClient
            _web_search_client = WebSearchClient()
        except ImportError:
            logger.warning("Atlas WebSearchClient not available")
            _web_search_client = None
    return _web_search_client


def _get_context7_client():
    """Get or create Context7Client singleton."""
    global _context7_client
    if _context7_client is None:
        try:
            from atlas.rag.context7 import Context7Client
            _context7_client = Context7Client()
        except ImportError:
            logger.warning("Atlas Context7Client not available")
            _context7_client = None
    return _context7_client


async def web_search_handler(
    query: str,
    num_results: int = 10,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
    fetch_full_content: bool = True,
) -> Dict[str, Any]:
    """
    Web search handler using Atlas WebSearchClient.

    Searches the web using DuckDuckGo or SerpAPI backend.
    By default, fetches FULL page content (4,000-6,000 chars per result)
    for rich context instead of just snippets (200-300 chars).

    Args:
        query: Search query
        num_results: Maximum number of results
        allowed_domains: Optional list of allowed domains
        blocked_domains: Optional list of blocked domains
        fetch_full_content: If True (default), fetch full page content for rich context

    Returns:
        Dict with query and results (with rich content if fetch_full_content=True)
    """
    client = _get_web_search_client()
    if not client:
        return {
            "query": query,
            "results": [],
            "error": "WebSearchClient not available",
        }

    try:
        if fetch_full_content:
            # Use search_for_code_context for rich content (4,000-6,000 chars per result)
            results = await client.search_for_code_context(query)

            # Filter blocked domains
            if blocked_domains:
                results.results = [
                    r for r in results.results
                    if not any(blocked in r.url for blocked in blocked_domains)
                ]

            # Sort by content length (rich content first) to prioritize full page fetches
            # Stack Overflow snippets are ~200-300 chars, fetched pages are 4,000-6,000 chars
            results.results.sort(key=lambda r: len(r.snippet), reverse=True)

            # Limit to requested number
            results.results = results.results[:num_results]
        else:
            # Basic search with short snippets (200-300 chars)
            site_filter = allowed_domains[0] if allowed_domains else None

            results = await client.search(
                query=query,
                max_results=num_results,
                site_filter=site_filter,
                filter_domains=not allowed_domains,
            )

            # Filter blocked domains
            if blocked_domains:
                results.results = [
                    r for r in results.results
                    if not any(blocked in r.url for blocked in blocked_domains)
                ]

        return {
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,  # Now 4,000-6,000 chars when fetch_full_content=True
                    "source": r.source,
                }
                for r in results.results[:num_results]
            ],
            "error": results.error if hasattr(results, 'error') else None,
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "query": query,
            "results": [],
            "error": str(e),
        }


async def context7_handler(
    library: str,
    query: str,
    max_tokens: int = 5000,
) -> Dict[str, Any]:
    """
    Context7 documentation search handler.

    Retrieves up-to-date documentation from Context7 MCP service.

    Args:
        library: Library name or Context7 ID (e.g., "react", "python/cpython")
        query: Documentation query/topic
        max_tokens: Maximum tokens to return

    Returns:
        Dict with library, query, and documentation chunks
    """
    client = _get_context7_client()
    if not client:
        return {
            "library": library,
            "query": query,
            "results": [],
            "error": "Context7Client not available",
        }

    try:
        result = await client.get_documentation(
            library_name=library,
            query=query,
            max_tokens=max_tokens,
        )

        return {
            "library": library,
            "library_id": result.library_id,
            "query": query,
            "results": [
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "relevance": chunk.relevance_score,
                }
                for chunk in result.chunks
            ],
            "combined_content": result.combined_content,
            "total_tokens": result.total_tokens,
        }

    except Exception as e:
        logger.error(f"Context7 query failed: {e}")
        return {
            "library": library,
            "query": query,
            "results": [],
            "error": str(e),
        }


async def repo_search_handler(
    query: str,
    repo_path: str,
    file_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Repository search handler.

    Searches for code/content within a repository using ripgrep or grep.

    Args:
        query: Search pattern
        repo_path: Path to repository
        file_pattern: Optional file glob pattern (e.g., "*.py")

    Returns:
        Dict with query, repo_path, and matching files
    """
    import subprocess
    from pathlib import Path

    results = []
    try:
        # Use ripgrep if available, fallback to grep
        cmd = ["rg", "--json", "-l", query, repo_path]
        if file_pattern:
            cmd.extend(["-g", file_pattern])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "match":
                            results.append(data["data"]["path"]["text"])
                    except json.JSONDecodeError:
                        results.append(line)

    except FileNotFoundError:
        # ripgrep not available, use grep
        cmd = ["grep", "-rl", query, repo_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        results = result.stdout.strip().split("\n") if result.stdout else []
    except Exception as e:
        logger.error(f"repo_search error: {e}")

    return {
        "query": query,
        "repo_path": repo_path,
        "matches": results[:50],  # Limit results
    }


async def agent_mcp_rag_handler(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Agent-MCP RAG query handler.

    Searches the local RAG system for relevant project context.

    Args:
        query: Search query
        project_id: Optional project ID to scope search
        limit: Maximum number of results
        min_score: Minimum relevance score (0-1)

    Returns:
        Dict with query and matching chunks
    """
    try:
        # Import Agent-MCP RAG query function
        from ...features.rag.query import query_rag_system

        results = await query_rag_system(
            query=query,
            project_id=project_id,
            limit=limit,
            min_score=min_score,
        )

        return {
            "query": query,
            "project_id": project_id,
            "results": [
                {
                    "content": r.get("content", ""),
                    "source": r.get("source", ""),
                    "score": r.get("score", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ],
            "count": len(results),
        }

    except ImportError:
        logger.warning("Agent-MCP RAG not available")
        return {
            "query": query,
            "results": [],
            "error": "RAG system not available",
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "query": query,
            "results": [],
            "error": str(e),
        }


def create_default_router(
    config: Optional[ToolConfig] = None,
    run_id: Optional[str] = None,
) -> ToolRouter:
    """
    Create a tool router with default handlers.

    Args:
        config: Tool configuration
        run_id: Swarm run ID

    Returns:
        Configured ToolRouter
    """
    router = ToolRouter(config, run_id)

    # Register handlers with appropriate TTLs
    router.register_tool(
        "web_search",
        web_search_handler,
        cache_ttl=604800,  # 7 days
        rate_limit=2.0,
    )

    router.register_tool(
        "context7",
        context7_handler,
        cache_ttl=604800,  # 7 days
        rate_limit=1.0,
    )

    router.register_tool(
        "repo_search",
        repo_search_handler,
        cache_ttl=600,  # 10 minutes
        rate_limit=10.0,
    )

    router.register_tool(
        "agent_mcp_rag",
        agent_mcp_rag_handler,
        cache_ttl=600,  # 10 minutes
        rate_limit=5.0,
    )

    return router
