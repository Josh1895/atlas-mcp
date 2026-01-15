"""Agentic Gemini client with tool access.

Gives Gemini autonomous access to tools like Context7 and web search.
The AI decides what to search, when to search, and how much to search.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai
from google.genai import types

from atlas.core.config import Config, get_config
from atlas.rag.context7 import Context7Client
from atlas.rag.web_search import WebSearchClient
from atlas.scout.repo_tools import RepoTools

logger = logging.getLogger(__name__)


# Tool definitions for Gemini function calling
# Includes both Context7 (documentation) and Web Search tools
AGENT_TOOLS = [
    types.Tool(
        function_declarations=[
            # Context7 tools for official documentation
            types.FunctionDeclaration(
                name="resolve_library_id",
                description=(
                    "Resolve a library name to its Context7 library ID. "
                    "Use this first to find the correct ID for any library you want to search. "
                    "Examples: 'react' -> '/facebook/react', 'nextjs' -> '/vercel/next.js', "
                    "'python asyncio' -> '/python/cpython', 'fastapi' -> '/tiangolo/fastapi'"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "library_name": types.Schema(
                            type=types.Type.STRING,
                            description="The library name to resolve (e.g., 'react', 'asyncio', 'fastapi', 'tanstack query')",
                        ),
                    },
                    required=["library_name"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_library_docs",
                description=(
                    "Get official documentation from a library using its Context7 ID. "
                    "Use resolve_library_id first to get the correct ID. "
                    "Best for: API references, official guides, function signatures. "
                    "You can call this multiple times with different queries."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "library_id": types.Schema(
                            type=types.Type.STRING,
                            description="The Context7 library ID (e.g., '/facebook/react', '/python/cpython')",
                        ),
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="What to search for in the documentation (e.g., 'hooks useState', 'asyncio Lock synchronization')",
                        ),
                        "max_tokens": types.Schema(
                            type=types.Type.INTEGER,
                            description="Maximum tokens to return (default 5000, max 10000)",
                        ),
                    },
                    required=["library_id", "query"],
                ),
            ),
            # Web search tool for broader research
            types.FunctionDeclaration(
                name="web_search",
                description=(
                    "Search the web for coding information, best practices, Stack Overflow answers, "
                    "tutorials, and real-world examples. Use this to find: "
                    "1. Best practices and common patterns "
                    "2. Stack Overflow solutions to similar problems "
                    "3. Tutorial explanations and examples "
                    "4. GitHub code examples "
                    "5. Blog posts explaining concepts "
                    "You can optionally filter to a specific site like stackoverflow.com or github.com."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The search query (e.g., 'python asyncio Lock best practices', 'time.monotonic vs time.time')",
                        ),
                        "site_filter": types.Schema(
                            type=types.Type.STRING,
                            description="Optional: restrict to specific site (e.g., 'stackoverflow.com', 'github.com', 'docs.python.org')",
                        ),
                        "max_results": types.Schema(
                            type=types.Type.INTEGER,
                            description="Maximum number of results (default 5, max 10)",
                        ),
                    },
                    required=["query"],
                ),
            ),
            # Repo tools for local codebase research
            types.FunctionDeclaration(
                name="repo_search",
                description=(
                    "Search the local repository for a query. "
                    "Use this to find symbols, component names, and patterns in the codebase."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Search query (e.g., 'DataGrid', 'ColumnManager', 'TaskDAG')",
                        ),
                        "glob": types.Schema(
                            type=types.Type.STRING,
                            description="Optional glob filter (e.g., 'src/**/*.py')",
                        ),
                        "max_hits": types.Schema(
                            type=types.Type.INTEGER,
                            description="Maximum number of results (default 10)",
                        ),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="repo_open_file",
                description="Open a file from the local repository with optional line range.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "path": types.Schema(
                            type=types.Type.STRING,
                            description="Repository-relative file path",
                        ),
                        "start_line": types.Schema(
                            type=types.Type.INTEGER,
                            description="Start line (1-based)",
                        ),
                        "end_line": types.Schema(
                            type=types.Type.INTEGER,
                            description="End line (1-based)",
                        ),
                    },
                    required=["path"],
                ),
            ),
            types.FunctionDeclaration(
                name="repo_find_symbol",
                description="Find symbol definitions in the local repository.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "symbol": types.Schema(
                            type=types.Type.STRING,
                            description="Symbol name to search for",
                        ),
                    },
                    required=["symbol"],
                ),
            ),
            types.FunctionDeclaration(
                name="repo_list_tests",
                description="List test files in the local repository.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "related_path": types.Schema(
                            type=types.Type.STRING,
                            description="Optional substring to filter test files",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="repo_get_style_guide",
                description="Get style guide and lint configuration snippets from the local repo.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="repo_get_component_map",
                description="Get file paths where a component or symbol is defined.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "component_name": types.Schema(
                            type=types.Type.STRING,
                            description="Component or symbol name",
                        ),
                    },
                    required=["component_name"],
                ),
            ),
        ]
    )
]

# Keep old name for backward compatibility
CONTEXT7_TOOLS = AGENT_TOOLS


@dataclass
class AgentResult:
    """Result from an agentic generation."""

    text: str
    tool_calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    iterations: int = 0


class AgenticGeminiClient:
    """Gemini client with autonomous tool access.

    The AI decides when and how to use tools based on the task.
    Tools available:
    - Context7: Official documentation search
    - Web Search: Stack Overflow, tutorials, best practices, GitHub examples
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._client: genai.Client | None = None
        self._context7 = Context7Client(config)
        self._web_search = WebSearchClient(config)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    async def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        repo_tools: RepoTools | None = None,
    ) -> str:
        """Execute a tool call and return the result."""
        logger.info(f"Executing tool: {name} with args: {args}")

        if name == "resolve_library_id":
            library_name = args.get("library_name", "")
            result = await self._context7.resolve_library_id(library_name)
            if result:
                return f"Resolved '{library_name}' to library ID: {result}"
            return f"Could not resolve library '{library_name}'. Try a different name or check spelling."

        elif name == "get_library_docs":
            library_id = args.get("library_id", "")
            query = args.get("query", "")
            max_tokens = args.get("max_tokens", 5000)

            result = await self._context7.query_docs(library_id, query, max_tokens)
            if result.chunks:
                return result.combined_content
            return f"No documentation found for '{query}' in {library_id}. Try a different query."

        elif name == "web_search":
            query = args.get("query", "")
            site_filter = args.get("site_filter")
            max_results = min(args.get("max_results", 5), 10)

            result = await self._web_search.search(
                query=query,
                max_results=max_results,
                site_filter=site_filter,
            )

            if result.error:
                return f"Web search failed: {result.error}"

            if not result.results:
                return f"No results found for '{query}'. Try a different query."

            # Format results for the AI
            output_parts = [f"Web search results for: {query}\n"]
            for i, r in enumerate(result.results, 1):
                output_parts.append(f"\n--- Result {i} ---")
                output_parts.append(f"Title: {r.title}")
                output_parts.append(f"URL: {r.url}")
                output_parts.append(f"Content: {r.snippet[:2000]}")  # Limit per result

            return "\n".join(output_parts)

        elif name.startswith("repo_"):
            if not repo_tools:
                return "Repo tools unavailable. Provide repo context before searching."

            if name == "repo_search":
                query = args.get("query", "")
                glob = args.get("glob")
                max_hits = min(args.get("max_hits", 10), 20)
                hits = await repo_tools.search(query, glob=glob, max_hits=max_hits)
                if not hits:
                    return f"No matches for '{query}'."
                output = [f"Repo search results for '{query}':"]
                for hit in hits:
                    output.append(f"- {hit.path}:{hit.line} {hit.preview}")
                return "\n".join(output)

            if name == "repo_open_file":
                path = args.get("path", "")
                start_line = int(args.get("start_line", 1) or 1)
                end_line = args.get("end_line")
                end = int(end_line) if end_line is not None else None
                content = await repo_tools.open_file(path, start_line=start_line, end_line=end)
                return f"# {path}\n{content}"

            if name == "repo_find_symbol":
                symbol = args.get("symbol", "")
                locations = await repo_tools.find_symbol(symbol)
                if not locations:
                    return f"No symbol '{symbol}' found."
                output = [f"Symbol locations for '{symbol}':"]
                for loc in locations:
                    output.append(f"- {loc.path}:{loc.line} ({loc.kind})")
                return "\n".join(output)

            if name == "repo_list_tests":
                related_path = args.get("related_path")
                tests = repo_tools.list_tests(related_path=related_path)
                if not tests:
                    return "No tests found."
                return "Test files:\n" + "\n".join(f"- {t}" for t in tests[:50])

            if name == "repo_get_style_guide":
                return repo_tools.get_style_guide()

            if name == "repo_get_component_map":
                component_name = args.get("component_name", "")
                component_map = repo_tools.get_component_map(component_name)
                if not component_map:
                    return f"No component '{component_name}' found."
                return "\n".join(
                    f"{name}: {', '.join(paths)}" for name, paths in component_map.items()
                )

            return f"Unknown repo tool: {name}"

        return f"Unknown tool: {name}"

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_iterations: int = 10,
        repo_tools: RepoTools | None = None,
    ) -> AgentResult:
        """Generate with autonomous tool access.

        The AI can call Context7 tools as many times as needed.
        It decides what libraries to search and what queries to make.

        Args:
            prompt: The user prompt / task
            system_prompt: Optional system instruction
            temperature: Optional temperature
            max_tokens: Optional max output tokens
            max_iterations: Maximum tool call iterations (safety limit)

        Returns:
            AgentResult with final text and tool call history
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_output_tokens

        # Build system prompt that encourages tool use
        full_system = (
            "You are an expert coding assistant with access to research tools. "
            "ALWAYS use the tools to research before answering coding questions. "
            "\n\n"
            "Available tools:\n"
            "1. resolve_library_id - Find library IDs for Context7 documentation\n"
            "2. get_library_docs - Fetch official documentation from Context7\n"
            "3. web_search - Search the web for best practices, Stack Overflow, tutorials, examples\n"
            "4. repo_search - Search the local repository\n"
            "5. repo_open_file - Open local files with line ranges\n"
            "6. repo_find_symbol - Find symbol definitions in the repo\n"
            "7. repo_list_tests - List relevant tests\n"
            "8. repo_get_style_guide - Fetch style/lint configs\n"
            "9. repo_get_component_map - Locate component definitions\n"
            "\n"
            "Research strategy:\n"
            "1. Use resolve_library_id + get_library_docs for official API documentation\n"
            "2. Use web_search for best practices, common patterns, and real-world examples\n"
            "3. Use repo_search + repo_open_file to ground changes in the codebase\n"
            "4. ALWAYS search both Context7 AND the web - they complement each other\n"
            "5. Make multiple tool calls to gather comprehensive information\n"
            "6. After researching, provide a complete, production-ready answer\n"
            "\n\n"
        )
        if system_prompt:
            full_system += system_prompt

        # Initialize conversation - just use the prompt string directly
        contents = [prompt]

        generation_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=full_system,
            tools=AGENT_TOOLS,
        )

        total_input = 0
        total_output = 0
        tool_calls = []

        loop = asyncio.get_running_loop()

        for iteration in range(max_iterations):
            # Call Gemini with retry logic for server errors (500, 502, 503, 504)
            response = None
            for retry in range(5):
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.config.model,
                            contents=contents,
                            config=generation_config,
                        ),
                    )
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    is_server_error = any(code in error_str for code in ["500", "502", "503", "504", "overloaded", "unavailable", "internal"])
                    if is_server_error:
                        wait_time = (2 ** retry) * 3 + 1  # Longer backoff: 4s, 7s, 13s, 25s, 49s
                        logger.warning(f"Gemini server error ({e}), retry {retry+1}/5 in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            if response is None:
                raise Exception("Failed after 5 retries")

            # Track tokens
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_input += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                total_output += getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            # Check for function calls
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                break

            # Get all parts from the response
            has_function_calls = False
            function_responses = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_function_calls = True
                    fc = part.function_call

                    # Extract args
                    args = {}
                    if hasattr(fc, "args") and fc.args:
                        # Convert protobuf Struct to dict
                        for key, value in fc.args.items():
                            args[key] = value

                    # Execute the tool
                    tool_result = await self._execute_tool(fc.name, args, repo_tools=repo_tools)

                    tool_calls.append({
                        "name": fc.name,
                        "args": args,
                        "result_preview": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result,
                    })

                    logger.info(f"Tool {fc.name} returned {len(tool_result)} chars")

                    # Build function response
                    function_responses.append({
                        "name": fc.name,
                        "response": {"result": tool_result},
                    })

            if has_function_calls:
                # Add assistant's function call to conversation
                contents.append(candidate.content)

                # Add function responses as a new turn
                function_response_parts = []
                for fr in function_responses:
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=fr["name"],
                            response=fr["response"],
                        )
                    )
                contents.append(types.Content(role="user", parts=function_response_parts))
            else:
                # No function calls - we have the final response
                text = response.text if response.text else ""
                cost = self.config.calculate_token_cost(total_input, total_output)

                return AgentResult(
                    text=text,
                    tool_calls=tool_calls,
                    total_input_tokens=total_input,
                    total_output_tokens=total_output,
                    total_cost=cost,
                    iterations=iteration + 1,
                )

        # Max iterations reached
        return AgentResult(
            text="Max iterations reached. Please try a simpler query.",
            tool_calls=tool_calls,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=self.config.calculate_token_cost(total_input, total_output),
            iterations=max_iterations,
        )
