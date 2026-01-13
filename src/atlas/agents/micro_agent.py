"""Micro-agent implementation for patch generation.

Two agent types:
1. MicroAgent - Pre-fetches RAG context then generates (faster, less autonomous)
2. AgenticMicroAgent - AI autonomously decides what to search (more thorough)
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from atlas.agents.gemini_client import GeminiClient
from atlas.agents.prompt_styles import BASE_SYSTEM_PROMPT, PromptStyle
from atlas.core.config import Config, get_config
from atlas.core.task import Solution, TaskSubmission
from atlas.rag.context7 import Context7Client, Context7Result
from atlas.rag.web_search import WebSearchClient, WebSearchResults

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atlas.scout.repo_tools import RepoTools


@dataclass
class AgentContext:
    """Context provided to an agent for generation."""

    task: TaskSubmission
    repository_content: str  # Relevant code from the repo
    repo_tools: "RepoTools | None" = None
    rag_context: Context7Result | None = None
    web_search_context: WebSearchResults | None = None
    additional_context: str = ""


class MicroAgent:
    """A micro-agent that generates patches using a specific prompt style."""

    def __init__(
        self,
        agent_id: str | None = None,
        prompt_style: PromptStyle | None = None,
        config: Config | None = None,
    ):
        """Initialize a micro-agent.

        Args:
            agent_id: Unique identifier for this agent
            prompt_style: The prompt style to use
            config: Optional Config instance
        """
        self.agent_id = agent_id or str(uuid4())[:8]
        self.prompt_style = prompt_style
        self.config = config or get_config()

        self._gemini_client: GeminiClient | None = None
        self._context7_client: Context7Client | None = None
        self._web_search_client: WebSearchClient | None = None

    @property
    def gemini_client(self) -> GeminiClient:
        """Get or create the Gemini client."""
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(self.config)
        return self._gemini_client

    @property
    def context7_client(self) -> Context7Client:
        """Get or create the Context7 client."""
        if self._context7_client is None:
            self._context7_client = Context7Client(self.config)
        return self._context7_client

    @property
    def web_search_client(self) -> WebSearchClient:
        """Get or create the web search client."""
        if self._web_search_client is None:
            self._web_search_client = WebSearchClient(self.config)
        return self._web_search_client

    async def generate(self, context: AgentContext) -> Solution:
        """Generate a patch for the given task.

        Args:
            context: The agent context with task and code

        Returns:
            Solution with the generated patch
        """
        # Build the prompt
        prompt = self._build_prompt(context)
        system_prompt = self._build_system_prompt()

        # Calculate temperature with style offset
        temperature = self.config.temperature
        if self.prompt_style:
            temperature += self.prompt_style.temperature_offset
        temperature = max(0.0, min(2.0, temperature))  # Clamp to valid range

        try:
            # Generate the response
            result = await self.gemini_client.generate_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )

            # Extract the patch from the response
            patch = self._extract_patch(result.text)

            # Get RAG sources
            rag_sources = []
            if context.rag_context:
                rag_sources = [
                    chunk.source for chunk in context.rag_context.chunks
                ]

            return Solution(
                agent_id=self.agent_id,
                prompt_style=self.prompt_style.name.value if self.prompt_style else "default",
                patch=patch,
                explanation=self._extract_explanation(result.text),
                model=result.model,
                tokens_used=result.input_tokens + result.output_tokens,
                cost=result.cost,
                rag_sources=rag_sources,
                is_valid=bool(patch),
                validation_errors=[] if patch else ["No patch found in response"],
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate: {e}")
            return Solution(
                agent_id=self.agent_id,
                prompt_style=self.prompt_style.name.value if self.prompt_style else "default",
                patch="",
                explanation=f"Generation failed: {e}",
                is_valid=False,
                validation_errors=[str(e)],
            )

    async def gather_rag_context(
        self,
        task: TaskSubmission,
    ) -> tuple[Context7Result | None, WebSearchResults | None]:
        """Gather RAG context for the task from both Context7 and web search.

        Args:
            task: The task submission

        Returns:
            Tuple of (Context7Result, WebSearchResults) - either can be None
        """
        context7_result = None
        web_search_result = None

        # Skip RAG if prompt style disables it
        if self.prompt_style and not self.prompt_style.use_web_rag:
            logger.debug(f"Agent {self.agent_id} skipping RAG (style: {self.prompt_style.name})")
            return None, None

        # Extract libraries from the issue
        libraries = self.context7_client.extract_libraries_from_issue(task.description)

        # Gather Context7 documentation (if libraries found)
        if libraries:
            try:
                context7_result = await self.context7_client.get_documentation(
                    library_name=libraries[0],
                    query=task.description,
                    max_tokens=3000,
                )
            except Exception as e:
                logger.warning(f"Agent {self.agent_id} Context7 query failed: {e}")

        # Gather web search results
        try:
            web_search_result = await self.web_search_client.search_for_code_context(
                issue_description=task.description,
                libraries=libraries,
            )
        except Exception as e:
            logger.warning(f"Agent {self.agent_id} web search failed: {e}")

        return context7_result, web_search_result

    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        if self.prompt_style:
            return self.prompt_style.get_system_prompt(BASE_SYSTEM_PROMPT)
        return BASE_SYSTEM_PROMPT

    def _build_prompt(self, context: AgentContext) -> str:
        """Build the user prompt for generation."""
        parts = []

        # Issue description
        parts.append("## Issue Description")
        parts.append(context.task.description)
        parts.append("")

        # Repository context
        if context.repository_content:
            parts.append("## Repository Code")
            parts.append("```")
            parts.append(context.repository_content)
            parts.append("```")
            parts.append("")

        # RAG context (documentation from Context7)
        if context.rag_context and context.rag_context.chunks:
            parts.append("## Relevant Documentation (Context7)")
            parts.append(context.rag_context.combined_content)
            parts.append("")

        # Web search context
        if context.web_search_context and context.web_search_context.results:
            parts.append("## Web Search Results")
            parts.append(context.web_search_context.combined_content)
            parts.append("")

        # Additional context
        if context.additional_context:
            parts.append("## Additional Context")
            parts.append(context.additional_context)
            parts.append("")

        # Relevant files hint
        if context.task.relevant_files:
            parts.append("## Relevant Files")
            parts.append(", ".join(context.task.relevant_files))
            parts.append("")

        # Extract file names from repository content
        file_names = []
        if context.repository_content:
            import re
            file_names = re.findall(r"# File: (\S+)", context.repository_content)

        # Final instruction with explicit format requirements
        parts.append("## Output Requirements")
        parts.append("**IMPORTANT**: Generate a COMPLETE unified diff patch that fixes ALL the issues listed above.")
        parts.append("")
        parts.append("The patch format MUST be:")
        parts.append("1. Start with `--- a/filename` and `+++ b/filename` headers")
        parts.append("2. Include `@@ -start,count +start,count @@` hunk headers for EACH change")
        parts.append("3. Use `-` for removed lines, `+` for added lines, ` ` (space) for context")
        parts.append("")
        if file_names:
            parts.append(f"**File path**: Use exactly `{file_names[0]}` in the patch headers.")
            parts.append("")
            parts.append("Example format:")
            parts.append("```diff")
            parts.append(f"--- a/{file_names[0]}")
            parts.append(f"+++ b/{file_names[0]}")
            parts.append("@@ -8,6 +8,10 @@")
            parts.append(" // context before")
            parts.append("+  // new validation code")
            parts.append(" // context after")
            parts.append("@@ -20,6 +24,10 @@")
            parts.append(" // second location context")
            parts.append("+  // more new validation code")
            parts.append(" // more context")
            parts.append("```")
        parts.append("")
        parts.append("Include ALL necessary hunks to fix ALL issues. Do not stop early.")

        return "\n".join(parts)

    def _extract_patch(self, response: str) -> str:
        """Extract a unified diff patch from the response.

        Args:
            response: The full response text

        Returns:
            The extracted patch, or empty string if not found
        """
        # Method 1: Look for diff in code blocks (improved pattern)
        # Match code blocks with optional diff language specifier
        code_block_pattern = r"```(?:diff|patch)?\s*\n([\s\S]*?)```"
        code_blocks = re.findall(code_block_pattern, response)

        for block in code_blocks:
            # Check if this block looks like a diff
            if ("--- " in block or "---\t" in block) and ("@@ " in block or "@@-" in block):
                return block.strip()

        # Method 2: Extract diff lines directly from response
        lines = response.split("\n")
        diff_lines = []
        in_diff = False
        found_header = False

        for i, line in enumerate(lines):
            # Start of diff: --- line
            if line.startswith("--- ") or line.startswith("---\t"):
                in_diff = True
                found_header = True
                diff_lines = [line]  # Start fresh
            elif in_diff:
                # +++ line must follow ---
                if line.startswith("+++ ") or line.startswith("+++\t"):
                    diff_lines.append(line)
                # @@ hunk header
                elif line.startswith("@@ "):
                    diff_lines.append(line)
                # Diff content lines
                elif line.startswith("+") or line.startswith("-") or line.startswith(" "):
                    diff_lines.append(line)
                # Empty lines in context
                elif line == "":
                    # Check if next line continues the diff
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if (next_line.startswith("+") or next_line.startswith("-") or
                            next_line.startswith(" ") or next_line.startswith("@@ ") or
                            next_line.startswith("--- ")):
                            diff_lines.append(line)
                            continue
                    # End of diff - check if we have enough
                    if len(diff_lines) > 5:
                        break
                elif line.startswith("\\ No newline"):
                    # Git diff marker, include it
                    diff_lines.append(line)
                else:
                    # Non-diff line encountered
                    # If we have a valid diff, stop here
                    if len(diff_lines) > 5 and found_header:
                        break
                    # Otherwise reset and try again
                    if not found_header:
                        diff_lines = []
                        in_diff = False

        if diff_lines and len(diff_lines) > 5:
            return "\n".join(diff_lines).strip()

        return ""

    def _extract_explanation(self, response: str) -> str:
        """Extract the explanation from the response.

        Args:
            response: The full response text

        Returns:
            The explanation text
        """
        # Remove code blocks to get just the explanation
        explanation = re.sub(r"```[\s\S]*?```", "", response)
        explanation = explanation.strip()

        # Limit length
        if len(explanation) > 1000:
            explanation = explanation[:1000] + "..."

        return explanation


class AgenticMicroAgent:
    """A micro-agent with autonomous tool access.

    Unlike MicroAgent which pre-fetches RAG context, this agent
    autonomously decides what to search using Context7 and web search.
    The AI controls its own research process.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        prompt_style: PromptStyle | None = None,
        config: Config | None = None,
    ):
        """Initialize an agentic micro-agent.

        Args:
            agent_id: Unique identifier for this agent
            prompt_style: The prompt style to use
            config: Optional Config instance
        """
        self.agent_id = agent_id or str(uuid4())[:8]
        self.prompt_style = prompt_style
        self.config = config or get_config()

        # Lazy import to avoid circular dependency
        self._agentic_client = None

    @property
    def agentic_client(self):
        """Get or create the agentic Gemini client."""
        if self._agentic_client is None:
            from atlas.agents.agentic_client import AgenticGeminiClient
            self._agentic_client = AgenticGeminiClient(self.config)
        return self._agentic_client

    async def generate(self, context: AgentContext) -> Solution:
        """Generate a patch using autonomous tool access.

        The AI decides what to search in Context7 and web based on the task.

        Args:
            context: The agent context with task and code

        Returns:
            Solution with the generated patch
        """
        # Build the prompt
        prompt = self._build_prompt(context)
        system_prompt = self._build_system_prompt()

        # Calculate temperature with style offset
        temperature = self.config.temperature
        if self.prompt_style:
            temperature += self.prompt_style.temperature_offset
        temperature = max(0.0, min(2.0, temperature))

        try:
            # Generate with autonomous tool access
            result = await self.agentic_client.generate_with_tools(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=self.config.max_output_tokens,
                max_iterations=10,
                repo_tools=context.repo_tools,
            )

            # Extract the patch from the response
            patch = self._extract_patch(result.text)

            # Get RAG sources from tool calls
            rag_sources = [
                tc["name"] for tc in result.tool_calls
            ]

            return Solution(
                agent_id=self.agent_id,
                prompt_style=self.prompt_style.name.value if self.prompt_style else "default",
                patch=patch,
                explanation=self._extract_explanation(result.text),
                model=self.config.model,
                tokens_used=result.total_input_tokens + result.total_output_tokens,
                cost=result.total_cost,
                rag_sources=rag_sources,
                is_valid=bool(patch),
                validation_errors=[] if patch else ["No patch found in response"],
            )

        except Exception as e:
            logger.error(f"Agentic agent {self.agent_id} failed to generate: {e}")
            return Solution(
                agent_id=self.agent_id,
                prompt_style=self.prompt_style.name.value if self.prompt_style else "default",
                patch="",
                explanation=f"Generation failed: {e}",
                is_valid=False,
                validation_errors=[str(e)],
            )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        base = (
            "You are an expert software engineer. Your task is to fix bugs and implement features.\n\n"
            "IMPORTANT: You MUST output your fix as a unified diff patch in a code block:\n"
            "```diff\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -line,count +line,count @@\n"
            " context line\n"
            "-removed line\n"
            "+added line\n"
            "```\n\n"
            "Research thoroughly using the available tools before generating your solution.\n"
        )

        if self.prompt_style:
            return self.prompt_style.get_system_prompt(base)
        return base

    def _build_prompt(self, context: AgentContext) -> str:
        """Build the user prompt for generation."""
        parts = []

        # Issue description
        parts.append("## Issue Description")
        parts.append(context.task.description)
        parts.append("")

        # Repository context
        if context.repository_content:
            parts.append("## Repository Code")
            parts.append("```")
            # Limit to avoid token overflow
            content = context.repository_content
            if len(content) > 15000:
                content = content[:15000] + "\n... (truncated)"
            parts.append(content)
            parts.append("```")
            parts.append("")

        # Relevant files hint
        if context.task.relevant_files:
            parts.append("## Relevant Files")
            parts.append(", ".join(context.task.relevant_files))
            parts.append("")

        # Final instruction
        parts.append("## Instructions")
        parts.append(
            "1. Use the tools to research the relevant libraries and best practices\n"
            "2. Search both Context7 (official docs) AND the web (Stack Overflow, tutorials)\n"
            "3. Generate a production-ready fix as a unified diff patch\n"
            "4. Explain your changes briefly"
        )

        return "\n".join(parts)

    def _extract_patch(self, response: str) -> str:
        """Extract a unified diff patch from the response."""
        # Look for diff in code blocks
        diff_pattern = r"```(?:diff)?\s*\n((?:---|\+\+\+|@@|[-+ ].*?\n)+)```"
        matches = re.findall(diff_pattern, response, re.MULTILINE | re.DOTALL)

        if matches:
            for match in matches:
                if "---" in match or "+++" in match or "@@" in match:
                    return match.strip()

        # Try to find diff without code blocks
        lines = response.split("\n")
        diff_lines = []
        in_diff = False

        for line in lines:
            if line.startswith("---") or line.startswith("+++"):
                in_diff = True
                diff_lines.append(line)
            elif in_diff:
                if line.startswith("@@") or line.startswith("+") or line.startswith("-") or line.startswith(" "):
                    diff_lines.append(line)
                elif line.strip() == "" and diff_lines:
                    diff_lines.append(line)
                else:
                    if len(diff_lines) > 3:
                        break
                    diff_lines = []
                    in_diff = False

        if diff_lines:
            return "\n".join(diff_lines).strip()

        return ""

    def _extract_explanation(self, response: str) -> str:
        """Extract the explanation from the response."""
        explanation = re.sub(r"```[\s\S]*?```", "", response)
        explanation = explanation.strip()
        if len(explanation) > 1000:
            explanation = explanation[:1000] + "..."
        return explanation
