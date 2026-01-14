"""Gemini API client wrapper for ATLAS."""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from google import genai
from google.genai import types

from atlas.core.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from a Gemini generation request."""

    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str


class GeminiClient:
    """Async wrapper for Gemini API with cost tracking."""

    def __init__(self, config: Config | None = None):
        """Initialize the Gemini client.

        Args:
            config: Optional Config instance. Uses global config if not provided.
        """
        self.config = config or get_config()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Get or create the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text using Gemini.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            GenerationResult with text and usage info
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_output_tokens

        # Build the request
        contents = [prompt]

        generation_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system_prompt,
        )

        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            ),
        )

        # Extract usage metadata
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        # Calculate cost
        cost = self.config.calculate_token_cost(input_tokens, output_tokens)

        # Get text from response
        text = ""
        if response.text:
            text = response.text

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model=self.config.model,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Generate text using Gemini with streaming.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            Text chunks as they are generated
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_output_tokens

        contents = [prompt]

        generation_config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system_prompt,
        )

        # Run in executor to get the stream
        loop = asyncio.get_running_loop()
        response_stream = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=generation_config,
            ),
        )

        # Yield chunks
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> GenerationResult:
        """Generate text with automatic retries on failure.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)

        Returns:
            GenerationResult with text and usage info

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit or server overload errors
                is_rate_limit = any(code in error_str for code in ["429", "rate", "quota", "resource_exhausted"])
                is_server_error = any(code in error_str for code in ["503", "500", "502", "504", "unavailable", "overloaded"])

                if is_rate_limit:
                    # Longer backoff for rate limits (start at 5x base delay)
                    delay = retry_delay * 5 * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s: {e}"
                    )
                elif is_server_error:
                    # Server errors: moderate backoff (start at 3x base delay)
                    delay = retry_delay * 3 * (2 ** attempt)
                    logger.warning(
                        f"Server error (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s: {e}"
                    )
                else:
                    # Standard exponential backoff for other errors
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Generation attempt {attempt + 1}/{max_retries} failed: {e}"
                    )

                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

        raise last_error or Exception("Generation failed after all retries")

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        This is a rough estimate. Actual tokenization may differ.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 1000) -> float:
        """Estimate the cost of a generation request.

        Args:
            prompt: The prompt text
            expected_output_tokens: Expected number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_tokens = self.estimate_tokens(prompt)
        return self.config.calculate_token_cost(input_tokens, expected_output_tokens)
