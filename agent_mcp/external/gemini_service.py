# Agent-MCP Gemini Service
# Replaces OpenAI for embeddings and chat, using the same GEMINI_API_KEY

import os
from typing import Optional, List

try:
    import google.generativeai as genai
except ImportError:
    from ..core.config import logger as temp_logger
    temp_logger.error(
        "Google Generative AI library not found. "
        "Please install it using 'pip install google-generativeai'. "
        "Gemini-dependent features will be unavailable."
    )
    genai = None

from ..core.config import logger
from ..core import globals as g

# Gemini API key from environment
GEMINI_API_KEY_ENV: Optional[str] = os.environ.get("GEMINI_API_KEY")

# Embedding model configuration
# text-embedding-004 is Google's latest embedding model
GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
GEMINI_EMBEDDING_DIMENSION: int = 768  # text-embedding-004 outputs 768 dimensions

# Chat model for task analysis (uses same key as swarm agents)
GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"


def initialize_gemini_client() -> bool:
    """
    Initializes the Gemini API client.
    Returns True if successful, False otherwise.
    """
    if getattr(g, 'gemini_initialized', False):
        logger.info("Gemini client already initialized.")
        return True

    if genai is None:
        logger.error("Google Generative AI library failed to import. Cannot initialize client.")
        return False

    if not GEMINI_API_KEY_ENV:
        logger.error("GEMINI_API_KEY not found in environment variables. Cannot initialize Gemini client.")
        return False

    logger.info("Initializing Gemini client...")
    try:
        # Configure the API key globally
        genai.configure(api_key=GEMINI_API_KEY_ENV)

        # Test the connection by listing models
        models = list(genai.list_models())
        if not models:
            raise Exception("No models available")

        logger.info(f"Gemini client initialized successfully. {len(models)} models available.")
        g.gemini_initialized = True
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
        g.gemini_initialized = False
        return False


def get_gemini_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for a single text using Gemini's embedding model.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding, or None if failed
    """
    if genai is None or not getattr(g, 'gemini_initialized', False):
        if not initialize_gemini_client():
            return None

    try:
        result = genai.embed_content(
            model=f"models/{GEMINI_EMBEDDING_MODEL}",
            content=text,
            task_type="retrieval_document",
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to get Gemini embedding: {e}")
        return None


def get_gemini_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
    """
    Get embeddings for multiple texts using Gemini's embedding model.

    Args:
        texts: List of texts to embed
        batch_size: Maximum texts per API call

    Returns:
        List of embeddings (or None for failed items)
    """
    if genai is None or not getattr(g, 'gemini_initialized', False):
        if not initialize_gemini_client():
            return [None] * len(texts)

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Gemini supports batch embedding
            batch_result = genai.embed_content(
                model=f"models/{GEMINI_EMBEDDING_MODEL}",
                content=batch,
                task_type="retrieval_document",
            )
            # batch_result['embedding'] is a list of embeddings when content is a list
            if isinstance(batch_result.get('embedding'), list):
                if batch_result['embedding'] and isinstance(batch_result['embedding'][0], list):
                    # Multiple embeddings returned
                    results.extend(batch_result['embedding'])
                else:
                    # Single embedding returned (shouldn't happen with batch)
                    results.append(batch_result['embedding'])
            else:
                results.extend([None] * len(batch))
        except Exception as e:
            logger.error(f"Failed to get Gemini batch embeddings: {e}")
            results.extend([None] * len(batch))

    return results


def get_gemini_chat_response(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> Optional[str]:
    """
    Get a chat response from Gemini.

    Args:
        prompt: The user prompt
        system_instruction: Optional system instruction
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        The model's response text, or None if failed
    """
    if genai is None or not getattr(g, 'gemini_initialized', False):
        if not initialize_gemini_client():
            return None

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        logger.error(f"Failed to get Gemini chat response: {e}")
        return None


# Compatibility layer - these mirror OpenAI service functions
def get_embedding_model() -> str:
    """Returns the embedding model name."""
    return GEMINI_EMBEDDING_MODEL


def get_embedding_dimension() -> int:
    """Returns the embedding dimension."""
    return GEMINI_EMBEDDING_DIMENSION


def get_chat_model() -> str:
    """Returns the chat model name."""
    return GEMINI_CHAT_MODEL


# For backward compatibility with code expecting OpenAI-style client
class GeminiEmbeddingClient:
    """
    Compatibility wrapper that mimics OpenAI embedding interface.
    """

    def __init__(self):
        self.initialized = initialize_gemini_client()

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for single text."""
        return get_gemini_embedding(text)

    def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts."""
        return get_gemini_embeddings_batch(texts)


# Global client instance for compatibility
_embedding_client: Optional[GeminiEmbeddingClient] = None


def get_embedding_client() -> Optional[GeminiEmbeddingClient]:
    """Get or create the embedding client."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = GeminiEmbeddingClient()
    return _embedding_client if _embedding_client.initialized else None
