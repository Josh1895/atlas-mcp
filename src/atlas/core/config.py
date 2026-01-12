"""Configuration management for ATLAS."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """ATLAS configuration loaded from environment variables."""

    # API Keys
    gemini_api_key: str = ""
    context7_api_key: str = ""
    serpapi_api_key: str = ""  # Optional - enables better web search

    # Model configuration
    model: str = "gemini-3-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192

    # Voting parameters
    voting_k: int = 3
    max_samples: int = 30
    initial_samples: int = 10  # 2 of each of 5 prompt styles

    # Cost and timeout limits
    max_cost_usd: float = 2.0
    timeout_minutes: int = 15

    # Paths
    work_dir: Path = field(default_factory=lambda: Path.cwd() / ".atlas")

    # Gemini pricing (per 1M tokens) - Gemini 3 Flash
    gemini_input_cost_per_m: float = 0.50
    gemini_output_cost_per_m: float = 3.00

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to .env file. If not provided,
                         will look for .env in current directory.

        Returns:
            Config instance with values from environment.
        """
        # Load .env file if it exists
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        return cls(
            # API Keys
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            context7_api_key=os.getenv("CONTEXT7_API_KEY", ""),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY", ""),
            # Model configuration
            model=os.getenv("ATLAS_MODEL", "gemini-3-flash"),
            temperature=float(os.getenv("ATLAS_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.getenv("ATLAS_MAX_OUTPUT_TOKENS", "8192")),
            # Voting parameters
            voting_k=int(os.getenv("ATLAS_VOTING_K", "3")),
            max_samples=int(os.getenv("ATLAS_MAX_SAMPLES", "30")),
            initial_samples=int(os.getenv("ATLAS_INITIAL_SAMPLES", "10")),
            # Cost and timeout limits
            max_cost_usd=float(os.getenv("ATLAS_MAX_COST_USD", "2.0")),
            timeout_minutes=int(os.getenv("ATLAS_TIMEOUT_MINUTES", "15")),
            # Paths
            work_dir=Path(os.getenv("ATLAS_WORK_DIR", ".atlas")),
        )

    def validate(self) -> list[str]:
        """Validate the configuration.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required")

        if not self.context7_api_key:
            errors.append("CONTEXT7_API_KEY is required")

        if self.voting_k < 1:
            errors.append("ATLAS_VOTING_K must be at least 1")

        if self.max_samples < self.voting_k:
            errors.append("ATLAS_MAX_SAMPLES must be >= ATLAS_VOTING_K")

        if self.max_cost_usd <= 0:
            errors.append("ATLAS_MAX_COST_USD must be positive")

        if self.timeout_minutes <= 0:
            errors.append("ATLAS_TIMEOUT_MINUTES must be positive")

        return errors

    def is_valid(self) -> bool:
        """Check if the configuration is valid."""
        return len(self.validate()) == 0

    def ensure_work_dir(self) -> Path:
        """Ensure the work directory exists and return its path."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir

    def calculate_token_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.gemini_input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * self.gemini_output_cost_per_m
        return input_cost + output_cost


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.

    Creates a new instance from environment if not already loaded.
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    Useful for testing or programmatic configuration.
    """
    global _config
    _config = config
