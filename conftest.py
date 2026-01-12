"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
