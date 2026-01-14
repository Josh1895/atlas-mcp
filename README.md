# ATLAS MCP Server

**ATLAS** (Aggregate Testing, Learning, and Agentic Synthesis) is an MCP server for reliable code generation using multi-agent consensus voting.

## Features

- Multi-agent code generation with diverse prompt styles
- First-to-ahead-by-K consensus voting for reliability
- Context7 integration for up-to-date documentation
- Gemini 3 Flash as the primary LLM
- Static analysis and patch validation
- Early stopping when consensus is reached

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/atlas-mcp.git
cd atlas-mcp

# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your API keys:
   - `GEMINI_API_KEY`: Get from https://ai.google.dev/
   - `CONTEXT7_API_KEY`: Get from https://context7.com/dashboard

## Usage

### As an MCP Server

Add to your Claude Code config:

```json
{
  "mcpServers": {
    "atlas": {
      "command": "python",
      "args": ["-m", "atlas.server"],
      "env": {
        "GEMINI_API_KEY": "your-key",
        "CONTEXT7_API_KEY": "your-key"
      }
    }
  }
}
```

### MCP Tools

- `solve_issue`: Generate a patch for a GitHub issue
- `check_status`: Check the status of a running task
- `get_result`: Get the final result with patch

## Architecture

```
ATLAS MCP Server
├── MCP Tools (FastMCP)
├── Orchestrator (Core Logic)
├── Micro-Agents (5 parallel, diverse styles)
├── Context7 RAG (Live documentation)
└── Verification Layer (Static analysis, clustering)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## License

MIT
