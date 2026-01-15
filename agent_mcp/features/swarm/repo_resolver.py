"""
Repository Resolver for swarm operations.

Supports local paths only - no remote cloning.
Local mode uses MCP_PROJECT_DIR or explicit paths.
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

from ...core.config import logger, get_project_dir
from .schemas import RepoConfig


@dataclass
class ResolvedRepo:
    """Result of resolving a repository."""
    path: Path
    branch: Optional[str] = None
    commit: Optional[str] = None

    def cleanup(self) -> None:
        """Clean up resources (no-op for local repos)."""
        pass


class RepoResolverError(Exception):
    """Raised when repository resolution fails."""
    pass


class RepoResolver:
    """
    Resolves repository paths for swarm operations.

    Supports local mode only - uses MCP_PROJECT_DIR or explicit paths.
    """

    def __init__(self, config: Optional[RepoConfig] = None):
        """
        Initialize resolver with configuration.

        Args:
            config: Repository configuration. Defaults to local mode.
        """
        self.config = config or RepoConfig()

    def resolve(self) -> ResolvedRepo:
        """
        Resolve the repository based on configuration.

        Returns:
            ResolvedRepo with path and metadata

        Raises:
            RepoResolverError: If resolution fails
        """
        return self._resolve_local()

    def _resolve_local(self) -> ResolvedRepo:
        """Resolve using local project directory."""
        project_dir = get_project_dir()

        if not project_dir.exists():
            raise RepoResolverError(f"Project directory not found: {project_dir}")

        # Check if it's a git repo (optional, not required)
        git_dir = project_dir / ".git"
        if not git_dir.exists():
            logger.debug(f"Project directory is not a git repo: {project_dir}")

        # Get current branch and commit if available
        branch = self._get_current_branch(project_dir)
        commit = self._get_current_commit(project_dir)

        logger.debug(f"Resolved local repo: {project_dir} (branch={branch}, commit={commit[:8] if commit else 'N/A'})")

        return ResolvedRepo(
            path=project_dir,
            branch=branch,
            commit=commit,
        )

    def _get_current_branch(self, repo_path: Path) -> Optional[str]:
        """Get the current branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                return branch if branch != "HEAD" else None
        except Exception:
            pass
        return None

    def _get_current_commit(self, repo_path: Path) -> Optional[str]:
        """Get the current commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def get_test_command(self, resolved: ResolvedRepo) -> Optional[str]:
        """
        Get the test command for the repository.

        Checks config first, then tries to discover from repo.

        Returns:
            Test command string or None
        """
        # Use configured test command if provided
        if self.config.test_command:
            return self.config.test_command

        # Try to discover test command
        return self._discover_test_command(resolved.path)

    def _discover_test_command(self, repo_path: Path) -> Optional[str]:
        """
        Try to discover the test command from repo configuration.

        Checks common patterns:
        - package.json scripts.test
        - pyproject.toml [tool.pytest]
        - Makefile test target
        """
        # Check package.json
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    data = json.load(f)
                    if "scripts" in data and "test" in data["scripts"]:
                        return "npm test"
            except Exception:
                pass

        # Check pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                if "[tool.pytest" in content or "pytest" in content.lower():
                    return "pytest"
            except Exception:
                pass

        # Check for pytest.ini
        if (repo_path / "pytest.ini").exists():
            return "pytest"

        # Check for setup.py with test
        if (repo_path / "setup.py").exists():
            return "python setup.py test"

        # Check Makefile
        makefile = repo_path / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text()
                if "test:" in content:
                    return "make test"
            except Exception:
                pass

        logger.debug(f"Could not discover test command for {repo_path}")
        return None


@contextmanager
def temporary_repo(config: RepoConfig):
    """
    Context manager for working with a resolved repository.

    For local repos, this is a simple passthrough with no cleanup needed.

    Usage:
        with temporary_repo(config) as resolved:
            # Work with resolved.path
            pass
    """
    resolver = RepoResolver(config)
    resolved = resolver.resolve()
    try:
        yield resolved
    finally:
        resolved.cleanup()
