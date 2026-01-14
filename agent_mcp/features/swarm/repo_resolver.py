"""
Repository Resolver for swarm operations.

Supports both local (MCP_PROJECT_DIR) and remote (git clone) modes (AD-005).
Local mode is default for Agent-MCP compatibility.
"""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from contextlib import contextmanager

from ...core.config import logger, get_project_dir
from .schemas import RepoConfig


@dataclass
class ResolvedRepo:
    """Result of resolving a repository."""
    path: Path
    branch: Optional[str] = None
    commit: Optional[str] = None
    is_temp: bool = False
    worktree_path: Optional[Path] = None

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self.worktree_path and self.worktree_path.exists():
            try:
                # Remove git worktree
                subprocess.run(
                    ["git", "worktree", "remove", str(self.worktree_path), "--force"],
                    cwd=str(self.path),
                    capture_output=True,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"Failed to remove worktree: {e}")

            # Force remove directory if still exists
            if self.worktree_path.exists():
                shutil.rmtree(self.worktree_path, ignore_errors=True)

        if self.is_temp and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            logger.debug(f"Cleaned up temp repo at {self.path}")


class RepoResolverError(Exception):
    """Raised when repository resolution fails."""
    pass


class RepoResolver:
    """
    Resolves repository paths for swarm operations.

    Supports:
    - Local mode: Uses MCP_PROJECT_DIR
    - Remote mode: Clones from URL to temp directory
    - Worktree support for isolated patch application
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
        if self.config.mode == "local":
            return self._resolve_local()
        elif self.config.mode == "remote":
            return self._resolve_remote()
        else:
            raise RepoResolverError(f"Unknown repo mode: {self.config.mode}")

    def _resolve_local(self) -> ResolvedRepo:
        """Resolve using local project directory."""
        project_dir = get_project_dir()

        if not project_dir.exists():
            raise RepoResolverError(f"Project directory not found: {project_dir}")

        # Check if it's a git repo
        git_dir = project_dir / ".git"
        if not git_dir.exists():
            logger.warning(f"Project directory is not a git repo: {project_dir}")

        # Get current branch and commit
        branch = self._get_current_branch(project_dir)
        commit = self._get_current_commit(project_dir)

        logger.debug(f"Resolved local repo: {project_dir} (branch={branch}, commit={commit[:8] if commit else 'N/A'})")

        return ResolvedRepo(
            path=project_dir,
            branch=branch,
            commit=commit,
            is_temp=False,
        )

    def _resolve_remote(self) -> ResolvedRepo:
        """Clone remote repository to temp directory."""
        if not self.config.url:
            raise RepoResolverError("Remote mode requires a URL")

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="swarm_repo_"))
        logger.debug(f"Cloning {self.config.url} to {temp_dir}")

        try:
            # Clone the repo
            clone_args = ["git", "clone"]

            if self.config.branch:
                clone_args.extend(["--branch", self.config.branch])

            clone_args.extend(["--depth", "1", self.config.url, str(temp_dir)])

            result = subprocess.run(
                clone_args,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for clone
            )

            if result.returncode != 0:
                raise RepoResolverError(f"Git clone failed: {result.stderr}")

            # Checkout specific commit if provided
            if self.config.commit:
                # Need to fetch the specific commit first (since we did shallow clone)
                subprocess.run(
                    ["git", "fetch", "--depth", "1", "origin", self.config.commit],
                    cwd=str(temp_dir),
                    capture_output=True,
                    timeout=60,
                )

                result = subprocess.run(
                    ["git", "checkout", self.config.commit],
                    cwd=str(temp_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    raise RepoResolverError(f"Git checkout failed: {result.stderr}")

            branch = self._get_current_branch(temp_dir)
            commit = self._get_current_commit(temp_dir)

            logger.info(f"Cloned remote repo to {temp_dir} (branch={branch}, commit={commit[:8] if commit else 'N/A'})")

            return ResolvedRepo(
                path=temp_dir,
                branch=branch,
                commit=commit,
                is_temp=True,
            )

        except subprocess.TimeoutExpired:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RepoResolverError("Git clone timed out")
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RepoResolverError(f"Failed to clone repo: {e}")

    def create_worktree(self, resolved: ResolvedRepo, name: str) -> Path:
        """
        Create an isolated git worktree for patch application.

        Args:
            resolved: The resolved repository
            name: Name for the worktree (used as suffix)

        Returns:
            Path to the worktree directory
        """
        worktree_dir = resolved.path.parent / f"{resolved.path.name}_worktree_{name}"

        try:
            # Create worktree
            result = subprocess.run(
                ["git", "worktree", "add", str(worktree_dir), "HEAD", "--detach"],
                cwd=str(resolved.path),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RepoResolverError(f"Failed to create worktree: {result.stderr}")

            resolved.worktree_path = worktree_dir
            logger.debug(f"Created worktree at {worktree_dir}")

            return worktree_dir

        except subprocess.TimeoutExpired:
            raise RepoResolverError("Git worktree creation timed out")

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

    Automatically cleans up temp resources on exit.

    Usage:
        with temporary_repo(config) as resolved:
            # Work with resolved.path
            pass
        # Cleanup happens automatically
    """
    resolver = RepoResolver(config)
    resolved = resolver.resolve()
    try:
        yield resolved
    finally:
        resolved.cleanup()
