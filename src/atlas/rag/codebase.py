"""Local codebase indexing and search utilities for repository context."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".atlas",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".cache",
}

DEFAULT_TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".rb",
    ".php",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


@dataclass
class CodebaseIndexConfig:
    """Configuration for codebase indexing."""

    max_file_size_bytes: int = 250_000
    max_files: int = 15_000
    exclude_dirs: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDE_DIRS))
    text_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_TEXT_EXTENSIONS))
    max_symbols_per_file: int = 200


@dataclass
class CodebaseFile:
    """Metadata about a file in the repository."""

    path: str
    size: int
    extension: str
    is_binary: bool = False


@dataclass
class SymbolLocation:
    """Location of a symbol in the codebase."""

    name: str
    kind: str
    path: str
    line: int


@dataclass
class CodeSearchResult:
    """Result from a code search query."""

    path: str
    line: int
    preview: str


@dataclass
class CodebaseIndex:
    """Index of a codebase for retrieval."""

    root_path: Path | None = None
    files_indexed: int = 0
    is_ready: bool = False
    files: list[CodebaseFile] = field(default_factory=list)
    symbol_index: dict[str, list[SymbolLocation]] = field(default_factory=dict)


class CodebaseIndexer:
    """Indexer for local codebase context."""

    def __init__(self, config: CodebaseIndexConfig | None = None):
        self.config = config or CodebaseIndexConfig()
        self._index: CodebaseIndex | None = None

    async def index_repository(self, repo_path: Path) -> CodebaseIndex:
        """Index a repository for semantic search."""
        return await asyncio.to_thread(self._index_repository_sync, repo_path)

    def _index_repository_sync(self, repo_path: Path) -> CodebaseIndex:
        logger.info("Indexing repository: %s", repo_path)

        files: list[CodebaseFile] = []
        symbol_index: dict[str, list[SymbolLocation]] = {}

        file_count = 0
        for file_path in self._iter_files(repo_path):
            file_count += 1
            if file_count > self.config.max_files:
                logger.warning("Reached max file limit (%s)", self.config.max_files)
                break

            try:
                stat = file_path.stat()
            except OSError:
                continue

            extension = file_path.suffix.lower()
            is_binary = self._is_binary(file_path)

            files.append(
                CodebaseFile(
                    path=str(file_path.relative_to(repo_path)),
                    size=stat.st_size,
                    extension=extension,
                    is_binary=is_binary,
                )
            )

            if is_binary or stat.st_size > self.config.max_file_size_bytes:
                continue

            if extension in self.config.text_extensions:
                self._extract_symbols(file_path, repo_path, symbol_index)

        self._index = CodebaseIndex(
            root_path=repo_path,
            files_indexed=len(files),
            is_ready=True,
            files=files,
            symbol_index=symbol_index,
        )

        return self._index

    async def search(
        self,
        query: str,
        max_results: int = 5,
        glob: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search the indexed codebase."""
        if not self._index or not self._index.is_ready or not self._index.root_path:
            logger.warning("Codebase not indexed")
            return []

        return await asyncio.to_thread(
            self._search_sync,
            self._index.root_path,
            query,
            max_results,
            glob,
        )

    async def open_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """Read a file segment from the indexed repository."""
        if not self._index or not self._index.is_ready or not self._index.root_path:
            raise ValueError("Codebase not indexed")

        return await asyncio.to_thread(
            self._open_file_sync,
            self._index.root_path,
            path,
            start_line,
            end_line,
        )

    async def find_symbol(self, symbol: str) -> list[SymbolLocation]:
        """Find symbol locations by name."""
        if not self._index or not self._index.is_ready:
            return []

        return list(self._index.symbol_index.get(symbol, []))

    def clear(self) -> None:
        """Clear the index."""
        self._index = None

    def _iter_files(self, repo_path: Path) -> Iterable[Path]:
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            if self._is_excluded(file_path):
                continue

            yield file_path

    def _is_excluded(self, file_path: Path) -> bool:
        for part in file_path.parts:
            if part in self.config.exclude_dirs:
                return True
        return False

    def _is_binary(self, file_path: Path) -> bool:
        try:
            with file_path.open("rb") as handle:
                chunk = handle.read(2048)
            return b"\x00" in chunk
        except OSError:
            return True

    def _extract_symbols(
        self,
        file_path: Path,
        repo_path: Path,
        symbol_index: dict[str, list[SymbolLocation]],
    ) -> None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        extension = file_path.suffix.lower()
        rel_path = str(file_path.relative_to(repo_path))

        patterns = self._symbol_patterns_for_extension(extension)
        if not patterns:
            return

        lines = content.splitlines()
        count = 0
        for idx, line in enumerate(lines, start=1):
            for kind, pattern in patterns:
                match = pattern.search(line)
                if match:
                    name = match.group(1)
                    symbol_index.setdefault(name, []).append(
                        SymbolLocation(
                            name=name,
                            kind=kind,
                            path=rel_path,
                            line=idx,
                        )
                    )
                    count += 1
                    if count >= self.config.max_symbols_per_file:
                        return

    def _symbol_patterns_for_extension(
        self,
        extension: str,
    ) -> list[tuple[str, re.Pattern[str]]]:
        if extension == ".py":
            return [
                ("function", re.compile(r"^\s*(?:async\s+def|def)\s+(\w+)")),
                ("class", re.compile(r"^\s*class\s+(\w+)")),
            ]
        if extension in {".js", ".ts", ".jsx", ".tsx"}:
            return [
                ("function", re.compile(r"\bfunction\s+(\w+)")),
                ("class", re.compile(r"\bclass\s+(\w+)")),
                ("interface", re.compile(r"\binterface\s+(\w+)")),
                ("type", re.compile(r"\btype\s+(\w+)")),
                ("const", re.compile(r"\bconst\s+(\w+)\s*=")),
            ]
        if extension == ".go":
            return [
                ("function", re.compile(r"^\s*func\s+(\w+)")),
                ("type", re.compile(r"^\s*type\s+(\w+)")),
            ]
        if extension == ".rs":
            return [
                ("function", re.compile(r"^\s*fn\s+(\w+)")),
                ("struct", re.compile(r"^\s*struct\s+(\w+)")),
                ("enum", re.compile(r"^\s*enum\s+(\w+)")),
            ]
        if extension in {".java", ".cs"}:
            return [
                ("class", re.compile(r"\bclass\s+(\w+)")),
                ("interface", re.compile(r"\binterface\s+(\w+)")),
                ("enum", re.compile(r"\benum\s+(\w+)")),
                (
                    "method",
                    re.compile(
                        r"\b(?:public|private|protected)?\s*(?:static\s+)?\w+\s+(\w+)\s*\("
                    ),
                ),
            ]
        if extension == ".rb":
            return [
                ("class", re.compile(r"^\s*class\s+(\w+)")),
                ("method", re.compile(r"^\s*def\s+(\w+)")),
            ]
        if extension == ".php":
            return [
                ("function", re.compile(r"\bfunction\s+(\w+)")),
                ("class", re.compile(r"\bclass\s+(\w+)")),
            ]
        return []

    def _search_sync(
        self,
        repo_path: Path,
        query: str,
        max_results: int,
        glob: str | None,
    ) -> list[CodeSearchResult]:
        if not query:
            return []

        rg_path = shutil.which("rg")
        if rg_path:
            return self._search_with_ripgrep(repo_path, query, max_results, glob)

        return self._search_with_scan(repo_path, query, max_results, glob)

    def _search_with_ripgrep(
        self,
        repo_path: Path,
        query: str,
        max_results: int,
        glob: str | None,
    ) -> list[CodeSearchResult]:
        import subprocess

        command = ["rg", "-n", "--no-heading", "--color", "never", query, str(repo_path)]
        if glob:
            command.extend(["-g", glob])

        try:
            output = subprocess.check_output(command, text=True, errors="replace")
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return []
            logger.warning("ripgrep failed: %s", exc)
            return []

        results: list[CodeSearchResult] = []
        for line in output.splitlines():
            if len(results) >= max_results:
                break
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path, line_no, preview = parts[0], parts[1], parts[2]
            results.append(
                CodeSearchResult(
                    path=str(Path(path).relative_to(repo_path)),
                    line=int(line_no),
                    preview=preview,
                )
            )
        return results

    def _search_with_scan(
        self,
        repo_path: Path,
        query: str,
        max_results: int,
        glob: str | None,
    ) -> list[CodeSearchResult]:
        results: list[CodeSearchResult] = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for file_path in self._iter_files(repo_path):
            if glob and not file_path.match(glob):
                continue

            if file_path.suffix.lower() not in self.config.text_extensions:
                continue

            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for idx, line in enumerate(lines, start=1):
                if pattern.search(line):
                    results.append(
                        CodeSearchResult(
                            path=str(file_path.relative_to(repo_path)),
                            line=idx,
                            preview=line.strip(),
                        )
                    )
                    if len(results) >= max_results:
                        return results

        return results

    def _open_file_sync(
        self,
        repo_path: Path,
        path: str,
        start_line: int,
        end_line: int | None,
    ) -> str:
        target_path = repo_path / path
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = target_path.read_text(encoding="utf-8", errors="replace").splitlines()

        if start_line < 1:
            start_line = 1
        if end_line is None or end_line < start_line:
            end_line = start_line + 200

        slice_lines = content[start_line - 1 : end_line]
        return "\n".join(slice_lines)
