"""Repository scouting utilities for deterministic context packs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from atlas.rag.codebase import CodebaseIndexer, CodebaseIndex, CodeSearchResult
from atlas.verification.test_runner import TestFrameworkDetector

logger = logging.getLogger(__name__)


DEFAULT_STYLE_FILES = [
    ".editorconfig",
    "pyproject.toml",
    "ruff.toml",
    "setup.cfg",
    "setup.py",
    "package.json",
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.yml",
    ".prettierrc.yaml",
    ".eslintrc",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
    ".eslintignore",
    ".stylelintrc",
    ".stylelintrc.json",
    ".stylelintrc.yml",
    ".stylelintrc.yaml",
]

DEFAULT_ENTRYPOINTS = [
    "main.py",
    "app.py",
    "server.py",
    "index.js",
    "index.ts",
    "src/index.js",
    "src/index.ts",
    "src/main.py",
    "src/app.py",
    "src/server.py",
]


@dataclass
class RepoSearchHit:
    """A simplified search hit for RepoScout output."""

    path: str
    line: int
    preview: str


@dataclass
class RepoScoutReport:
    """Structured output from RepoScout."""

    root_path: Path
    codebase_index: CodebaseIndex
    languages: dict[str, int] = field(default_factory=dict)
    entrypoints: list[str] = field(default_factory=list)
    style_guides: dict[str, str] = field(default_factory=dict)
    test_frameworks: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    keyword_hits: dict[str, list[RepoSearchHit]] = field(default_factory=dict)
    component_map: dict[str, list[str]] = field(default_factory=dict)

    def to_context_pack(self, max_chars: int = 8000) -> str:
        """Create a compact context pack for LLM prompts."""
        parts: list[str] = []

        parts.append("## RepoScout Summary")
        parts.append(f"Root: {self.root_path}")
        parts.append(f"Files indexed: {self.codebase_index.files_indexed}")

        if self.languages:
            parts.append("Languages:")
            for ext, count in sorted(self.languages.items(), key=lambda x: x[1], reverse=True)[:10]:
                parts.append(f"- {ext or '<no_ext>'}: {count}")

        if self.entrypoints:
            parts.append("Entrypoints:")
            parts.append(", ".join(self.entrypoints[:10]))

        if self.test_frameworks:
            parts.append("Test frameworks: " + ", ".join(self.test_frameworks))

        if self.test_files:
            parts.append("Test files (sample):")
            parts.append(", ".join(self.test_files[:10]))

        if self.style_guides:
            parts.append("Style guide files:")
            parts.append(", ".join(sorted(self.style_guides.keys())))

        if self.component_map:
            parts.append("Component map (symbols found):")
            for name, paths in self.component_map.items():
                parts.append(f"- {name}: {', '.join(paths[:5])}")

        if self.keyword_hits:
            parts.append("Keyword hits:")
            for keyword, hits in self.keyword_hits.items():
                parts.append(f"- {keyword}: {len(hits)} hits")

        summary = "\n".join(parts)
        if len(summary) > max_chars:
            return summary[: max_chars - 3] + "..."
        return summary


class RepoScout:
    """Deterministic repository scanner for decomposition context."""

    def __init__(self, indexer: CodebaseIndexer | None = None):
        self.indexer = indexer or CodebaseIndexer()
        self._detector = TestFrameworkDetector()

    async def scan(
        self,
        repo_path: Path,
        keywords: list[str] | None = None,
        component_names: list[str] | None = None,
    ) -> RepoScoutReport:
        """Scan the repository and return a structured report."""
        index = await self.indexer.index_repository(repo_path)
        languages = self._count_languages(index)
        entrypoints = self._detect_entrypoints(repo_path)
        style_guides = self._collect_style_guides(repo_path)
        test_frameworks, test_files = self._detect_tests(repo_path, index)
        keyword_hits = await self._search_keywords(keywords or [])
        component_map = self._map_components(component_names or [], index)

        return RepoScoutReport(
            root_path=repo_path,
            codebase_index=index,
            languages=languages,
            entrypoints=entrypoints,
            style_guides=style_guides,
            test_frameworks=test_frameworks,
            test_files=test_files,
            keyword_hits=keyword_hits,
            component_map=component_map,
        )

    def _count_languages(self, index: CodebaseIndex) -> dict[str, int]:
        counts: dict[str, int] = {}
        for file_info in index.files:
            counts[file_info.extension] = counts.get(file_info.extension, 0) + 1
        return counts

    def _detect_entrypoints(self, repo_path: Path) -> list[str]:
        entrypoints = []
        for rel_path in DEFAULT_ENTRYPOINTS:
            candidate = repo_path / rel_path
            if candidate.exists():
                entrypoints.append(rel_path)
        return entrypoints

    def _collect_style_guides(self, repo_path: Path) -> dict[str, str]:
        guides: dict[str, str] = {}
        for rel_path in DEFAULT_STYLE_FILES:
            candidate = repo_path / rel_path
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                guides[rel_path] = content[:4000]
        return guides

    def _detect_tests(self, repo_path: Path, index: CodebaseIndex) -> tuple[list[str], list[str]]:
        frameworks = [cfg.framework.value for cfg in self._detector.detect(repo_path)]
        test_files = []
        for file_info in index.files:
            path = file_info.path
            if "/tests/" in path.replace("\\", "/") or "/test/" in path.replace("\\", "/"):
                test_files.append(path)
            elif path.startswith("tests/") or path.startswith("test/"):
                test_files.append(path)
            elif path.endswith("_test.py") or path.endswith("test.py"):
                test_files.append(path)
            elif path.endswith(".spec.ts") or path.endswith(".spec.tsx") or path.endswith(".spec.js"):
                test_files.append(path)
        return frameworks, sorted(set(test_files))

    async def _search_keywords(
        self,
        keywords: Iterable[str],
    ) -> dict[str, list[RepoSearchHit]]:
        results: dict[str, list[RepoSearchHit]] = {}

        for keyword in keywords:
            hits = await self.indexer.search(keyword, max_results=15)
            results[keyword] = [
                RepoSearchHit(path=hit.path, line=hit.line, preview=hit.preview)
                for hit in hits
            ]

        return results

    def _map_components(
        self,
        component_names: Iterable[str],
        index: CodebaseIndex,
    ) -> dict[str, list[str]]:
        component_map: dict[str, list[str]] = {}
        for name in component_names:
            locations = index.symbol_index.get(name, [])
            paths = sorted({loc.path for loc in locations})
            if paths:
                component_map[name] = paths
        return component_map
