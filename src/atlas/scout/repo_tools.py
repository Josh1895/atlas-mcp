"""Repo-aware tools for agentic codebase research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas.rag.codebase import CodebaseIndexer, SymbolLocation
from atlas.scout.repo_scout import RepoScoutReport, RepoSearchHit


@dataclass
class RepoTools:
    """Tooling surface for repo search and file access."""

    indexer: CodebaseIndexer
    report: RepoScoutReport

    async def search(
        self,
        query: str,
        glob: str | None = None,
        max_hits: int = 10,
    ) -> list[RepoSearchHit]:
        results = await self.indexer.search(query, max_results=max_hits, glob=glob)
        return [RepoSearchHit(path=r.path, line=r.line, preview=r.preview) for r in results]

    async def open_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        return await self.indexer.open_file(path, start_line=start_line, end_line=end_line)

    async def find_symbol(self, symbol: str) -> list[SymbolLocation]:
        return await self.indexer.find_symbol(symbol)

    def list_tests(self, related_path: str | None = None) -> list[str]:
        if not related_path:
            return list(self.report.test_files)
        return [path for path in self.report.test_files if related_path in path]

    def get_style_guide(self) -> str:
        if not self.report.style_guides:
            return "No style guide files detected."

        parts = []
        for path, content in self.report.style_guides.items():
            parts.append(f"# {path}\n{content}")
        return "\n\n".join(parts)

    def get_component_map(self, component_name: str) -> dict[str, list[str]]:
        if component_name in self.report.component_map:
            return {component_name: self.report.component_map[component_name]}

        locations = self.report.codebase_index.symbol_index.get(component_name, [])
        if not locations:
            return {}

        paths = sorted({loc.path for loc in locations})
        return {component_name: paths}

    def repo_root(self) -> Path:
        return self.report.root_path
