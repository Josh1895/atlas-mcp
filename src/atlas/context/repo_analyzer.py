"""Repository analyzer for intelligent context extraction.

This module replaces the naive "first 10 files" approach with intelligent
context selection based on:
1. Issue description keywords
2. File importance (entry points, configs, etc.)
3. Symbol/import analysis
4. Test file association
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FileRelevance:
    """Relevance score and metadata for a file."""

    path: str
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    # Metadata
    size_bytes: int = 0
    line_count: int = 0
    is_test: bool = False
    is_config: bool = False
    is_entry_point: bool = False

    # Content preview
    symbols: List[str] = field(default_factory=list)  # Functions/classes defined


@dataclass
class RepoContext:
    """Extracted context from a repository."""

    # Selected files with content
    files: Dict[str, str] = field(default_factory=dict)  # {path: content}
    file_relevance: Dict[str, FileRelevance] = field(default_factory=dict)

    # Repository metadata
    primary_language: str = ""
    test_command: str = ""
    entry_points: List[str] = field(default_factory=list)

    # Symbols found
    relevant_symbols: List[str] = field(default_factory=list)
    import_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Statistics
    total_files_scanned: int = 0
    files_selected: int = 0


@dataclass
class AnalysisConfig:
    """Configuration for repository analysis."""

    max_files: int = 15
    max_file_size: int = 50000  # bytes
    max_total_size: int = 200000  # bytes for all files combined
    include_tests: bool = True
    test_file_limit: int = 3


class RepoAnalyzer:
    """Analyzes repositories to extract relevant context."""

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": ["*.py", "pyproject.toml", "setup.py", "requirements.txt"],
        "javascript": ["*.js", "*.jsx", "package.json"],
        "typescript": ["*.ts", "*.tsx", "tsconfig.json"],
        "rust": ["*.rs", "Cargo.toml"],
        "go": ["*.go", "go.mod"],
        "java": ["*.java", "pom.xml", "build.gradle"],
        "ruby": ["*.rb", "Gemfile"],
        "csharp": ["*.cs", "*.csproj"],
    }

    # Important file patterns
    ENTRY_POINT_PATTERNS = [
        "main.py", "__main__.py", "app.py", "server.py", "cli.py",
        "index.js", "index.ts", "main.js", "main.ts", "app.js", "app.ts",
        "main.rs", "lib.rs",
        "main.go", "cmd/**/main.go",
        "Main.java", "Application.java",
    ]

    CONFIG_PATTERNS = [
        "*.toml", "*.yaml", "*.yml", "*.json", "*.ini", "*.cfg",
        "Makefile", "Dockerfile", ".env.example",
    ]

    TEST_PATTERNS = [
        "test_*.py", "*_test.py", "tests/**/*.py",
        "*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts",
        "*_test.go", "*_test.rs",
        "Test*.java", "*Test.java",
        "*_spec.rb", "*_test.rb",
    ]

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        repo_path: Path,
        issue_description: str,
        relevant_files: Optional[List[str]] = None,
    ) -> RepoContext:
        """Analyze a repository and extract relevant context.

        Args:
            repo_path: Path to the repository
            issue_description: Description of the issue/task
            relevant_files: Optional user-specified relevant files

        Returns:
            RepoContext with extracted information
        """
        context = RepoContext()

        # Detect primary language
        context.primary_language = self._detect_language(repo_path)
        logger.info(f"Detected primary language: {context.primary_language}")

        # Extract keywords from issue
        keywords = self._extract_keywords(issue_description)
        logger.info(f"Extracted keywords: {keywords[:10]}")

        # Score all files
        all_files = self._scan_files(repo_path)
        context.total_files_scanned = len(all_files)

        scored_files = []
        for file_path in all_files:
            rel_path = str(file_path.relative_to(repo_path))
            relevance = self._score_file(
                file_path,
                rel_path,
                keywords,
                relevant_files or [],
                context.primary_language,
            )
            scored_files.append((file_path, relevance))

        # Sort by score
        scored_files.sort(key=lambda x: x[1].score, reverse=True)

        # Select top files within size budget
        total_size = 0
        selected_count = 0
        test_count = 0

        for file_path, relevance in scored_files:
            # Check limits
            if selected_count >= self.config.max_files:
                break

            if relevance.is_test:
                if test_count >= self.config.test_file_limit:
                    continue
                test_count += 1

            if relevance.size_bytes > self.config.max_file_size:
                continue

            if total_size + relevance.size_bytes > self.config.max_total_size:
                continue

            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(repo_path))

                # Truncate if still too long
                if len(content) > self.config.max_file_size:
                    content = content[:self.config.max_file_size] + "\n# ... (truncated)"

                context.files[rel_path] = content
                context.file_relevance[rel_path] = relevance
                total_size += relevance.size_bytes
                selected_count += 1

                # Track entry points
                if relevance.is_entry_point:
                    context.entry_points.append(rel_path)

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        context.files_selected = selected_count
        logger.info(f"Selected {selected_count} files, {total_size} bytes total")

        return context

    def _detect_language(self, repo_path: Path) -> str:
        """Detect the primary language of the repository."""
        file_counts = defaultdict(int)

        for pattern, lang in [
            ("*.py", "python"),
            ("*.js", "javascript"),
            ("*.ts", "typescript"),
            ("*.rs", "rust"),
            ("*.go", "go"),
            ("*.java", "java"),
            ("*.rb", "ruby"),
            ("*.cs", "csharp"),
        ]:
            count = len(list(repo_path.rglob(pattern)))
            file_counts[lang] = count

        if not file_counts:
            return "unknown"

        return max(file_counts, key=file_counts.get)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from issue description."""
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "and", "but", "if", "or", "because", "until", "while",
            "this", "that", "these", "those", "what", "which", "who", "whom",
            "fix", "bug", "issue", "error", "problem", "please", "add", "update",
            "change", "modify", "implement", "feature", "request", "i", "we", "you",
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        # Filter and score
        keyword_counts = defaultdict(int)
        for word in words:
            if word not in stopwords and len(word) > 2:
                keyword_counts[word] += 1

        # Sort by frequency
        sorted_keywords = sorted(
            keyword_counts.keys(),
            key=lambda w: keyword_counts[w],
            reverse=True,
        )

        return sorted_keywords

    def _scan_files(self, repo_path: Path) -> List[Path]:
        """Scan repository for relevant files."""
        files = []

        # Directories to skip
        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".tox", ".pytest_cache", ".mypy_cache",
            "target", "vendor", ".idea", ".vscode",
        }

        for file_path in repo_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip hidden files
            if file_path.name.startswith("."):
                continue

            # Skip files in ignored directories
            if any(part in skip_dirs for part in file_path.parts):
                continue

            # Skip binary/non-text files
            if file_path.suffix.lower() in {
                ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
                ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg",
                ".pdf", ".zip", ".tar", ".gz", ".7z",
                ".woff", ".woff2", ".ttf", ".eot",
                ".mp3", ".mp4", ".avi", ".mov",
                ".db", ".sqlite", ".sqlite3",
            }:
                continue

            files.append(file_path)

        return files

    def _score_file(
        self,
        file_path: Path,
        rel_path: str,
        keywords: List[str],
        user_files: List[str],
        language: str,
    ) -> FileRelevance:
        """Score a file's relevance to the issue."""
        relevance = FileRelevance(path=rel_path)

        try:
            relevance.size_bytes = file_path.stat().st_size
        except OSError:
            return relevance

        # Check if user explicitly mentioned this file
        if rel_path in user_files or any(rel_path.endswith(f) for f in user_files):
            relevance.score += 100
            relevance.reasons.append("User specified")

        # Check if it's a test file
        for pattern in self.TEST_PATTERNS:
            if file_path.match(pattern):
                relevance.is_test = True
                relevance.score += 5
                relevance.reasons.append("Test file")
                break

        # Check if it's an entry point
        for pattern in self.ENTRY_POINT_PATTERNS:
            if file_path.match(pattern):
                relevance.is_entry_point = True
                relevance.score += 20
                relevance.reasons.append("Entry point")
                break

        # Check if it's a config file
        for pattern in self.CONFIG_PATTERNS:
            if file_path.match(pattern):
                relevance.is_config = True
                relevance.score += 10
                relevance.reasons.append("Config file")
                break

        # Check filename against keywords
        filename_lower = file_path.name.lower()
        for keyword in keywords[:20]:  # Check top 20 keywords
            if keyword in filename_lower:
                relevance.score += 15
                relevance.reasons.append(f"Filename matches '{keyword}'")

        # Read file content for deeper analysis
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            relevance.line_count = content.count("\n") + 1

            # Check content against keywords
            content_lower = content.lower()
            keyword_hits = 0
            for keyword in keywords[:20]:
                if keyword in content_lower:
                    keyword_hits += 1

            if keyword_hits > 0:
                relevance.score += keyword_hits * 3
                relevance.reasons.append(f"Content matches {keyword_hits} keywords")

            # Extract symbols (functions, classes)
            relevance.symbols = self._extract_symbols(content, language)
            if relevance.symbols:
                # Check if any symbols match keywords
                for symbol in relevance.symbols:
                    symbol_lower = symbol.lower()
                    for keyword in keywords[:10]:
                        if keyword in symbol_lower:
                            relevance.score += 10
                            relevance.reasons.append(f"Symbol '{symbol}' matches")
                            break

        except Exception:
            pass

        return relevance

    def _extract_symbols(self, content: str, language: str) -> List[str]:
        """Extract function/class names from file content."""
        symbols = []

        if language == "python":
            # Match def and class
            for match in re.finditer(r'(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content):
                symbols.append(match.group(1))

        elif language in ("javascript", "typescript"):
            # Match function declarations and exports
            for match in re.finditer(r'(?:function|class|const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content):
                symbols.append(match.group(1))

        elif language == "rust":
            # Match fn, struct, impl, enum
            for match in re.finditer(r'(?:fn|struct|enum|impl)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content):
                symbols.append(match.group(1))

        elif language == "go":
            # Match func and type
            for match in re.finditer(r'(?:func|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content):
                symbols.append(match.group(1))

        elif language == "java":
            # Match class, interface, method
            for match in re.finditer(r'(?:class|interface|public|private|protected)?\s*(?:static)?\s*(?:\w+)\s+([A-Z][a-zA-Z0-9_]*)', content):
                symbols.append(match.group(1))

        return symbols[:50]  # Limit to 50 symbols


def extract_repo_context(
    repo_path: Path,
    issue_description: str,
    relevant_files: Optional[List[str]] = None,
    max_files: int = 15,
) -> str:
    """Convenience function to extract context as a formatted string.

    Args:
        repo_path: Path to the repository
        issue_description: Description of the issue/task
        relevant_files: Optional user-specified relevant files
        max_files: Maximum files to include

    Returns:
        Formatted string with file contents
    """
    config = AnalysisConfig(max_files=max_files)
    analyzer = RepoAnalyzer(config)

    context = analyzer.analyze(repo_path, issue_description, relevant_files)

    # Format as string
    parts = []
    parts.append(f"# Repository Context (Language: {context.primary_language})")
    parts.append(f"# Files: {context.files_selected} selected from {context.total_files_scanned} scanned")
    parts.append("")

    for path, content in context.files.items():
        relevance = context.file_relevance.get(path)
        if relevance:
            reasons = ", ".join(relevance.reasons[:3])
            parts.append(f"# File: {path} (Score: {relevance.score:.1f}, {reasons})")
        else:
            parts.append(f"# File: {path}")
        parts.append(content)
        parts.append("")

    return "\n".join(parts)


def find_related_test_files(
    repo_path: Path,
    source_files: List[str],
) -> List[str]:
    """Find test files related to given source files.

    Args:
        repo_path: Path to the repository
        source_files: List of source file paths

    Returns:
        List of related test file paths
    """
    test_files = []

    for source_file in source_files:
        source_path = Path(source_file)
        source_name = source_path.stem

        # Common test file patterns
        patterns = [
            f"test_{source_name}.py",
            f"{source_name}_test.py",
            f"tests/test_{source_name}.py",
            f"tests/{source_name}_test.py",
            f"{source_name}.test.js",
            f"{source_name}.spec.js",
            f"{source_name}.test.ts",
            f"{source_name}.spec.ts",
            f"{source_name}_test.go",
            f"{source_name}_test.rs",
        ]

        for pattern in patterns:
            matches = list(repo_path.glob(f"**/{pattern}"))
            for match in matches:
                rel_path = str(match.relative_to(repo_path))
                if rel_path not in test_files:
                    test_files.append(rel_path)

    return test_files
