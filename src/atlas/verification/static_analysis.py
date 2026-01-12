"""Static analysis for validating generated patches."""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from static analysis."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(warning)


class StaticAnalyzer:
    """Static analysis for Python code."""

    def __init__(self):
        """Initialize the static analyzer."""
        self.max_line_length = 120
        self.max_function_lines = 100

    def analyze_python(self, code: str, filename: str = "<unknown>") -> AnalysisResult:
        """Analyze Python code for syntax and basic issues.

        Args:
            code: The Python code to analyze
            filename: The filename for error messages

        Returns:
            AnalysisResult with any errors or warnings
        """
        result = AnalysisResult()

        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.add_error(f"Syntax error at line {e.lineno}: {e.msg}")
            return result

        # Check for common issues
        self._check_line_lengths(code, result)
        self._check_trailing_whitespace(code, result)
        self._check_common_mistakes(code, result)

        return result

    def analyze_patch(self, patch: str) -> AnalysisResult:
        """Analyze a unified diff patch.

        Args:
            patch: The unified diff patch

        Returns:
            AnalysisResult with any errors or warnings
        """
        result = AnalysisResult()

        if not patch:
            result.add_error("Empty patch")
            return result

        # Check patch structure
        if "---" not in patch and "+++" not in patch:
            result.add_warning("Patch may be malformed: missing --- or +++ headers")

        if "@@" not in patch:
            result.add_warning("Patch may be malformed: missing @@ hunk headers")

        # Extract and analyze added Python code
        added_lines = self._extract_added_lines(patch)
        if added_lines:
            # Try to parse as Python if it looks like Python
            if self._looks_like_python(added_lines):
                # We can't fully parse partial code, but check for obvious issues
                self._check_added_code(added_lines, result)

        return result

    def _check_line_lengths(self, code: str, result: AnalysisResult) -> None:
        """Check for lines that are too long."""
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > self.max_line_length:
                result.add_warning(
                    f"Line {i} exceeds {self.max_line_length} characters"
                )

    def _check_trailing_whitespace(self, code: str, result: AnalysisResult) -> None:
        """Check for trailing whitespace."""
        for i, line in enumerate(code.split("\n"), 1):
            if line != line.rstrip():
                result.add_warning(f"Line {i} has trailing whitespace")

    def _check_common_mistakes(self, code: str, result: AnalysisResult) -> None:
        """Check for common Python mistakes."""
        # Check for print statements (Python 2 style)
        if re.search(r"\bprint\s+['\"]", code):
            result.add_warning("Found Python 2 style print statement")

        # Check for bare except
        if re.search(r"\bexcept\s*:", code):
            result.add_warning("Found bare except clause")

        # Check for mutable default arguments
        if re.search(r"def\s+\w+\([^)]*=\s*(\[\]|\{\})", code):
            result.add_warning("Found mutable default argument")

    def _extract_added_lines(self, patch: str) -> str:
        """Extract lines added in the patch."""
        added = []
        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
        return "\n".join(added)

    def _looks_like_python(self, code: str) -> bool:
        """Check if code looks like Python."""
        python_keywords = [
            "def ", "class ", "import ", "from ", "return ",
            "if ", "else:", "elif ", "for ", "while ", "try:",
            "except", "with ", "async ", "await ",
        ]
        return any(kw in code for kw in python_keywords)

    def _check_added_code(self, code: str, result: AnalysisResult) -> None:
        """Check added code for issues."""
        # Check for obvious issues
        self._check_common_mistakes(code, result)

        # Check indentation consistency
        lines = code.split("\n")
        indents = set()
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.add(indent)

        # Check for mixed indentation
        if len(indents) > 1:
            min_indent = min(indents)
            for indent in indents:
                if indent % min_indent != 0:
                    result.add_warning("Inconsistent indentation detected")
                    break


def analyze_file(file_path: Path) -> AnalysisResult:
    """Analyze a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        AnalysisResult with any errors or warnings
    """
    analyzer = StaticAnalyzer()

    try:
        code = file_path.read_text()
        return analyzer.analyze_python(code, str(file_path))
    except Exception as e:
        result = AnalysisResult()
        result.add_error(f"Failed to read file: {e}")
        return result
