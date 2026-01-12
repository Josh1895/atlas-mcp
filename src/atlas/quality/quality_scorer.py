"""Objective quality scoring for patches.

Computes measurable quality metrics without using AI:
- Style compliance (linting, formatting, naming)
- Maintainability (complexity, size, readability)
- Risk assessment (dangerous patterns, sensitive areas)
"""

import ast
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Comprehensive quality score for a patch."""

    patch_id: str

    # Style Fit (0-100)
    style_score: float = 0.0
    linter_issues: int = 0
    formatter_compliant: bool = True
    naming_consistent: bool = True
    import_order_correct: bool = True

    # Maintainability (0-100)
    maintainability_score: float = 0.0
    cyclomatic_complexity_delta: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_touched: int = 0
    functions_modified: int = 0
    max_function_length: int = 0

    # Risk Flags
    risk_score: float = 0.0  # 0 = no risk, 100 = very risky
    risk_flags: List[str] = field(default_factory=list)

    # Computed overall score
    overall_score: float = 0.0

    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Compute weighted overall score.

        Args:
            weights: Optional weight overrides for each category
        """
        if weights is None:
            weights = {
                "style": 0.25,
                "maintainability": 0.35,
                "risk": 0.40,
            }

        # Risk score is inverted (high risk = low score)
        risk_component = 100 - self.risk_score

        self.overall_score = (
            weights["style"] * self.style_score
            + weights["maintainability"] * self.maintainability_score
            + weights["risk"] * risk_component
        )


class QualityScorer:
    """Computes objective quality metrics for patches.

    No AI involved - just static analysis and heuristics.
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        language: str = "python",
    ):
        """Initialize the quality scorer.

        Args:
            repo_path: Path to the repository (for context)
            language: Programming language of the code
        """
        self.repo_path = repo_path
        self.language = language

    async def score(
        self,
        patch: str,
        patch_id: str,
        original_files: Dict[str, str],
        patched_files: Dict[str, str],
    ) -> QualityScore:
        """Score a patch on all quality dimensions.

        Args:
            patch: The unified diff patch
            patch_id: Identifier for the patch
            original_files: Dict of {filepath: original_content}
            patched_files: Dict of {filepath: patched_content}

        Returns:
            QualityScore with all metrics computed
        """
        # Style scoring
        style_result = await self._score_style(patched_files)

        # Maintainability scoring
        maint_result = self._score_maintainability(patch, original_files, patched_files)

        # Risk scoring
        risk_result = self._score_risk(patch, patched_files)

        score = QualityScore(
            patch_id=patch_id,
            # Style
            style_score=style_result["score"],
            linter_issues=style_result["linter_issues"],
            formatter_compliant=style_result["formatter_compliant"],
            naming_consistent=style_result["naming_consistent"],
            import_order_correct=style_result["import_order"],
            # Maintainability
            maintainability_score=maint_result["score"],
            cyclomatic_complexity_delta=maint_result["complexity_delta"],
            lines_added=maint_result["lines_added"],
            lines_removed=maint_result["lines_removed"],
            files_touched=maint_result["files_touched"],
            functions_modified=maint_result["functions_modified"],
            max_function_length=maint_result["max_func_length"],
            # Risk
            risk_score=risk_result["score"],
            risk_flags=risk_result["flags"],
        )

        score.compute_overall()
        return score

    async def _score_style(self, patched_files: Dict[str, str]) -> Dict:
        """Score code style compliance."""
        total_issues = 0
        formatter_ok = True
        naming_ok = True
        imports_ok = True

        for filepath, content in patched_files.items():
            if not filepath.endswith(".py"):
                continue

            # Run linter (ruff if available)
            linter_issues = await self._run_linter(content)
            total_issues += linter_issues

            # Check formatter compliance
            formatter_ok = formatter_ok and self._check_formatter(content)

            # Check naming conventions
            naming_ok = naming_ok and self._check_naming(content)

            # Check import order
            imports_ok = imports_ok and self._check_imports(content)

        # Convert issues to score (more issues = lower score)
        # 0 issues = 100, 10+ issues = 0
        issue_score = max(0, 100 - total_issues * 10)

        # Combine factors
        score = (
            issue_score * 0.5
            + (100 if formatter_ok else 50) * 0.2
            + (100 if naming_ok else 50) * 0.15
            + (100 if imports_ok else 50) * 0.15
        )

        return {
            "score": score,
            "linter_issues": total_issues,
            "formatter_compliant": formatter_ok,
            "naming_consistent": naming_ok,
            "import_order": imports_ok,
        }

    async def _run_linter(self, code: str) -> int:
        """Run linter and count issues."""
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            # Try ruff first (faster)
            try:
                result = subprocess.run(
                    ["ruff", "check", temp_path, "--output-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                import json

                issues = json.loads(result.stdout) if result.stdout else []
                return len(issues)
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # Fallback: try flake8
            try:
                result = subprocess.run(
                    ["flake8", temp_path, "--count", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # flake8 outputs count on last line with -q
                lines = result.stdout.strip().split("\n")
                if lines and lines[-1].isdigit():
                    return int(lines[-1])
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # No linter available
            return 0

        except Exception as e:
            logger.debug(f"Linter check failed: {e}")
            return 0
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    def _check_formatter(self, code: str) -> bool:
        """Check if code is properly formatted."""
        try:
            import black

            formatted = black.format_str(code, mode=black.Mode())
            return formatted.strip() == code.strip()
        except ImportError:
            # black not available, assume ok
            return True
        except Exception:
            # Parse error or other issue
            return True

    def _check_naming(self, code: str) -> bool:
        """Check if naming follows Python conventions."""
        violations = [
            # camelCase variable (should be snake_case)
            r"\b[a-z]+[A-Z][a-z]+\s*=",
            # lowercase class name
            r"class\s+[a-z][a-z_]*\s*[:\(]",
            # SCREAMING_CASE for non-constants (simplified check)
            # Skip this as it's hard to detect accurately
        ]

        for pattern in violations:
            if re.search(pattern, code):
                return False

        return True

    def _check_imports(self, code: str) -> bool:
        """Check import ordering (stdlib, third-party, local)."""
        # Simplified check - just verify imports are at the top
        lines = code.split("\n")
        seen_non_import = False

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            is_import = stripped.startswith("import ") or stripped.startswith("from ")

            if is_import and seen_non_import:
                # Import after code - bad
                return False

            if not is_import and not stripped.startswith('"""'):
                seen_non_import = True

        return True

    def _score_maintainability(
        self,
        patch: str,
        original_files: Dict[str, str],
        patched_files: Dict[str, str],
    ) -> Dict:
        """Score maintainability metrics."""
        # Count changes from patch
        lines_added = sum(1 for line in patch.split("\n") if line.startswith("+") and not line.startswith("+++"))
        lines_removed = sum(1 for line in patch.split("\n") if line.startswith("-") and not line.startswith("---"))
        files_touched = len(patched_files)

        # Compute complexity delta
        original_complexity = sum(
            self._compute_complexity(code) for code in original_files.values()
        )
        patched_complexity = sum(
            self._compute_complexity(code) for code in patched_files.values()
        )
        complexity_delta = patched_complexity - original_complexity

        # Find max function length in patched code
        max_func_length = max(
            (self._max_function_length(code) for code in patched_files.values()),
            default=0,
        )

        # Count modified functions
        functions_modified = len(re.findall(r"^[-+]\s*def ", patch, re.MULTILINE))

        # Compute score
        # Prefer: small diffs, low complexity increase, reasonable function lengths
        size_penalty = min(50, (lines_added + lines_removed) / 2)
        complexity_penalty = min(30, max(0, complexity_delta * 5))
        length_penalty = min(20, max(0, (max_func_length - 50) / 2))

        score = max(0, 100 - size_penalty - complexity_penalty - length_penalty)

        return {
            "score": score,
            "complexity_delta": complexity_delta,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "files_touched": files_touched,
            "functions_modified": functions_modified,
            "max_func_length": max_func_length,
        }

    def _compute_complexity(self, code: str) -> int:
        """Compute cyclomatic complexity (simplified)."""
        # Count decision points
        decision_keywords = [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "except",
            "and",
            "or",
        ]
        complexity = 1  # Base complexity

        for keyword in decision_keywords:
            complexity += len(re.findall(rf"\b{keyword}\b", code))

        return complexity

    def _max_function_length(self, code: str) -> int:
        """Find the longest function in the code."""
        try:
            tree = ast.parse(code)
            max_length = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, "end_lineno") and node.end_lineno:
                        length = node.end_lineno - node.lineno + 1
                        max_length = max(max_length, length)

            return max_length
        except Exception:
            return 0

    def _score_risk(self, patch: str, patched_files: Dict[str, str]) -> Dict:
        """Score risk factors."""
        flags = []
        risk_points = 0

        # Check for risky patterns
        risk_patterns = [
            (r"except\s*:", "bare_except", 15, "Bare except clause catches all errors"),
            (
                r"except.*:\s*pass",
                "silent_exception",
                20,
                "Silent exception swallowing",
            ),
            (r"#\s*TODO", "todo_comment", 5, "TODO comment left in code"),
            (r"#\s*HACK", "hack_comment", 15, "HACK comment indicates workaround"),
            (r"#\s*FIXME", "fixme_comment", 10, "FIXME comment indicates known issue"),
            (r"eval\s*\(", "eval_usage", 25, "eval() is a security risk"),
            (r"exec\s*\(", "exec_usage", 25, "exec() is a security risk"),
            (r"__import__\s*\(", "dynamic_import", 15, "Dynamic import can be risky"),
            (
                r"subprocess\.(call|run|Popen)",
                "subprocess",
                10,
                "Subprocess execution",
            ),
            (r"os\.system\s*\(", "os_system", 20, "os.system() is risky"),
            (
                r"password|secret|api_key|token",
                "sensitive_keyword",
                10,
                "Possible sensitive data",
            ),
            (r"pickle\.load", "pickle_load", 15, "pickle.load() is a security risk"),
            (r"yaml\.load\(", "yaml_load", 10, "yaml.load() without Loader is risky"),
        ]

        combined_code = "\n".join(patched_files.values()) + "\n" + patch

        for pattern, flag_name, points, description in risk_patterns:
            if re.search(pattern, combined_code, re.IGNORECASE):
                flags.append(f"{flag_name}: {description}")
                risk_points += points

        # Check for file sensitivity
        sensitive_paths = [
            "auth",
            "security",
            "login",
            "password",
            "config",
            "secret",
            "credential",
        ]
        for filepath in patched_files.keys():
            if any(s in filepath.lower() for s in sensitive_paths):
                flags.append(f"sensitive_file: Modifies {filepath}")
                risk_points += 10

        # Check for dependency changes
        if "requirements.txt" in patched_files or "setup.py" in patched_files:
            flags.append("dependency_change: Modifies dependencies")
            risk_points += 15

        if "pyproject.toml" in patched_files:
            flags.append("dependency_change: Modifies pyproject.toml")
            risk_points += 10

        # Cap at 100
        risk_score = min(100, risk_points)

        return {
            "score": risk_score,
            "flags": flags,
        }


async def score_patches(
    patches: Dict[str, str],
    original_files: Dict[str, str],
    patched_files_map: Dict[str, Dict[str, str]],
) -> Dict[str, QualityScore]:
    """Score multiple patches.

    Args:
        patches: Dict of {patch_id: patch_content}
        original_files: Dict of {filepath: original_content}
        patched_files_map: Dict of {patch_id: {filepath: patched_content}}

    Returns:
        Dict of {patch_id: QualityScore}
    """
    scorer = QualityScorer()
    scores = {}

    for patch_id, patch_content in patches.items():
        patched_files = patched_files_map.get(patch_id, {})
        score = await scorer.score(patch_id, patch_id, original_files, patched_files)
        scores[patch_id] = score

    return scores
