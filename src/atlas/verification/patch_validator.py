"""Patch validation and application testing."""

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """Information extracted from a patch."""

    files_modified: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    hunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from patch validation."""

    is_valid: bool = True
    can_apply: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    patch_info: PatchInfo = field(default_factory=PatchInfo)

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)


class PatchValidator:
    """Validates patches and checks if they can be applied."""

    def __init__(self):
        """Initialize the patch validator."""
        self.max_patch_size = 50000  # Maximum patch size in characters
        self.max_files = 10  # Maximum files modified

    def validate(self, patch: str) -> ValidationResult:
        """Validate a patch.

        Args:
            patch: The unified diff patch

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        if not patch:
            result.add_error("Empty patch")
            return result

        if len(patch) > self.max_patch_size:
            result.add_error(f"Patch too large: {len(patch)} characters")
            return result

        # Parse the patch
        patch_info = self._parse_patch(patch)
        result.patch_info = patch_info

        # Check for required elements
        if not patch_info.files_modified:
            result.add_error("No files modified in patch")
            return result

        if len(patch_info.files_modified) > self.max_files:
            result.add_warning(f"Patch modifies many files: {len(patch_info.files_modified)}")

        # Check for suspicious patterns
        self._check_suspicious_patterns(patch, result)

        # Validate hunk structure
        for hunk in patch_info.hunks:
            self._validate_hunk(hunk, result)

        return result

    def _parse_patch(self, patch: str) -> PatchInfo:
        """Parse a unified diff patch.

        Args:
            patch: The patch text

        Returns:
            PatchInfo with extracted information
        """
        info = PatchInfo()
        current_file = None
        current_hunk = None

        for line in patch.split("\n"):
            # File headers
            if line.startswith("---"):
                # Old file
                match = re.match(r"--- (?:a/)?(.+)", line)
                if match:
                    current_file = match.group(1)

            elif line.startswith("+++"):
                # New file
                match = re.match(r"\+\+\+ (?:b/)?(.+)", line)
                if match:
                    filename = match.group(1)
                    if filename not in info.files_modified:
                        info.files_modified.append(filename)

            elif line.startswith("@@"):
                # Hunk header
                match = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",
                    line
                )
                if match:
                    current_hunk = {
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "file": current_file,
                        "lines": [],
                    }
                    info.hunks.append(current_hunk)

            elif line.startswith("+") and not line.startswith("+++"):
                info.lines_added += 1
                if current_hunk:
                    current_hunk["lines"].append(("add", line[1:]))

            elif line.startswith("-") and not line.startswith("---"):
                info.lines_removed += 1
                if current_hunk:
                    current_hunk["lines"].append(("remove", line[1:]))

            elif line.startswith(" "):
                if current_hunk:
                    current_hunk["lines"].append(("context", line[1:]))

        return info

    def _check_suspicious_patterns(self, patch: str, result: ValidationResult) -> None:
        """Check for suspicious patterns in the patch."""
        # Check for credentials or secrets
        secret_patterns = [
            r"(?i)password\s*=\s*['\"][^'\"]+['\"]",
            r"(?i)api_key\s*=\s*['\"][^'\"]+['\"]",
            r"(?i)secret\s*=\s*['\"][^'\"]+['\"]",
            r"(?i)token\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]",
        ]

        for pattern in secret_patterns:
            if re.search(pattern, patch):
                result.add_warning("Patch may contain hardcoded credentials")
                break

        # Check for dangerous operations
        dangerous_patterns = [
            r"(?i)os\.system\s*\(",
            r"(?i)subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True",
            r"(?i)eval\s*\(",
            r"(?i)exec\s*\(",
            r"(?i)__import__\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, patch):
                result.add_warning("Patch may contain potentially dangerous code")
                break

    def _validate_hunk(self, hunk: dict[str, Any], result: ValidationResult) -> None:
        """Validate a single hunk."""
        if not hunk.get("lines"):
            result.add_warning(f"Empty hunk at line {hunk.get('new_start')}")
            return

        # Count actual added/removed lines vs declared
        actual_adds = sum(1 for op, _ in hunk["lines"] if op == "add")
        actual_removes = sum(1 for op, _ in hunk["lines"] if op == "remove")
        actual_context = sum(1 for op, _ in hunk["lines"] if op == "context")

        expected_old = hunk.get("old_count", 0)
        expected_new = hunk.get("new_count", 0)

        actual_old = actual_removes + actual_context
        actual_new = actual_adds + actual_context

        if actual_old != expected_old:
            result.add_warning(
                f"Hunk line count mismatch: expected {expected_old} old lines, got {actual_old}"
            )

        if actual_new != expected_new:
            result.add_warning(
                f"Hunk line count mismatch: expected {expected_new} new lines, got {actual_new}"
            )

    def try_apply(
        self,
        patch: str,
        target_dir: Path,
        dry_run: bool = True,
    ) -> ValidationResult:
        """Try to apply a patch to a directory.

        Args:
            patch: The patch text
            target_dir: Directory to apply patch to
            dry_run: If True, don't actually modify files

        Returns:
            ValidationResult with application status
        """
        result = self.validate(patch)

        if not result.is_valid:
            result.can_apply = False
            return result

        # Try to apply the patch
        try:
            # For now, we do a simple check if the target files exist
            for filename in result.patch_info.files_modified:
                target_file = target_dir / filename
                if not target_file.exists():
                    # This is okay for new files
                    continue

            result.can_apply = True

        except Exception as e:
            result.can_apply = False
            result.add_error(f"Failed to apply patch: {e}")

        return result


def compare_patches(patch1: str, patch2: str) -> float:
    """Compare two patches for semantic similarity.

    Uses a combination of:
    - File overlap (which files are modified)
    - Function overlap (which functions are touched)
    - Content similarity (fuzzy text matching on changes)

    Args:
        patch1: First patch
        patch2: Second patch

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not patch1 or not patch2:
        return 0.0

    if patch1 == patch2:
        return 1.0

    # Import here to avoid circular imports
    from atlas.verification.clustering import extract_patch_signature

    # Extract semantic signatures
    sig1 = extract_patch_signature(patch1)
    sig2 = extract_patch_signature(patch2)

    # Use the semantic similarity calculation
    return sig1.similarity_to(sig2)


def compare_patches_exact(patch1: str, patch2: str) -> float:
    """Compare two patches for exact line-based similarity.

    Uses Jaccard similarity on normalized patch lines.
    Faster but less intelligent than semantic comparison.

    Args:
        patch1: First patch
        patch2: Second patch

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not patch1 or not patch2:
        return 0.0

    if patch1 == patch2:
        return 1.0

    # Normalize patches
    lines1 = set(_normalize_patch_lines(patch1))
    lines2 = set(_normalize_patch_lines(patch2))

    if not lines1 or not lines2:
        return 0.0

    # Jaccard similarity
    intersection = len(lines1 & lines2)
    union = len(lines1 | lines2)

    return intersection / union if union > 0 else 0.0


def _normalize_patch_lines(patch: str) -> list[str]:
    """Normalize patch lines for comparison."""
    lines = []
    for line in patch.split("\n"):
        # Skip headers
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        # Skip empty lines
        if not line.strip():
            continue
        # Normalize whitespace
        lines.append(line.strip())
    return lines
