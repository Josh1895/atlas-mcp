"""Ownership validation for task-level patch constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch

from atlas.core.task_dag import OwnershipRules
from atlas.verification.patch_applier import PatchParser


@dataclass
class OwnershipValidationResult:
    """Result of ownership validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_checked: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False


class OwnershipValidator:
    """Validate that patches respect ownership rules."""

    def __init__(self):
        self._parser = PatchParser()

    def validate(self, patch: str, ownership: OwnershipRules) -> OwnershipValidationResult:
        result = OwnershipValidationResult()

        file_patches = self._parser.parse(patch)
        for file_patch in file_patches:
            target = file_patch.target_path
            result.files_checked.append(target)

            if self._is_blocked(target, ownership):
                result.add_error(f"File {target} is explicitly blocked")
                continue

            if not self._is_allowed(target, ownership):
                result.add_error(f"File {target} is outside ownership scope")

        return result

    def _is_allowed(self, path: str, ownership: OwnershipRules) -> bool:
        if ownership.allowed_files and path in ownership.allowed_files:
            return True

        for allowed_dir in ownership.allowed_dirs:
            if path.startswith(allowed_dir.rstrip("/") + "/"):
                return True

        for pattern in ownership.allowed_globs:
            # Handle **/* specially - it means "all files"
            if pattern == "**/*" or pattern == "*":
                return True
            if fnmatch(path, pattern):
                return True

        return False

    def _is_blocked(self, path: str, ownership: OwnershipRules) -> bool:
        for pattern in ownership.blocked_globs:
            if fnmatch(path, pattern):
                return True
        return False

