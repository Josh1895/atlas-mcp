"""Patch composition and diff utilities."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from atlas.verification.patch_applier import PatchApplier, PatchApplyResult


@dataclass
class PatchComposeResult:
    """Result from composing multiple patches."""

    success: bool
    patched_files: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)


def apply_patches(
    base_files: dict[str, str],
    patches: list[str],
) -> PatchComposeResult:
    """Apply patches sequentially to a file map."""
    applier = PatchApplier()
    current_files = dict(base_files)
    result = PatchComposeResult(success=True, patched_files=current_files)

    for patch in patches:
        apply_result = applier.apply_patch(patch, current_files)
        result.errors.extend(apply_result.errors)
        result.warnings.extend(apply_result.warnings)
        result.files_created.extend(apply_result.files_created)
        result.files_modified.extend(apply_result.files_modified)
        result.files_deleted.extend(apply_result.files_deleted)

        if not apply_result.success:
            result.success = False
            return result

        current_files = dict(apply_result.patched_files)
        result.patched_files = current_files

    return result


def diff_files(
    original_files: dict[str, str],
    patched_files: dict[str, str],
) -> str:
    """Generate a unified diff from original to patched files."""
    diff_chunks: list[str] = []
    all_paths = sorted(set(original_files.keys()) | set(patched_files.keys()))

    for path in all_paths:
        original = original_files.get(path, "")
        patched = patched_files.get(path, "")
        if original == patched:
            continue

        original_lines = original.splitlines()
        patched_lines = patched.splitlines()

        diff = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        diff_chunks.extend(list(diff))

    return "\n".join(diff_chunks)

