"""Patch application utility for ATLAS.

This module handles applying unified diff patches to files,
enabling test execution and proper quality scoring on patched code.
"""

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HunkHeader:
    """Parsed hunk header from unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass
class Hunk:
    """A single hunk from a unified diff."""

    header: HunkHeader
    lines: List[str]  # Lines with their prefixes (+, -, space)

    def get_context_lines(self) -> List[str]:
        """Get context lines (no prefix)."""
        return [line[1:] for line in self.lines if line.startswith(' ')]

    def get_added_lines(self) -> List[str]:
        """Get added lines (+ prefix)."""
        return [line[1:] for line in self.lines if line.startswith('+')]

    def get_removed_lines(self) -> List[str]:
        """Get removed lines (- prefix)."""
        return [line[1:] for line in self.lines if line.startswith('-')]


@dataclass
class FilePatch:
    """Patch for a single file."""

    old_path: str  # Path from --- line (may be /dev/null for new files)
    new_path: str  # Path from +++ line (may be /dev/null for deleted files)
    hunks: List[Hunk] = field(default_factory=list)

    @property
    def is_new_file(self) -> bool:
        return self.old_path == "/dev/null" or self.old_path.startswith("a//dev/null")

    @property
    def is_deleted_file(self) -> bool:
        return self.new_path == "/dev/null" or self.new_path.startswith("b//dev/null")

    @property
    def target_path(self) -> str:
        """Get the actual file path (stripping a/ or b/ prefix)."""
        path = self.new_path if not self.is_deleted_file else self.old_path
        # Strip common prefixes
        if path.startswith("b/"):
            return path[2:]
        if path.startswith("a/"):
            return path[2:]
        return path


@dataclass
class PatchApplyResult:
    """Result of applying a patch."""

    success: bool
    patched_files: Dict[str, str] = field(default_factory=dict)  # {path: content}
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)


class PatchParser:
    """Parser for unified diff format."""

    # Regex patterns
    FILE_HEADER_OLD = re.compile(r'^--- (.+?)(?:\t.*)?$')
    FILE_HEADER_NEW = re.compile(r'^\+\+\+ (.+?)(?:\t.*)?$')
    HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    def parse(self, patch_content: str) -> List[FilePatch]:
        """Parse a unified diff into FilePatch objects.

        Args:
            patch_content: The unified diff content

        Returns:
            List of FilePatch objects
        """
        patches = []
        lines = patch_content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for file header
            old_match = self.FILE_HEADER_OLD.match(line)
            if old_match:
                old_path = old_match.group(1)

                # Next line should be +++
                i += 1
                if i >= len(lines):
                    break

                new_match = self.FILE_HEADER_NEW.match(lines[i])
                if not new_match:
                    i += 1
                    continue

                new_path = new_match.group(1)

                # Parse hunks for this file
                file_patch = FilePatch(old_path=old_path, new_path=new_path)
                i += 1

                while i < len(lines):
                    hunk_match = self.HUNK_HEADER.match(lines[i])
                    if hunk_match:
                        header = HunkHeader(
                            old_start=int(hunk_match.group(1)),
                            old_count=int(hunk_match.group(2) or 1),
                            new_start=int(hunk_match.group(3)),
                            new_count=int(hunk_match.group(4) or 1),
                        )

                        # Collect hunk lines
                        hunk_lines = []
                        i += 1

                        while i < len(lines):
                            hunk_line = lines[i]

                            # Check for end of hunk
                            if (hunk_line.startswith('---') or
                                hunk_line.startswith('@@') or
                                hunk_line.startswith('diff ')):
                                break

                            # Valid hunk line prefixes
                            if hunk_line and hunk_line[0] in ' +-':
                                hunk_lines.append(hunk_line)
                            elif hunk_line == '':
                                # Empty line in context
                                hunk_lines.append(' ')
                            elif hunk_line.startswith('\\'):
                                # "\ No newline at end of file"
                                pass
                            else:
                                # Unknown line, likely end of hunk
                                break

                            i += 1

                        file_patch.hunks.append(Hunk(header=header, lines=hunk_lines))
                    elif lines[i].startswith('---'):
                        # Next file patch
                        break
                    else:
                        i += 1

                patches.append(file_patch)
            else:
                i += 1

        return patches


class PatchApplier:
    """Applies unified diff patches to files."""

    def __init__(self):
        self.parser = PatchParser()

    def apply_to_content(
        self,
        original_content: str,
        file_patch: FilePatch,
    ) -> Tuple[str, List[str]]:
        """Apply a file patch to content.

        Args:
            original_content: The original file content
            file_patch: The patch to apply

        Returns:
            Tuple of (patched_content, errors)
        """
        errors = []

        if file_patch.is_new_file:
            # New file - construct from added lines
            new_lines = []
            for hunk in file_patch.hunks:
                new_lines.extend(hunk.get_added_lines())
            return '\n'.join(new_lines), errors

        original_lines = original_content.split('\n')
        result_lines = list(original_lines)

        # Apply hunks in reverse order to preserve line numbers
        for hunk in reversed(file_patch.hunks):
            hunk_result, hunk_errors = self._apply_hunk(
                result_lines,
                hunk,
                file_patch.target_path,
            )
            result_lines = hunk_result
            errors.extend(hunk_errors)

        return '\n'.join(result_lines), errors

    def _apply_hunk(
        self,
        lines: List[str],
        hunk: Hunk,
        filepath: str,
    ) -> Tuple[List[str], List[str]]:
        """Apply a single hunk to lines.

        Args:
            lines: Current file lines
            hunk: The hunk to apply
            filepath: File path for error messages

        Returns:
            Tuple of (modified_lines, errors)
        """
        errors = []
        result = list(lines)

        # Find the correct position (0-indexed)
        start_pos = hunk.header.old_start - 1

        # Try to find context match
        match_pos = self._find_hunk_position(result, hunk, start_pos)

        if match_pos is None:
            # Try fuzzy matching
            match_pos = self._fuzzy_find_hunk(result, hunk)
            if match_pos is not None:
                errors.append(
                    f"Hunk applied with offset in {filepath} "
                    f"(expected line {hunk.header.old_start}, found at {match_pos + 1})"
                )

        if match_pos is None:
            errors.append(
                f"Could not apply hunk to {filepath} at line {hunk.header.old_start}"
            )
            return result, errors

        # Build new content for this section
        new_section = []
        for line in hunk.lines:
            if line.startswith('+'):
                new_section.append(line[1:])
            elif line.startswith(' '):
                new_section.append(line[1:])
            # Skip removed lines (they're being replaced)

        # Calculate how many lines to remove
        lines_to_remove = sum(
            1 for line in hunk.lines if line.startswith('-') or line.startswith(' ')
        )

        # Replace the section
        result = result[:match_pos] + new_section + result[match_pos + lines_to_remove:]

        return result, errors

    def _find_hunk_position(
        self,
        lines: List[str],
        hunk: Hunk,
        expected_pos: int,
    ) -> Optional[int]:
        """Find exact position for hunk based on context.

        Args:
            lines: File lines
            hunk: Hunk to apply
            expected_pos: Expected position (0-indexed)

        Returns:
            Position if found, None otherwise
        """
        # Get lines that should be present (context + removed)
        expected_lines = []
        for line in hunk.lines:
            if line.startswith(' ') or line.startswith('-'):
                expected_lines.append(line[1:])

        if not expected_lines:
            return expected_pos if expected_pos < len(lines) else None

        # Check at expected position
        if self._matches_at_position(lines, expected_lines, expected_pos):
            return expected_pos

        return None

    def _fuzzy_find_hunk(
        self,
        lines: List[str],
        hunk: Hunk,
        search_range: int = 50,
    ) -> Optional[int]:
        """Fuzzy search for hunk position.

        Args:
            lines: File lines
            hunk: Hunk to find
            search_range: How many lines to search around expected position

        Returns:
            Position if found, None otherwise
        """
        expected_pos = hunk.header.old_start - 1

        expected_lines = []
        for line in hunk.lines:
            if line.startswith(' ') or line.startswith('-'):
                expected_lines.append(line[1:])

        if not expected_lines:
            return None

        # Search around expected position
        for offset in range(search_range):
            for pos in [expected_pos + offset, expected_pos - offset]:
                if pos < 0:
                    continue
                if self._matches_at_position(lines, expected_lines, pos):
                    return pos

        return None

    def _matches_at_position(
        self,
        lines: List[str],
        expected: List[str],
        pos: int,
    ) -> bool:
        """Check if expected lines match at position.

        Args:
            lines: File lines
            expected: Expected lines
            pos: Position to check

        Returns:
            True if matches
        """
        if pos < 0 or pos + len(expected) > len(lines):
            return False

        for i, expected_line in enumerate(expected):
            if lines[pos + i].rstrip() != expected_line.rstrip():
                return False

        return True

    def apply_patch(
        self,
        patch_content: str,
        original_files: Dict[str, str],
    ) -> PatchApplyResult:
        """Apply a patch to a set of files (in memory).

        Args:
            patch_content: The unified diff
            original_files: Dict of {filepath: content}

        Returns:
            PatchApplyResult with patched files
        """
        result = PatchApplyResult(success=True, patched_files=dict(original_files))

        try:
            file_patches = self.parser.parse(patch_content)
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to parse patch: {e}")
            return result

        for file_patch in file_patches:
            target_path = file_patch.target_path

            if file_patch.is_deleted_file:
                # Delete file
                if target_path in result.patched_files:
                    del result.patched_files[target_path]
                    result.files_deleted.append(target_path)
                continue

            if file_patch.is_new_file:
                # New file
                patched_content, errors = self.apply_to_content("", file_patch)
                result.patched_files[target_path] = patched_content
                result.files_created.append(target_path)
                result.errors.extend(errors)
                continue

            # Modify existing file
            original_content = original_files.get(target_path, "")

            if not original_content:
                result.warnings.append(
                    f"File {target_path} not found in original files, treating as new"
                )
                patched_content, errors = self.apply_to_content("", file_patch)
            else:
                patched_content, errors = self.apply_to_content(
                    original_content, file_patch
                )

            result.patched_files[target_path] = patched_content
            result.files_modified.append(target_path)
            result.errors.extend(errors)

        # Mark as failed if there were critical errors
        if any("Could not apply hunk" in e for e in result.errors):
            result.success = False

        return result

    def apply_patch_to_directory(
        self,
        patch_content: str,
        repo_path: Path,
        output_path: Optional[Path] = None,
    ) -> PatchApplyResult:
        """Apply a patch to a directory on disk.

        Args:
            patch_content: The unified diff
            repo_path: Path to the repository
            output_path: Optional output path (if None, modifies in place)

        Returns:
            PatchApplyResult
        """
        result = PatchApplyResult(success=True)

        # If output_path specified, copy the repo first
        if output_path:
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(repo_path, output_path)
            target_path = output_path
        else:
            target_path = repo_path

        try:
            file_patches = self.parser.parse(patch_content)
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to parse patch: {e}")
            return result

        for file_patch in file_patches:
            file_target = file_patch.target_path
            full_path = target_path / file_target

            if file_patch.is_deleted_file:
                if full_path.exists():
                    full_path.unlink()
                    result.files_deleted.append(file_target)
                continue

            if file_patch.is_new_file:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                patched_content, errors = self.apply_to_content("", file_patch)
                full_path.write_text(patched_content)
                result.patched_files[file_target] = patched_content
                result.files_created.append(file_target)
                result.errors.extend(errors)
                continue

            # Modify existing file
            if not full_path.exists():
                result.warnings.append(f"File {file_target} not found, skipping")
                continue

            original_content = full_path.read_text()
            patched_content, errors = self.apply_to_content(
                original_content, file_patch
            )

            full_path.write_text(patched_content)
            result.patched_files[file_target] = patched_content
            result.files_modified.append(file_target)
            result.errors.extend(errors)

        if any("Could not apply hunk" in e for e in result.errors):
            result.success = False

        return result


def create_patched_checkout(
    repo_path: Path,
    patch_content: str,
) -> Tuple[Optional[Path], PatchApplyResult]:
    """Create a temporary checkout with a patch applied.

    Args:
        repo_path: Path to the original repository
        patch_content: The unified diff to apply

    Returns:
        Tuple of (temp_path, result) where temp_path is None on failure
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="atlas_patched_")
    temp_path = Path(temp_dir)

    applier = PatchApplier()
    result = applier.apply_patch_to_directory(
        patch_content,
        repo_path,
        temp_path,
    )

    if not result.success:
        # Clean up on failure
        shutil.rmtree(temp_dir)
        return None, result

    return temp_path, result
