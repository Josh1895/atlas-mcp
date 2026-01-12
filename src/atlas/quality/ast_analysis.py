"""AST-based analysis for precise structural comparison of patches.

Uses Python's built-in ast module for parsing, with optional tree-sitter
support for more languages.
"""

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ASTChangeSignature:
    """Represents the structural changes in a patch."""

    patch_id: str

    # What kinds of changes
    functions_added: List[str] = field(default_factory=list)
    functions_modified: List[str] = field(default_factory=list)
    functions_removed: List[str] = field(default_factory=list)

    classes_added: List[str] = field(default_factory=list)
    classes_modified: List[str] = field(default_factory=list)
    classes_removed: List[str] = field(default_factory=list)

    imports_added: List[str] = field(default_factory=list)
    imports_removed: List[str] = field(default_factory=list)

    # Structural patterns detected
    added_guard_clause: bool = False
    added_try_except: bool = False
    added_loop: bool = False
    changed_return_type: bool = False
    added_validation: bool = False
    added_assertion: bool = False

    # Edit operations (high-level)
    edit_operations: List[str] = field(default_factory=list)

    # Node type counts for comparison
    node_type_delta: Dict[str, int] = field(default_factory=dict)


class ASTAnalyzer:
    """Analyzes patches at the AST level for structural comparison.

    Uses Python's built-in ast module. Can be extended with tree-sitter
    for multi-language support.
    """

    def __init__(self, language: str = "python"):
        """Initialize the AST analyzer.

        Args:
            language: Programming language to analyze
        """
        self.language = language
        self._tree_sitter_available = self._check_tree_sitter()

    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            import tree_sitter_python  # noqa: F401

            return True
        except ImportError:
            return False

    def analyze_patch(
        self,
        original_code: str,
        patched_code: str,
        patch_id: str,
    ) -> ASTChangeSignature:
        """Analyze what changed between original and patched code.

        Args:
            original_code: The original source code
            patched_code: The patched source code
            patch_id: Identifier for the patch

        Returns:
            ASTChangeSignature describing the structural changes
        """
        try:
            # Parse both versions
            original_tree = ast.parse(original_code)
            patched_tree = ast.parse(patched_code)

            # Extract structural elements
            original_funcs = self._extract_functions(original_tree)
            patched_funcs = self._extract_functions(patched_tree)

            original_classes = self._extract_classes(original_tree)
            patched_classes = self._extract_classes(patched_tree)

            original_imports = self._extract_imports(original_tree)
            patched_imports = self._extract_imports(patched_tree)

            # Compute diffs
            funcs_added = [f for f in patched_funcs if f not in original_funcs]
            funcs_removed = [f for f in original_funcs if f not in patched_funcs]
            funcs_modified = self._find_modified_functions(
                original_code, patched_code, original_funcs, patched_funcs
            )

            classes_added = [c for c in patched_classes if c not in original_classes]
            classes_removed = [c for c in original_classes if c not in patched_classes]

            imports_added = [i for i in patched_imports if i not in original_imports]
            imports_removed = [i for i in original_imports if i not in patched_imports]

            # Detect patterns
            patterns = self._detect_patterns(original_tree, patched_tree)

            # Generate edit operations
            edit_ops = self._compute_edit_operations(original_tree, patched_tree)

            # Compute node type delta
            node_delta = self._compute_node_delta(original_tree, patched_tree)

            return ASTChangeSignature(
                patch_id=patch_id,
                functions_added=funcs_added,
                functions_modified=funcs_modified,
                functions_removed=funcs_removed,
                classes_added=classes_added,
                classes_modified=[],  # Simplified for now
                classes_removed=classes_removed,
                imports_added=imports_added,
                imports_removed=imports_removed,
                added_guard_clause=patterns.get("guard_clause", False),
                added_try_except=patterns.get("try_except", False),
                added_loop=patterns.get("loop", False),
                changed_return_type=patterns.get("return_change", False),
                added_validation=patterns.get("validation", False),
                added_assertion=patterns.get("assertion", False),
                edit_operations=edit_ops,
                node_type_delta=node_delta,
            )

        except SyntaxError as e:
            logger.warning(f"Failed to parse code for patch {patch_id}: {e}")
            return self._empty_signature(patch_id)

    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Extract function names from AST."""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """Extract class names from AST."""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")

        return imports

    def _detect_patterns(
        self, original_tree: ast.AST, patched_tree: ast.AST
    ) -> Dict[str, bool]:
        """Detect common code patterns added in patch."""
        patterns = {}

        original_types = self._count_node_types(original_tree)
        patched_types = self._count_node_types(patched_tree)

        # Guard clause: new if statement (early return pattern)
        patterns["guard_clause"] = patched_types.get("If", 0) > original_types.get(
            "If", 0
        )

        # Try-except added
        patterns["try_except"] = patched_types.get("Try", 0) > original_types.get(
            "Try", 0
        )

        # Loop added
        patterns["loop"] = (
            patched_types.get("For", 0)
            + patched_types.get("While", 0)
            + patched_types.get("AsyncFor", 0)
            > original_types.get("For", 0)
            + original_types.get("While", 0)
            + original_types.get("AsyncFor", 0)
        )

        # Return statement changed
        patterns["return_change"] = patched_types.get("Return", 0) != original_types.get(
            "Return", 0
        )

        # Validation (assert or raise)
        patterns["validation"] = patched_types.get("Raise", 0) > original_types.get(
            "Raise", 0
        )

        # Assertion added
        patterns["assertion"] = patched_types.get("Assert", 0) > original_types.get(
            "Assert", 0
        )

        return patterns

    def _count_node_types(self, tree: ast.AST) -> Dict[str, int]:
        """Count occurrences of each node type in AST."""
        counts: Dict[str, int] = defaultdict(int)

        for node in ast.walk(tree):
            counts[type(node).__name__] += 1

        return dict(counts)

    def _compute_edit_operations(
        self, original_tree: ast.AST, patched_tree: ast.AST
    ) -> List[str]:
        """Compute high-level edit operations."""
        original_types = self._count_node_types(original_tree)
        patched_types = self._count_node_types(patched_tree)

        ops = []

        all_types = set(original_types.keys()) | set(patched_types.keys())

        for node_type in all_types:
            orig_count = original_types.get(node_type, 0)
            patch_count = patched_types.get(node_type, 0)

            if patch_count > orig_count:
                ops.append(f"insert:{node_type}")
            elif patch_count < orig_count:
                ops.append(f"delete:{node_type}")

        return ops

    def _compute_node_delta(
        self, original_tree: ast.AST, patched_tree: ast.AST
    ) -> Dict[str, int]:
        """Compute the difference in node type counts."""
        original_types = self._count_node_types(original_tree)
        patched_types = self._count_node_types(patched_tree)

        delta = {}
        all_types = set(original_types.keys()) | set(patched_types.keys())

        for node_type in all_types:
            diff = patched_types.get(node_type, 0) - original_types.get(node_type, 0)
            if diff != 0:
                delta[node_type] = diff

        return delta

    def _find_modified_functions(
        self,
        original_code: str,
        patched_code: str,
        original_funcs: List[str],
        patched_funcs: List[str],
    ) -> List[str]:
        """Find functions that exist in both but were modified."""
        common = set(original_funcs) & set(patched_funcs)

        # For now, assume common functions may have been modified
        # A more sophisticated version would extract and compare function bodies
        return list(common)

    def _empty_signature(self, patch_id: str) -> ASTChangeSignature:
        """Return empty signature when AST analysis fails."""
        return ASTChangeSignature(patch_id=patch_id)


def compute_ast_similarity(sig1: ASTChangeSignature, sig2: ASTChangeSignature) -> float:
    """Compute similarity between two AST change signatures.

    Args:
        sig1: First signature
        sig2: Second signature

    Returns:
        Similarity score between 0 and 1
    """
    # Compare edit operations
    ops1 = set(sig1.edit_operations)
    ops2 = set(sig2.edit_operations)

    if not ops1 and not ops2:
        return 1.0  # Both empty = same
    if not ops1 or not ops2:
        return 0.0

    jaccard = len(ops1 & ops2) / len(ops1 | ops2)

    # Bonus for same structural patterns
    pattern_matches = sum(
        [
            sig1.added_guard_clause == sig2.added_guard_clause,
            sig1.added_try_except == sig2.added_try_except,
            sig1.added_loop == sig2.added_loop,
            sig1.added_validation == sig2.added_validation,
            sig1.added_assertion == sig2.added_assertion,
        ]
    )
    pattern_match = pattern_matches / 5

    return 0.7 * jaccard + 0.3 * pattern_match


def extract_code_from_patch(
    patch: str,
    original_files: Dict[str, str],
) -> Dict[str, str]:
    """Apply a patch and return the resulting file contents.

    This is a simplified version that extracts the patched content
    from the diff itself. For real application, use a proper patch tool.

    Args:
        patch: The unified diff patch
        original_files: Dict of {filepath: original_content}

    Returns:
        Dict of {filepath: patched_content}
    """
    # This is a simplified implementation
    # In production, use unidiff library to properly apply patches
    patched_files = dict(original_files)

    current_file = None
    current_content = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("+++ "):
            match = re.match(r"\+\+\+ (?:b/)?(.+)", line)
            if match:
                current_file = match.group(1)
                if current_file in original_files:
                    current_content = original_files[current_file].split("\n")
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and current_file:
            if line.startswith("+") and not line.startswith("+++"):
                current_content.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                # Remove line (simplified - just skip)
                pass
            elif line.startswith(" "):
                current_content.append(line[1:])

    if current_file and current_content:
        patched_files[current_file] = "\n".join(current_content)

    return patched_files
