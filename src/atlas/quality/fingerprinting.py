"""Token-based fingerprinting for fast patch similarity detection.

Uses winnowing algorithm (same as MOSS plagiarism detection) to create
fingerprints for fast comparison of code patches.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PatchFingerprint:
    """Fingerprint for a code patch."""

    patch_id: str
    raw_diff: str

    # Computed fields
    files_touched: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    token_hashes: Set[str] = field(default_factory=set)  # Winnowing fingerprints
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)

    # Metrics
    lines_added: int = 0
    lines_removed: int = 0
    hunks_count: int = 0


class TokenFingerprinter:
    """Creates token-based fingerprints for code patches.

    Uses winnowing algorithm (same as MOSS plagiarism detection).
    This provides fast, coarse-grained similarity detection.
    """

    def __init__(
        self,
        ngram_size: int = 5,
        window_size: int = 4,
    ):
        """Initialize the fingerprinter.

        Args:
            ngram_size: Size of n-grams to create
            window_size: Winnowing window size
        """
        self.ngram_size = ngram_size
        self.window_size = window_size

    def fingerprint(self, patch: str, patch_id: str = "") -> PatchFingerprint:
        """Create fingerprint for a patch.

        Args:
            patch: The unified diff patch text
            patch_id: Optional identifier for the patch

        Returns:
            PatchFingerprint with computed fingerprints
        """
        # Parse diff to extract actual code changes
        files_touched = self._extract_files(patch)
        added_lines, removed_lines = self._extract_changes(patch)

        # Normalize code (remove whitespace variations, standardize)
        normalized = self._normalize_code("\n".join(added_lines))

        # Tokenize
        tokens = self._tokenize(normalized)

        # Compute winnowing fingerprints
        token_hashes = self._winnow(tokens)

        return PatchFingerprint(
            patch_id=patch_id,
            raw_diff=patch,
            files_touched=files_touched,
            tokens=tokens,
            token_hashes=token_hashes,
            added_lines=added_lines,
            removed_lines=removed_lines,
            lines_added=len(added_lines),
            lines_removed=len(removed_lines),
            hunks_count=patch.count("@@"),
        )

    def _normalize_code(self, code: str) -> str:
        """Normalize code to reduce superficial differences.

        - Removes comments
        - Standardizes whitespace
        - Lowercases for comparison
        """
        # Remove single-line comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line strings/comments (simplified)
        code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r"\s+", " ", code)

        # Lowercase for comparison
        code = code.lower()

        return code.strip()

    def _tokenize(self, code: str) -> List[str]:
        """Split code into tokens."""
        # Tokenize by splitting on non-alphanumeric chars
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s\w]", code)
        return tokens

    def _winnow(self, tokens: List[str]) -> Set[str]:
        """Winnowing algorithm for fingerprint selection.

        Creates n-grams, hashes them, then uses sliding window
        to select representative fingerprints.
        """
        if len(tokens) < self.ngram_size:
            # Too short, just hash the whole thing
            if tokens:
                return {self._hash(" ".join(tokens))}
            return set()

        # Create n-gram hashes
        ngram_hashes = []
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i : i + self.ngram_size])
            ngram_hashes.append(self._hash(ngram))

        if len(ngram_hashes) < self.window_size:
            return set(ngram_hashes)

        # Winnowing: select minimum hash in each window
        fingerprints = set()
        for i in range(len(ngram_hashes) - self.window_size + 1):
            window = ngram_hashes[i : i + self.window_size]
            fingerprints.add(min(window))

        return fingerprints

    def _hash(self, text: str) -> str:
        """Hash a string to 16-character hex digest."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _extract_files(self, patch: str) -> List[str]:
        """Extract list of files touched by patch."""
        files = re.findall(r"^(?:\+\+\+|---) [ab]/(.+)$", patch, re.MULTILINE)
        return list(set(files))

    def _extract_changes(self, patch: str) -> Tuple[List[str], List[str]]:
        """Extract added and removed lines from patch."""
        added = []
        removed = []

        for line in patch.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])

        return added, removed


def compute_similarity(fp1: PatchFingerprint, fp2: PatchFingerprint) -> float:
    """Compute Jaccard similarity between two fingerprints.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint

    Returns:
        Similarity between 0 (completely different) and 1 (identical)
    """
    if not fp1.token_hashes or not fp2.token_hashes:
        return 0.0

    intersection = len(fp1.token_hashes & fp2.token_hashes)
    union = len(fp1.token_hashes | fp2.token_hashes)

    if union == 0:
        return 0.0

    return intersection / union


def cluster_by_similarity(
    fingerprints: List[PatchFingerprint],
    threshold: float = 0.3,
) -> List[List[PatchFingerprint]]:
    """Cluster fingerprints by similarity using agglomerative approach.

    Args:
        fingerprints: List of patch fingerprints
        threshold: Minimum Jaccard similarity to be in same cluster

    Returns:
        List of clusters, where each cluster is a list of fingerprints
    """
    if not fingerprints:
        return []

    if len(fingerprints) == 1:
        return [fingerprints]

    # Start with each fingerprint in its own cluster
    clusters: List[Set[int]] = [{i} for i in range(len(fingerprints))]

    # Compute pairwise similarities and merge
    n = len(fingerprints)
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(fingerprints[i], fingerprints[j])

            if sim >= threshold:
                # Find clusters containing i and j
                cluster_i = None
                cluster_j = None

                for c in clusters:
                    if i in c:
                        cluster_i = c
                    if j in c:
                        cluster_j = c

                # Merge clusters if they're different
                if cluster_i is not None and cluster_j is not None:
                    if cluster_i is not cluster_j:
                        cluster_i.update(cluster_j)
                        clusters.remove(cluster_j)

    # Convert to list of fingerprints
    return [[fingerprints[i] for i in sorted(cluster)] for cluster in clusters]


def fingerprint_patches(
    patches: dict[str, str],
    ngram_size: int = 5,
    window_size: int = 4,
) -> List[PatchFingerprint]:
    """Fingerprint multiple patches.

    Args:
        patches: Dict of {patch_id: patch_content}
        ngram_size: N-gram size for fingerprinting
        window_size: Winnowing window size

    Returns:
        List of PatchFingerprint objects
    """
    fingerprinter = TokenFingerprinter(ngram_size, window_size)

    fingerprints = []
    for patch_id, patch_content in patches.items():
        fp = fingerprinter.fingerprint(patch_content, patch_id)
        fingerprints.append(fp)

    return fingerprints
