"""
Answer Clustering for swarm answer mode.

Uses embedding-based clustering for answer mode responses.
Integrates with Agent-MCP's embedding infrastructure (AD-006).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re

from ...core.config import logger as config_logger

# Try to import numpy/sklearn for clustering
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    config_logger.warning("sklearn not available, using fallback clustering")

from .schemas import AgentOutput, ClusterInfo

logger = logging.getLogger(__name__)


@dataclass
class AnswerCluster:
    """A cluster of similar answers."""
    cluster_id: str
    answers: List[AgentOutput] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    medoid_idx: int = 0  # Index of the most representative answer

    @property
    def size(self) -> int:
        return len(self.answers)

    @property
    def medoid(self) -> Optional[AgentOutput]:
        """Get the medoid (most representative answer)."""
        if self.answers:
            return self.answers[self.medoid_idx]
        return None


@dataclass
class CitationInfo:
    """Extracted citation from an answer."""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


def extract_citations(text: str) -> List[CitationInfo]:
    """
    Extract citations/URLs from answer text.

    Looks for:
    - URLs in markdown links: [title](url)
    - Raw URLs: http(s)://...
    - Reference markers: [1], [2], etc.
    """
    citations = []

    # Markdown links
    md_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    for match in re.finditer(md_pattern, text):
        citations.append(CitationInfo(
            url=match.group(2),
            title=match.group(1),
        ))

    # Raw URLs (not already captured)
    captured_urls = {c.url for c in citations}
    url_pattern = r'https?://[^\s\)\]<>]+'
    for match in re.finditer(url_pattern, text):
        url = match.group(0).rstrip('.,;:')
        if url not in captured_urls:
            citations.append(CitationInfo(url=url))
            captured_urls.add(url)

    return citations


def has_citations(text: str) -> bool:
    """Check if text contains citations."""
    return len(extract_citations(text)) > 0


class AnswerClusterer:
    """
    Clusters answers by semantic similarity using embeddings.

    Uses k-means clustering with cosine similarity on embeddings.
    Falls back to simple text hashing if sklearn is not available.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        min_cluster_size: int = 1,
    ):
        """
        Initialize the answer clusterer.

        Args:
            n_clusters: Target number of clusters (will be adjusted based on data)
            min_cluster_size: Minimum answers per cluster
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

    async def cluster(
        self,
        outputs: List[AgentOutput],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[ClusterInfo]:
        """
        Cluster answers by similarity.

        Args:
            outputs: Agent outputs to cluster
            embeddings: Optional pre-computed embeddings. If not provided,
                       will use fallback text-based clustering.

        Returns:
            List of ClusterInfo with cluster assignments
        """
        if not outputs:
            return []

        if len(outputs) == 1:
            # Single answer - one cluster
            outputs[0].cluster_id = "cluster_0"
            return [ClusterInfo(
                cluster_id="cluster_0",
                size=1,
                vote_count=1,
                representative_output=outputs[0].output_text,
                member_agent_ids=[outputs[0].agent_id],
            )]

        if embeddings and SKLEARN_AVAILABLE and len(embeddings) == len(outputs):
            return self._cluster_with_embeddings(outputs, embeddings)
        else:
            return self._cluster_by_text_hash(outputs)

    def _cluster_with_embeddings(
        self,
        outputs: List[AgentOutput],
        embeddings: List[List[float]],
    ) -> List[ClusterInfo]:
        """Cluster using k-means on embeddings."""
        # Convert to numpy array
        X = np.array(embeddings)

        # Adjust n_clusters based on data size
        n_clusters = min(self.n_clusters, len(outputs))

        # Run k-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(X)

        # Group outputs by cluster
        cluster_outputs: Dict[int, List[tuple[int, AgentOutput]]] = {}
        for idx, (output, label) in enumerate(zip(outputs, labels)):
            if label not in cluster_outputs:
                cluster_outputs[label] = []
            cluster_outputs[label].append((idx, output))

        # Create ClusterInfo objects
        clusters = []
        for label, members in cluster_outputs.items():
            cluster_id = f"cluster_{label}"

            # Find medoid (point closest to centroid)
            centroid = kmeans.cluster_centers_[label]
            member_indices = [m[0] for m in members]
            member_embeddings = X[member_indices]

            # Compute distances to centroid
            distances = np.linalg.norm(member_embeddings - centroid, axis=1)
            medoid_local_idx = int(np.argmin(distances))
            medoid_output = members[medoid_local_idx][1]

            # Assign cluster IDs to outputs
            for _, output in members:
                output.cluster_id = cluster_id

            cluster = ClusterInfo(
                cluster_id=cluster_id,
                size=len(members),
                vote_count=len(members),
                representative_output=medoid_output.output_text,
                member_agent_ids=[m[1].agent_id for m in members],
            )
            clusters.append(cluster)

        return sorted(clusters, key=lambda c: -c.size)

    def _cluster_by_text_hash(
        self,
        outputs: List[AgentOutput],
    ) -> List[ClusterInfo]:
        """
        Fallback clustering by text content hash.

        Groups identical or very similar answers together.
        """
        # Normalize and hash answers
        groups: Dict[str, List[AgentOutput]] = {}

        for output in outputs:
            # Normalize: lowercase, strip whitespace, remove punctuation
            normalized = output.output_text.lower().strip()
            # Take first 500 chars for grouping
            key_text = normalized[:500]
            key_hash = hashlib.md5(key_text.encode()).hexdigest()[:8]

            if key_hash not in groups:
                groups[key_hash] = []
            groups[key_hash].append(output)

        # Convert to ClusterInfo
        clusters = []
        for idx, (hash_key, members) in enumerate(groups.items()):
            cluster_id = f"cluster_{idx}"

            # Pick shortest answer as representative
            representative = min(members, key=lambda o: len(o.output_text))

            # Assign cluster IDs
            for output in members:
                output.cluster_id = cluster_id

            cluster = ClusterInfo(
                cluster_id=cluster_id,
                size=len(members),
                vote_count=len(members),
                representative_output=representative.output_text,
                member_agent_ids=[m.agent_id for m in members],
            )
            clusters.append(cluster)

        return sorted(clusters, key=lambda c: -c.size)

    def select_best_cluster(
        self,
        clusters: List[ClusterInfo],
        prefer_citations: bool = True,
    ) -> Optional[ClusterInfo]:
        """
        Select the best cluster based on voting rules.

        Args:
            clusters: Clusters to choose from
            prefer_citations: Prefer clusters with citations in answers

        Returns:
            Best cluster or None
        """
        if not clusters:
            return None

        # Sort by size (votes)
        sorted_clusters = sorted(clusters, key=lambda c: -c.size)

        if prefer_citations:
            # Among top clusters, prefer those with citations
            top_size = sorted_clusters[0].size
            top_clusters = [c for c in sorted_clusters if c.size == top_size]

            for cluster in top_clusters:
                if has_citations(cluster.representative_output):
                    return cluster

        return sorted_clusters[0]


async def cluster_answers(
    outputs: List[AgentOutput],
    embeddings: Optional[List[List[float]]] = None,
    n_clusters: int = 3,
) -> List[ClusterInfo]:
    """
    Convenience function to cluster answers.

    Args:
        outputs: Agent outputs to cluster
        embeddings: Optional embeddings
        n_clusters: Target number of clusters

    Returns:
        List of ClusterInfo
    """
    clusterer = AnswerClusterer(n_clusters=n_clusters)
    return await clusterer.cluster(outputs, embeddings)
