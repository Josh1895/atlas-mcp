"""
Atlas Swarm Integration for Agent-MCP

This module provides multi-agent consensus capabilities by integrating
Atlas swarm functionality into Agent-MCP.

Feature-flagged via SWARM_ENABLED environment variable.
"""

from .schemas import (
    SwarmMode,
    SwarmRequest,
    SwarmResult,
    BudgetConfig,
    ToolConfig,
    MemoryConfig,
    RepoConfig,
    SwarmConfig,
    AgentOutput,
    ClusterInfo,
    SwarmMetrics,
)

from .budget_manager import (
    BudgetManager,
    BudgetState,
    BudgetExceededError,
    IncrementalBudgetChecker,
)

from .repo_resolver import (
    RepoResolver,
    ResolvedRepo,
    RepoResolverError,
    temporary_repo,
)

from .context_builder import (
    ContextBuilder,
    SwarmContext,
    ContextChunk,
    build_swarm_context,
)

from .memory_writer import (
    MemoryWriter,
    write_swarm_result,
)

from .tool_router import (
    ToolRouter,
    CircuitBreaker,
    RateLimiter,
    ToolRouterError,
    ToolNotAllowedError,
    CircuitOpenError,
    RateLimitExceededError,
    create_default_router,
)

from .atlas_adapter import (
    AtlasAdapter,
)

from .consensus_engine import (
    ConsensusEngine,
    ConsensusResult,
)

from .answer_clustering import (
    AnswerClusterer,
    AnswerCluster,
    CitationInfo,
    extract_citations,
    has_citations,
    cluster_answers,
)

from .swarm_manager import (
    SwarmManager,
    SwarmLogger,
    run_swarm,
)

from .metrics import (
    SwarmMetricsCollector,
    LatencyHistogram,
    get_metrics_collector,
    reset_metrics,
)

__all__ = [
    # Schemas
    "SwarmMode",
    "SwarmRequest",
    "SwarmResult",
    "BudgetConfig",
    "ToolConfig",
    "MemoryConfig",
    "RepoConfig",
    "SwarmConfig",
    "AgentOutput",
    "ClusterInfo",
    "SwarmMetrics",
    # Budget Manager
    "BudgetManager",
    "BudgetState",
    "BudgetExceededError",
    "IncrementalBudgetChecker",
    # Repo Resolver
    "RepoResolver",
    "ResolvedRepo",
    "RepoResolverError",
    "temporary_repo",
    # Context Builder
    "ContextBuilder",
    "SwarmContext",
    "ContextChunk",
    "build_swarm_context",
    # Memory Writer
    "MemoryWriter",
    "write_swarm_result",
    # Tool Router
    "ToolRouter",
    "CircuitBreaker",
    "RateLimiter",
    "ToolRouterError",
    "ToolNotAllowedError",
    "CircuitOpenError",
    "RateLimitExceededError",
    "create_default_router",
    # Atlas Adapter
    "AtlasAdapter",
    # Consensus Engine
    "ConsensusEngine",
    "ConsensusResult",
    # Answer Clustering
    "AnswerClusterer",
    "AnswerCluster",
    "CitationInfo",
    "extract_citations",
    "has_citations",
    "cluster_answers",
    # Swarm Manager
    "SwarmManager",
    "SwarmLogger",
    "run_swarm",
    # Metrics
    "SwarmMetricsCollector",
    "LatencyHistogram",
    "get_metrics_collector",
    "reset_metrics",
]
