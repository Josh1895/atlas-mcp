"""
Schema definitions for Atlas Swarm Integration.

Defines dataclasses for swarm requests, results, and intermediate data structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime


class SwarmMode(Enum):
    """Mode of swarm operation."""
    PATCH = "patch"
    ANSWER = "answer"


@dataclass
class BudgetConfig:
    """Budget limits for a swarm run (P-005: Budget Enforcement)."""
    timeout_seconds: int = 900
    max_cost_usd: float = 5.0
    max_tokens: int = 500000
    max_tool_calls: int = 100


@dataclass
class ToolConfig:
    """Tool availability configuration for swarm agents."""
    enable_web_search: bool = True  # Enabled by default (free)
    enable_context7: bool = True  # Default True; actually enabled only if CONTEXT7_API_KEY present (via config flag)
    enable_repo_search: bool = True
    enable_agent_mcp_rag: bool = True
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    """Memory write-back configuration."""
    write_back_to_task: bool = True
    write_back_to_project_context: bool = False
    index_into_rag: bool = False


@dataclass
class RepoConfig:
    """Repository configuration for patch mode (local paths only)."""
    branch: Optional[str] = None
    commit: Optional[str] = None
    test_command: Optional[str] = None


@dataclass
class SwarmConfig:
    """Configuration for swarm agent behavior."""
    agent_count: int = 10  # 10 agents for robust K=3 consensus
    consensus_k: int = 2  # Votes ahead needed
    prompt_styles: List[str] = field(default_factory=lambda: [
        "SENIOR_ENGINEER",
        "SECURITY_FOCUSED",
        "PERFORMANCE_EXPERT",
        "JUNIOR_DEVELOPER",
        "CODE_REVIEWER",
    ])
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.7


@dataclass
class SwarmRequest:
    """
    Input schema for run_swarm_consensus tool.

    Matches the API contract in spec.md.
    """
    token: str
    task_id: Optional[str] = None
    description: Optional[str] = None
    mode: SwarmMode = SwarmMode.PATCH
    repo: RepoConfig = field(default_factory=RepoConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


@dataclass
class AgentOutput:
    """Output from a single swarm agent."""
    agent_id: str
    prompt_style: str
    output_text: str
    explanation: Optional[str] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None
    test_result: Optional[Dict[str, Any]] = None
    quality_score: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0


@dataclass
class ClusterInfo:
    """Information about a cluster of similar outputs."""
    cluster_id: str
    size: int
    vote_count: int
    representative_output: str
    member_agent_ids: List[str] = field(default_factory=list)
    behavioral_pass: Optional[bool] = None  # True if tests pass for this cluster


@dataclass
class SwarmMetrics:
    """Metrics collected during a swarm run."""
    duration_ms: int = 0
    cost_usd: float = 0.0
    tokens_total: int = 0
    tool_calls_count: int = 0
    cache_hits: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0


@dataclass
class SwarmResult:
    """
    Output schema for run_swarm_consensus tool.

    Matches the API contract in spec.md.
    """
    run_id: str
    task_id: Optional[str] = None
    mode: SwarmMode = SwarmMode.PATCH
    status: str = "completed"  # running, completed, failed, timeout, budget_exceeded
    consensus_reached: bool = False
    selected_output: str = ""
    selected_variant_id: Optional[str] = None
    confidence_score: float = 0.0
    vote_counts: Dict[str, int] = field(default_factory=dict)
    clusters: List[ClusterInfo] = field(default_factory=list)
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    metrics: SwarmMetrics = field(default_factory=SwarmMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "mode": self.mode.value,
            "status": self.status,
            "consensus_reached": self.consensus_reached,
            "selected_output": self.selected_output,
            "selected_variant_id": self.selected_variant_id,
            "confidence_score": self.confidence_score,
            "vote_counts": self.vote_counts,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "vote_count": c.vote_count,
                    "representative_output": c.representative_output,
                    "member_agent_ids": c.member_agent_ids,
                    "behavioral_pass": c.behavioral_pass,
                }
                for c in self.clusters
            ],
            "metrics": {
                "duration_ms": self.metrics.duration_ms,
                "cost_usd": self.metrics.cost_usd,
                "tokens_total": self.metrics.tokens_total,
                "tool_calls_count": self.metrics.tool_calls_count,
                "cache_hits": self.metrics.cache_hits,
                "agents_succeeded": self.metrics.agents_succeeded,
                "agents_failed": self.metrics.agents_failed,
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
