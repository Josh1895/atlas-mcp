"""
Metrics collection for swarm operations.

Provides detailed metrics tracking for latency, cost, and consensus rates.
Thread-safe and supports aggregation across multiple runs.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class LatencyHistogram:
    """
    Simple histogram for tracking latency distributions.

    Buckets are in milliseconds.
    """
    buckets: List[float] = field(default_factory=lambda: [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    counts: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    total_count: int = 0
    total_sum: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0

    def observe(self, value_ms: float) -> None:
        """Record an observation."""
        self.total_count += 1
        self.total_sum += value_ms
        self.min_value = min(self.min_value, value_ms)
        self.max_value = max(self.max_value, value_ms)

        # Find bucket
        for bucket in self.buckets:
            if value_ms <= bucket:
                self.counts[bucket] += 1
                break
        else:
            # Larger than all buckets
            self.counts[float('inf')] += 1

    @property
    def mean(self) -> float:
        """Get mean latency."""
        if self.total_count == 0:
            return 0.0
        return self.total_sum / self.total_count

    def percentile(self, p: float) -> float:
        """Estimate percentile from histogram buckets."""
        if self.total_count == 0:
            return 0.0

        target = self.total_count * (p / 100.0)
        running_count = 0

        for bucket in sorted(self.buckets + [float('inf')]):
            running_count += self.counts.get(bucket, 0)
            if running_count >= target:
                return bucket

        return self.max_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "count": self.total_count,
            "sum_ms": self.total_sum,
            "mean_ms": self.mean,
            "min_ms": self.min_value if self.total_count > 0 else 0,
            "max_ms": self.max_value,
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
        }


@dataclass
class Counter:
    """Thread-safe counter."""
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        with self._lock:
            self._value += value

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value


@dataclass
class Gauge:
    """Thread-safe gauge."""
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= value

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value


class SwarmMetricsCollector:
    """
    Collects and aggregates metrics across swarm runs.

    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Run counters
        self.runs_total = Counter()
        self.runs_successful = Counter()
        self.runs_failed = Counter()
        self.runs_timeout = Counter()
        self.runs_budget_exceeded = Counter()

        # Consensus metrics
        self.consensus_reached_total = Counter()
        self.consensus_not_reached_total = Counter()
        self.confidence_scores: List[float] = []

        # Latency histograms
        self.run_duration_histogram = LatencyHistogram()
        self.generation_latency_histogram = LatencyHistogram()
        self.consensus_latency_histogram = LatencyHistogram()

        # Cost tracking
        self.total_cost_usd = Counter()
        self.total_tokens = Counter()
        self.cost_per_run: List[float] = []

        # Agent metrics
        self.agents_succeeded = Counter()
        self.agents_failed = Counter()
        self.agents_per_run: List[int] = []

        # Cluster metrics
        self.clusters_per_run: List[int] = []

        # Active runs gauge
        self.active_runs = Gauge()

        # Timestamps
        self._start_time = time.time()

    def record_run_started(self) -> None:
        """Record that a run has started."""
        self.runs_total.inc()
        self.active_runs.inc()

    def record_run_completed(
        self,
        status: str,
        duration_ms: int,
        consensus_reached: bool,
        confidence_score: float,
        cost_usd: float,
        tokens_used: int,
        agents_succeeded: int,
        agents_failed: int,
        cluster_count: int,
    ) -> None:
        """Record run completion metrics."""
        self.active_runs.dec()

        # Status counters
        if status == "completed":
            self.runs_successful.inc()
        elif status == "failed":
            self.runs_failed.inc()
        elif status == "timeout":
            self.runs_timeout.inc()
        elif status == "budget_exceeded":
            self.runs_budget_exceeded.inc()

        # Latency
        self.run_duration_histogram.observe(duration_ms)

        # Consensus
        if consensus_reached:
            self.consensus_reached_total.inc()
        else:
            self.consensus_not_reached_total.inc()

        with self._lock:
            self.confidence_scores.append(confidence_score)

        # Cost
        self.total_cost_usd.inc(cost_usd)
        self.total_tokens.inc(tokens_used)
        with self._lock:
            self.cost_per_run.append(cost_usd)

        # Agents
        self.agents_succeeded.inc(agents_succeeded)
        self.agents_failed.inc(agents_failed)
        with self._lock:
            self.agents_per_run.append(agents_succeeded + agents_failed)

        # Clusters
        with self._lock:
            self.clusters_per_run.append(cluster_count)

    def record_generation_latency(self, duration_ms: int) -> None:
        """Record agent generation latency."""
        self.generation_latency_histogram.observe(duration_ms)

    def record_consensus_latency(self, duration_ms: int) -> None:
        """Record consensus calculation latency."""
        self.consensus_latency_histogram.observe(duration_ms)

    def get_consensus_rate(self) -> float:
        """Get the consensus success rate."""
        total = self.consensus_reached_total.value + self.consensus_not_reached_total.value
        if total == 0:
            return 0.0
        return self.consensus_reached_total.value / total

    def get_success_rate(self) -> float:
        """Get the run success rate."""
        total = self.runs_total.value
        if total == 0:
            return 0.0
        return self.runs_successful.value / total

    def get_mean_confidence(self) -> float:
        """Get mean confidence score."""
        with self._lock:
            if not self.confidence_scores:
                return 0.0
            return sum(self.confidence_scores) / len(self.confidence_scores)

    def get_mean_cost(self) -> float:
        """Get mean cost per run."""
        with self._lock:
            if not self.cost_per_run:
                return 0.0
            return sum(self.cost_per_run) / len(self.cost_per_run)

    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        uptime_seconds = time.time() - self._start_time

        return {
            "uptime_seconds": uptime_seconds,
            "runs": {
                "total": self.runs_total.value,
                "successful": self.runs_successful.value,
                "failed": self.runs_failed.value,
                "timeout": self.runs_timeout.value,
                "budget_exceeded": self.runs_budget_exceeded.value,
                "active": self.active_runs.value,
                "success_rate": self.get_success_rate(),
            },
            "consensus": {
                "reached": self.consensus_reached_total.value,
                "not_reached": self.consensus_not_reached_total.value,
                "rate": self.get_consensus_rate(),
                "mean_confidence": self.get_mean_confidence(),
            },
            "latency": {
                "run_duration": self.run_duration_histogram.to_dict(),
                "generation": self.generation_latency_histogram.to_dict(),
                "consensus": self.consensus_latency_histogram.to_dict(),
            },
            "cost": {
                "total_usd": self.total_cost_usd.value,
                "mean_per_run": self.get_mean_cost(),
                "total_tokens": self.total_tokens.value,
            },
            "agents": {
                "total_succeeded": self.agents_succeeded.value,
                "total_failed": self.agents_failed.value,
                "success_rate": (
                    self.agents_succeeded.value /
                    max(1, self.agents_succeeded.value + self.agents_failed.value)
                ),
            },
        }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self.__init__()


# Global metrics collector instance
_global_metrics: Optional[SwarmMetricsCollector] = None


def get_metrics_collector() -> SwarmMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SwarmMetricsCollector()
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics (for testing)."""
    global _global_metrics
    _global_metrics = None
