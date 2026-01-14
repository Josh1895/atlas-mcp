"""
Tests for swarm feature.

Test modules:
- test_swarm_manager: Unit tests for SwarmManager orchestration
- test_consensus_engine: Unit tests for consensus voting/clustering
- test_tool_router: Unit tests for tool routing and permissions
- test_integration: End-to-end integration tests
- test_replay_harness: Tests for golden test replay functionality
- test_backward_compatibility: Tests for SWARM_ENABLED=false behavior
"""

from .replay_harness import (
    SwarmRecorder,
    ReplayHarness,
    GoldenTestCase,
    GoldenTestRunner,
    create_golden_from_outputs,
)

__all__ = [
    "SwarmRecorder",
    "ReplayHarness",
    "GoldenTestCase",
    "GoldenTestRunner",
    "create_golden_from_outputs",
]
