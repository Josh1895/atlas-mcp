"""
Offline Replay Harness for Golden Tests.

Provides infrastructure for recording swarm runs and replaying them
deterministically for regression testing. Golden tests ensure that
swarm behavior remains consistent across code changes.

Usage:
    # Record a run
    recorder = SwarmRecorder()
    with recorder.recording("test_case_001"):
        result = await run_swarm(request)
    recorder.save("golden/test_case_001.json")

    # Replay and verify
    harness = ReplayHarness("golden/test_case_001.json")
    result = await harness.replay()
    assert harness.matches_golden(result)
"""

import json
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Awaitable
from unittest.mock import AsyncMock, MagicMock, patch
import copy

from ..schemas import (
    SwarmRequest,
    SwarmResult,
    SwarmMode,
    AgentOutput,
    ClusterInfo,
)
from ..swarm_manager import SwarmManager
from ..consensus_engine import ConsensusResult


@dataclass
class RecordedCall:
    """A recorded function call with inputs and outputs."""

    function_name: str
    timestamp: str
    inputs: Dict[str, Any]
    output: Any
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class GoldenTestCase:
    """A complete recorded swarm run for golden testing."""

    name: str
    description: str
    recorded_at: str
    request: Dict[str, Any]
    expected_result: Dict[str, Any]
    recorded_calls: List[RecordedCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "recorded_at": self.recorded_at,
            "request": self.request,
            "expected_result": self.expected_result,
            "recorded_calls": [asdict(c) for c in self.recorded_calls],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenTestCase":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            recorded_at=data["recorded_at"],
            request=data["request"],
            expected_result=data["expected_result"],
            recorded_calls=[
                RecordedCall(**c) for c in data.get("recorded_calls", [])
            ],
            metadata=data.get("metadata", {}),
        )


class SwarmRecorder:
    """Records swarm run execution for replay testing."""

    def __init__(self):
        self._recording = False
        self._current_case: Optional[GoldenTestCase] = None
        self._calls: List[RecordedCall] = []
        self._start_time: Optional[datetime] = None

    def start_recording(self, name: str, description: str = "") -> None:
        """Start recording a new test case."""
        if self._recording:
            raise RuntimeError("Already recording")

        self._recording = True
        self._calls = []
        self._start_time = datetime.utcnow()
        self._current_case = GoldenTestCase(
            name=name,
            description=description,
            recorded_at=self._start_time.isoformat(),
            request={},
            expected_result={},
        )

    def stop_recording(self, request: SwarmRequest, result: SwarmResult) -> GoldenTestCase:
        """Stop recording and return the test case."""
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False
        self._current_case.request = self._serialize_request(request)
        self._current_case.expected_result = self._serialize_result(result)
        self._current_case.recorded_calls = self._calls.copy()

        case = self._current_case
        self._current_case = None
        self._calls = []

        return case

    def record_call(
        self,
        function_name: str,
        inputs: Dict[str, Any],
        output: Any,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Record a function call."""
        if not self._recording:
            return

        self._calls.append(RecordedCall(
            function_name=function_name,
            timestamp=datetime.utcnow().isoformat(),
            inputs=self._make_serializable(inputs),
            output=self._make_serializable(output),
            duration_ms=duration_ms,
            error=error,
        ))

    def save(self, case: GoldenTestCase, filepath: Path) -> None:
        """Save a golden test case to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(case.to_dict(), f, indent=2)

    def load(self, filepath: Path) -> GoldenTestCase:
        """Load a golden test case from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return GoldenTestCase.from_dict(data)

    def _serialize_request(self, request: SwarmRequest) -> Dict[str, Any]:
        """Serialize SwarmRequest to dict."""
        return {
            "token": "[REDACTED]",  # Don't store tokens
            "task_id": request.task_id,
            "description": request.description,
            "mode": request.mode.value if request.mode else "patch",
            "budgets": {
                "max_cost_usd": request.budgets.max_cost_usd,
                "max_tokens": request.budgets.max_tokens,
                "timeout_seconds": request.budgets.timeout_seconds,
            } if request.budgets else {},
            "swarm": {
                "agent_count": request.swarm.agent_count,
                "consensus_k": request.swarm.consensus_k,
            } if request.swarm else {},
        }

    def _serialize_result(self, result: SwarmResult) -> Dict[str, Any]:
        """Serialize SwarmResult to dict."""
        return {
            "run_id": result.run_id,
            "mode": result.mode.value if result.mode else "patch",
            "status": result.status,
            "consensus_reached": result.consensus_reached,
            "confidence_score": result.confidence_score,
            "selected_output": result.selected_output,
            "agent_outputs": [
                {
                    "agent_id": o.agent_id,
                    "prompt_style": o.prompt_style,
                    "output_text": o.output_text,
                    "is_valid": o.is_valid,
                }
                for o in (result.agent_outputs or [])
            ],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "representative_output": c.representative_output,
                }
                for c in (result.clusters or [])
            ],
        }

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        return str(obj)


class ReplayHarness:
    """Replays recorded swarm runs for golden testing."""

    def __init__(self, golden_case: GoldenTestCase):
        self.golden = golden_case
        self._call_index = 0
        self._recorded_outputs: Dict[str, Any] = {}

        # Build lookup of recorded outputs by function name
        for call in golden_case.recorded_calls:
            key = f"{call.function_name}_{self._recorded_outputs.get(call.function_name, 0)}"
            self._recorded_outputs[key] = call.output

    @classmethod
    def from_file(cls, filepath: Path) -> "ReplayHarness":
        """Create harness from golden test file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(GoldenTestCase.from_dict(data))

    async def replay(self) -> SwarmResult:
        """Replay the recorded run with mocked responses."""
        request = self._reconstruct_request()

        manager = SwarmManager(request)

        # Mock external calls with recorded responses
        with self._mock_atlas_adapter(manager):
            with self._mock_database_calls():
                with patch("agent_mcp.features.swarm.swarm_manager.SWARM_ENABLED", True):
                    with patch.object(manager.memory_writer, "write_result", new_callable=AsyncMock):
                        result = await manager.run()

        return result

    def matches_golden(
        self,
        result: SwarmResult,
        check_outputs: bool = True,
        check_consensus: bool = True,
        check_clusters: bool = False,
    ) -> bool:
        """Check if result matches the golden expected result."""
        expected = self.golden.expected_result

        # Check status
        if result.status != expected.get("status"):
            return False

        # Check consensus
        if check_consensus:
            if result.consensus_reached != expected.get("consensus_reached"):
                return False

        # Check outputs match
        if check_outputs:
            if result.selected_output != expected.get("selected_output"):
                return False

        # Check clusters
        if check_clusters:
            expected_clusters = expected.get("clusters", [])
            if len(result.clusters or []) != len(expected_clusters):
                return False

        return True

    def get_diff(self, result: SwarmResult) -> Dict[str, Any]:
        """Get differences between result and golden."""
        expected = self.golden.expected_result
        diff = {}

        if result.status != expected.get("status"):
            diff["status"] = {
                "expected": expected.get("status"),
                "actual": result.status,
            }

        if result.consensus_reached != expected.get("consensus_reached"):
            diff["consensus_reached"] = {
                "expected": expected.get("consensus_reached"),
                "actual": result.consensus_reached,
            }

        if result.selected_output != expected.get("selected_output"):
            diff["selected_output"] = {
                "expected": expected.get("selected_output"),
                "actual": result.selected_output,
            }

        return diff

    def _reconstruct_request(self) -> SwarmRequest:
        """Reconstruct SwarmRequest from golden data."""
        req_data = self.golden.request

        from ..schemas import BudgetConfig, SwarmConfig

        return SwarmRequest(
            token="replay_token",  # Dummy token for replay
            task_id=req_data.get("task_id", "replay_task"),
            description=req_data.get("description", ""),
            mode=SwarmMode(req_data.get("mode", "patch")),
            budgets=BudgetConfig(
                max_cost_usd=req_data.get("budgets", {}).get("max_cost_usd", 10.0),
                max_tokens=req_data.get("budgets", {}).get("max_tokens", 100000),
                timeout_seconds=req_data.get("budgets", {}).get("timeout_seconds", 300),
            ),
            swarm=SwarmConfig(
                agent_count=req_data.get("swarm", {}).get("agent_count", 3),
                consensus_k=req_data.get("swarm", {}).get("consensus_k", 2),
            ),
        )

    def _mock_atlas_adapter(self, manager: SwarmManager):
        """Create mock Atlas adapter returning recorded outputs."""
        # Find recorded agent outputs
        expected_outputs = self.golden.expected_result.get("agent_outputs", [])

        mock_outputs = [
            AgentOutput(
                agent_id=o["agent_id"],
                prompt_style=o["prompt_style"],
                output_text=o["output_text"],
                is_valid=o["is_valid"],
            )
            for o in expected_outputs
        ]

        mock_adapter = MagicMock()
        mock_adapter.generate_candidates = AsyncMock(return_value=mock_outputs)
        mock_adapter.validate_and_score_patch_candidates = AsyncMock(return_value={
            "results": {
                o["agent_id"]: {"passed": True, "patch_applied": True}
                for o in expected_outputs
            },
            "any_passed": True,
        })

        return patch.object(manager, "_atlas_adapter", mock_adapter)

    def _mock_database_calls(self):
        """Mock database calls during replay."""
        return patch.multiple(
            "agent_mcp.features.swarm.swarm_manager",
            save_swarm_run=AsyncMock(),
            update_swarm_run_status=AsyncMock(),
            save_swarm_agent=AsyncMock(),
            save_swarm_output=AsyncMock(),
        )


class GoldenTestRunner:
    """Runs golden tests from a directory."""

    def __init__(self, golden_dir: Path):
        self.golden_dir = Path(golden_dir)
        self.results: List[Dict[str, Any]] = []

    async def run_all(self) -> Dict[str, Any]:
        """Run all golden tests in the directory."""
        self.results = []

        test_files = list(self.golden_dir.glob("*.json"))

        for test_file in test_files:
            result = await self.run_single(test_file)
            self.results.append(result)

        return self.get_summary()

    async def run_single(self, test_file: Path) -> Dict[str, Any]:
        """Run a single golden test."""
        harness = ReplayHarness.from_file(test_file)

        test_result = {
            "name": harness.golden.name,
            "file": str(test_file),
            "passed": False,
            "diff": {},
            "error": None,
        }

        try:
            result = await harness.replay()
            test_result["passed"] = harness.matches_golden(result)

            if not test_result["passed"]:
                test_result["diff"] = harness.get_diff(result)

        except Exception as e:
            test_result["error"] = str(e)

        return test_result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"] and not r["error"])
        errors = sum(1 for r in self.results if r["error"])

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": self.results,
        }


def create_golden_from_outputs(
    name: str,
    description: str,
    request: SwarmRequest,
    outputs: List[AgentOutput],
    selected_output: str,
    consensus_reached: bool = True,
) -> GoldenTestCase:
    """Helper to create a golden test case from known outputs."""
    recorder = SwarmRecorder()

    result = SwarmResult(
        run_id=f"golden_{name}",
        mode=request.mode,
        status="completed",
        consensus_reached=consensus_reached,
        selected_output=selected_output,
        agent_outputs=outputs,
    )

    return GoldenTestCase(
        name=name,
        description=description,
        recorded_at=datetime.utcnow().isoformat(),
        request=recorder._serialize_request(request),
        expected_result=recorder._serialize_result(result),
    )


# Example golden test cases for reference
EXAMPLE_GOLDEN_CASES = [
    {
        "name": "simple_patch_consensus",
        "description": "Three identical patches reach consensus",
        "request": {
            "task_id": "fix_001",
            "description": "Fix null pointer exception",
            "mode": "patch",
            "swarm": {"agent_count": 3, "consensus_k": 2},
        },
        "expected_result": {
            "status": "completed",
            "consensus_reached": True,
            "selected_output": "def fix(): return None",
        },
    },
    {
        "name": "answer_mode_consensus",
        "description": "Two similar answers reach consensus",
        "request": {
            "task_id": "question_001",
            "description": "What is the capital of France?",
            "mode": "answer",
            "swarm": {"agent_count": 2, "consensus_k": 1},
        },
        "expected_result": {
            "status": "completed",
            "consensus_reached": True,
            "selected_output": "The capital of France is Paris.",
        },
    },
]
