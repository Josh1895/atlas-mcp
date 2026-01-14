"""
Tests for the offline replay harness for golden tests.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ..schemas import (
    SwarmRequest,
    SwarmResult,
    SwarmMode,
    SwarmConfig,
    BudgetConfig,
    AgentOutput,
)
from .replay_harness import (
    SwarmRecorder,
    ReplayHarness,
    GoldenTestCase,
    GoldenTestRunner,
    RecordedCall,
    create_golden_from_outputs,
)


class TestSwarmRecorder:
    """Tests for SwarmRecorder class."""

    def test_start_recording(self):
        """Test starting a recording session."""
        recorder = SwarmRecorder()
        recorder.start_recording("test_001", "Test description")

        assert recorder._recording is True
        assert recorder._current_case is not None
        assert recorder._current_case.name == "test_001"

    def test_cannot_start_double_recording(self):
        """Test that starting recording twice raises error."""
        recorder = SwarmRecorder()
        recorder.start_recording("test_001")

        with pytest.raises(RuntimeError, match="Already recording"):
            recorder.start_recording("test_002")

    def test_stop_recording(self):
        """Test stopping a recording session."""
        recorder = SwarmRecorder()
        recorder.start_recording("test_001")

        request = SwarmRequest(
            token="test",
            task_id="task_001",
            mode=SwarmMode.PATCH,
        )
        result = SwarmResult(
            run_id="run_001",
            mode=SwarmMode.PATCH,
            status="completed",
        )

        case = recorder.stop_recording(request, result)

        assert recorder._recording is False
        assert case.name == "test_001"
        assert case.request["task_id"] == "task_001"
        assert case.expected_result["status"] == "completed"

    def test_stop_without_recording_raises_error(self):
        """Test stopping without recording raises error."""
        recorder = SwarmRecorder()

        with pytest.raises(RuntimeError, match="Not recording"):
            recorder.stop_recording(SwarmRequest(token="x"), SwarmResult(run_id="x", mode=SwarmMode.PATCH, status="x"))

    def test_record_call(self):
        """Test recording a function call."""
        recorder = SwarmRecorder()
        recorder.start_recording("test")

        recorder.record_call(
            function_name="generate_candidates",
            inputs={"description": "test"},
            output=[{"agent_id": "a1"}],
            duration_ms=100.0,
        )

        assert len(recorder._calls) == 1
        assert recorder._calls[0].function_name == "generate_candidates"
        assert recorder._calls[0].duration_ms == 100.0

    def test_record_call_without_recording_is_ignored(self):
        """Test recording calls when not recording does nothing."""
        recorder = SwarmRecorder()

        recorder.record_call("test_func", {}, None)

        assert len(recorder._calls) == 0

    def test_save_and_load(self):
        """Test saving and loading golden test cases."""
        recorder = SwarmRecorder()
        recorder.start_recording("save_test")

        request = SwarmRequest(token="test", task_id="save_001")
        result = SwarmResult(run_id="save_run", mode=SwarmMode.PATCH, status="completed")

        case = recorder.stop_recording(request, result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "test.json"
            recorder.save(case, filepath)

            loaded = recorder.load(filepath)

        assert loaded.name == "save_test"
        assert loaded.request["task_id"] == "save_001"

    def test_token_is_redacted(self):
        """Test that tokens are redacted in saved data."""
        recorder = SwarmRecorder()
        recorder.start_recording("token_test")

        request = SwarmRequest(token="secret_token_12345", task_id="task")
        result = SwarmResult(run_id="run", mode=SwarmMode.PATCH, status="completed")

        case = recorder.stop_recording(request, result)

        assert case.request["token"] == "[REDACTED]"
        assert "secret_token" not in json.dumps(case.to_dict())


class TestGoldenTestCase:
    """Tests for GoldenTestCase dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        case = GoldenTestCase(
            name="test",
            description="desc",
            recorded_at="2024-01-01T00:00:00",
            request={"task_id": "t1"},
            expected_result={"status": "completed"},
        )

        d = case.to_dict()

        assert d["name"] == "test"
        assert d["request"]["task_id"] == "t1"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test",
            "description": "desc",
            "recorded_at": "2024-01-01T00:00:00",
            "request": {"task_id": "t1"},
            "expected_result": {"status": "completed"},
            "recorded_calls": [
                {
                    "function_name": "test_func",
                    "timestamp": "2024-01-01T00:00:01",
                    "inputs": {},
                    "output": None,
                    "duration_ms": 0.0,
                    "error": None,
                }
            ],
        }

        case = GoldenTestCase.from_dict(data)

        assert case.name == "test"
        assert len(case.recorded_calls) == 1


class TestReplayHarness:
    """Tests for ReplayHarness class."""

    @pytest.fixture
    def simple_golden(self):
        """Create a simple golden test case."""
        return GoldenTestCase(
            name="simple_test",
            description="Simple test case",
            recorded_at="2024-01-01T00:00:00",
            request={
                "task_id": "replay_task",
                "description": "Test task",
                "mode": "patch",
                "budgets": {"max_cost_usd": 1.0, "timeout_seconds": 60},
                "swarm": {"agent_count": 2, "consensus_k": 1},
            },
            expected_result={
                "run_id": "golden_run",
                "mode": "patch",
                "status": "completed",
                "consensus_reached": True,
                "selected_output": "def fix(): pass",
                "agent_outputs": [
                    {
                        "agent_id": "a1",
                        "prompt_style": "direct",
                        "output_text": "def fix(): pass",
                        "is_valid": True,
                    },
                    {
                        "agent_id": "a2",
                        "prompt_style": "cot",
                        "output_text": "def fix(): pass",
                        "is_valid": True,
                    },
                ],
                "clusters": [],
            },
        )

    def test_matches_golden_status(self, simple_golden):
        """Test matching by status."""
        harness = ReplayHarness(simple_golden)

        matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=True,
            selected_output="def fix(): pass",
        )

        not_matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="failed",
            consensus_reached=True,
            selected_output="def fix(): pass",
        )

        assert harness.matches_golden(matching) is True
        assert harness.matches_golden(not_matching) is False

    def test_matches_golden_consensus(self, simple_golden):
        """Test matching by consensus."""
        harness = ReplayHarness(simple_golden)

        matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=True,
            selected_output="def fix(): pass",
        )

        not_matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=False,
            selected_output="def fix(): pass",
        )

        assert harness.matches_golden(matching, check_consensus=True) is True
        assert harness.matches_golden(not_matching, check_consensus=True) is False

    def test_matches_golden_output(self, simple_golden):
        """Test matching by selected output."""
        harness = ReplayHarness(simple_golden)

        matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=True,
            selected_output="def fix(): pass",
        )

        not_matching = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="completed",
            consensus_reached=True,
            selected_output="different output",
        )

        assert harness.matches_golden(matching, check_outputs=True) is True
        assert harness.matches_golden(not_matching, check_outputs=True) is False

    def test_get_diff(self, simple_golden):
        """Test getting differences from golden."""
        harness = ReplayHarness(simple_golden)

        result = SwarmResult(
            run_id="test",
            mode=SwarmMode.PATCH,
            status="failed",
            consensus_reached=False,
            selected_output="wrong output",
        )

        diff = harness.get_diff(result)

        assert "status" in diff
        assert diff["status"]["expected"] == "completed"
        assert diff["status"]["actual"] == "failed"

        assert "consensus_reached" in diff
        assert "selected_output" in diff

    def test_reconstruct_request(self, simple_golden):
        """Test request reconstruction."""
        harness = ReplayHarness(simple_golden)

        request = harness._reconstruct_request()

        assert request.task_id == "replay_task"
        assert request.mode == SwarmMode.PATCH
        assert request.swarm.agent_count == 2

    @pytest.mark.asyncio
    async def test_replay(self, simple_golden):
        """Test replay execution."""
        harness = ReplayHarness(simple_golden)

        result = await harness.replay()

        assert result is not None
        assert result.status == "completed"


class TestGoldenTestRunner:
    """Tests for GoldenTestRunner class."""

    @pytest.fixture
    def golden_dir(self):
        """Create a temporary directory with golden tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            golden_path = Path(tmp_dir)

            # Create test files
            case1 = GoldenTestCase(
                name="test_001",
                description="First test",
                recorded_at="2024-01-01T00:00:00",
                request={"task_id": "t1", "mode": "patch"},
                expected_result={
                    "status": "completed",
                    "consensus_reached": True,
                    "selected_output": "output1",
                    "agent_outputs": [],
                },
            )

            case2 = GoldenTestCase(
                name="test_002",
                description="Second test",
                recorded_at="2024-01-01T00:00:00",
                request={"task_id": "t2", "mode": "patch"},
                expected_result={
                    "status": "completed",
                    "consensus_reached": True,
                    "selected_output": "output2",
                    "agent_outputs": [],
                },
            )

            with open(golden_path / "test_001.json", "w") as f:
                json.dump(case1.to_dict(), f)

            with open(golden_path / "test_002.json", "w") as f:
                json.dump(case2.to_dict(), f)

            yield golden_path

    @pytest.mark.asyncio
    async def test_run_all(self, golden_dir):
        """Test running all golden tests."""
        runner = GoldenTestRunner(golden_dir)

        summary = await runner.run_all()

        assert summary["total"] == 2

    def test_get_summary_empty(self):
        """Test summary with no results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = GoldenTestRunner(Path(tmp_dir))
            runner.results = []

            summary = runner.get_summary()

            assert summary["total"] == 0
            assert summary["pass_rate"] == 0.0


class TestCreateGoldenFromOutputs:
    """Tests for helper function."""

    def test_create_golden_from_outputs(self):
        """Test creating golden test case from outputs."""
        request = SwarmRequest(
            token="test",
            task_id="helper_test",
            mode=SwarmMode.PATCH,
            swarm=SwarmConfig(agent_count=2),
        )

        outputs = [
            AgentOutput(agent_id="a1", prompt_style="s1", output_text="patch", is_valid=True),
            AgentOutput(agent_id="a2", prompt_style="s2", output_text="patch", is_valid=True),
        ]

        case = create_golden_from_outputs(
            name="helper_case",
            description="Created by helper",
            request=request,
            outputs=outputs,
            selected_output="patch",
            consensus_reached=True,
        )

        assert case.name == "helper_case"
        assert case.expected_result["selected_output"] == "patch"
        assert len(case.expected_result["agent_outputs"]) == 2
