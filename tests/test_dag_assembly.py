"""Tests for DAG assembly behavior."""

import pytest

from atlas.core.dag_orchestrator import RepoState, TaskCandidate, TaskDAGOrchestrator
from atlas.core.task_dag import OwnershipRules, TaskDAG, TaskSpec


@pytest.mark.asyncio
async def test_dag_assembly_combines_patches(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x\n")

    task_a = TaskSpec(
        task_id="a",
        title="A",
        description="A desc",
        contract="A contract",
        ownership=OwnershipRules(allowed_files=["a.txt"]),
        dependencies=[],
    )
    task_b = TaskSpec(
        task_id="b",
        title="B",
        description="B desc",
        contract="B contract",
        ownership=OwnershipRules(allowed_files=["a.txt"]),
        dependencies=["a"],
    )
    dag = TaskDAG.from_list([task_a, task_b])

    patch_one = "\n".join([
        "--- a/a.txt",
        "+++ b/a.txt",
        "@@ -1 +1,2 @@",
        " x",
        "+one",
    ])
    patch_two = "\n".join([
        "--- a/a.txt",
        "+++ b/a.txt",
        "@@ -1 +1,2 @@",
        " x",
        "+two",
    ])

    candidates = {
        "a": [
            TaskCandidate(task_id="a", candidate_id="a1", patch=patch_one, score=10.0),
        ],
        "b": [
            TaskCandidate(task_id="b", candidate_id="b1", patch=patch_two, score=10.0),
        ],
    }

    orchestrator = TaskDAGOrchestrator()
    base_state = RepoState(tmp_path)
    result = await orchestrator._assemble(dag, candidates, base_state)

    assert result.success
    assert "+one" in result.combined_patch
    assert "+two" in result.combined_patch
