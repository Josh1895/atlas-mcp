"""Tests for TaskDAG models."""

from atlas.core.task_dag import OwnershipRules, TaskDAG, TaskSpec


def test_task_dag_validation_missing_ownership():
    task = TaskSpec(
        task_id="t1",
        title="Task 1",
        description="Do something",
        contract="Contract",
        ownership=OwnershipRules(),
    )
    dag = TaskDAG.from_list([task])
    errors = dag.validate()
    assert any("ownership" in error for error in errors)


def test_task_dag_topological_order():
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
        ownership=OwnershipRules(allowed_files=["b.txt"]),
        dependencies=["a"],
    )
    dag = TaskDAG.from_list([task_a, task_b])
    ordered = [task.task_id for task in dag.topological_order()]
    assert ordered == ["a", "b"]
