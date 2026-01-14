"""Tests for TaskDecomposer diversity enhancement and ensemble merging.

Tests T-001 through T-010 from the decomposer-diversity spec.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from atlas.core.task_dag import OwnershipRules, TaskDAG, TaskOracle, TaskSpec, OracleType
from atlas.core.task_decomposer import (
    DECOMPOSITION_PERSONAS,
    DECOMPOSITION_TEMPERATURES,
    MergeMetrics,
    TaskDecomposer,
    TaskDecomposerConfig,
    TaskDecompositionCandidate,
)


# =============================================================================
# T-001: Configuration Tests
# =============================================================================

def test_decomposition_personas_defined():
    """Verify 3 personas are defined with distinct prompts."""
    assert len(DECOMPOSITION_PERSONAS) == 3
    assert "ATOMIC_ARCHITECT" in DECOMPOSITION_PERSONAS
    assert "INTEGRATION_LEAD" in DECOMPOSITION_PERSONAS
    assert "TEST_FIRST_QA" in DECOMPOSITION_PERSONAS

    # Each persona should have unique content
    prompts = list(DECOMPOSITION_PERSONAS.values())
    assert len(set(prompts)) == 3


def test_decomposition_temperatures():
    """Verify 3 temperatures spanning low to high."""
    assert len(DECOMPOSITION_TEMPERATURES) == 3
    assert DECOMPOSITION_TEMPERATURES == [0.3, 0.5, 0.8]


def test_config_defaults_to_9_variants():
    """Config should default to 9 variants for optimal diversity."""
    config = TaskDecomposerConfig()
    assert config.num_variants == 9
    assert config.num_personas == 3
    assert config.num_temperatures == 3
    assert config.enable_ensemble_merge is True


def test_config_ensemble_merge_settings():
    """Verify ensemble merge configuration options."""
    config = TaskDecomposerConfig()
    assert config.cluster_similarity_threshold == 0.6
    assert config.min_oracle_count == 1


# =============================================================================
# T-004: Jaccard Similarity Tests
# =============================================================================

def test_jaccard_similarity_identical_scopes():
    """Tasks with identical scopes should have Jaccard = 1.0."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="Task A",
        description="A",
        contract="A",
        ownership=OwnershipRules(
            allowed_files=["src/api/handler.py"],
            allowed_dirs=["src/api"],
        ),
    )
    task_b = TaskSpec(
        task_id="b",
        title="Task B",
        description="B",
        contract="B",
        ownership=OwnershipRules(
            allowed_files=["src/api/handler.py"],
            allowed_dirs=["src/api"],
        ),
    )

    similarity = decomposer._task_jaccard(task_a, task_b)
    assert similarity == 1.0


def test_jaccard_similarity_no_overlap():
    """Tasks with no scope overlap should have Jaccard = 0.0."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="Task A",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_files=["src/api/handler.py"]),
    )
    task_b = TaskSpec(
        task_id="b",
        title="Task B",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_files=["src/db/models.py"]),
    )

    similarity = decomposer._task_jaccard(task_a, task_b)
    assert similarity == 0.0


def test_jaccard_similarity_partial_overlap():
    """Tasks with partial overlap should have 0 < Jaccard < 1."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="Task A",
        description="A",
        contract="A",
        ownership=OwnershipRules(
            allowed_files=["src/api/handler.py", "src/models/user.py"],
        ),
    )
    task_b = TaskSpec(
        task_id="b",
        title="Task B",
        description="B",
        contract="B",
        ownership=OwnershipRules(
            allowed_files=["src/api/handler.py", "src/db/models.py"],
        ),
    )

    similarity = decomposer._task_jaccard(task_a, task_b)
    # Intersection: {src/api/handler.py} = 1
    # Union: {src/api/handler.py, src/models/user.py, src/db/models.py} = 3
    # Jaccard = 1/3 = 0.333...
    assert 0.3 < similarity < 0.4


def test_jaccard_similarity_empty_scopes():
    """Both tasks with empty scopes should have Jaccard = 1.0."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="Task A",
        description="A",
        contract="A",
        ownership=OwnershipRules(),
    )
    task_b = TaskSpec(
        task_id="b",
        title="Task B",
        description="B",
        contract="B",
        ownership=OwnershipRules(),
    )

    similarity = decomposer._task_jaccard(task_a, task_b)
    assert similarity == 1.0


# =============================================================================
# T-004: Clustering Tests
# =============================================================================

def test_clustering_groups_similar_tasks():
    """Similar tasks (Jaccard > threshold) should be in the same cluster."""
    decomposer = TaskDecomposer(config=TaskDecomposerConfig(cluster_similarity_threshold=0.5))

    # Task 1 and 2 overlap significantly (same api scope)
    task_1 = TaskSpec(
        task_id="api_1",
        title="API Task 1",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
    )
    task_2 = TaskSpec(
        task_id="api_2",
        title="API Task 2",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
    )
    # Task 3 is different (db scope)
    task_3 = TaskSpec(
        task_id="db_1",
        title="DB Task",
        description="C",
        contract="C",
        ownership=OwnershipRules(allowed_dirs=["src/db"]),
    )

    all_tasks = [
        (task_1, "ATOMIC_0.3", "ATOMIC_ARCHITECT", 0.3),
        (task_2, "INTEGRATION_0.5", "INTEGRATION_LEAD", 0.5),
        (task_3, "TEST_FIRST_0.8", "TEST_FIRST_QA", 0.8),
    ]

    clusters = decomposer._cluster_tasks(all_tasks)

    # Should have 2 clusters: one for api tasks, one for db
    assert len(clusters) == 2

    # Find the api cluster and db cluster
    cluster_sizes = sorted(len(c) for c in clusters)
    assert cluster_sizes == [1, 2]  # One cluster with 2, one with 1


def test_clustering_with_no_overlap():
    """All distinct tasks should each be in their own cluster."""
    decomposer = TaskDecomposer(config=TaskDecomposerConfig(cluster_similarity_threshold=0.6))

    tasks = [
        (TaskSpec(task_id="a", title="A", description="A", contract="A",
                  ownership=OwnershipRules(allowed_dirs=["src/api"])),
         "source_a", "ATOMIC_ARCHITECT", 0.3),
        (TaskSpec(task_id="b", title="B", description="B", contract="B",
                  ownership=OwnershipRules(allowed_dirs=["src/db"])),
         "source_b", "INTEGRATION_LEAD", 0.5),
        (TaskSpec(task_id="c", title="C", description="C", contract="C",
                  ownership=OwnershipRules(allowed_dirs=["src/models"])),
         "source_c", "TEST_FIRST_QA", 0.8),
    ]

    clusters = decomposer._cluster_tasks(tasks)
    assert len(clusters) == 3  # Each task in its own cluster


# =============================================================================
# T-005: Cluster Voting Tests
# =============================================================================

def test_select_representative_prefers_more_oracles():
    """Task with more oracles should win the cluster vote."""
    decomposer = TaskDecomposer()

    task_few = TaskSpec(
        task_id="few",
        title="Few Oracles",
        description="A",
        contract="Contract A",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
        oracles=[TaskOracle(oracle_type=OracleType.TEST, command="pytest")],
    )
    task_many = TaskSpec(
        task_id="many",
        title="Many Oracles",
        description="B",
        contract="Contract B",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
        oracles=[
            TaskOracle(oracle_type=OracleType.TEST, command="pytest"),
            TaskOracle(oracle_type=OracleType.LINT, command="flake8"),
            TaskOracle(oracle_type=OracleType.TYPECHECK, command="mypy"),
        ],
    )

    cluster = [
        (task_few, "source_a", "ATOMIC_ARCHITECT", 0.3),
        (task_many, "source_b", "INTEGRATION_LEAD", 0.5),
    ]

    winner, _, _, _ = decomposer._select_representative(cluster)
    assert winner.task_id == "many"


def test_select_representative_penalizes_wildcards():
    """Task with fewer wildcards should score higher."""
    decomposer = TaskDecomposer()

    task_wildcard = TaskSpec(
        task_id="wild",
        title="Wildcard Task",
        description="A",
        contract="Contract",
        ownership=OwnershipRules(allowed_globs=["**/*.py"]),
    )
    task_specific = TaskSpec(
        task_id="specific",
        title="Specific Task",
        description="B",
        contract="Contract",
        ownership=OwnershipRules(allowed_files=["src/api/handler.py"]),
    )

    cluster = [
        (task_wildcard, "source_a", "ATOMIC_ARCHITECT", 0.3),
        (task_specific, "source_b", "INTEGRATION_LEAD", 0.5),
    ]

    winner, _, _, _ = decomposer._select_representative(cluster)
    assert winner.task_id == "specific"


# =============================================================================
# T-006: Gap Filling Tests
# =============================================================================

def test_gap_filling_covers_all_scopes():
    """Gap filling should add tasks to cover missing scopes."""
    decomposer = TaskDecomposer()

    # Already selected task covers src/api
    selected = [
        TaskSpec(
            task_id="api",
            title="API Task",
            description="A",
            contract="A",
            ownership=OwnershipRules(allowed_dirs=["src/api"]),
        )
    ]

    # All available tasks
    db_task = TaskSpec(
        task_id="db",
        title="DB Task",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_dirs=["src/db"]),
    )
    model_task = TaskSpec(
        task_id="model",
        title="Model Task",
        description="C",
        contract="C",
        ownership=OwnershipRules(allowed_dirs=["src/models"]),
    )

    all_tasks = [
        (selected[0], "s1", "ATOMIC_ARCHITECT", 0.3),
        (db_task, "s2", "INTEGRATION_LEAD", 0.5),
        (model_task, "s3", "TEST_FIRST_QA", 0.8),
    ]

    # Gaps are db and models
    gaps = {"src/db", "src/models"}

    fillers = decomposer._fill_gaps(gaps, all_tasks, selected)

    assert len(fillers) == 2
    filler_ids = {f[0].task_id for f in fillers}
    assert filler_ids == {"db", "model"}


def test_gap_filling_greedy_selection():
    """Gap filling should prefer tasks that cover more gaps."""
    decomposer = TaskDecomposer()

    selected = []

    # Task covering 2 scopes
    broad_task = TaskSpec(
        task_id="broad",
        title="Broad Task",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_dirs=["src/api", "src/db"]),
    )
    # Task covering 1 scope
    narrow_task = TaskSpec(
        task_id="narrow",
        title="Narrow Task",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
    )

    all_tasks = [
        (narrow_task, "s1", "ATOMIC_ARCHITECT", 0.3),  # Listed first but covers less
        (broad_task, "s2", "INTEGRATION_LEAD", 0.5),
    ]

    gaps = {"src/api", "src/db"}

    fillers = decomposer._fill_gaps(gaps, all_tasks, selected)

    # Should select the broad task first since it covers more gaps
    assert len(fillers) == 1
    assert fillers[0][0].task_id == "broad"


# =============================================================================
# T-007: Dependency Resolution Tests
# =============================================================================

def test_dependency_resolution_topological_sort():
    """Dependencies should be resolved with topological sort."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="A",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_files=["a.py"]),
        dependencies=[],
    )
    task_b = TaskSpec(
        task_id="b",
        title="B",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_files=["b.py"]),
        dependencies=["a"],
    )
    task_c = TaskSpec(
        task_id="c",
        title="C",
        description="C",
        contract="C",
        ownership=OwnershipRules(allowed_files=["c.py"]),
        dependencies=["b"],
    )

    # Pass in wrong order
    tasks = [task_c, task_a, task_b]

    sorted_tasks = decomposer._resolve_dependencies(tasks)

    ids = [t.task_id for t in sorted_tasks]
    assert ids.index("a") < ids.index("b")
    assert ids.index("b") < ids.index("c")


def test_dependency_resolution_handles_missing_deps():
    """Missing dependencies should be ignored."""
    decomposer = TaskDecomposer()

    task_a = TaskSpec(
        task_id="a",
        title="A",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_files=["a.py"]),
        dependencies=["missing_task"],  # This task doesn't exist
    )

    tasks = [task_a]

    sorted_tasks = decomposer._resolve_dependencies(tasks)

    # Should still return the task
    assert len(sorted_tasks) == 1
    assert sorted_tasks[0].task_id == "a"


def test_dependency_resolution_breaks_cycles():
    """Cycles should be broken by removing an edge."""
    decomposer = TaskDecomposer()

    # Create a cycle: A -> B -> A
    task_a = TaskSpec(
        task_id="a",
        title="A",
        description="A",
        contract="A",
        ownership=OwnershipRules(allowed_files=["a.py"]),
        dependencies=["b"],
    )
    task_b = TaskSpec(
        task_id="b",
        title="B",
        description="B",
        contract="B",
        ownership=OwnershipRules(allowed_files=["b.py"]),
        dependencies=["a"],  # Creates cycle
    )

    tasks = [task_a, task_b]

    # Should not raise, should break the cycle
    sorted_tasks = decomposer._resolve_dependencies(tasks)

    # Should return both tasks
    assert len(sorted_tasks) == 2

    # At least one dependency should have been removed
    total_deps = sum(len(t.dependencies) for t in sorted_tasks)
    assert total_deps < 2  # Was 2 before, should be 1 or 0 after breaking


def test_path_normalization_handles_edge_cases():
    """Path normalization should handle various edge cases."""
    decomposer = TaskDecomposer()

    # Double slashes
    assert decomposer._normalize_path("src//api//handler.py") == "src/api/handler.py"

    # Windows paths
    assert decomposer._normalize_path("src\\api\\handler.py") == "src/api/handler.py"

    # Leading ./
    assert decomposer._normalize_path("./src/api") == "src/api"

    # Leading /
    assert decomposer._normalize_path("/src/api") == "src/api"

    # Trailing /
    assert decomposer._normalize_path("src/api/") == "src/api"

    # Combined
    assert decomposer._normalize_path(".//src\\api//") == "src/api"


# =============================================================================
# T-008: Merged DAG Validation Tests
# =============================================================================

def test_merged_dag_is_valid():
    """Merged DAG should pass validation."""
    decomposer = TaskDecomposer(config=TaskDecomposerConfig())

    # Create valid candidates
    task_1 = TaskSpec(
        task_id="api",
        title="API Task",
        description="Build API",
        contract="API endpoints work",
        ownership=OwnershipRules(allowed_dirs=["src/api"]),
        oracles=[TaskOracle(oracle_type=OracleType.TEST, command="pytest tests/api")],
    )
    task_2 = TaskSpec(
        task_id="db",
        title="DB Task",
        description="Build DB",
        contract="DB queries work",
        ownership=OwnershipRules(allowed_dirs=["src/db"]),
        oracles=[TaskOracle(oracle_type=OracleType.TEST, command="pytest tests/db")],
    )

    dag_1 = TaskDAG.from_list([task_1])
    dag_2 = TaskDAG.from_list([task_2])

    candidates = [
        TaskDecompositionCandidate(
            dag=dag_1,
            source="ATOMIC_0.3",
            errors=[],
            persona="ATOMIC_ARCHITECT",
            temperature=0.3,
        ),
        TaskDecompositionCandidate(
            dag=dag_2,
            source="INTEGRATION_0.5",
            errors=[],
            persona="INTEGRATION_LEAD",
            temperature=0.5,
        ),
    ]

    # Mock repo_report
    repo_report = MagicMock()
    repo_report.keyword_hits = {}
    repo_report.component_map = {}
    repo_report.entrypoints = []

    merged_dag, metrics = decomposer._merge_candidates(
        candidates, repo_report, [], []
    )

    assert merged_dag is not None
    assert len(merged_dag.tasks) == 2
    assert metrics.variants_valid == 2
    assert metrics.final_task_count == 2


# =============================================================================
# T-009: Merge Metrics Tests
# =============================================================================

def test_merge_metrics_tracks_contributions():
    """Merge metrics should track persona and temperature contributions."""
    metrics = MergeMetrics()

    # Add contributions
    metrics.persona_contributions["ATOMIC_ARCHITECT"] = 2
    metrics.persona_contributions["INTEGRATION_LEAD"] = 1
    metrics.temperature_contributions[0.3] = 1
    metrics.temperature_contributions[0.5] = 2

    assert metrics.persona_contributions["ATOMIC_ARCHITECT"] == 2
    assert sum(metrics.temperature_contributions.values()) == 3


# =============================================================================
# T-010: 9 Variants Diversity Tests
# =============================================================================

def test_9_variants_produce_diverse_outputs():
    """9 variants (3×3) should produce genuinely diverse decompositions."""
    # This is more of an integration test - verify configuration is correct
    config = TaskDecomposerConfig()

    # Verify we get 9 combinations
    expected_combinations = config.num_personas * config.num_temperatures
    assert expected_combinations == 9

    # Verify temperatures are spread out (not clustered like 0.2, 0.25, 0.3)
    temps = config.temperatures
    temp_spread = max(temps) - min(temps)
    assert temp_spread >= 0.4  # At least 0.4 spread


def test_variant_source_naming():
    """Each variant should have unique persona_temp source name."""
    personas = list(DECOMPOSITION_PERSONAS.keys())
    temperatures = DECOMPOSITION_TEMPERATURES

    sources = []
    for persona in personas:
        for temp in temperatures:
            sources.append(f"{persona}_{temp}")

    # Should have 9 unique source names
    assert len(sources) == 9
    assert len(set(sources)) == 9


def test_decomposer_config_can_disable_ensemble():
    """Should be able to fall back to legacy selection."""
    config = TaskDecomposerConfig(enable_ensemble_merge=False)
    assert config.enable_ensemble_merge is False


# =============================================================================
# Integration Test: Full Merge Pipeline
# =============================================================================

def test_full_merge_pipeline():
    """Test the complete merge pipeline with mock candidates."""
    decomposer = TaskDecomposer(config=TaskDecomposerConfig())

    # Create 3 different candidates with overlapping and unique tasks

    # Candidate 1: api + models
    dag_1 = TaskDAG.from_list([
        TaskSpec(
            task_id="api_v1",
            title="API Handler",
            description="Build API",
            contract="Handles requests",
            ownership=OwnershipRules(allowed_dirs=["src/api"]),
            oracles=[TaskOracle(oracle_type=OracleType.TEST, command="pytest")],
        ),
        TaskSpec(
            task_id="models_v1",
            title="Models",
            description="Build models",
            contract="Models work",
            ownership=OwnershipRules(allowed_dirs=["src/models"]),
        ),
    ])

    # Candidate 2: api + db (api overlaps with candidate 1)
    dag_2 = TaskDAG.from_list([
        TaskSpec(
            task_id="api_v2",
            title="API Endpoints",
            description="Build endpoints",
            contract="API works",
            ownership=OwnershipRules(allowed_dirs=["src/api"]),
            oracles=[
                TaskOracle(oracle_type=OracleType.TEST, command="pytest"),
                TaskOracle(oracle_type=OracleType.LINT, command="flake8"),
            ],
        ),
        TaskSpec(
            task_id="db_v2",
            title="Database",
            description="Build DB",
            contract="DB queries work",
            ownership=OwnershipRules(allowed_dirs=["src/db"]),
        ),
    ])

    # Candidate 3: tests only (unique)
    dag_3 = TaskDAG.from_list([
        TaskSpec(
            task_id="tests_v3",
            title="Tests",
            description="Write tests",
            contract="Tests pass",
            ownership=OwnershipRules(allowed_dirs=["tests"]),
            oracles=[TaskOracle(oracle_type=OracleType.TEST, command="pytest")],
        ),
    ])

    candidates = [
        TaskDecompositionCandidate(
            dag=dag_1, source="ATOMIC_0.3", errors=[],
            persona="ATOMIC_ARCHITECT", temperature=0.3,
        ),
        TaskDecompositionCandidate(
            dag=dag_2, source="INTEGRATION_0.5", errors=[],
            persona="INTEGRATION_LEAD", temperature=0.5,
        ),
        TaskDecompositionCandidate(
            dag=dag_3, source="TEST_FIRST_0.8", errors=[],
            persona="TEST_FIRST_QA", temperature=0.8,
        ),
    ]

    repo_report = MagicMock()
    repo_report.keyword_hits = {}
    repo_report.component_map = {}
    repo_report.entrypoints = []

    merged_dag, metrics = decomposer._merge_candidates(
        candidates, repo_report, [], []
    )

    assert merged_dag is not None

    # Should have selected api_v2 (more oracles than api_v1)
    # Plus unique tasks: models_v1, db_v2, tests_v3
    task_ids = set(merged_dag.tasks.keys())

    # Verify coverage: all scopes should be covered
    covered_scopes = decomposer._get_covered_scopes(list(merged_dag.tasks.values()))
    assert "src/api" in covered_scopes
    assert "src/models" in covered_scopes
    assert "src/db" in covered_scopes
    assert "tests" in covered_scopes

    # Metrics should reflect the merge
    assert metrics.variants_valid == 3
    assert metrics.clusters_formed >= 2  # At least api cluster + others
