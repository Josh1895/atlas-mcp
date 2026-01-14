# Spec: Task Decomposer Diversity Enhancement

## Problem Statement

The current task decomposer generates 3 variants using:
- Same prompt
- Same system persona
- Nearly identical temperatures (0.2, 0.25, 0.3)

This produces **correlated samples**, not true independent variants.

## Goal

1. Increase to **9 variants** (3 personas × 3 temperatures) for maximum diversity
2. **Mathematically merge** the best parts instead of just picking one winner

---

## Part 1: 9-Variant Generation

### Personas (3)

| Persona | System Prompt | Thinking Style |
|---------|---------------|----------------|
| ATOMIC_ARCHITECT | "Decompose into smallest independent units. Each task completable in isolation." | Bottom-up, granular |
| INTEGRATION_LEAD | "Think about interfaces and data flow. Define tasks around integration points." | Middle-out, API-first |
| TEST_FIRST_QA | "Start from test scenarios. Work backward from 'how do we verify this?'" | Top-down, verification-first |

### Temperatures (3)

| Level | Value | Behavior |
|-------|-------|----------|
| Focused | 0.3 | Conservative, safe |
| Balanced | 0.5 | Middle ground |
| Creative | 0.8 | Exploratory, novel |

### Matrix (9 variants)

```
           Temp 0.3    Temp 0.5    Temp 0.8
ATOMIC       V1          V2          V3
INTEGRATION  V4          V5          V6
TEST_FIRST   V7          V8          V9
```

---

## Part 2: Mathematical Ensemble Merging

Instead of picking one winner, we **mathematically combine** all valid variants.

### Algorithm: Set Cover + Voting

```
INPUT: 9 TaskDAG candidates
OUTPUT: 1 merged TaskDAG with maximum coverage

Step 1: COLLECT all unique scopes
  - Union of all file paths across all variants
  - This is the "ground truth" of what needs to be covered

Step 2: CLUSTER similar tasks (Jaccard similarity on file scopes)
  - Task A covers {src/api/*, src/models/*}
  - Task B covers {src/api/*, src/db/*}
  - Jaccard(A,B) = |intersection| / |union| = 1/3 = 0.33
  - Group tasks with Jaccard > 0.6 into same cluster

Step 3: VOTE within each cluster
  - For each cluster of similar tasks, pick representative by:
    a) Most oracles (verification coverage)
    b) Most specific ownership (not wildcards)
    c) Highest priority score

Step 4: FILL GAPS using set cover
  - Compute: covered_scopes = union of selected task scopes
  - Compute: missing_scopes = all_scopes - covered_scopes
  - Greedily add tasks that cover missing scopes

Step 5: RESOLVE dependencies
  - Build dependency graph from merged tasks
  - Topological sort to ensure valid ordering
```

### Mathematical Operations (No LLM)

| Operation | Method | Library |
|-----------|--------|---------|
| Scope similarity | Jaccard index | Pure Python sets |
| Task clustering | Agglomerative clustering | sklearn or manual |
| Gap detection | Set difference | Python set operations |
| Dependency resolution | Topological sort | graphlib.TopologicalSorter |

### Example

```
Variant 1 proposes: [TaskA: src/api/*, TaskB: src/db/*]
Variant 2 proposes: [TaskX: src/api/*, TaskY: src/models/*]
Variant 3 proposes: [TaskZ: src/api/*, TaskW: src/db/*, TaskQ: tests/*]

Step 1: All scopes = {src/api/*, src/db/*, src/models/*, tests/*}

Step 2: Cluster by similarity
  - Cluster 1: {TaskA, TaskX, TaskZ} - all cover src/api/*
  - Cluster 2: {TaskB, TaskW} - both cover src/db/*
  - Cluster 3: {TaskY} - unique (src/models/*)
  - Cluster 4: {TaskQ} - unique (tests/*)

Step 3: Pick best from each cluster
  - Cluster 1 → TaskZ (has 2 oracles vs 1)
  - Cluster 2 → TaskW (more specific ownership)
  - Cluster 3 → TaskY (only option)
  - Cluster 4 → TaskQ (only option)

Step 4: Check coverage
  - Merged covers: {src/api/*, src/db/*, src/models/*, tests/*}
  - Missing: {} (nothing!)

OUTPUT: [TaskZ, TaskW, TaskY, TaskQ] - merged from 3 variants
```

---

## Implementation Tasks

### T-001: Add 9-Variant Configuration
- Add `DECOMPOSITION_PERSONAS` dict (3 personas)
- Add `DECOMPOSITION_TEMPERATURES` list [0.3, 0.5, 0.8]
- Update config to default `num_variants=9`

### T-002: Implement Variant Generation Matrix
- Generate all 9 combinations (3×3)
- Run in parallel for speed
- Track which persona+temp produced each variant

### T-003: Implement Scope Extraction
- `extract_all_scopes(candidates) -> Set[str]`
- Union of all file paths, dirs, globs across all variants

### T-004: Implement Task Clustering
- `cluster_tasks_by_scope(all_tasks) -> List[List[TaskSpec]]`
- Use Jaccard similarity on ownership scopes
- Threshold: 0.6 similarity to be in same cluster

### T-005: Implement Cluster Voting
- `select_representative(cluster: List[TaskSpec]) -> TaskSpec`
- Score by: oracle_count × specificity × priority
- Pick highest scorer

### T-006: Implement Gap Filling
- `find_coverage_gaps(selected_tasks, all_scopes) -> Set[str]`
- `fill_gaps(gaps, all_tasks) -> List[TaskSpec]`
- Greedy set cover to minimize added tasks

### T-007: Implement Dependency Resolution
- `resolve_dependencies(tasks) -> List[TaskSpec]`
- Topological sort
- Handle cycles by breaking weakest link

### T-008: Integrate Ensemble Merge
- Replace `_select_candidate()` with `_merge_candidates()`
- Return merged DAG instead of single winner

### T-009: Add Merge Metrics
- Track: variants_used, tasks_merged, gaps_filled
- Log merge decisions for debugging

### T-010: Unit Tests
- Test clustering produces expected groups
- Test gap filling covers all scopes
- Test merged DAG is valid

---

## Configuration

```python
@dataclass
class TaskDecomposerConfig:
    # Variant generation
    num_personas: int = 3
    num_temperatures: int = 3
    temperatures: List[float] = [0.3, 0.5, 0.8]

    # Ensemble merging
    enable_ensemble_merge: bool = True
    cluster_similarity_threshold: float = 0.6
    min_oracle_count: int = 1

    # Existing config...
    max_tasks: int = 12
    require_oracles: bool = True
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Variant diversity (avg Jaccard) | ~0.95 | < 0.5 |
| Scope coverage | ~80% | > 95% |
| Oracle coverage | ~70% | > 90% |
| Variants used in merge | 1 | 3-5 |

---

## Cost Analysis

| Config | LLM Calls | Cost Multiplier |
|--------|-----------|-----------------|
| Current (3 similar) | 3 | 1x |
| **Proposed (9 diverse)** | 9 | 3x |

**Tradeoff**: 3x cost for significantly better decomposition quality.

---

## Decision: APPROVED

- [x] 9 variants (3 personas × 3 temps)
- [x] Mathematical ensemble merging
- [x] No additional LLM calls for merging (pure math)
