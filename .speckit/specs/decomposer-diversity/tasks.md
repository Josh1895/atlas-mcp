# Task Decomposer Diversity - Implementation Tasks

## Status: Ready to Implement

## Summary

- **9 LLM calls** (3 personas × 3 temperatures)
- **Mathematical merging** (no additional LLM, pure set operations)
- **Result**: Best tasks from all 9 variants combined

---

## Task List

### Phase 1: Configuration (T-001)
- [ ] **T-001**: Add 9-Variant Configuration
  ```python
  DECOMPOSITION_PERSONAS = {
      "ATOMIC_ARCHITECT": "You decompose into smallest independent units...",
      "INTEGRATION_LEAD": "You think about interfaces and data flow...",
      "TEST_FIRST_QA": "You start from test scenarios and work backward...",
  }
  DECOMPOSITION_TEMPERATURES = [0.3, 0.5, 0.8]
  ```

### Phase 2: Variant Generation (T-002)
- [ ] **T-002**: Implement 3×3 Variant Matrix
  - Generate all 9 persona+temp combinations
  - Run in parallel (asyncio.gather)
  - Track source: `"ATOMIC_0.3"`, `"INTEGRATION_0.5"`, etc.

### Phase 3: Mathematical Merging (T-003 to T-007)
- [ ] **T-003**: Scope Extraction
  ```python
  def extract_all_scopes(candidates: List[TaskDAG]) -> Set[str]:
      """Union of all file scopes across all variants."""
  ```

- [ ] **T-004**: Task Clustering (Jaccard)
  ```python
  def jaccard_similarity(task_a: TaskSpec, task_b: TaskSpec) -> float:
      """Jaccard index on ownership scopes."""
      scopes_a = set(task_a.ownership.allowed_files + task_a.ownership.allowed_dirs)
      scopes_b = set(task_b.ownership.allowed_files + task_b.ownership.allowed_dirs)
      if not scopes_a and not scopes_b:
          return 1.0
      return len(scopes_a & scopes_b) / len(scopes_a | scopes_b)

  def cluster_tasks(all_tasks: List[TaskSpec], threshold: float = 0.6) -> List[List[TaskSpec]]:
      """Group similar tasks into clusters."""
  ```

- [ ] **T-005**: Cluster Voting
  ```python
  def select_representative(cluster: List[TaskSpec]) -> TaskSpec:
      """Pick best task from cluster by scoring."""
      def score(task):
          oracle_score = len(task.oracles) * 10
          specificity = 10 - task.ownership.allowed_globs.count("**")  # Penalize wildcards
          priority = task.priority
          return oracle_score + specificity + priority
      return max(cluster, key=score)
  ```

- [ ] **T-006**: Gap Filling (Greedy Set Cover)
  ```python
  def find_gaps(selected: List[TaskSpec], all_scopes: Set[str]) -> Set[str]:
      """Find scopes not covered by selected tasks."""
      covered = set()
      for task in selected:
          covered.update(task.ownership.allowed_files)
          covered.update(task.ownership.allowed_dirs)
      return all_scopes - covered

  def fill_gaps(gaps: Set[str], all_tasks: List[TaskSpec]) -> List[TaskSpec]:
      """Greedily add tasks to cover gaps."""
  ```

- [ ] **T-007**: Dependency Resolution
  ```python
  from graphlib import TopologicalSorter

  def resolve_dependencies(tasks: List[TaskSpec]) -> List[TaskSpec]:
      """Topological sort, handle cycles."""
  ```

### Phase 4: Integration (T-008, T-009)
- [ ] **T-008**: Replace Selection with Merge
  - Change `_select_candidate()` to `_merge_candidates()`
  - Return merged DAG

- [ ] **T-009**: Add Merge Metrics
  ```python
  @dataclass
  class MergeMetrics:
      variants_generated: int  # 9
      variants_valid: int      # e.g., 7
      clusters_formed: int     # e.g., 5
      tasks_selected: int      # e.g., 5
      gaps_filled: int         # e.g., 1
      final_task_count: int    # e.g., 6
  ```

### Phase 5: Testing (T-010)
- [ ] **T-010**: Unit Tests
  - `test_jaccard_similarity()`
  - `test_clustering_groups_similar_tasks()`
  - `test_gap_filling_covers_all_scopes()`
  - `test_merged_dag_is_valid()`
  - `test_9_variants_produce_diverse_outputs()`

---

## Algorithm Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                    9 VARIANT GENERATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ATOMIC_ARCHITECT ───┬─── temp 0.3 ──→ Variant 1              │
│                       ├─── temp 0.5 ──→ Variant 2              │
│                       └─── temp 0.8 ──→ Variant 3              │
│                                                                 │
│   INTEGRATION_LEAD ───┬─── temp 0.3 ──→ Variant 4              │
│                       ├─── temp 0.5 ──→ Variant 5              │
│                       └─── temp 0.8 ──→ Variant 6              │
│                                                                 │
│   TEST_FIRST_QA ──────┬─── temp 0.3 ──→ Variant 7              │
│                       ├─── temp 0.5 ──→ Variant 8              │
│                       └─── temp 0.8 ──→ Variant 9              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MATHEMATICAL MERGING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Extract all scopes ─────────────────────────────────  │
│          {src/api/*, src/db/*, src/models/*, tests/*}          │
│                                                                 │
│  Step 2: Cluster similar tasks (Jaccard > 0.6) ──────────────  │
│          Cluster A: [V1.task1, V4.task1, V7.task1] (api)       │
│          Cluster B: [V1.task2, V5.task2] (db)                  │
│          Cluster C: [V3.task3] (models)                        │
│          Cluster D: [V7.task2, V8.task2] (tests)               │
│                                                                 │
│  Step 3: Vote for best in each cluster ──────────────────────  │
│          A → V4.task1 (most oracles)                           │
│          B → V5.task2 (most specific)                          │
│          C → V3.task3 (only option)                            │
│          D → V7.task2 (higher priority)                        │
│                                                                 │
│  Step 4: Fill gaps (set cover) ──────────────────────────────  │
│          Missing: {} (all covered!)                            │
│                                                                 │
│  Step 5: Resolve dependencies (topological sort) ────────────  │
│          [V3.task3] → [V4.task1] → [V5.task2] → [V7.task2]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL MERGED DAG                            │
├─────────────────────────────────────────────────────────────────┤
│  4 tasks from 4 different variants, 100% scope coverage        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insight

**Current**: Pick 1 winner from 3 similar variants
**Proposed**: Mathematically combine best parts from 9 diverse variants

This is like ensemble learning in ML - multiple weak learners combined produce a strong result.
