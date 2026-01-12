#!/usr/bin/env python3
"""Test ATLAS on a real SWE-bench style problem.

Based on SymPy Issue #17309: "ceiling(pos) > 0 should be true"
https://github.com/sympy/sympy/issues/17309

This is a simplified version of the actual SymPy problem that tests
ATLAS's ability to:
1. Understand symbolic math semantics
2. Implement comparison operators correctly
3. Handle multiple related methods that need changes
"""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.core.orchestrator import ATLASOrchestrator
from atlas.core.task import TaskSubmission


# This is a simplified version of SymPy's floor/ceiling with the SAME BUG
# The real SymPy code is much larger, but this captures the essence
BUGGY_CODE = '''"""Symbolic floor and ceiling functions with comparison bug."""

from abc import ABC, abstractmethod


class Symbol:
    """A symbolic variable with optional assumptions."""

    def __init__(self, name: str, **assumptions):
        self.name = name
        self.assumptions = assumptions

    @property
    def is_positive(self) -> bool | None:
        return self.assumptions.get('positive')

    @property
    def is_negative(self) -> bool | None:
        return self.assumptions.get('negative')

    @property
    def is_nonnegative(self) -> bool | None:
        if self.is_positive:
            return True
        return self.assumptions.get('nonnegative')

    @property
    def is_nonpositive(self) -> bool | None:
        if self.is_negative:
            return True
        return self.assumptions.get('nonpositive')

    @property
    def is_real(self) -> bool | None:
        return self.assumptions.get('real', True)

    def __repr__(self):
        return self.name


class RoundingFunction(ABC):
    """Base class for floor and ceiling."""

    def __init__(self, arg):
        self.arg = arg

    @abstractmethod
    def _direction(self) -> str:
        """Return 'floor' or 'ceiling'."""
        pass

    # BUG: These comparison methods are missing or incomplete!
    # The bug is that comparisons like ceiling(pos) > 0 don't evaluate to True
    # when mathematically they should.

    # Currently returns unevaluated comparison (the bug)
    def __lt__(self, other):
        return (self, '<', other)  # Returns tuple instead of evaluating

    def __le__(self, other):
        return (self, '<=', other)  # Returns tuple instead of evaluating

    def __gt__(self, other):
        return (self, '>', other)  # Returns tuple instead of evaluating

    def __ge__(self, other):
        return (self, '>=', other)  # Returns tuple instead of evaluating


class floor(RoundingFunction):
    """
    Floor function - returns the largest integer <= arg.

    BUG: Currently does not properly evaluate comparisons.

    Expected behavior:
    - floor(negative) < 0 should return True
    - floor(positive) >= 0 should return True
    - floor(nonnegative) >= 0 should return True

    But currently these return unevaluated tuple comparisons.
    """

    def _direction(self) -> str:
        return 'floor'

    def __repr__(self):
        return f"floor({self.arg})"

    # Missing: _eval_is_negative, _eval_is_nonnegative methods
    # Missing: proper __lt__, __le__, __gt__, __ge__ implementations


class ceiling(RoundingFunction):
    """
    Ceiling function - returns the smallest integer >= arg.

    BUG: Currently does not properly evaluate comparisons.

    Expected behavior:
    - ceiling(positive) > 0 should return True
    - ceiling(negative) <= 0 should return True
    - ceiling(nonpositive) <= 0 should return True

    But currently these return unevaluated tuple comparisons.
    """

    def _direction(self) -> str:
        return 'ceiling'

    def __repr__(self):
        return f"ceiling({self.arg})"

    # Missing: _eval_is_positive, _eval_is_nonpositive methods
    # Missing: proper __lt__, __le__, __gt__, __ge__ implementations


# Test cases that should work after fix
def test_floor_ceiling_comparisons():
    """Test that floor/ceiling comparisons evaluate correctly."""
    pos = Symbol('pos', positive=True, real=True)
    neg = Symbol('neg', negative=True, real=True)
    nn = Symbol('nn', nonnegative=True, real=True)
    np = Symbol('np', nonpositive=True, real=True)

    # These should all return True after fix
    tests = [
        (ceiling(pos) > 0, True, "ceiling(positive) > 0"),
        (ceiling(pos) >= 1, True, "ceiling(positive) >= 1"),
        (floor(neg) < 0, True, "floor(negative) < 0"),
        (floor(neg) <= -1, True, "floor(negative) <= -1"),
        (floor(pos) >= 0, True, "floor(positive) >= 0"),
        (floor(nn) >= 0, True, "floor(nonnegative) >= 0"),
        (ceiling(neg) <= 0, True, "ceiling(negative) <= 0"),
        (ceiling(np) <= 0, True, "ceiling(nonpositive) <= 0"),
    ]

    print("Running floor/ceiling comparison tests:")
    passed = 0
    failed = 0

    for result, expected, desc in tests:
        if result == expected:
            print(f"  PASS: {desc}")
            passed += 1
        else:
            print(f"  FAIL: {desc} - got {result}, expected {expected}")
            failed += 1

    print(f"\\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    test_floor_ceiling_comparisons()
'''


def create_test_repo(temp_dir: Path) -> tuple[str, str]:
    """Create a temporary git repository with the buggy code."""
    repo_dir = temp_dir / "sympy_test"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Write the buggy code
    code_file = repo_dir / "integers.py"
    code_file.write_text(BUGGY_CODE)

    # Initialize git repo with main branch
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit with buggy floor/ceiling"], cwd=repo_dir, capture_output=True)

    result = subprocess.run(["git", "branch", "--show-current"], cwd=repo_dir, capture_output=True, text=True)
    branch = result.stdout.strip() or "main"

    return str(repo_dir), branch


async def main():
    print("=" * 100)
    print(" ATLAS SWE-BENCH STYLE TEST")
    print(" Based on SymPy Issue #17309: ceiling(pos) > 0 should be true")
    print(" https://github.com/sympy/sympy/issues/17309")
    print("=" * 100)

    config = Config.from_env()

    errors = config.validate()
    if errors:
        print(f"\nConfiguration errors: {errors}")
        return

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Voting K: {config.voting_k}")
    print(f"  Max Samples: {config.max_samples}")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        print(f"\nCreating test repository in: {temp_dir}")
        repo_path, branch = create_test_repo(temp_dir)
        print(f"Test repo created at: {repo_path}")
        print(f"Branch: {branch}")

        # The problem statement - mimics what an AI assistant would receive
        problem_statement = """Fix the floor and ceiling comparison operators in integers.py.

## Problem
The `floor` and `ceiling` classes have broken comparison operators. Currently:
- `ceiling(positive_symbol) > 0` returns a tuple instead of `True`
- `floor(negative_symbol) < 0` returns a tuple instead of `True`

## Mathematical Background
- `ceiling(x)` returns the smallest integer >= x
- `floor(x)` returns the largest integer <= x

Therefore:
- If x is positive, ceiling(x) >= 1, so ceiling(x) > 0 is always True
- If x is negative, floor(x) <= -1, so floor(x) < 0 is always True
- If x is nonnegative, floor(x) >= 0 is always True
- If x is nonpositive, ceiling(x) <= 0 is always True

## Required Fix
1. Override `__lt__`, `__le__`, `__gt__`, `__ge__` in both `floor` and `ceiling` classes
2. Check the argument's assumptions (is_positive, is_negative, is_nonnegative, is_nonpositive)
3. Return True/False when the comparison can be determined mathematically
4. Return the tuple (unevaluated) only when the comparison cannot be determined

## Tests That Should Pass After Fix
```python
pos = Symbol('pos', positive=True)
neg = Symbol('neg', negative=True)

assert ceiling(pos) > 0 == True
assert floor(neg) < 0 == True
assert floor(pos) >= 0 == True
assert ceiling(neg) <= 0 == True
```

Generate a unified diff patch to fix all comparison operators in both floor and ceiling classes."""

        task = TaskSubmission(
            description=problem_statement,
            repository_url=repo_path,
            branch=branch,
            relevant_files=["integers.py"],
            max_cost_usd=2.0,
            timeout_minutes=15,
            initial_samples=10,  # Full 10 agents
            max_samples=15,
            voting_k=3,
        )

        print(f"\nTask ID: {task.task_id}")
        print(f"Initial samples: {task.initial_samples}")
        print(f"Max samples: {task.max_samples}")
        print(f"Voting K: {task.voting_k}")

        print("\n" + "=" * 100)
        print(" STARTING ORCHESTRATOR")
        print(" 10 Agents will autonomously search Context7 AND the web")
        print(" Problem: SymPy floor/ceiling comparison (SWE-bench style)")
        print("=" * 100)

        orchestrator = ATLASOrchestrator(
            config=config,
            enable_quality_selection=True,
            use_agentic=True,
        )

        print("\nRunning solve()...")
        print("(This will take several minutes as agents research Python comparison operators,")
        print(" symbolic math semantics, and SymPy best practices)\n")

        try:
            result = await asyncio.wait_for(
                orchestrator.solve(task),
                timeout=task.timeout_minutes * 60,
            )

            print("\n" + "=" * 100)
            print(" RESULTS")
            print("=" * 100)

            print(f"\nStatus: {result.status.value}")
            print(f"Consensus Reached: {result.consensus_reached}")
            print(f"Confidence Score: {result.confidence_score:.2%}")
            print(f"Samples Generated: {result.samples_generated}")
            print(f"Cost: ${result.cost_usd:.4f}")
            print(f"Duration: {result.duration_seconds:.1f}s")

            if result.patch:
                print(f"\nWinning Patch ({len(result.patch)} chars):")
                print("-" * 80)
                print(result.patch)
                print("-" * 80)

                # Check for key patterns that should be in a correct fix
                patch_content = result.patch.lower()
                patterns = {
                    "__lt__": "__lt__" in result.patch,
                    "__le__": "__le__" in result.patch,
                    "__gt__": "__gt__" in result.patch,
                    "__ge__": "__ge__" in result.patch,
                    "is_positive": "is_positive" in result.patch,
                    "is_negative": "is_negative" in result.patch,
                    "return True": "return true" in patch_content or "return True" in result.patch,
                }

                print("\nKey patterns in patch:")
                all_found = True
                for pattern, found in patterns.items():
                    status = "YES" if found else "NO"
                    if not found:
                        all_found = False
                    print(f"  {pattern}: {status}")

                if all_found:
                    print("\n*** EXCELLENT: Patch contains all expected patterns! ***")
                else:
                    print("\n*** WARNING: Some expected patterns missing ***")
            else:
                print("\nNo patch generated!")
                if result.error_message:
                    print(f"Error: {result.error_message}")

            # Show execution trace
            if result.execution_trace:
                trace = result.execution_trace
                print(f"\nExecution Trace:")
                print(f"  Phases: {[p['phase'] for p in trace.phases]}")
                print(f"  Agent outputs: {len(trace.agent_outputs)}")
                print(f"  Voting rounds: {len(trace.voting_rounds)}")

                if trace.voting_rounds:
                    print(f"\n  Last voting round:")
                    last_round = trace.voting_rounds[-1]
                    print(f"    Clusters: {last_round.get('clusters', {})}")
                    print(f"    Winner: {last_round.get('winner')}")
                    print(f"    Consensus: {last_round.get('consensus_reached')}")

                if trace.errors:
                    print(f"\n  Errors: {trace.errors}")

        except asyncio.TimeoutError:
            print(f"\nTask timed out after {task.timeout_minutes} minutes")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    finally:
        print(f"\nCleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 100)
    print(" TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
