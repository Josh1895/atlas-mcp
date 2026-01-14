#!/usr/bin/env python3
"""Full ATLAS Integration Test - Complete Pipeline.

Integrates:
1. AgenticGeminiClient - Autonomous Context7 tool access
2. SimilarityClustering - Groups solutions by patch similarity
3. VotingManager - First-to-ahead-by-K consensus voting
4. Quality selection - Picks the best solution

This is the complete ATLAS system working end-to-end.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas.core.config import Config
from atlas.core.task import Solution
from atlas.agents.agentic_client import AgenticGeminiClient
from atlas.agents.prompt_styles import ALL_STYLES
from atlas.verification.clustering import SimilarityClustering, extract_patch_signature
from atlas.voting.consensus import VotingManager, IncrementalVoter

# The buggy code to fix
BUGGY_CODE = '''import time

class AsyncRateLimiter:
    def __init__(self, rate: float = 10.0):
        self.tokens = 5
        self.rate = rate
        self.last_update = time.time()
        self._lock = None  # BUG: Never initialized!

    async def acquire(self):
        # BUG: Race condition - no lock protection!
        now = time.time()
        self.tokens += (now - self.last_update) * self.rate
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def wait_for_token(self):
        while not await self.acquire():
            time.sleep(0.01)  # BUG: Blocking call in async!
        return True'''

ISSUE = """Fix the following 3 bugs:
1. Race condition - multiple coroutines can access self.tokens simultaneously without synchronization
2. time.sleep(0.01) blocks the event loop - should use async sleep
3. self._lock = None is never initialized as an actual asyncio.Lock()"""

BASE_SYSTEM_PROMPT = """You are an expert Python developer specializing in async/await patterns.
Research the relevant documentation using the tools provided, then generate a production-ready fix.

IMPORTANT: Output your fix as a unified diff patch in a code block. Example format:
```diff
--- a/file.py
+++ b/file.py
@@ -1,5 +1,6 @@
 existing line
-removed line
+added line
 context line
```"""


def check_patterns(text):
    """Check which best practice patterns are present in the text."""
    text_lower = text.lower()
    return {
        "time.monotonic": "monotonic" in text_lower,
        "asyncio.Lock": "asyncio.lock" in text_lower or "asyncio lock" in text_lower,
        "asyncio.sleep": "asyncio.sleep" in text_lower,
        "async with": "async with" in text_lower,
    }


def extract_patch(response: str) -> str:
    """Extract a unified diff patch from the response."""
    import re

    # Look for diff in code blocks
    diff_pattern = r"```(?:diff)?\s*\n((?:---|\+\+\+|@@|[-+ ].*?\n)+)```"
    matches = re.findall(diff_pattern, response, re.MULTILINE | re.DOTALL)

    if matches:
        for match in matches:
            if "---" in match or "+++" in match or "@@" in match:
                return match.strip()

    # Try to find diff without code blocks
    lines = response.split("\n")
    diff_lines = []
    in_diff = False

    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            in_diff = True
            diff_lines.append(line)
        elif in_diff:
            if line.startswith("@@") or line.startswith("+") or line.startswith("-") or line.startswith(" "):
                diff_lines.append(line)
            elif line.strip() == "" and diff_lines:
                diff_lines.append(line)
            else:
                if len(diff_lines) > 3:
                    break
                diff_lines = []
                in_diff = False

    if diff_lines:
        return "\n".join(diff_lines).strip()

    return ""


async def run_agent(agent_client, style, agent_num, delay_seconds=0):
    """Run a single agent with autonomous tool access."""
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    style_name = str(style.name).replace("PromptStyleName.", "")
    agent_id = f"{style_name}_{agent_num}"

    prompt = f"""{style.get_system_prompt(BASE_SYSTEM_PROMPT)}

## Buggy Code
```python
{BUGGY_CODE}
```

## Issue
{ISSUE}

---
IMPORTANT: Use the tools to research Python asyncio documentation before answering.
Then provide a production-ready fix as a unified diff patch."""

    temp = 0.7 + style.temperature_offset + (agent_num * 0.02)

    try:
        result = await agent_client.generate_with_tools(
            prompt=prompt,
            temperature=temp,
            max_tokens=4000,
            max_iterations=8,
        )

        patch = extract_patch(result.text)

        return {
            "agent_id": agent_id,
            "style": style_name,
            "success": True,
            "response": result.text,
            "patch": patch,
            "tool_calls": result.tool_calls,
            "iterations": result.iterations,
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "cost": result.total_cost,
            "patterns": check_patterns(result.text),
            "context7_chars": sum(
                len(tc.get("result_preview", ""))
                for tc in result.tool_calls
                if tc["name"] == "get_library_docs"
            ),
        }
    except Exception as e:
        return {
            "agent_id": agent_id,
            "style": style_name,
            "success": False,
            "error": str(e),
            "response": "",
            "patch": "",
            "tool_calls": [],
            "patterns": {},
            "context7_chars": 0,
        }


async def main():
    print("=" * 100)
    print(" ATLAS FULL INTEGRATION TEST")
    print(" Complete Pipeline: Agentic RAG -> Clustering -> Voting -> Consensus")
    print("=" * 100)

    config = Config.from_env()
    agent_client = AgenticGeminiClient(config)

    # Run 10 agents (2 per style) with staggered delays
    print("\n" + "=" * 100)
    print(" PHASE 1: DEPLOYING AGENTIC SWARM")
    print(" Each agent has autonomous access to Context7 tools")
    print("=" * 100)

    tasks = []
    agent_num = 0
    for style in ALL_STYLES:
        for i in range(2):
            delay = agent_num * 2.0  # 2 second stagger
            tasks.append(run_agent(agent_client, style, i, delay_seconds=delay))
            agent_num += 1

    print(f"\n  Running {len(tasks)} agents with staggered delays...")
    print("  (This may take a few minutes due to rate limiting)\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    agent_results = []
    for r in results:
        if isinstance(r, Exception):
            agent_results.append({
                "agent_id": "Unknown",
                "success": False,
                "error": str(r),
                "response": "",
                "patch": "",
                "tool_calls": [],
                "patterns": {},
                "context7_chars": 0,
            })
        else:
            agent_results.append(r)

    # Show agent summary
    print("\n" + "=" * 100)
    print(" PHASE 1 RESULTS: AGENT RESPONSES")
    print("=" * 100)

    for i, result in enumerate(agent_results):
        print(f"\n{'─' * 80}")
        print(f" Agent {i+1}: {result['agent_id']}")
        print(f"{'─' * 80}")

        if result["success"]:
            print(f"  Tool Calls: {len(result['tool_calls'])}")
            for tc in result["tool_calls"][:3]:  # Show first 3
                print(f"    • {tc['name']}({tc.get('args', {})})")
            if len(result["tool_calls"]) > 3:
                print(f"    ... and {len(result['tool_calls']) - 3} more")

            print(f"  Iterations: {result.get('iterations', 'N/A')}")
            print(f"  Context7 chars: {result['context7_chars']}")
            print(f"  Patch length: {len(result['patch'])} chars")
            print(f"  Patterns: ", end="")
            patterns = [k for k, v in result["patterns"].items() if v]
            print(", ".join(patterns) if patterns else "none found")
        else:
            print(f"  ERROR: {result.get('error', 'Unknown')[:80]}")

    # Create Solution objects for clustering/voting
    print("\n" + "=" * 100)
    print(" PHASE 2: CREATING SOLUTIONS FOR CLUSTERING")
    print("=" * 100)

    solutions = []
    for result in agent_results:
        if result["success"] and result["patch"]:
            solution = Solution(
                agent_id=result["agent_id"],
                prompt_style=result["style"],
                patch=result["patch"],
                explanation=result["response"][:500],
                model="gemini-3-flash-preview",
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0.0),
                rag_sources=[tc["name"] for tc in result["tool_calls"]],
                is_valid=True,
            )
            solutions.append(solution)
            print(f"  + {result['agent_id']}: {len(result['patch'])} char patch")
        else:
            print(f"  - {result['agent_id']}: No valid patch")

    print(f"\n  Total valid solutions: {len(solutions)}/{len(agent_results)}")

    if not solutions:
        print("\n  ERROR: No valid solutions to cluster!")
        return

    # Phase 3: Clustering
    print("\n" + "=" * 100)
    print(" PHASE 3: SIMILARITY CLUSTERING")
    print(" Grouping solutions by patch similarity")
    print("=" * 100)

    clustering = SimilarityClustering(similarity_threshold=0.6)
    clustering_result = clustering.cluster(solutions)
    clustering_result = clustering.merge_similar_clusters(clustering_result)

    print(f"\n  Total clusters: {clustering_result.cluster_count}")
    print(f"  Similarity threshold: {clustering_result.similarity_threshold}")

    for cluster in clustering_result.clusters:
        print(f"\n  Cluster '{cluster.cluster_id}':")
        print(f"    Size: {cluster.size} solutions")
        print(f"    Valid: {cluster.is_valid}")
        print(f"    Members:")
        for sol in cluster.solutions:
            print(f"      - {sol.agent_id} ({sol.prompt_style})")
        if cluster.representative:
            sig = extract_patch_signature(cluster.representative.patch)
            print(f"    Representative: {cluster.representative.agent_id}")
            print(f"    Files modified: {sig.files_modified or 'N/A'}")
            print(f"    Functions touched: {sig.functions_touched or 'N/A'}")

    # Phase 4: Voting
    print("\n" + "=" * 100)
    print(" PHASE 4: CONSENSUS VOTING")
    print(" First-to-ahead-by-K algorithm (k=3)")
    print("=" * 100)

    voter = VotingManager(k=3, similarity_threshold=0.8)
    voting_result = voter.vote(solutions)

    print(f"\n  Total votes: {voting_result.total_votes}")
    print(f"  Vote counts by cluster:")
    for cluster_id, count in sorted(voting_result.vote_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {cluster_id}: {count} votes {bar}")

    print(f"\n  Current leader: {voting_result.winner.cluster_id if voting_result.winner else 'None'}")
    print(f"  Margin: {voting_result.margin}")
    print(f"  Consensus reached: {voting_result.consensus_reached}")
    print(f"  Confidence score: {voting_result.confidence_score:.2%}")

    # Phase 5: Final Selection
    print("\n" + "=" * 100)
    print(" PHASE 5: FINAL SELECTION")
    print("=" * 100)

    if voting_result.winning_solution:
        winner = voting_result.winning_solution
        print(f"\n  WINNER: {winner.agent_id}")
        print(f"  Style: {winner.prompt_style}")
        print(f"  Cluster: {winner.cluster_id}")
        print(f"  Consensus: {'YES' if voting_result.consensus_reached else 'NO (best effort)'}")

        print(f"\n  Winning Patch ({len(winner.patch)} chars):")
        print("  " + "─" * 76)
        for line in winner.patch.split("\n")[:30]:
            print(f"  {line}")
        if len(winner.patch.split("\n")) > 30:
            print(f"  ... [{len(winner.patch.split(chr(10))) - 30} more lines]")

        # Check patterns in winning solution
        winning_patterns = check_patterns(winner.patch + winner.explanation)
        print(f"\n  Patterns in winning solution:")
        for pattern, found in winning_patterns.items():
            status = "✓" if found else "✗"
            print(f"    [{status}] {pattern}")
    else:
        print("\n  No winner could be determined.")

    # Summary statistics
    print("\n" + "=" * 100)
    print(" FINAL STATISTICS")
    print("=" * 100)

    successful = [r for r in agent_results if r["success"]]
    total_cost = sum(r.get("cost", 0) for r in successful)
    total_tool_calls = sum(len(r["tool_calls"]) for r in successful)
    total_ctx7 = sum(r["context7_chars"] for r in successful)

    print(f"\n  Agents deployed: {len(agent_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Valid patches: {len(solutions)}")
    print(f"  Clusters formed: {clustering_result.cluster_count}")
    print(f"  Consensus reached: {voting_result.consensus_reached}")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Total Context7 chars: {total_ctx7}")
    print(f"  Total cost: ${total_cost:.4f}")

    # Pattern adoption across all agents
    print("\n  Pattern adoption (all agents):")
    pattern_names = ["asyncio.Lock", "asyncio.sleep", "async with", "time.monotonic"]
    for pattern in pattern_names:
        count = sum(1 for r in successful if r["patterns"].get(pattern))
        pct = (count / len(successful) * 100) if successful else 0
        bar = "█" * int(pct / 5)
        print(f"    {pattern:<20} {count:>2}/{len(successful)} ({pct:>5.1f}%) {bar}")

    print("\n" + "=" * 100)
    print(" ATLAS INTEGRATION TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
