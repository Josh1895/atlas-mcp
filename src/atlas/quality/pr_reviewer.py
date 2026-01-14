"""LLM-based PR reviewer for quality assessment.

Uses a DIFFERENT model than the generators to provide a "senior dev" review.
This acts as a tiebreaker, not the primary decision maker.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PRReviewResult:
    """Result of LLM PR review."""

    patch_id: str

    # Overall assessment
    verdict: str = "needs_discussion"  # "approve", "request_changes", "needs_discussion"
    quality_score: float = 5.0  # 0-10
    confidence: float = 0.5  # 0-1

    # Detailed feedback
    top_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Specific checks
    follows_best_practices: bool = True
    matches_codebase_style: bool = True
    has_code_smells: bool = False
    is_over_engineered: bool = False
    is_hacky: bool = False


@dataclass
class PairwiseResult:
    """Result of pairwise comparison."""

    winner_id: str
    loser_id: str
    reason: str
    confidence: float


class LLMPRReviewer:
    """Uses an LLM to review patches like a senior developer.

    IMPORTANT: Use a DIFFERENT model than the one that generated the patches
    to avoid self-bias.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gemini-3-flash-preview",  # Could use a different model
    ):
        """Initialize the PR reviewer.

        Args:
            llm_client: The LLM client to use for reviews
            model: Model name to use (should differ from generators)
        """
        self.llm_client = llm_client
        self.model = model

    async def review(
        self,
        patch: str,
        patch_id: str,
        issue_description: str,
        surrounding_code: str = "",
        repo_style_guide: Optional[str] = None,
    ) -> PRReviewResult:
        """Review a patch as a senior developer would.

        Args:
            patch: The unified diff patch
            patch_id: Identifier for the patch
            issue_description: Description of the issue being fixed
            surrounding_code: Context code from the repository
            repo_style_guide: Optional repository style guide

        Returns:
            PRReviewResult with the review assessment
        """
        # Truncate long patches
        truncated_patch = patch[:3000] if len(patch) > 3000 else patch
        truncated_context = surrounding_code[:2000] if len(surrounding_code) > 2000 else surrounding_code

        style_section = ""
        if repo_style_guide:
            style_section = f"\n## Repository Style Guide\n{repo_style_guide}"

        prompt = f"""You are a senior software engineer reviewing a pull request.

## The Issue Being Fixed
{issue_description}

## The Patch
```diff
{truncated_patch}
```

## Surrounding Code Context
```
{truncated_context}
```
{style_section}

## Your Task
Review this patch as if it were a PR from a colleague. Be critical but fair.

Respond in this exact JSON format:
```json
{{
    "verdict": "approve" | "request_changes" | "needs_discussion",
    "quality_score": <0-10>,
    "confidence": <0.0-1.0>,
    "top_issues": ["issue 1", "issue 2", "issue 3"],
    "strengths": ["strength 1", "strength 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "follows_best_practices": true | false,
    "matches_codebase_style": true | false,
    "has_code_smells": true | false,
    "is_over_engineered": true | false,
    "is_hacky": true | false
}}
```

Focus on:
1. Does this fix the issue correctly?
2. Is this the idiomatic way to solve this in this codebase/framework?
3. Are there any red flags (silent exceptions, hacky workarounds, etc.)?
4. Would you approve this PR as-is?

JSON response:"""

        try:
            result = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,  # Lower temperature for consistent reviews
            )

            return self._parse_response(result.text, patch_id)

        except Exception as e:
            logger.error(f"PR review failed for {patch_id}: {e}")
            return PRReviewResult(
                patch_id=patch_id,
                verdict="needs_discussion",
                quality_score=5.0,
                confidence=0.3,
                top_issues=["Review failed"],
            )

    def _parse_response(self, response: str, patch_id: str) -> PRReviewResult:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            return PRReviewResult(
                patch_id=patch_id,
                verdict=data.get("verdict", "needs_discussion"),
                quality_score=float(data.get("quality_score", 5)),
                confidence=float(data.get("confidence", 0.5)),
                top_issues=data.get("top_issues", [])[:3],
                strengths=data.get("strengths", [])[:3],
                suggestions=data.get("suggestions", [])[:3],
                follows_best_practices=data.get("follows_best_practices", True),
                matches_codebase_style=data.get("matches_codebase_style", True),
                has_code_smells=data.get("has_code_smells", False),
                is_over_engineered=data.get("is_over_engineered", False),
                is_hacky=data.get("is_hacky", False),
            )

        except Exception as e:
            logger.warning(f"Failed to parse review response: {e}")
            return PRReviewResult(
                patch_id=patch_id,
                verdict="needs_discussion",
                quality_score=5.0,
                confidence=0.3,
                top_issues=["Failed to parse review"],
            )


class TournamentReviewer:
    """Runs a tournament-style comparison between patches.

    More consistent than absolute scoring because LLMs are
    better at relative comparisons.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gemini-3-flash-preview",
    ):
        """Initialize the tournament reviewer.

        Args:
            llm_client: The LLM client to use
            model: Model name to use
        """
        self.llm_client = llm_client
        self.model = model

    async def run_tournament(
        self,
        patches: List[Tuple[str, str]],  # [(patch_id, patch_content), ...]
        issue_description: str,
        context: str = "",
    ) -> List[str]:
        """Run tournament and return patches ranked best to worst.

        Args:
            patches: List of (patch_id, patch_content) tuples
            issue_description: Description of the issue being fixed
            context: Additional context code

        Returns:
            List of patch_ids ordered from best to worst
        """
        if len(patches) <= 1:
            return [p[0] for p in patches]

        # Track wins for each patch
        wins = {patch_id: 0 for patch_id, _ in patches}

        # Round-robin: compare each pair
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                result = await self._compare_pair(
                    patches[i],
                    patches[j],
                    issue_description,
                    context,
                )
                wins[result.winner_id] += 1

        # Sort by wins (descending)
        ranked = sorted(wins.keys(), key=lambda x: wins[x], reverse=True)
        return ranked

    async def _compare_pair(
        self,
        patch_a: Tuple[str, str],
        patch_b: Tuple[str, str],
        issue_description: str,
        context: str,
    ) -> PairwiseResult:
        """Compare two patches head-to-head.

        Args:
            patch_a: First (patch_id, patch_content) tuple
            patch_b: Second (patch_id, patch_content) tuple
            issue_description: The issue being fixed
            context: Additional context

        Returns:
            PairwiseResult with the winner
        """
        id_a, content_a = patch_a
        id_b, content_b = patch_b

        # Randomize order to avoid position bias
        if random.random() < 0.5:
            first, second = (id_a, content_a), (id_b, content_b)
            first_label, second_label = "A", "B"
        else:
            first, second = (id_b, content_b), (id_a, content_a)
            first_label, second_label = "B", "A"

        # Truncate for prompt
        first_content = first[1][:2000] if len(first[1]) > 2000 else first[1]
        second_content = second[1][:2000] if len(second[1]) > 2000 else second[1]
        context_truncated = context[:1000] if len(context) > 1000 else context

        prompt = f"""You are a senior engineer choosing between two patches.

## Issue Being Fixed
{issue_description}

## Patch A
```diff
{first_content}
```

## Patch B
```diff
{second_content}
```

## Context
{context_truncated}

Which patch is better? Consider:
- Correctness (both pass tests, but which is more robust?)
- Code quality (cleaner, more readable, more maintainable)
- Best practices (idiomatic, follows conventions)
- Avoiding hacks (no workarounds, no code smells)

Respond with ONLY this JSON:
```json
{{
    "winner": "A" | "B",
    "reason": "brief explanation",
    "confidence": <0.5-1.0>
}}
```

JSON:"""

        try:
            result = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2,
            )

            # Parse response
            json_match = re.search(r"\{[\s\S]*\}", result.text)
            if json_match:
                data = json.loads(json_match.group())

                winner_label = data.get("winner", "A")

                # Map back to actual IDs
                if winner_label == first_label:
                    winner_id, loser_id = first[0], second[0]
                else:
                    winner_id, loser_id = second[0], first[0]

                return PairwiseResult(
                    winner_id=winner_id,
                    loser_id=loser_id,
                    reason=data.get("reason", ""),
                    confidence=float(data.get("confidence", 0.7)),
                )

        except Exception as e:
            logger.warning(f"Pairwise comparison failed: {e}")

        # Default to first on failure
        return PairwiseResult(
            winner_id=first[0],
            loser_id=second[0],
            reason="Parse failure, defaulted",
            confidence=0.5,
        )


async def label_approach_family(
    patches: List[str],
    llm_client: Any,
) -> str:
    """Use LLM to generate a human-readable label for an approach family.

    This is for understanding only, not for selection decisions.

    Args:
        patches: List of patch contents in this family
        llm_client: The LLM client to use

    Returns:
        A short descriptive label for the approach
    """
    # Use first patch as representative
    representative = patches[0][:2000] if patches else ""

    prompt = f"""Look at this code patch and describe the approach in one short phrase (under 10 words).

Examples of good labels:
- "added null check before access"
- "rewrote query to use join"
- "wrapped call in try-except"
- "added input validation"
- "refactored to use helper method"

Patch:
```
{representative}
```

Label:"""

    try:
        result = await llm_client.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.3,
        )

        # Clean up response
        label = result.text.strip().strip('"').strip("'")
        return label[:50]  # Cap length

    except Exception as e:
        logger.warning(f"Failed to label approach family: {e}")
        return "unknown approach"
