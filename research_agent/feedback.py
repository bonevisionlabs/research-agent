"""
Research Agent Toolkit -- Self-Learning Feedback Loop.

Provides a review-learn-improve cycle for iterative paper writing.
The orchestrator (Claude Code) writes a section, reviews it against
criteria, records lessons, and applies accumulated knowledge to
subsequent tasks.

Key concepts
------------
* **ReviewResult** -- multi-dimensional quality scores for a task's output.
* **Lesson** -- a reusable insight extracted from a review (anti-pattern +
  correction).  Confidence grows each time the lesson is reinforced.
* **FeedbackLoop** -- manages the full cycle: record reviews, accumulate
  lessons, decide whether to retry, and surface relevant guidance before
  each task.

All state persists to JSON on disk.  Zero API calls -- pure computation.

Usage
-----
    from research_agent.feedback import FeedbackLoop, create_review, create_lesson

    loop = FeedbackLoop(state_dir=Path("state/feedback"))

    # After completing a task, review it
    review = create_review(
        task_id="write_intro",
        iteration=0,
        scores={"methodology": 0.8, "writing_quality": 0.6, "completeness": 0.9},
    )
    review.strengths = ["Clear problem statement"]
    review.weaknesses = ["Gap statement too vague"]
    review.suggestions = ["Add specific metrics comparing prior work"]
    loop.record_review(review)

    # Extract a lesson
    lesson = create_lesson(
        source_task="write_intro",
        category="writing_quality",
        pattern="Gap statement lacks quantitative comparison with prior work",
        correction="Include specific metrics (e.g., accuracy, latency) when describing limitations of existing approaches",
    )
    loop.record_lesson(lesson)

    # Before starting the next task, get guidance
    guidance = loop.get_task_guidance("write_discussion")

    # Check if a task should be retried
    if loop.should_retry("write_intro"):
        # workflow.retry_task("write_intro")
        pass

    loop.save()
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import review_criteria as rc


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    """Multi-dimensional quality assessment of a task's output.

    Attributes
    ----------
    task_id : str
        The workflow task that was reviewed.
    iteration : int
        Which attempt this review covers (0 = first try).
    scores : dict[str, float]
        Criterion name to score (0.0--1.0).  Keys should match
        ``REVIEW_CRITERIA`` names (e.g., ``"methodology"``,
        ``"writing_quality"``).
    weighted_score : float
        Overall quality score (weighted average of *scores*).
    passed : bool
        Whether *weighted_score* meets the quality threshold.
    strengths : list[str]
        What the output did well.
    weaknesses : list[str]
        What needs improvement.
    suggestions : list[str]
        Actionable next steps for revision.
    timestamp : str
        ISO-8601 creation time.
    """

    task_id: str
    iteration: int
    scores: Dict[str, float]
    weighted_score: float = 0.0
    passed: bool = False
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewResult:
        return cls(**data)


@dataclass
class Lesson:
    """A reusable insight extracted from a review.

    Lessons accumulate over iterations.  Confidence starts at 0.5 and
    increases each time :meth:`FeedbackLoop.reinforce_lesson` is called,
    meaning the lesson has been validated again.

    Attributes
    ----------
    id : str
        UUID identifier.
    source_task : str
        The task this lesson originated from.
    category : str
        Review criterion category (e.g., ``"writing_quality"``,
        ``"methodology"``, ``"results"``).
    pattern : str
        The anti-pattern or mistake observed.
    correction : str
        How to fix or avoid the pattern.
    confidence : float
        0.0--1.0.  Increases with reinforcement, decreases with
        deprecation.
    times_applied : int
        How many times this lesson was surfaced as guidance.
    created_at : str
        ISO-8601 timestamp.
    last_used : str
        ISO-8601 timestamp of last application.
    """

    id: str
    source_task: str
    category: str
    pattern: str
    correction: str
    confidence: float = 0.5
    times_applied: int = 0
    created_at: str = ""
    last_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Lesson:
        return cls(**data)


# ---------------------------------------------------------------------------
# FeedbackLoop manager
# ---------------------------------------------------------------------------

class FeedbackLoop:
    """Manages the review-learn-improve cycle.

    All state is persisted to ``state_dir`` as two JSON files:
    ``reviews.json`` and ``learnings.json``.

    Parameters
    ----------
    state_dir : Path
        Directory for persisting feedback state.
    quality_threshold : float
        Minimum *weighted_score* for a review to pass (default 0.7).
    max_iterations : int
        Maximum retry attempts before accepting output (default 3).
    """

    def __init__(
        self,
        state_dir: Path,
        quality_threshold: float = 0.7,
        max_iterations: int = 3,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

        self._reviews: List[ReviewResult] = []
        self._lessons: List[Lesson] = []

        # Auto-load if state exists
        if (self.state_dir / "reviews.json").exists():
            self.load()

    # -- Review management --------------------------------------------------

    def record_review(self, review: ReviewResult) -> None:
        """Record a review result."""
        self._reviews.append(review)

    def get_reviews(self, task_id: Optional[str] = None) -> List[ReviewResult]:
        """Get reviews, optionally filtered by task."""
        if task_id is None:
            return list(self._reviews)
        return [r for r in self._reviews if r.task_id == task_id]

    def latest_review(self, task_id: str) -> Optional[ReviewResult]:
        """Get the most recent review for a task."""
        task_reviews = self.get_reviews(task_id)
        if not task_reviews:
            return None
        return max(task_reviews, key=lambda r: r.iteration)

    # -- Learning management ------------------------------------------------

    def record_lesson(self, lesson: Lesson) -> None:
        """Add a lesson to the knowledge base."""
        self._lessons.append(lesson)

    def get_lessons(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Lesson]:
        """Retrieve lessons, optionally filtered by category and confidence.

        Returns lessons sorted by confidence (highest first).
        """
        filtered = self._lessons
        if category is not None:
            filtered = [l for l in filtered if l.category == category]
        if min_confidence > 0:
            filtered = [l for l in filtered if l.confidence >= min_confidence]
        return sorted(filtered, key=lambda l: l.confidence, reverse=True)

    def reinforce_lesson(self, lesson_id: str) -> None:
        """Increase confidence for a validated lesson.

        Bumps confidence by 0.1 (capped at 1.0) and increments
        ``times_applied``.
        """
        for lesson in self._lessons:
            if lesson.id == lesson_id:
                lesson.confidence = min(1.0, lesson.confidence + 0.1)
                lesson.times_applied += 1
                lesson.last_used = _now_iso()
                return
        raise KeyError(f"Lesson '{lesson_id}' not found.")

    def deprecate_lesson(self, lesson_id: str) -> None:
        """Decrease confidence for a lesson that didn't help.

        Drops confidence by 0.15 (floored at 0.0).
        """
        for lesson in self._lessons:
            if lesson.id == lesson_id:
                lesson.confidence = max(0.0, lesson.confidence - 0.15)
                return
        raise KeyError(f"Lesson '{lesson_id}' not found.")

    # -- Decision support ---------------------------------------------------

    def should_retry(self, task_id: str) -> bool:
        """Decide whether a task should be retried.

        Returns ``True`` if the latest review failed AND the iteration
        count is below ``max_iterations``.
        """
        review = self.latest_review(task_id)
        if review is None:
            return False
        return not review.passed and review.iteration < self.max_iterations

    def get_task_guidance(
        self,
        task_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compile guidance for a task based on accumulated knowledge.

        Returns a dict with:
        - ``past_reviews``: previous reviews for this task (if retrying)
        - ``relevant_lessons``: lessons matching the task's agent category
        - ``checklist``: applicable review criteria
        - ``iteration``: which attempt this will be

        Parameters
        ----------
        task_id : str
            The task about to be started.
        agent : str, optional
            Agent role (used to filter lessons by category).
        """
        past_reviews = self.get_reviews(task_id)
        latest = self.latest_review(task_id)
        iteration = (latest.iteration + 1) if latest else 0

        # Gather relevant lessons
        relevant_lessons: List[Dict] = []

        # If we have past reviews, get lessons matching weak categories
        if latest and latest.weaknesses:
            weak_categories = [
                cat for cat, score in latest.scores.items()
                if score < self.quality_threshold
            ]
            for cat in weak_categories:
                for lesson in self.get_lessons(category=cat, min_confidence=0.3):
                    relevant_lessons.append(lesson.to_dict())
        else:
            # No past reviews: surface high-confidence general lessons
            for lesson in self.get_lessons(min_confidence=0.6):
                relevant_lessons.append(lesson.to_dict())

        # Map agent role to primary criteria categories
        _agent_criteria = {
            "literature": ["novelty", "completeness"],
            "evaluation": ["results", "methodology"],
            "figures": ["figures"],
            "writer": ["writing_quality", "completeness", "methodology"],
            "reviewer": ["completeness", "results", "writing_quality"],
        }
        applicable_criteria = _agent_criteria.get(agent or "", list(rc.REVIEW_CRITERIA.keys()))

        return {
            "task_id": task_id,
            "iteration": iteration,
            "past_reviews": [r.to_dict() for r in past_reviews],
            "relevant_lessons": relevant_lessons,
            "applicable_criteria": {
                k: rc.REVIEW_CRITERIA[k]
                for k in applicable_criteria
                if k in rc.REVIEW_CRITERIA
            },
            "quality_threshold": self.quality_threshold,
        }

    # -- Reporting ----------------------------------------------------------

    def iteration_summary(self) -> Dict[str, Any]:
        """Summary of all reviews and retry progress.

        Returns
        -------
        dict
            Keys: ``total_reviews``, ``tasks_reviewed``, ``tasks_passed``,
            ``tasks_failed``, ``tasks_retrying``, ``average_score``,
            ``per_task`` breakdown.
        """
        per_task: Dict[str, Dict] = {}

        for review in self._reviews:
            tid = review.task_id
            if tid not in per_task or review.iteration > per_task[tid]["iteration"]:
                per_task[tid] = {
                    "iteration": review.iteration,
                    "weighted_score": review.weighted_score,
                    "passed": review.passed,
                }

        tasks_passed = sum(1 for t in per_task.values() if t["passed"])
        tasks_failed = sum(1 for t in per_task.values() if not t["passed"])
        scores = [t["weighted_score"] for t in per_task.values()]
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0

        return {
            "total_reviews": len(self._reviews),
            "tasks_reviewed": len(per_task),
            "tasks_passed": tasks_passed,
            "tasks_failed": tasks_failed,
            "average_score": avg,
            "per_task": per_task,
        }

    def learning_summary(self) -> Dict[str, Any]:
        """Summary of accumulated lessons.

        Returns
        -------
        dict
            Keys: ``total_lessons``, ``by_category`` (with counts and
            avg confidence), ``top_lessons`` (top 5 by confidence).
        """
        by_category: Dict[str, Dict] = {}
        for lesson in self._lessons:
            cat = lesson.category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "total_confidence": 0.0}
            by_category[cat]["count"] += 1
            by_category[cat]["total_confidence"] += lesson.confidence

        for cat_data in by_category.values():
            cat_data["avg_confidence"] = round(
                cat_data["total_confidence"] / cat_data["count"], 3
            )
            del cat_data["total_confidence"]

        top = sorted(self._lessons, key=lambda l: l.confidence, reverse=True)[:5]

        return {
            "total_lessons": len(self._lessons),
            "by_category": by_category,
            "top_lessons": [l.to_dict() for l in top],
        }

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        """Persist reviews and lessons to JSON files."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

        reviews_path = self.state_dir / "reviews.json"
        reviews_path.write_text(
            json.dumps(
                [r.to_dict() for r in self._reviews],
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

        learnings_path = self.state_dir / "learnings.json"
        learnings_path.write_text(
            json.dumps(
                [l.to_dict() for l in self._lessons],
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

    def load(self) -> None:
        """Restore reviews and lessons from JSON files."""
        reviews_path = self.state_dir / "reviews.json"
        if reviews_path.exists():
            data = json.loads(reviews_path.read_text(encoding="utf-8"))
            self._reviews = [ReviewResult.from_dict(d) for d in data]

        learnings_path = self.state_dir / "learnings.json"
        if learnings_path.exists():
            data = json.loads(learnings_path.read_text(encoding="utf-8"))
            self._lessons = [Lesson.from_dict(d) for d in data]

    def clear(self) -> None:
        """Reset all reviews and lessons (useful in tests)."""
        self._reviews.clear()
        self._lessons.clear()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_review(
    task_id: str,
    iteration: int,
    scores: Dict[str, float],
    criteria: Optional[Dict[str, Dict]] = None,
    threshold: float = 0.7,
) -> ReviewResult:
    """Create a ReviewResult with computed weighted score.

    Parameters
    ----------
    task_id : str
        The workflow task being reviewed.
    iteration : int
        Which attempt (0 = first try).
    scores : dict[str, float]
        Criterion name to score (0.0--1.0).
    criteria : dict, optional
        Criteria dict with weights.  Defaults to ``REVIEW_CRITERIA``.
    threshold : float
        Passing threshold (default 0.7).

    Returns
    -------
    ReviewResult
        With ``weighted_score`` and ``passed`` computed.
        Fill in ``strengths``, ``weaknesses``, ``suggestions`` manually.
    """
    if criteria is None:
        criteria = rc.REVIEW_CRITERIA

    # Compute weighted average (only for scored criteria)
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in scores.items():
        weight = criteria.get(name, {}).get("weight", 0.0)
        weighted_sum += score * weight
        total_weight += weight

    weighted_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    return ReviewResult(
        task_id=task_id,
        iteration=iteration,
        scores=scores,
        weighted_score=weighted_score,
        passed=weighted_score >= threshold,
        timestamp=_now_iso(),
    )


def create_lesson(
    source_task: str,
    category: str,
    pattern: str,
    correction: str,
    confidence: float = 0.5,
) -> Lesson:
    """Create a Lesson with a generated UUID and timestamps.

    Parameters
    ----------
    source_task : str
        Task this lesson originated from.
    category : str
        Review criterion category (e.g., ``"writing_quality"``).
    pattern : str
        What was wrong (the anti-pattern observed).
    correction : str
        How to fix or avoid it in future.
    confidence : float
        Initial confidence (default 0.5).
    """
    now = _now_iso()
    return Lesson(
        id=str(uuid.uuid4())[:8],
        source_task=source_task,
        category=category,
        pattern=pattern,
        correction=correction,
        confidence=confidence,
        times_applied=0,
        created_at=now,
        last_used=now,
    )


def build_review_checklist(
    section_name: str,
    criteria: Optional[Dict[str, Dict]] = None,
    checklist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a section-specific review checklist.

    Combines review criteria (with section-appropriate weight boosts)
    and completeness checklist items relevant to the section.

    Parameters
    ----------
    section_name : str
        Section being reviewed (e.g., ``"methods"``, ``"results"``).
    criteria : dict, optional
        Base criteria.  Defaults to ``REVIEW_CRITERIA``.
    checklist : list[str], optional
        Base checklist.  Defaults to ``COMPLETENESS_CHECKLIST``.

    Returns
    -------
    dict
        Keys: ``section``, ``criteria`` (with section weights),
        ``checklist`` (filtered items), ``guidelines`` (from
        ``SECTION_GUIDELINES``).
    """
    if criteria is None:
        criteria = rc.REVIEW_CRITERIA
    if checklist is None:
        checklist = rc.COMPLETENESS_CHECKLIST

    # Map sections to their primary criteria for weight boosting
    _section_emphasis: Dict[str, List[str]] = {
        "abstract": ["completeness", "writing_quality"],
        "introduction": ["novelty", "writing_quality"],
        "related_work": ["completeness", "novelty"],
        "methods": ["methodology", "completeness"],
        "experiments": ["methodology", "results"],
        "results": ["results", "figures"],
        "discussion": ["impact", "writing_quality"],
        "conclusion": ["writing_quality", "impact"],
    }

    # Boost weights for emphasized criteria
    import copy
    adjusted = copy.deepcopy(criteria)
    emphasis = _section_emphasis.get(section_name, [])
    for crit_name in emphasis:
        if crit_name in adjusted:
            adjusted[crit_name]["weight"] *= 1.5

    # Normalize weights
    total_weight = sum(c["weight"] for c in adjusted.values())
    if total_weight > 0:
        for c in adjusted.values():
            c["weight"] = round(c["weight"] / total_weight, 4)

    # Filter checklist to section-relevant items
    section_label = section_name.replace("_", " ").title()
    relevant_items = [
        item for item in checklist
        if section_label in item
        or item.startswith("All ")
        or item.startswith("Figures")
        or item.startswith("References")
    ]

    # Get guidelines if available
    guidelines = rc.SECTION_GUIDELINES.get(section_name, {})

    return {
        "section": section_name,
        "criteria": adjusted,
        "checklist": relevant_items if relevant_items else checklist,
        "guidelines": guidelines,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
